use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;

use crate::domain::ports::{LlmPort, LlmResponse};
use crate::domain::session::{Message, ToolCall, ToolCallFunction};
use crate::infra::llm::shared::{
    chat_response_to_llm, send_json_with_retry, tool_function, ChatChoice, ChatResponse,
    PromptTokensDetails, ProviderReply, Usage, RESERVED_KEYS,
};

const ANTHROPIC_API_VERSION: &str = "2023-06-01";
const ANTHROPIC_DEFAULT_MAX_TOKENS: u64 = 8192;

/// Client for Anthropic's Messages API (native, non-OpenAI-compatible wire format).
pub struct AnthropicClient {
    pub(crate) client: reqwest::Client,
    pub(crate) base_url: String,
    pub(crate) api_key: String,
    /// What the agent declared, applied after this API's own. Never one of those:
    /// `validate` refuses a definition naming a header Atoma sets.
    pub(crate) extra_headers: Vec<(String, String)>,
}

impl AnthropicClient {
    pub fn new(
        client: reqwest::Client,
        base_url: String,
        api_key: String,
        extra_headers: Vec<(String, String)>,
    ) -> Self {
        AnthropicClient {
            client,
            base_url,
            api_key,
            extra_headers,
        }
    }

    async fn call_anthropic(
        &self,
        model: &str,
        messages: &[Message],
        tools: Option<&[Value]>,
        extra_body: &std::collections::HashMap<String, Value>,
    ) -> Result<ProviderReply<ChatResponse>> {
        let url = format!("{}/v1/messages", self.base_url.trim_end_matches('/'));
        let body = build_request_body(model, messages, tools, extra_body)?;

        tracing::debug!("Request URL: {}", url);
        tracing::debug!("Request body: {}", serde_json::to_string_pretty(&body)?);

        let raw: ProviderReply<AnthropicResponse> = send_json_with_retry("Anthropic", || {
            let mut request = self
                .client
                .post(&url)
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", ANTHROPIC_API_VERSION)
                .header("Content-Type", "application/json");
            for (name, value) in &self.extra_headers {
                request = request.header(name.as_str(), value.as_str());
            }
            request.json(&body)
        })
        .await?;

        // The id travels beside the translation rather than through it: this API's
        // reply has no field for it, and the translation's output is the shared
        // chat-completions shape, which has none either. It is a property of the HTTP
        // exchange, and that is the level it is kept at.
        Ok(ProviderReply {
            body: anthropic_to_chat_response(raw.body),
            request_id: raw.request_id,
        })
    }
}

#[async_trait]
impl LlmPort for AnthropicClient {
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[Message],
        tools: Option<&[Value]>,
        extra_body: &std::collections::HashMap<String, Value>,
    ) -> Result<LlmResponse> {
        let reply = self
            .call_anthropic(model, messages, tools, extra_body)
            .await?;
        Ok(chat_response_to_llm(reply.body, reply.request_id))
    }
}

// ── Anthropic wire types ──────────────────────────────────────────────────────

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContentBlock>,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Deserialize)]
struct AnthropicUsage {
    input_tokens: u64,
    output_tokens: u64,
    /// Anthropic counts this SEPARATELY from `input_tokens` rather than as a part of
    /// it, which is the opposite of how OpenAI reports the same thing. Kept as what
    /// this API means by it; the translation is at the one call site.
    #[serde(default)]
    cache_read_input_tokens: Option<u64>,
    /// Counted separately from `input_tokens` as well -- and this one is charged at
    /// 1.25 times an ordinary input token, where a read is charged at a tenth.
    ///
    /// Every request this adapter sends marks a cache breakpoint, so leaving this
    /// out drops the most expensive part of an Anthropic prompt from its own total,
    /// on every run.
    #[serde(default)]
    cache_creation_input_tokens: Option<u64>,
}

// ── Anthropic translation helpers ─────────────────────────────────────────────

/// Builds the full Anthropic Messages API request body, including
/// prompt-cache breakpoints.
///
/// Anthropic only caches a prompt when a content block is explicitly marked
/// `cache_control: {"type": "ephemeral"}` -- omitting it (the previous
/// behavior here) means EVERY request is billed as fully fresh, even when
/// the bulk of it is byte-identical to the previous call. This matters a
/// lot for this codebase: a single `atoma run` can iterate its tool-calling
/// loop up to `max_iterations` times (100-200 for some agents), and each
/// iteration re-sends the ENTIRE growing conversation so far.
///
/// Two breakpoints are set, matching Anthropic's own recommended pattern
/// for multi-turn tool-using agents:
///   1. the system prompt -- identical across every call for the same
///      agent/run (agent role + tool descriptions + colleagues), and
///      typically the largest static chunk of the prompt.
///   2. the last message in the conversation -- captures the entire
///      accumulated history up to this point. Each subsequent call within
///      the same run's iteration loop only appends new messages after this
///      point, so the cached prefix keeps growing and being reused across
///      iterations instead of being re-billed as fresh input every time.
fn build_request_body(
    model: &str,
    messages: &[Message],
    tools: Option<&[Value]>,
    extra_body: &std::collections::HashMap<String, Value>,
) -> Result<Value> {
    let (system, mut anthropic_messages) = messages_to_anthropic(messages)?;
    mark_last_message_cacheable(&mut anthropic_messages);

    let max_tokens = extra_body
        .get("max_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(ANTHROPIC_DEFAULT_MAX_TOKENS);

    let mut body = serde_json::json!({
        "model": model,
        "max_tokens": max_tokens,
        "messages": anthropic_messages,
    });

    if let Some(sys) = system {
        body["system"] = serde_json::json!([
            { "type": "text", "text": sys, "cache_control": { "type": "ephemeral" } }
        ]);
    }

    if let Some(tools) = tools {
        body["tools"] = Value::Array(tools_to_anthropic(tools)?);
        body["tool_choice"] = serde_json::json!({ "type": "auto" });
    }

    // The shared reservations plus this dialect's own, rather than a written-out list
    // that happens to overlap. `tools` and `tool_choice` are reserved HERE and not in
    // the shared set on purpose: Anthropic's tool shape uses `input_schema`, not
    // `parameters`, so appending an OpenAI-shaped definition from `extra_body` would
    // send this endpoint something it cannot read. `max_tokens` is required by this API
    // and computed above.
    const ALSO_RESERVED: [&str; 3] = ["max_tokens", "tools", "tool_choice"];
    if let Some(obj) = body.as_object_mut() {
        for (k, v) in extra_body {
            if !RESERVED_KEYS.contains(&k.as_str()) && !ALSO_RESERVED.contains(&k.as_str()) {
                obj.insert(k.clone(), v.clone());
            }
        }
    }

    Ok(body)
}

/// Attaches an ephemeral cache-control breakpoint to the LAST content block
/// of the last message. `cache_control` can only be attached to a content
/// BLOCK, not directly to a message, so a bare string `content` is first
/// converted into Anthropic's block-array form (a no-op for the model:
/// `"content": "text"` and `"content": [{"type":"text","text":"text"}]` are
/// equivalent other than allowing a `cache_control` field on the latter).
/// A no-op if `messages` is empty.
fn mark_last_message_cacheable(messages: &mut [Value]) {
    let Some(last) = messages.last_mut() else {
        return;
    };
    let Some(content) = last.get_mut("content") else {
        return;
    };
    match content {
        Value::String(s) => {
            *content = serde_json::json!([
                { "type": "text", "text": s, "cache_control": { "type": "ephemeral" } }
            ]);
        }
        Value::Array(blocks) => {
            if let Some(last_block) = blocks.last_mut().and_then(Value::as_object_mut) {
                last_block.insert(
                    "cache_control".to_string(),
                    serde_json::json!({ "type": "ephemeral" }),
                );
            }
        }
        _ => {}
    }
}

fn messages_to_anthropic(messages: &[Message]) -> Result<(Option<String>, Vec<Value>)> {
    let mut system_content: Option<String> = None;
    let mut out: Vec<Value> = Vec::new();

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                if let Some(Value::String(s)) = &msg.content {
                    system_content = Some(s.clone());
                }
            }
            "user" => {
                let content = msg.content.clone().unwrap_or(Value::String(String::new()));
                let content = mcp_blocks_to_anthropic(content);
                out.push(serde_json::json!({ "role": "user", "content": content }));
            }
            "assistant" => {
                let mut blocks: Vec<Value> = Vec::new();
                if let Some(Value::String(text)) = &msg.content {
                    if !text.is_empty() {
                        blocks.push(serde_json::json!({ "type": "text", "text": text }));
                    }
                }
                for tc in msg.tool_calls.iter().flatten() {
                    let input: Value = serde_json::from_str(&tc.function.arguments)
                        .unwrap_or(Value::Object(Default::default()));
                    blocks.push(serde_json::json!({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": input,
                    }));
                }
                if !blocks.is_empty() {
                    out.push(serde_json::json!({ "role": "assistant", "content": blocks }));
                }
            }
            "tool" => {
                let content = msg.content.clone().unwrap_or(Value::String(String::new()));
                let content = mcp_blocks_to_anthropic(content);
                let tool_use_id = msg.tool_call_id_for_result()?;
                out.push(serde_json::json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": content,
                    }],
                }));
            }
            _ => {}
        }
    }

    Ok((system_content, out))
}

/// Rewrite MCP image blocks into Anthropic's shape, leaving everything else be.
///
/// MCP says `{"type":"image","data":...,"mimeType":...}`; Anthropic wants
/// `{"type":"image","source":{"type":"base64","media_type":...,"data":...}}`.
/// Content that is a plain string — every message that carries no picture —
/// passes through untouched.
///
/// Applied to user messages as well as tool results, because a picture reaches a
/// run from two directions: a tool that returns one, and a person who attached
/// one to the issue being worked on.
fn mcp_blocks_to_anthropic(content: Value) -> Value {
    let Value::Array(blocks) = content else {
        return content;
    };
    Value::Array(
        blocks
            .into_iter()
            .map(|block| {
                if block.get("type").and_then(Value::as_str) != Some("image") {
                    return block;
                }
                let (Some(data), Some(media_type)) = (
                    block.get("data").and_then(Value::as_str),
                    block.get("mimeType").and_then(Value::as_str),
                ) else {
                    // Not the shape we know how to move. Leaving it as it is
                    // sends something Anthropic will reject with a clear
                    // message, which beats silently dropping the picture.
                    return block;
                };
                serde_json::json!({
                    "type": "image",
                    "source": { "type": "base64", "media_type": media_type, "data": data },
                })
            })
            .collect::<Vec<_>>(),
    )
}

fn tools_to_anthropic(tools: &[Value]) -> Result<Vec<Value>> {
    tools
        .iter()
        .map(|tool| {
            let func = tool_function(tool)?;
            let name = func.get("name").and_then(Value::as_str).unwrap_or_default();
            let description = func
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let input_schema = func
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| serde_json::json!({ "type": "object", "properties": {} }));
            Ok(serde_json::json!({
                "name": name,
                "description": description,
                "input_schema": input_schema,
            }))
        })
        .collect()
}

fn anthropic_to_chat_response(raw: AnthropicResponse) -> ChatResponse {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    for block in raw.content {
        match block {
            AnthropicContentBlock::Text { text } => text_parts.push(text),
            AnthropicContentBlock::ToolUse { id, name, input } => {
                tool_calls.push(ToolCall {
                    id,
                    type_: "function".to_string(),
                    function: ToolCallFunction {
                        name,
                        arguments: serde_json::to_string(&input).unwrap_or_default(),
                    },
                });
            }
        }
    }

    let text = if text_parts.is_empty() {
        None
    } else {
        Some(text_parts.join(""))
    };
    let tool_calls = if tool_calls.is_empty() {
        None
    } else {
        Some(tool_calls)
    };

    // Every stop reason this API documents, translated here because this is the only
    // place that knows Anthropic's vocabulary. The last arm used to pass anything
    // unrecognised through as-is, which turned a successful completion into "unexpected
    // finish_reason: stop_sequence" in the runner -- reachable from an agent's
    // `extra_body: stop_sequences`, since this adapter does not reserve that key.
    let finish_reason = match raw.stop_reason.as_deref() {
        // A stop sequence firing is the model finishing, as asked.
        Some("end_turn" | "stop_sequence") => Some("stop".to_string()),
        Some("tool_use") => Some("tool_calls".to_string()),
        Some("max_tokens") => Some("length".to_string()),
        // `refusal` is this API's name for what the canonical vocabulary calls a content
        // filter, and the runner reports that as its own outcome.
        Some("refusal") => Some("content_filter".to_string()),
        Some(other) => {
            tracing::warn!(
                "Anthropic returned stop_reason '{}', which this adapter does not map; \
                 treating it as a normal finish",
                other
            );
            Some("stop".to_string())
        }
        None => None,
    };

    // Anthropic's `input_tokens` excludes BOTH cached figures rather than containing
    // them, so the prompt is the three added up. Leaving the write out was leaving out
    // the one part of an Anthropic prompt that costs more than an ordinary input
    // token, on every run -- this adapter marks a cache breakpoint in every request.
    let prompt = raw.usage.input_tokens
        + raw.usage.cache_read_input_tokens.unwrap_or(0)
        + raw.usage.cache_creation_input_tokens.unwrap_or(0);

    ChatResponse {
        choices: vec![ChatChoice {
            message: Message::assistant(text.as_deref(), tool_calls),
            finish_reason,
        }],
        usage: Some(Usage {
            // Anthropic reports the cached part BESIDE the input rather than inside
            // it, and every other provider here reports it inside. `LlmUsage` defines
            // `cached_prompt_tokens` as a part of `prompt_tokens`, so the sum is the
            // input -- one meaning, translated at the single place the shapes differ.
            // Leaving it out would make an Anthropic run look cheaper in tokens than
            // an identical one anywhere else.
            prompt_tokens: prompt,
            completion_tokens: raw.usage.output_tokens,
            total_tokens: prompt + raw.usage.output_tokens,
            prompt_tokens_details: raw.usage.cache_read_input_tokens.map(|cached| {
                PromptTokensDetails {
                    cached_tokens: Some(cached),
                }
            }),
            prompt_cache_write_tokens: raw.usage.cache_creation_input_tokens,
            // Every field this API sends is read above; there is nothing to diagnose.
            unread: Default::default(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::session::ToolCall;

    fn cache_control_of(block: &Value) -> Option<&Value> {
        block.get("cache_control")
    }

    /// Anthropic counts the cached part beside the input; everyone else counts it
    /// inside. Translated at the one place the shapes differ, so a run against
    /// Anthropic does not look cheaper in tokens than an identical one elsewhere.
    /// The write, which `input_tokens` excludes as well.
    ///
    /// It is the part charged ABOVE an ordinary input token -- 1.25x, against a
    /// read's 0.1x -- and this adapter marks a cache breakpoint in every request, so
    /// dropping it understated the prompt on every Anthropic run that filled a cache.
    #[test]
    fn a_cached_write_is_part_of_the_prompt_and_is_named_apart_from_a_read() {
        let raw: AnthropicResponse = serde_json::from_value(serde_json::json!({
            "content": [],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 200,
                "output_tokens": 50,
                "cache_read_input_tokens": 500,
                "cache_creation_input_tokens": 300
            }
        }))
        .unwrap();
        let usage = anthropic_to_chat_response(raw).usage.expect("usage");
        // 200 fresh + 500 read + 300 written is the thousand tokens it processed.
        assert_eq!(usage.prompt_tokens, 1000);
        assert_eq!(usage.total_tokens, 1050);
        // Apart, because a read is a tenth of an input token and a write is 1.25 of
        // one: summed there is no price to apply to the result.
        assert_eq!(
            usage.prompt_tokens_details.and_then(|d| d.cached_tokens),
            Some(500)
        );
        assert_eq!(usage.prompt_cache_write_tokens, Some(300));
    }

    #[test]
    fn a_cached_read_is_added_into_the_prompt_rather_than_left_beside_it() {
        let raw: AnthropicResponse = serde_json::from_value(serde_json::json!({
            "content": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 200, "output_tokens": 50, "cache_read_input_tokens": 800}
        }))
        .unwrap();
        let usage = anthropic_to_chat_response(raw).usage.expect("usage");
        // 200 fresh + 800 cached is a thousand-token prompt, however it was served.
        assert_eq!(usage.prompt_tokens, 1000);
        assert_eq!(usage.total_tokens, 1050);
        assert_eq!(
            usage.prompt_tokens_details.and_then(|d| d.cached_tokens),
            Some(800)
        );
    }

    #[test]
    fn system_prompt_gets_a_cache_control_breakpoint() {
        let messages = vec![
            Message::system("you are a helpful agent"),
            Message::user("hi"),
        ];
        let body = build_request_body("claude-x", &messages, None, &Default::default())
            .expect("these messages name their calls");

        let system = body.get("system").expect("system should be present");
        let blocks = system.as_array().expect("system should be a block array");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["text"], "you are a helpful agent");
        assert_eq!(
            cache_control_of(&blocks[0]),
            Some(&serde_json::json!({ "type": "ephemeral" }))
        );
    }

    #[test]
    fn no_system_key_when_there_is_no_system_message() {
        let messages = vec![Message::user("hi")];
        let body = build_request_body("claude-x", &messages, None, &Default::default())
            .expect("these messages name their calls");
        assert!(body.get("system").is_none());
    }

    #[test]
    fn last_message_with_plain_string_content_is_converted_and_marked_cacheable() {
        let messages = vec![Message::user("first"), Message::user("second (latest)")];
        let body = build_request_body("claude-x", &messages, None, &Default::default())
            .expect("these messages name their calls");

        let out_messages = body["messages"].as_array().unwrap();
        assert_eq!(out_messages.len(), 2);
        // Earlier message untouched (still a plain string, no cache_control).
        assert_eq!(out_messages[0]["content"], "first");
        // Last message converted to block-array form with a cache_control breakpoint.
        let last_content = out_messages[1]["content"].as_array().unwrap();
        assert_eq!(last_content[0]["text"], "second (latest)");
        assert_eq!(
            cache_control_of(&last_content[0]),
            Some(&serde_json::json!({ "type": "ephemeral" }))
        );
    }

    #[test]
    fn tool_result_as_last_message_gets_cache_control_on_the_tool_result_block() {
        // A very common shape in real agent runs: the conversation's last
        // turn is a tool result (the model just made a tool call, the tool
        // ran, and its result was appended) -- per Anthropic's own
        // documented pattern for caching tool-using conversations, the
        // cache_control breakpoint belongs on the LAST block regardless of
        // its type, including "tool_result".
        let messages = vec![Message::tool("call_1", "issue #42: title, body...")];
        let body = build_request_body("claude-x", &messages, None, &Default::default())
            .expect("these messages name their calls");

        let out_messages = body["messages"].as_array().unwrap();
        assert_eq!(out_messages.len(), 1);
        assert_eq!(out_messages[0]["role"], "user");
        let blocks = out_messages[0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "tool_result");
        assert_eq!(blocks[0]["tool_use_id"], "call_1");
        assert_eq!(
            cache_control_of(&blocks[0]),
            Some(&serde_json::json!({ "type": "ephemeral" })),
            "cache_control must land on the tool_result block itself, not be skipped"
        );
    }

    #[test]
    fn last_message_with_existing_block_array_gets_cache_control_on_its_last_block() {
        let tool_call = ToolCall {
            id: "call_1".to_string(),
            type_: "function".to_string(),
            function: ToolCallFunction {
                name: "get_issue".to_string(),
                arguments: "{}".to_string(),
            },
        };
        let messages = vec![Message::assistant(Some("checking"), Some(vec![tool_call]))];
        let body = build_request_body("claude-x", &messages, None, &Default::default())
            .expect("these messages name their calls");

        let out_messages = body["messages"].as_array().unwrap();
        let blocks = out_messages[0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2, "expected a text block + a tool_use block");
        // cache_control lands on the LAST block only, not the first.
        assert_eq!(cache_control_of(&blocks[0]), None);
        assert_eq!(
            cache_control_of(&blocks[1]),
            Some(&serde_json::json!({ "type": "ephemeral" }))
        );
    }

    #[test]
    fn empty_messages_does_not_panic() {
        let body = build_request_body("claude-x", &[], None, &Default::default())
            .expect("these messages name their calls");
        assert_eq!(body["messages"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn extra_body_still_merges_alongside_cache_control_fields() {
        let mut extra = std::collections::HashMap::new();
        extra.insert("temperature".to_string(), serde_json::json!(0.5));
        let messages = vec![Message::user("hi")];
        let body = build_request_body("claude-x", &messages, None, &extra)
            .expect("these messages name their calls");
        assert_eq!(body["temperature"], 0.5);
    }
}

#[cfg(test)]
mod tool_image_tests {
    use super::mcp_blocks_to_anthropic;
    use serde_json::json;

    // MCP and Anthropic name the same thing differently; the picture is lost
    // unless something moves it across.
    #[test]
    fn an_mcp_image_block_becomes_an_anthropic_source_block() {
        let out = mcp_blocks_to_anthropic(json!([
            {"type": "text", "text": "Here is the screen:"},
            {"type": "image", "data": "AAAA", "mimeType": "image/png"},
        ]));
        assert_eq!(out[0]["type"], "text");
        assert_eq!(out[1]["source"]["type"], "base64");
        assert_eq!(out[1]["source"]["media_type"], "image/png");
        assert_eq!(out[1]["source"]["data"], "AAAA");
    }

    #[test]
    fn a_plain_string_result_passes_through() {
        let out = mcp_blocks_to_anthropic(json!("done"));
        assert_eq!(out, json!("done"));
    }

    // Sending something Anthropic rejects with a clear message beats dropping
    // the picture and reporting success.
    #[test]
    fn an_image_block_of_an_unknown_shape_is_left_alone() {
        let out = mcp_blocks_to_anthropic(json!([{"type": "image", "url": "http://x/y.png"}]));
        assert_eq!(out[0]["url"], "http://x/y.png");
    }
}

#[cfg(test)]
mod user_image_tests {
    use super::messages_to_anthropic;
    use crate::domain::session::Message;
    use serde_json::json;

    // The other direction a picture arrives from: attached to the issue, not
    // returned by a tool.
    #[test]
    fn a_user_message_picture_becomes_a_source_block() {
        let mut msg = Message::user("look at this");
        msg.content = Some(json!([
            {"type": "text", "text": "look at this"},
            {"type": "image", "data": "AAAA", "mimeType": "image/png"},
        ]));
        let (_, out) = messages_to_anthropic(&[msg]).expect("a user message names no call");
        assert_eq!(out[0]["role"], "user");
        assert_eq!(out[0]["content"][1]["source"]["media_type"], "image/png");
        assert_eq!(out[0]["content"][1]["source"]["data"], "AAAA");
    }
}

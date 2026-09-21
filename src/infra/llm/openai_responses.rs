//! Client for OpenAI's Responses API (`POST /v1/responses`).
//!
//! A separate endpoint from Chat Completions, not a newer version of it: the
//! request carries `input` items rather than `messages`, and the reply carries
//! `output` items rather than `choices`. Both are supported by OpenAI; this one
//! is what they recommend for new work, and it is the only one of the two that
//! can carry an image back from a tool.
//!
//! That is why it exists here. `openai.rs` reaches the same models over Chat
//! Completions and has to smuggle a picture through a following user message
//! (see `shared::split_images_out_of_tool_message`); `function_call_output`
//! takes image parts directly, so on this path a tool result is a tool result.
//!
//! Chat Completions stays the default. It is what the OpenAI-compatible
//! ecosystem speaks — vLLM, Ollama, LM Studio, Azure, and every gateway built
//! to that shape — and dropping it to gain one content type would cost far more
//! than it bought.
use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;

use crate::domain::ports::{FinishReason, LlmChoice, LlmPort, LlmResponse, LlmUsage};
use crate::domain::session::{Message, ToolCall, ToolCallFunction};
use crate::infra::llm::shared::{
    merge_extra_body, report_unread_usage, send_json_with_retry, tool_function, ProviderReply,
};

pub struct OpenAIResponsesClient {
    pub(crate) client: reqwest::Client,
    pub(crate) base_url: String,
    pub(crate) api_key: String,
    /// What the provider wants sent besides the credential -- attribution, mostly.
    ///
    /// This dialect had no such field, so a router reached through it learned nothing
    /// about who was asking. The same headers were going out over chat-completions the
    /// whole time, which is what made the gap invisible: switching dialect looked like a
    /// change of wire format and was also a change of identity.
    pub(crate) extra_headers: Vec<(String, String)>,
}

impl OpenAIResponsesClient {
    /// Endpoint and credential come from the caller: the vendor is one row of the
    /// provider table, and this dialect is another row of the same one.
    pub fn new(
        client: reqwest::Client,
        base_url: String,
        api_key: String,
        extra_headers: Vec<(String, String)>,
    ) -> Self {
        OpenAIResponsesClient {
            client,
            base_url,
            api_key,
            extra_headers,
        }
    }
}

#[async_trait]
impl LlmPort for OpenAIResponsesClient {
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[Message],
        tools: Option<&[Value]>,
        extra_body: &std::collections::HashMap<String, Value>,
    ) -> Result<LlmResponse> {
        let url = format!("{}/responses", self.base_url.trim_end_matches('/'));
        let body = build_request_body(model, messages, tools, extra_body)?;

        tracing::debug!("Request URL: {}", url);
        tracing::debug!("Request body: {}", serde_json::to_string_pretty(&body)?);

        let raw: ProviderReply<ResponsesReply> = send_json_with_retry("OpenAI Responses", || {
            let mut request = self
                .client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json");
            for (name, value) in &self.extra_headers {
                request = request.header(name, value);
            }
            request.json(&body)
        })
        .await?;

        Ok(reply_to_llm_response(raw.body, raw.request_id))
    }
}

// ── Wire types ────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ResponsesReply {
    #[serde(default)]
    output: Vec<Value>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    incomplete_details: Option<IncompleteDetails>,
    #[serde(default)]
    usage: Option<ResponsesUsage>,
}

#[derive(Debug, Deserialize)]
struct IncompleteDetails {
    #[serde(default)]
    reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponsesUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
    /// The cache breakdown, for a provider that sends one. Absent is the answer for
    /// one that does not -- see `LlmUsage::cached_prompt_tokens`.
    #[serde(default)]
    input_tokens_details: Option<InputTokensDetails>,
    /// Every usage field nothing above reads -- see `Usage::unread` in `shared`.
    #[serde(flatten)]
    unread: std::collections::BTreeMap<String, Value>,
}

#[derive(Debug, Deserialize)]
struct InputTokensDetails {
    #[serde(default)]
    cached_tokens: Option<u64>,
}

// ── Request ───────────────────────────────────────────────────────────────────

fn build_request_body(
    model: &str,
    messages: &[Message],
    tools: Option<&[Value]>,
    extra_body: &std::collections::HashMap<String, Value>,
) -> Result<Value> {
    let mut body = serde_json::json!({
        "model": model,
        "input": messages_to_input(messages)?,
        // Atoma keeps the whole conversation in its own session and resends it,
        // so the server has nothing to remember between calls. Storing it would
        // leave a copy on OpenAI's side that nothing here ever reads.
        "store": false,
    });

    if let Some(tools) = tools {
        body["tools"] = Value::Array(
            tools
                .iter()
                .map(chat_tool_to_responses)
                .collect::<Result<Vec<_>>>()?,
        );
        body["tool_choice"] = Value::String("auto".to_string());
    }

    // The shared policy, not a second one. This adapter's own version of this loop
    // inserted `tools` straight over the definitions written above, so an agent
    // carrying OpenRouter's server tools in `extra_body` sent those two and no MCP
    // schemas at all — leaving the model to infer argument shapes from the names in
    // the system prompt.
    if let Some(obj) = body.as_object_mut() {
        merge_extra_body(obj, extra_body)?;
    }

    Ok(body)
}

/// Flatten a Chat Completions tool definition into the Responses shape.
///
/// Chat Completions nests the callable under `function`; Responses puts its
/// fields at the top level. Everything else about the definition is the same, so
/// the tool registry produces one shape and this moves it.
fn chat_tool_to_responses(tool: &Value) -> Result<Value> {
    let func = tool_function(tool)?;
    Ok(serde_json::json!({
        "type": "function",
        "name": func.get("name").cloned().unwrap_or(Value::Null),
        "description": func.get("description").cloned().unwrap_or(Value::Null),
        "parameters": func.get("parameters").cloned().unwrap_or(Value::Null),
    }))
}

/// Turn the session's messages into Responses `input` items.
///
/// Three shapes come out, because Responses does not model an assistant's tool
/// call or a tool's result as messages at all:
///
/// - an ordinary message keeps its role and content;
/// - each of an assistant's tool calls becomes its own `function_call` item;
/// - a tool result becomes a `function_call_output`, which is where a picture
///   can finally travel as a picture.
fn messages_to_input(messages: &[Message]) -> Result<Vec<Value>> {
    let mut out = Vec::new();

    for msg in messages {
        match msg.role.as_str() {
            "tool" => {
                let call_id = msg.tool_call_id_for_result()?;
                out.push(serde_json::json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": tool_output(msg.content.as_ref()),
                }));
            }
            "assistant" => {
                // First, because that is the order they arrived in: the model reasons,
                // then says something or calls something. The API checks the order.
                out.extend(msg.provider_items.iter().flatten().cloned());
                if let Some(text) = msg.content.as_ref().and_then(Value::as_str) {
                    if !text.is_empty() {
                        out.push(serde_json::json!({
                            "role": "assistant",
                            "content": text,
                        }));
                    }
                }
                for call in msg.tool_calls.iter().flatten() {
                    out.push(serde_json::json!({
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    }));
                }
            }
            role => {
                let content = msg.content.clone().unwrap_or(Value::String(String::new()));
                out.push(serde_json::json!({
                    "role": role,
                    "content": mcp_blocks_to_input_parts(content),
                }));
            }
        }
    }

    Ok(out)
}

/// Rewrite MCP content blocks into the input parts a Responses message takes.
///
/// The same parts a tool result uses — `input_text` and `input_image` — because
/// Responses names them once for anything going in. A picture reaches a run from
/// two directions: a tool that returns one, and a person who attached one to the
/// issue. Content that is a plain string passes through untouched.
fn mcp_blocks_to_input_parts(content: Value) -> Value {
    let Value::Array(blocks) = &content else {
        return content;
    };
    if !blocks
        .iter()
        .any(|b| b.get("type").and_then(Value::as_str) == Some("image"))
    {
        return content;
    }
    Value::Array(blocks.iter().filter_map(block_to_input_part).collect())
}

/// One MCP block as a Responses input part, or `None` for a kind it has no place
/// for. Shared by the message path and the tool-result path, which take the same
/// parts.
fn block_to_input_part(block: &Value) -> Option<Value> {
    match block.get("type").and_then(Value::as_str) {
        Some("text") => Some(serde_json::json!({
            "type": "input_text",
            "text": block.get("text").cloned().unwrap_or(Value::Null),
        })),
        Some("image") => {
            let data = block.get("data").and_then(Value::as_str)?;
            let mime = block.get("mimeType").and_then(Value::as_str)?;
            Some(serde_json::json!({
                "type": "input_image",
                "image_url": format!("data:{};base64,{}", mime, data),
            }))
        }
        _ => None,
    }
}

/// The `output` of a `function_call_output`.
///
/// A plain string when the result is only text, which is the ordinary case and
/// the shape the API documents first. A result carrying pictures becomes the
/// array form, where MCP's image blocks are rewritten as `input_image` parts —
/// the reason this whole adapter exists.
fn tool_output(content: Option<&Value>) -> Value {
    let Some(Value::Array(blocks)) = content else {
        return content
            .cloned()
            .unwrap_or_else(|| Value::String(String::new()));
    };

    Value::Array(blocks.iter().filter_map(block_to_input_part).collect())
}

// ── Response ──────────────────────────────────────────────────────────────────

/// Collapse the `output` list into the one assistant message the runner expects.
///
/// The runner's loop is written against a single message carrying text and tool
/// calls together, which is the Chat Completions shape. Responses returns the
/// same information spread across items, so it is gathered here rather than
/// teaching the loop a second shape it would otherwise never need.
///
/// `request_id` arrives alongside rather than inside `raw`, because it is a header on
/// the HTTP exchange and not an item in this API's `output`.
fn reply_to_llm_response(raw: ResponsesReply, request_id: Option<String>) -> LlmResponse {
    let mut text = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    // Everything this adapter does NOT turn into text or a tool call. `reasoning` is
    // the one that matters -- the API requires it back in the next request's `input`
    // and answers 400 without it -- but the rule is the general one rather than that
    // name: an item kind nobody here has heard of is exactly the kind that must be
    // handed back untouched. `message` and `function_call` are excluded because they
    // are re-emitted from `text` and `tool_calls`, and keeping both would send each
    // of them twice.
    let mut carried: Vec<Value> = Vec::new();

    for item in &raw.output {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                for part in item
                    .get("content")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    if let Some(t) = part.get("text").and_then(Value::as_str) {
                        text.push_str(t);
                    }
                }
            }
            Some("function_call") => {
                tool_calls.push(ToolCall {
                    id: item
                        .get("call_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    type_: "function".to_string(),
                    function: ToolCallFunction {
                        name: item
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        arguments: item
                            .get("arguments")
                            .and_then(Value::as_str)
                            .unwrap_or("{}")
                            .to_string(),
                    },
                });
            }
            _ => carried.push(item.clone()),
        }
    }

    // This API's `incomplete` reasons, mapped to the canonical vocabulary. Collapsing
    // everything but `max_output_tokens` to `stop` is what made filtered output arrive as
    // an empty normal finish, reported as "LLM returned empty response … 3 times in a
    // row" after two paid retries -- an error naming the wrong cause.
    let finish_reason = match raw.status.as_deref() {
        Some("incomplete") => match raw.incomplete_details.and_then(|d| d.reason).as_deref() {
            Some("max_output_tokens") => FinishReason::Length,
            Some("content_filter") => FinishReason::ContentFilter,
            Some(other) => {
                tracing::warn!(
                    "Responses API returned incomplete reason '{}', which this adapter does \
                     not map; treating it as a normal finish",
                    other
                );
                FinishReason::Stop
            }
            None => FinishReason::Stop,
        },
        _ => FinishReason::Stop,
    };

    LlmResponse {
        choices: vec![LlmChoice {
            message: Message {
                provider_items: (!carried.is_empty()).then_some(carried),
                ..Message::assistant(
                    (!text.is_empty()).then_some(text.as_str()),
                    (!tool_calls.is_empty()).then_some(tool_calls),
                )
            },
            finish_reason: Some(finish_reason),
        }],
        usage: raw.usage.map(|u| {
            let cached = u.input_tokens_details.and_then(|d| d.cached_tokens);
            if cached.is_none() {
                report_unread_usage(&u.unread);
            }
            LlmUsage {
                prompt_tokens: u.input_tokens,
                completion_tokens: u.output_tokens,
                total_tokens: u.total_tokens,
                cached_prompt_tokens: cached,
                // This API does not charge for a cache write, so it reports none.
                written_prompt_tokens: None,
            }
        }),
        request_id,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool_message_with_image() -> Message {
        Message::tool_blocks(
            "call_1",
            "Here is the screen:",
            &[json!({"type": "image", "data": "AAAA", "mimeType": "image/png"})],
        )
    }

    // The reason this adapter exists: on Chat Completions this picture has to be
    // smuggled through a later user message; here it is part of the tool result.
    #[test]
    fn a_tool_result_carries_its_picture_as_an_input_image() {
        let input = messages_to_input(&[tool_message_with_image()])
            .expect("these messages name their calls");
        assert_eq!(input.len(), 1);
        assert_eq!(input[0]["type"], "function_call_output");
        assert_eq!(input[0]["call_id"], "call_1");
        assert_eq!(input[0]["output"][0]["type"], "input_text");
        assert_eq!(input[0]["output"][1]["type"], "input_image");
        assert_eq!(
            input[0]["output"][1]["image_url"],
            "data:image/png;base64,AAAA"
        );
    }

    #[test]
    fn a_text_only_tool_result_stays_a_plain_string() {
        let input = messages_to_input(&[Message::tool("call_1", "done")])
            .expect("these messages name their calls");
        assert_eq!(input[0]["output"], "done");
    }

    // Responses has no assistant message that carries tool calls; each call is
    // its own item.
    #[test]
    fn an_assistant_turn_splits_into_text_and_function_calls() {
        let msg = Message::assistant(
            Some("working on it"),
            Some(vec![ToolCall {
                id: "call_9".to_string(),
                type_: "function".to_string(),
                function: ToolCallFunction {
                    name: "shell".to_string(),
                    arguments: "{\"cmd\":\"ls\"}".to_string(),
                },
            }]),
        );
        let input = messages_to_input(&[msg]).expect("these messages name their calls");
        assert_eq!(input.len(), 2);
        assert_eq!(input[0]["role"], "assistant");
        assert_eq!(input[1]["type"], "function_call");
        assert_eq!(input[1]["call_id"], "call_9");
        assert_eq!(input[1]["name"], "shell");
    }

    #[test]
    fn a_user_message_keeps_its_role_and_content() {
        let input =
            messages_to_input(&[Message::user("hello")]).expect("these messages name their calls");
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"], "hello");
    }

    // The other direction a picture arrives from: attached to the issue, not
    // returned by a tool.
    #[test]
    fn a_user_message_carries_its_picture_as_an_input_image() {
        let mut msg = Message::user("look at this");
        msg.content = Some(json!([
            {"type": "text", "text": "look at this"},
            {"type": "image", "data": "AAAA", "mimeType": "image/png"},
        ]));
        let input = messages_to_input(&[msg]).expect("these messages name their calls");
        assert_eq!(input[0]["role"], "user");
        assert_eq!(input[0]["content"][0]["type"], "input_text");
        assert_eq!(input[0]["content"][1]["type"], "input_image");
        assert_eq!(
            input[0]["content"][1]["image_url"],
            "data:image/png;base64,AAAA"
        );
    }

    #[test]
    fn a_tool_definition_is_flattened_out_of_its_function_wrapper() {
        let tool = json!({
            "type": "function",
            "function": {"name": "shell", "description": "run", "parameters": {"type": "object"}},
        });
        let out = chat_tool_to_responses(&tool).expect("this definition has a function wrapper");
        assert_eq!(out["type"], "function");
        assert_eq!(out["name"], "shell");
        assert_eq!(out["description"], "run");
        assert_eq!(out["parameters"]["type"], "object");
        assert!(out.get("function").is_none());
    }

    #[test]
    fn output_items_collapse_into_one_assistant_message() {
        let raw: ResponsesReply = serde_json::from_value(json!({
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "done"}]},
                {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
            ],
            "status": "completed",
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }))
        .unwrap();

        let response = reply_to_llm_response(raw, None);
        let choice = &response.choices[0];
        assert_eq!(choice.message.content, Some(json!("done")));
        let calls = choice.message.tool_calls.as_ref().unwrap();
        assert_eq!(calls[0].id, "c1");
        assert_eq!(calls[0].function.name, "shell");
        assert_eq!(choice.finish_reason, Some(FinishReason::Stop));
        assert_eq!(response.usage.unwrap().total_tokens, 15);
    }

    // The runner reports a truncated completion, so the two APIs' names for it
    // have to meet somewhere.
    /// What the model thought, kept so it can be handed back.
    ///
    /// The Responses API in thinking mode requires its `reasoning` items in the next
    /// request and answers 400 without them. Dropping them cost an engineer run 49
    /// minutes and 122 tool calls on atomaton #766, and the failure arrives late: the
    /// conversation has to be long enough to carry one first.
    #[test]
    fn reasoning_is_carried_on_the_message_it_came_with() {
        let raw: ResponsesReply = serde_json::from_value(json!({
            "output": [
                {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "abc"},
                {"type": "message", "content": [{"text": "done"}]},
                {"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}"}
            ]
        }))
        .unwrap();
        let message = reply_to_llm_response(raw, None).choices.remove(0).message;

        let carried = message.provider_items.expect("the reasoning item is kept");
        assert_eq!(carried.len(), 1, "{carried:?}");
        assert_eq!(carried[0]["type"], "reasoning");

        // The two this adapter reads are NOT carried: they go back out of `content` and
        // `tool_calls`, and keeping them here would send each of them twice.
        assert_eq!(
            message.content.as_ref().and_then(Value::as_str),
            Some("done")
        );
        assert_eq!(message.tool_calls.as_ref().map(Vec::len), Some(1));
    }

    /// And handed back ahead of the turn it belongs to, because that is the order it
    /// arrived in and the API checks it.
    #[test]
    fn reasoning_goes_back_before_the_turn_it_belongs_to() {
        let message = Message {
            provider_items: Some(vec![json!({"type": "reasoning", "id": "rs_1"})]),
            ..Message::assistant(
                Some("here"),
                Some(vec![ToolCall {
                    id: "c1".to_string(),
                    type_: "function".to_string(),
                    function: ToolCallFunction {
                        name: "read".to_string(),
                        arguments: "{}".to_string(),
                    },
                }]),
            )
        };

        let input = messages_to_input(&[message]).expect("these messages name their calls");
        let kinds: Vec<&str> = input
            .iter()
            .map(|item| {
                item.get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_else(|| item.get("role").and_then(Value::as_str).unwrap_or("?"))
            })
            .collect();
        assert_eq!(kinds, vec!["reasoning", "assistant", "function_call"]);
    }

    /// A reply with nothing to carry carries nothing, rather than an empty list that
    /// would be serialised into every session for ever.
    #[test]
    fn a_reply_with_no_extra_items_carries_none() {
        let raw: ResponsesReply = serde_json::from_value(json!({
            "output": [{"type": "message", "content": [{"text": "hello"}]}]
        }))
        .unwrap();
        let message = reply_to_llm_response(raw, None).choices.remove(0).message;
        assert!(message.provider_items.is_none());
    }

    /// The cache figure, when the provider sends one.
    ///
    /// A run here is 99% prompt and a cached prompt token costs a fraction of an
    /// uncached one, so the bill is decided by a number that was being thrown away.
    #[test]
    fn a_cached_prompt_is_reported_as_the_part_of_the_prompt_it_is() {
        let raw: ResponsesUsage = serde_json::from_value(json!({
            "input_tokens": 1000,
            "output_tokens": 50,
            "total_tokens": 1050,
            "input_tokens_details": {"cached_tokens": 800}
        }))
        .unwrap();
        let reply = ResponsesReply {
            output: vec![],
            status: None,
            incomplete_details: None,
            usage: Some(raw),
        };
        let usage = reply_to_llm_response(reply, None).usage.expect("usage");
        assert_eq!(usage.prompt_tokens, 1000);
        // A part of the prompt, not extra beside it.
        assert_eq!(usage.cached_prompt_tokens, Some(800));
    }

    /// A provider that says nothing about its cache answers `None`, never zero.
    ///
    /// Zero is a claim that the cache did nothing, and would be indistinguishable
    /// afterwards from a provider that does not report. GitHub Copilot bills per
    /// request and reports no tokens at all.
    #[test]
    fn a_provider_that_says_nothing_about_its_cache_is_not_reported_as_zero() {
        let raw: ResponsesUsage = serde_json::from_value(json!({
            "input_tokens": 1000,
            "output_tokens": 50,
            "total_tokens": 1050
        }))
        .unwrap();
        let reply = ResponsesReply {
            output: vec![],
            status: None,
            incomplete_details: None,
            usage: Some(raw),
        };
        let usage = reply_to_llm_response(reply, None).usage.expect("usage");
        assert_eq!(usage.cached_prompt_tokens, None);
    }

    #[test]
    fn a_truncated_reply_reports_length() {
        let raw: ResponsesReply = serde_json::from_value(json!({
            "output": [],
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
        }))
        .unwrap();
        let response = reply_to_llm_response(raw, None);
        assert_eq!(
            response.choices[0].finish_reason,
            Some(FinishReason::Length)
        );
    }
}

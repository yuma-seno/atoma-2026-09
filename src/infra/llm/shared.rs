use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::error::Category;
use serde_json::Value;
use std::collections::BTreeMap;
use std::time::Duration;

use crate::domain::ports::{FinishReason, LlmChoice, LlmResponse, LlmUsage};
use crate::domain::session::Message;

const MAX_HTTP_ATTEMPTS: u8 = 3;
const RETRY_BASE_DELAY_MS: u64 = 1_000;
const RETRY_DELAY_FACTOR: u64 = 4;

fn is_transient(error: &reqwest::Error) -> bool {
    error.is_timeout() || error.is_connect() || error.is_body() || error.is_decode()
}

/// A body that was cut off mid-payload classifies as `Category::Eof`; a payload
/// whose *shape* the deserializer cannot accept classifies as `Data`/`Syntax`.
///
/// Only truncation earns another round trip. A structurally wrong response —
/// wrong field types, an envelope this provider adapter does not model — fails
/// identically on every attempt, so retrying it only multiplies the wait before
/// the same error surfaces.
fn is_truncated(error: &serde_json::Error) -> bool {
    matches!(error.classify(), Category::Eof)
}

fn retry_backoff(attempt: u8) -> Duration {
    let factor = RETRY_DELAY_FACTOR.saturating_pow(u32::from(attempt).saturating_sub(1));
    Duration::from_millis(RETRY_BASE_DELAY_MS.saturating_mul(factor))
}

async fn retry_delay(attempt: u8, reason: &str) {
    let delay = retry_backoff(attempt);
    tracing::warn!(
        "Transient LLM HTTP failure on attempt {}/{}: {}. Retrying in {:?}.",
        attempt,
        MAX_HTTP_ATTEMPTS,
        reason,
        delay,
    );
    tokio::time::sleep(delay).await;
}

/// Request keys Atoma owns outright; `extra_body` may not set them.
///
/// Every name any adapter assembles itself, not only the ones this dialect uses.
/// Each adapter used to carry its own list — `["model","messages"]` here,
/// `["model","input","messages"]` in the Responses adapter, six names inline in the
/// Anthropic one, and a fourth copy in `application::validator`. Four lists meant four
/// answers to "may an agent set this", and `atoma validate` was checking against the
/// one that did not apply.
///
/// `tools` is deliberately NOT here: an agent may add to it, which is what
/// [`reconcile_tools`] is for. Reserving it would break OpenRouter's server tools;
/// letting it through plainly would drop every MCP schema.
pub const RESERVED_KEYS: [&str; 5] = ["model", "messages", "input", "system", "store"];

/// The callable inside a tool definition.
///
/// Atoma's internal representation of a tool IS Chat Completions' shape --
/// `{"type": "function", "function": {…}}` -- and every producer emits it: the skill
/// loader (`application::tools`), the MCP registry (`infra::mcp::tool_definitions`),
/// and the test mock. `shared::chat_completion` sends that form verbatim, which is
/// why it is the internal one.
///
/// Both dialect adapters used to write `tool.get("function").unwrap_or(tool)`, which
/// says a bare definition is a second legitimate shape. Nothing produces one. What
/// the fallback actually did was turn a malformed definition into a tool named `""`
/// with an empty schema, sent to the provider as though it were real.
pub fn tool_function(tool: &Value) -> Result<&Value> {
    tool.get("function").ok_or_else(|| {
        anyhow::anyhow!(
            "a tool definition has no `function` object: {}",
            serde_json::to_string(tool).unwrap_or_else(|_| "<unserializable>".to_string())
        )
    })
}

/// Reconcile an agent's `extra_body.tools` with the runtime tool definitions.
///
/// Appending rather than replacing is the point. A plain insert would REPLACE the
/// runtime tools, silently stripping every MCP tool's JSON Schema from the request —
/// and because the system prompt lists only tool *names*, the model is then left to
/// guess argument shapes, observed in production as a stream of wrong-typed and
/// missing arguments.
///
/// OpenRouter's server tools (`{"type": "openrouter:web_search"}`) are declared in
/// this same array and are documented to work alongside user-defined tools, so both
/// sets belong in it.
///
/// Total, because its caller has already established that `extra` is an array. It
/// used to take a `Value` and, when that was not an array, log a warning and keep the
/// runtime tools — accepting a malformed declaration in place of refusing it, and
/// leaving the agent's `tools` silently absent from the request it was written for.
/// The check moved to [`merge_extra_body`], where it can be made once for every
/// shape of that key rather than only when there are runtime tools to protect.
fn reconcile_tools(runtime: Option<&Value>, extra: &[Value]) -> Value {
    // No runtime tools to protect: whatever the agent supplied stands alone.
    let Some(runtime) = runtime.and_then(Value::as_array) else {
        return Value::Array(extra.to_vec());
    };
    let mut merged = runtime.clone();
    merged.extend(extra.iter().cloned());
    Value::Array(merged)
}

/// Merge an agent's `extra_body` into an assembled request body.
///
/// Reserved keys are dropped; `tools` is merged with what the runtime already
/// put there; everything else overrides.
///
/// `pub` because it is the policy rather than this dialect's helper. The Responses
/// adapter assembled its own version of this loop and left out the `tools`
/// reconciliation, so on that path an agent's `extra_body.tools` REPLACED every MCP
/// tool definition — the exact failure documented on [`reconcile_tools`], on the path
/// this repository's own agents actually use.
pub fn merge_extra_body(
    body: &mut serde_json::Map<String, Value>,
    extra_body: &std::collections::HashMap<String, Value>,
) -> Result<()> {
    for (key, value) in extra_body {
        if RESERVED_KEYS.contains(&key.as_str()) {
            continue;
        }
        if key == "tools" {
            // Checked here rather than inside `reconcile_tools`, so the answer does not
            // depend on whether there happened to be runtime tools to merge with. A
            // `tools` that is not an array used to be a warning on one path and a plain
            // insert on the other -- the same malformed declaration, tolerated twice in
            // two different ways, with the agent's own tools going nowhere either time.
            let extra = value.as_array().ok_or_else(|| {
                anyhow::anyhow!(
                    "this agent's `extra_body.tools` is not an array, so it cannot be \
                     added to the tools this run offers. Write it as a list of tool \
                     definitions, or remove it."
                )
            })?;
            body.insert(key.clone(), reconcile_tools(body.get("tools"), extra));
            continue;
        }
        body.insert(key.clone(), value.clone());
    }
    Ok(())
}

/// Shared HTTP response types (OpenAI-compatible wire format).
#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub choices: Vec<ChatChoice>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct ChatChoice {
    pub message: Message,
    pub finish_reason: Option<String>,
}

/// Not `Copy`: `unread` is a map. Nothing takes this by value twice -- its one
/// consumer maps it into `LlmUsage`, which stays `Copy`.
#[derive(Debug, Deserialize, Default, Clone)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    /// The cache breakdown, for a provider that sends one.
    ///
    /// Absent for one that does not, and absent is the answer -- see
    /// `LlmUsage::cached_prompt_tokens` for why it must not become zero on the way
    /// through.
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// The cached part a provider WROTE, which nothing on any wire read here
    /// reports today.
    ///
    /// Anthropic's translation fills it in -- that API is the one charging for a
    /// cache write -- and it lives on this struct rather than beside it because this
    /// is the shape every chat-completions adapter hands to `chat_response_to_llm`.
    #[serde(default)]
    pub prompt_cache_write_tokens: Option<u64>,
    /// Every usage field nothing above reads.
    ///
    /// Kept so that "this provider reports no cache" can be told apart from "this
    /// provider spells it something we do not read". Without it the two are the same
    /// silence, and the second is diagnosed by guessing a field name and shipping a
    /// release to find out whether the guess was right.
    #[serde(flatten)]
    pub unread: BTreeMap<String, Value>,
}

/// What a provider says about the cached part of a prompt.
///
/// `Copy` because `LlmUsage` is: this is read out of a `Usage` by value on the way
/// through, and it is one machine word behind an `Option`.
#[derive(Debug, Deserialize, Default, Clone, Copy)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: Option<u64>,
}

/// POST a request and deserialize its JSON body, retrying transport-level
/// failures.
///
/// `build_request` is a closure rather than a prepared `RequestBuilder` because
/// each attempt needs a fresh one.
///
/// Retried: connect/timeout/body/decode errors, HTTP 429, HTTP 5xx, and a
/// truncated response body. Not retried: any other status, a provider error
/// object returned under HTTP 200, and a structurally invalid payload — see
/// [`is_truncated`].
pub(crate) async fn send_json_with_retry<T: DeserializeOwned>(
    label: &str,
    build_request: impl Fn() -> reqwest::RequestBuilder,
) -> Result<T> {
    for attempt in 1..=MAX_HTTP_ATTEMPTS {
        let response = match build_request().send().await {
            Ok(response) => response,
            Err(error) if attempt < MAX_HTTP_ATTEMPTS && is_transient(&error) => {
                retry_delay(attempt, &error.to_string()).await;
                continue;
            }
            Err(error) => {
                return Err(error).with_context(|| format!("Failed to send {label} request"))
            }
        };

        if !response.status().is_success() {
            let status = response.status();
            let retryable =
                status == reqwest::StatusCode::TOO_MANY_REQUESTS || status.is_server_error();
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "unknown error".to_string());
            if attempt < MAX_HTTP_ATTEMPTS && retryable {
                retry_delay(
                    attempt,
                    &format!("{label} API error ({status}): {error_text}"),
                )
                .await;
                continue;
            }
            anyhow::bail!("{} API error ({}): {}", label, status, error_text);
        }

        let body = match response.text().await {
            Ok(body) => body,
            Err(error) if attempt < MAX_HTTP_ATTEMPTS && is_transient(&error) => {
                retry_delay(attempt, &error.to_string()).await;
                continue;
            }
            Err(error) => return Err(error).context("Failed to read response body"),
        };

        // Some providers (e.g. OpenRouter) return HTTP 200 with an error object.
        // Detect this and provide a clear error message.
        if let Ok(val) = serde_json::from_str::<Value>(&body) {
            if let Some(error_obj) = val.get("error").filter(|e| !e.is_null()) {
                let msg = error_obj
                    .get("message")
                    .and_then(|m| m.as_str())
                    .unwrap_or("unknown provider error");
                let code = error_obj
                    .get("code")
                    .and_then(|c| c.as_i64())
                    .map(|c| format!(" (code: {})", c))
                    .unwrap_or_default();
                anyhow::bail!("LLM provider error{}: {}", code, msg);
            }
        }

        match serde_json::from_str::<T>(&body) {
            Ok(parsed) => return Ok(parsed),
            Err(error) if attempt < MAX_HTTP_ATTEMPTS && is_truncated(&error) => {
                retry_delay(attempt, &format!("truncated response body: {error}")).await;
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!(
                        "Failed to parse {label} response. \
                         The model may be unavailable or returning an unexpected format."
                    )
                })
            }
        }
    }

    // Unreachable: on the final attempt every arm above returns or bails.
    anyhow::bail!("{} HTTP retry loop exhausted without a response", label)
}

/// Text of the assistant message bridging a tool result to its pictures.
const IMAGE_BRIDGE: &str = "Passing along the image from that tool result.";

/// A `data:` URI for an MCP image block, or `None` if it is not one.
pub(crate) fn mcp_image_data_uri(block: &Value) -> Option<String> {
    if block.get("type").and_then(Value::as_str) != Some("image") {
        return None;
    }
    let data = block.get("data").and_then(Value::as_str)?;
    let mime = block.get("mimeType").and_then(Value::as_str)?;
    Some(format!("data:{};base64,{}", mime, data))
}

/// Rewrite MCP content blocks into the parts an OpenAI message takes.
///
/// A picture reaches a run from two directions: a tool that returns one, and a
/// person who attached one to the issue. Only the first needs the tool-message
/// workaround above — a `user` message carries `image_url` parts natively, which
/// is why this is a plain rewrite rather than a reshuffle.
///
/// Applied after that split, and harmless there: what the split emits is already
/// `image_url`, and only `type: "image"` is touched.
fn mcp_blocks_to_openai(message: Value) -> Value {
    let Some(blocks) = message.get("content").and_then(Value::as_array) else {
        return message;
    };
    if !blocks.iter().any(|b| mcp_image_data_uri(b).is_some()) {
        return message;
    }

    let parts: Vec<Value> = blocks
        .iter()
        .map(|block| match mcp_image_data_uri(block) {
            Some(url) => serde_json::json!({ "type": "image_url", "image_url": { "url": url } }),
            None => block.clone(),
        })
        .collect();

    let mut out = message;
    if let Some(obj) = out.as_object_mut() {
        obj.insert("content".to_string(), Value::Array(parts));
    }
    out
}

/// Move a tool result's pictures into a following `user` message.
///
/// A workaround, and worth naming as one. The Chat Completions schema has no
/// way to return an image from a tool — a `tool` message's content is text — so
/// on this API the only route to the model is a later message. The API that
/// does it properly is OpenAI's Responses API, whose `function_call_output`
/// carries image parts; `infra/llm/openai_responses.rs` takes that path.
///
/// Three messages, not two: tool result, a short assistant line, then the user
/// message with the pictures. The bridge exists because providers that enforce
/// strict role alternation reject a `user` straight after a `tool` with
/// `400 Unexpected role 'user' after role 'tool'` — reported against Mistral-
/// backed endpoints, and the same fix others landed. OpenAI and Google accept
/// either shape, so the stricter one is the one to send.
///
/// The synthetic messages are built here and nowhere else. The session keeps
/// the single tool message it recorded, so nothing about this reaches disk, and
/// a run resumed against Anthropic — which can carry an image in a tool result
/// — takes that path instead with no trace of this one.
///
/// Everything without pictures passes through as one message, unchanged.
fn split_images_out_of_tool_message(message: Value) -> Vec<Value> {
    if message.get("role").and_then(Value::as_str) != Some("tool") {
        return vec![message];
    }
    let Some(blocks) = message.get("content").and_then(Value::as_array) else {
        return vec![message];
    };

    let mut text = String::new();
    let mut images = Vec::new();
    for block in blocks {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(t) = block.get("text").and_then(Value::as_str) {
                    if !text.is_empty() {
                        text.push('\n');
                    }
                    text.push_str(t);
                }
            }
            Some("image") => {
                let Some(url) = mcp_image_data_uri(block) else {
                    continue;
                };
                images.push(serde_json::json!({
                    "type": "image_url",
                    "image_url": { "url": url },
                }));
            }
            _ => {}
        }
    }

    if images.is_empty() {
        return vec![message];
    }

    let mut tool_message = message;
    if let Some(obj) = tool_message.as_object_mut() {
        obj.insert("content".to_string(), Value::String(text));
    }
    vec![
        tool_message,
        serde_json::json!({ "role": "assistant", "content": IMAGE_BRIDGE }),
        serde_json::json!({ "role": "user", "content": images }),
    ]
}

/// Shared OpenAI-compatible HTTP call used by OpenAI and Copilot providers.
#[allow(clippy::too_many_arguments)]
pub async fn openai_compat_call(
    client: &reqwest::Client,
    base_url: &str,
    api_key: &str,
    extra_headers: &[(String, String)],
    model: &str,
    messages: &[Message],
    tools: Option<&[Value]>,
    extra_body: &std::collections::HashMap<String, Value>,
) -> Result<ChatResponse> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));

    let llm_messages: Vec<Value> = messages
        .iter()
        .flat_map(|m| split_images_out_of_tool_message(m.to_llm_value()))
        .map(mcp_blocks_to_openai)
        .collect();
    let mut body = serde_json::json!({
        "model": model,
        "messages": llm_messages,
    });

    if let Some(tools) = tools {
        body["tools"] = serde_json::json!(tools);
        body["tool_choice"] = serde_json::json!("auto");
    }

    if let Some(obj) = body.as_object_mut() {
        merge_extra_body(obj, extra_body)?;
    }

    tracing::debug!("Request URL: {}", url);
    tracing::debug!("Request body: {}", serde_json::to_string_pretty(&body)?);

    send_json_with_retry("LLM", || {
        let mut request = client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .header("Content-Type", "application/json");

        for (name, value) in extra_headers {
            request = request.header(name.as_str(), value.as_str());
        }

        request.json(&body)
    })
    .await
}

/// Turns a chat-completions reply into the domain reply.
///
/// Three adapters speak this dialect -- OpenAI, Copilot, and Anthropic after its own
/// translation -- and each used to carry its own copy of this mapping. A field added
/// to `LlmUsage` then had to be remembered in three places, and the one that forgot
/// would report the same call differently from the others.
/// Says once, per process, that a provider sent usage fields nothing here reads and
/// no cached-prompt figure under any name it does.
///
/// Those two together are the signature of a provider spelling the cache something
/// new, which is otherwise indistinguishable from a provider that has no cache to
/// report: the metrics downstream say `unknown` for both, correctly and uselessly.
/// Once per process because the answer is a field name -- seeing it a second time
/// adds nothing, and every inference of every run would say it.
///
/// Names only, never values. The names are what identifies the spelling, and this
/// line is printed into a workflow log anyone can read.
pub(crate) fn report_unread_usage(unread: &BTreeMap<String, Value>) {
    if unread.is_empty() {
        return;
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        tracing::warn!(
            "Provider reported usage but no cached-prompt figure under any name this \
             adapter reads. Usage fields it sent that nothing here reads: {}. If one of \
             them is the cache figure, it belongs in `Usage`.",
            unread.keys().cloned().collect::<Vec<_>>().join(", "),
        );
    });
}

pub fn chat_response_to_llm(resp: ChatResponse) -> LlmResponse {
    LlmResponse {
        choices: resp
            .choices
            .into_iter()
            .map(|c| LlmChoice {
                message: c.message,
                // The canonical spelling is this dialect's own, so a value that does not
                // read is a provider inventing one -- `None`, and the runner says so,
                // rather than being quietly taken for `stop`.
                finish_reason: c
                    .finish_reason
                    .as_deref()
                    .and_then(FinishReason::from_openai),
            })
            .collect(),
        usage: resp.usage.map(|u| {
            // A provider that sends no cache breakdown stays `None`: zero would be a
            // claim that the cache did nothing, indistinguishable afterwards from a
            // provider that never reports. Which of the two it is, the line below
            // answers -- rather than a guess at what this provider might call it.
            let cached = u.prompt_tokens_details.and_then(|d| d.cached_tokens);
            if cached.is_none() {
                report_unread_usage(&u.unread);
            }
            LlmUsage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
                cached_prompt_tokens: cached,
                written_prompt_tokens: u.prompt_cache_write_tokens,
            }
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    const VALID_BODY: &str = r#"{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":null}"#;

    fn http_200(body: &str) -> String {
        format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body,
        )
    }

    /// Serve one raw HTTP response per element of `responses`, counting requests.
    fn spawn_server(
        responses: Vec<String>,
    ) -> (
        tokio::task::JoinHandle<()>,
        std::net::SocketAddr,
        Arc<AtomicUsize>,
    ) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let requests = Arc::new(AtomicUsize::new(0));
        let server_requests = Arc::clone(&requests);

        let handle = tokio::spawn(async move {
            let listener = TcpListener::from_std(listener).unwrap();
            for response in responses {
                let (mut stream, _) = match listener.accept().await {
                    Ok(pair) => pair,
                    Err(_) => return,
                };
                let mut request = vec![0; 8192];
                let _ = stream.read(&mut request).await;
                server_requests.fetch_add(1, Ordering::SeqCst);
                let _ = stream.write_all(response.as_bytes()).await;
            }
        });

        (handle, address, requests)
    }

    async fn call(address: std::net::SocketAddr) -> Result<ChatResponse> {
        openai_compat_call(
            &reqwest::Client::new(),
            &format!("http://{address}"),
            "test-key",
            &[],
            "test-model",
            &[Message::user("hello")],
            None,
            &HashMap::new(),
        )
        .await
    }

    #[tokio::test]
    async fn retries_when_a_success_response_body_is_truncated() {
        // Content-Length promises more bytes than are sent: a transport-level
        // decode error, surfaced by `response.text()`.
        let (server, address, requests) = spawn_server(vec![
            "HTTP/1.1 200 OK\r\nContent-Length: 100\r\nConnection: close\r\n\r\n{".to_string(),
            http_200(VALID_BODY),
        ]);

        let response = call(address).await.unwrap();

        server.await.unwrap();
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        assert_eq!(response.choices.len(), 1);
    }

    #[tokio::test]
    async fn retries_when_a_complete_response_holds_truncated_json() {
        // A well-formed HTTP response whose body is valid up to a point and then
        // simply stops. `response.text()` succeeds; only the JSON parse fails.
        // This is the shape observed in production behind a stalling provider.
        let cut = &VALID_BODY[..40];
        let (server, address, requests) = spawn_server(vec![http_200(cut), http_200(VALID_BODY)]);

        let response = call(address).await.unwrap();

        server.await.unwrap();
        assert_eq!(
            requests.load(Ordering::SeqCst),
            2,
            "truncated JSON should be retried once"
        );
        assert_eq!(response.choices.len(), 1);
    }

    #[tokio::test]
    async fn does_not_retry_structurally_invalid_json() {
        // `choices` is an object where a sequence is required: no amount of
        // retrying changes the shape, so exactly one request must be made.
        let wrong_shape = r#"{"choices":{"message":"nope"},"usage":null}"#;
        let (server, address, requests) = spawn_server(vec![
            http_200(wrong_shape),
            http_200(VALID_BODY),
            http_200(VALID_BODY),
        ]);

        let error = call(address).await.unwrap_err();

        assert_eq!(
            requests.load(Ordering::SeqCst),
            1,
            "a structurally invalid payload must fail on the first attempt"
        );
        assert!(
            format!("{error:#}").contains("Failed to parse LLM response"),
            "unexpected error: {error:#}"
        );
        server.abort();
    }

    #[tokio::test]
    async fn surfaces_provider_error_object_returned_under_http_200() {
        let body = r#"{"error":{"message":"upstream timed out","code":504}}"#;
        let (server, address, requests) = spawn_server(vec![http_200(body), http_200(VALID_BODY)]);

        let error = call(address).await.unwrap_err();

        assert_eq!(requests.load(Ordering::SeqCst), 1);
        let rendered = format!("{error:#}");
        assert!(rendered.contains("upstream timed out"), "got: {rendered}");
        assert!(rendered.contains("504"), "got: {rendered}");
        server.abort();
    }

    fn body_with_runtime_tools() -> serde_json::Map<String, Value> {
        let body = serde_json::json!({
            "model": "test-model",
            "messages": [],
            "tools": [
                { "type": "function", "function": { "name": "github__get_issue" } },
                { "type": "function", "function": { "name": "atoma_builtin__load_skill" } },
            ],
            "tool_choice": "auto",
        });
        body.as_object().unwrap().clone()
    }

    fn tool_names(body: &serde_json::Map<String, Value>) -> Vec<String> {
        body["tools"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| {
                t.pointer("/function/name")
                    .or_else(|| t.get("type"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("?")
                    .to_string()
            })
            .collect()
    }

    #[test]
    fn extra_body_tools_are_appended_not_substituted_for_runtime_tools() {
        let mut body = body_with_runtime_tools();
        let extra = HashMap::from([(
            "tools".to_string(),
            serde_json::json!([
                { "type": "openrouter:web_search" },
                { "type": "openrouter:web_fetch" },
            ]),
        )]);

        merge_extra_body(&mut body, &extra).expect("these keys merge");

        assert_eq!(
            tool_names(&body),
            vec![
                "github__get_issue",
                "atoma_builtin__load_skill",
                "openrouter:web_search",
                "openrouter:web_fetch",
            ],
            "runtime tool schemas must survive alongside the agent's server tools"
        );
    }

    #[test]
    fn extra_body_tools_stand_alone_when_there_are_no_runtime_tools() {
        let mut body = serde_json::json!({ "model": "m", "messages": [] })
            .as_object()
            .unwrap()
            .clone();
        let extra = HashMap::from([(
            "tools".to_string(),
            serde_json::json!([{ "type": "openrouter:web_search" }]),
        )]);

        merge_extra_body(&mut body, &extra).expect("these keys merge");

        assert_eq!(tool_names(&body), vec!["openrouter:web_search"]);
    }

    /// It used to warn and keep the runtime tools, which reads as "handled" and is
    /// not: the agent declared tools that never reached the request, and only a log
    /// line nobody was watching said so. The same value with no runtime tools to
    /// protect was inserted verbatim, so one malformed declaration had two different
    /// tolerations and no refusal.
    #[test]
    fn a_non_array_extra_body_tools_is_refused_rather_than_absorbed() {
        let mut body = body_with_runtime_tools();
        let extra = HashMap::from([("tools".to_string(), serde_json::json!("web_search"))]);

        let refused =
            merge_extra_body(&mut body, &extra).expect_err("a string is not a tool list");
        assert!(refused.to_string().contains("not an array"), "{refused}");
    }

    #[test]
    fn extra_body_overrides_other_keys_and_never_reserved_ones() {
        let mut body = body_with_runtime_tools();
        let extra = HashMap::from([
            ("model".to_string(), serde_json::json!("hijacked")),
            ("messages".to_string(), serde_json::json!(["hijacked"])),
            ("tool_choice".to_string(), serde_json::json!("none")),
            ("temperature".to_string(), serde_json::json!(0)),
            (
                "provider".to_string(),
                serde_json::json!({ "order": ["Xiaomi"], "allow_fallbacks": false }),
            ),
        ]);

        merge_extra_body(&mut body, &extra).expect("these keys merge");

        assert_eq!(body["model"], serde_json::json!("test-model"));
        assert_eq!(body["messages"], serde_json::json!([]));
        assert_eq!(body["tool_choice"], serde_json::json!("none"));
        assert_eq!(body["temperature"], serde_json::json!(0));
        assert_eq!(body["provider"]["order"], serde_json::json!(["Xiaomi"]));
    }

    #[test]
    fn backoff_grows_between_attempts() {
        assert_eq!(retry_backoff(1), Duration::from_millis(1_000));
        assert_eq!(retry_backoff(2), Duration::from_millis(4_000));
        assert!(retry_backoff(2) > retry_backoff(1));
    }

    #[test]
    fn truncation_and_shape_errors_are_classified_apart() {
        let truncated = serde_json::from_str::<ChatResponse>(&VALID_BODY[..40]).unwrap_err();
        assert!(is_truncated(&truncated), "expected Eof classification");

        let wrong_shape = r#"{"choices":{"a":1},"usage":null}"#;
        let error = serde_json::from_str::<ChatResponse>(wrong_shape).unwrap_err();
        assert!(
            !is_truncated(&error),
            "wrong shape must not be treated as truncation"
        );
    }

    /// The cache figure a chat-completions provider sends, carried through as the
    /// part of the prompt it is.
    ///
    /// A run here is overwhelmingly prompt, re-sent in full every round trip, and a
    /// cached prompt token costs a fraction of a fresh one. Without this the bill
    /// cannot be read off the numbers the run records.
    #[test]
    fn a_cached_prompt_is_carried_through_as_part_of_the_prompt() {
        let resp: ChatResponse = serde_json::from_value(serde_json::json!({
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 50,
                "total_tokens": 1050,
                "prompt_tokens_details": {"cached_tokens": 800}
            }
        }))
        .unwrap();

        let usage = chat_response_to_llm(resp).usage.expect("usage");
        // 800 of the 1000, not 1800: this dialect counts the cached part inside.
        assert_eq!(usage.prompt_tokens, 1000);
        assert_eq!(usage.cached_prompt_tokens, Some(800));
    }

    /// A provider that says nothing about its cache is reported as saying nothing.
    ///
    /// Zero is a claim that the cache did nothing, which afterwards is
    /// indistinguishable from a provider that never reports -- and the two want
    /// opposite responses, one a prompt to fix, the other nothing at all.
    #[test]
    fn a_provider_that_reports_no_cache_is_not_recorded_as_a_cache_that_missed() {
        let resp: ChatResponse = serde_json::from_value(serde_json::json!({
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050}
        }))
        .unwrap();

        let usage = chat_response_to_llm(resp).usage.expect("usage");
        assert_eq!(usage.cached_prompt_tokens, None);
    }

    /// A usage object nothing here fully reads keeps what it could not read.

    /// Without this, a provider that spells the cache something new and a provider
    /// that has no cache are the same silence -- and the answer was being guessed at
    /// by shipping a field name and waiting to see whether a number appeared.
    #[test]
    fn a_usage_field_nothing_reads_is_kept_rather_than_dropped() {
        let resp: ChatResponse = serde_json::from_value(serde_json::json!({
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 50,
                "total_tokens": 1050,
                "prompt_cache_hit_tokens": 768,
                "prompt_cache_miss_tokens": 232
            }
        }))
        .unwrap();

        let usage = resp.usage.as_ref().expect("usage");
        assert_eq!(
            usage.unread.keys().cloned().collect::<Vec<_>>(),
            vec!["prompt_cache_hit_tokens", "prompt_cache_miss_tokens"],
        );
        // Still unknown, because nothing here reads those names -- but now the run
        // says which names it saw instead of leaving it to be guessed.
        assert_eq!(
            chat_response_to_llm(resp)
                .usage
                .expect("usage")
                .cached_prompt_tokens,
            None,
        );
    }

    /// Nothing unread is not a complaint. A provider reporting exactly what this
    /// adapter models, and no cache, has nothing to diagnose.
    #[test]
    fn a_provider_that_reports_only_what_is_modelled_leaves_nothing_unread() {
        let resp: ChatResponse = serde_json::from_value(serde_json::json!({
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050}
        }))
        .unwrap();

        assert!(resp.usage.as_ref().expect("usage").unread.is_empty());
    }
    /// The mapping is one function because every adapter speaking this dialect must
    /// report the same reply the same way, finish reason included.
    #[test]
    fn a_finish_reason_this_dialect_does_not_define_is_not_taken_for_a_normal_stop() {
        let resp: ChatResponse = serde_json::from_value(serde_json::json!({
            "choices": [{"message": {"role": "assistant", "content": "hi"}, "finish_reason": "banana"}],
            "usage": null
        }))
        .unwrap();

        assert_eq!(chat_response_to_llm(resp).choices[0].finish_reason, None);
    }
}

#[cfg(test)]
mod tool_image_tests {
    use super::split_images_out_of_tool_message;
    use serde_json::{json, Value};

    fn tool_with_image() -> Value {
        json!({
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [
                {"type": "text", "text": "Here is the screen:"},
                {"type": "image", "data": "AAAA", "mimeType": "image/png"},
            ],
        })
    }

    // The OpenAI schema has no way to return an image from a tool, so the only
    // route by which a model on this API ever sees one is a following user
    // message.
    #[test]
    fn a_tool_result_with_a_picture_becomes_three_messages() {
        let out = split_images_out_of_tool_message(tool_with_image());
        assert_eq!(out.len(), 3);
        assert_eq!(out[0]["role"], "tool");
        assert_eq!(out[0]["content"], "Here is the screen:");
        assert_eq!(out[0]["tool_call_id"], "call_1");
        // The bridge is what keeps a strict provider from rejecting the
        // sequence with "Unexpected role 'user' after role 'tool'".
        assert_eq!(out[1]["role"], "assistant");
        assert_eq!(out[2]["role"], "user");
        assert_eq!(
            out[2]["content"][0]["image_url"]["url"],
            "data:image/png;base64,AAAA"
        );
    }

    #[test]
    fn a_text_only_tool_result_stays_one_message() {
        let msg = json!({"role": "tool", "tool_call_id": "c", "content": "done"});
        assert_eq!(split_images_out_of_tool_message(msg.clone()), vec![msg]);
    }

    #[test]
    fn other_roles_are_untouched() {
        let msg = json!({"role": "user", "content": "hello"});
        assert_eq!(split_images_out_of_tool_message(msg.clone()), vec![msg]);
    }
}

#[cfg(test)]
mod user_image_tests {
    use super::mcp_blocks_to_openai;
    use serde_json::json;

    // The other direction a picture arrives from: attached to the issue, not
    // returned by a tool. A user message takes image parts natively, so this is
    // a rewrite rather than the reshuffle a tool result needs.
    #[test]
    fn a_user_message_picture_becomes_an_image_url_part() {
        let out = mcp_blocks_to_openai(json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image", "data": "AAAA", "mimeType": "image/png"},
            ],
        }));
        assert_eq!(out["content"][0]["type"], "text");
        assert_eq!(out["content"][1]["type"], "image_url");
        assert_eq!(
            out["content"][1]["image_url"]["url"],
            "data:image/png;base64,AAAA"
        );
    }

    #[test]
    fn a_string_content_message_is_untouched() {
        let msg = json!({"role": "user", "content": "hello"});
        assert_eq!(mcp_blocks_to_openai(msg.clone()), msg);
    }

    // It runs after the tool split, whose output is already `image_url`.
    #[test]
    fn parts_that_are_already_openai_shaped_are_left_alone() {
        let msg = json!({
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}],
        });
        assert_eq!(mcp_blocks_to_openai(msg.clone()), msg);
    }
}

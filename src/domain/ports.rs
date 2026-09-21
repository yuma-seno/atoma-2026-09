use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

use crate::domain::agent::ParsedAgentDef;
use crate::domain::session::{Message, Session};
use crate::domain::skill::{SkillCatalog, SkillMetadata};
use crate::domain::tool::ToolDef;

// ── LLM port ──────────────────────────────────────────────────────────────────

/// Response from a single LLM completion request.
pub struct LlmResponse {
    pub choices: Vec<LlmChoice>,
    pub usage: Option<LlmUsage>,
    /// The provider's own identifier for the request that produced this response.
    ///
    /// A provider that has one returns it in a response header, and every adapter here
    /// threw it away: `send_json_with_retry` read the body and dropped the headers. That
    /// left one half of a support conversation impossible to have. A run could record
    /// that an inference was billed as fresh, or that a 400 killed it, and could not say
    /// WHICH request on the provider's side that was -- so "why did the cache hit on one
    /// turn and miss on the next" stayed an inference from our own numbers rather than a
    /// question the provider could answer about a specific call.
    ///
    /// `None` when the provider returned no header any adapter here reads. That is a
    /// statement about the report rather than a value, and it is kept apart from an
    /// id the same way `LlmUsage::cached_prompt_tokens` is kept apart from zero: an
    /// empty string would read as an identifier, and would be carried into a support
    /// conversation that could never find it.
    ///
    /// Atoma records it and acts on nothing. What to do with a correlation key belongs
    /// to whoever reads the log.
    pub request_id: Option<String>,
}

/// A single choice returned by the LLM.
pub struct LlmChoice {
    pub message: Message,
    pub finish_reason: Option<FinishReason>,
}

/// Token usage statistics.
#[derive(Debug, Default, Clone, Copy)]
pub struct LlmUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    /// How much of `prompt_tokens` the provider served from its cache.
    ///
    /// `None` means the provider did not say, which is NOT zero and must never be
    /// recorded as it. GitHub Copilot bills per request and reports no tokens at all;
    /// a provider that reports its cache under a name this adapter does not read
    /// answers the same way. Zero would read as `the cache is doing nothing`, which
    /// is the one conclusion an absent measurement must not be allowed to support.
    ///
    /// It matters because a run here is 99% prompt -- the whole conversation is resent
    /// every turn -- and a cached prompt token costs between an eighth and a fiftieth
    /// of an uncached one. Without this the bill is not derivable from the counts, and
    /// `is this cheaper` cannot be answered about any change to what gets resent.
    pub cached_prompt_tokens: Option<u64>,
    /// How much of `prompt_tokens` the provider WROTE into its cache.
    ///
    /// Kept apart from `cached_prompt_tokens` because the two move the bill in
    /// opposite directions: Anthropic charges a cache read at a tenth of an input
    /// token and a cache write at 1.25 times one. Summed into a single `cache`
    /// figure there is no price to apply to the result, which is the whole reason
    /// these counts are kept.
    ///
    /// `None` on the same terms as `cached_prompt_tokens`: no chat-completions
    /// provider read here reports a write, and Anthropic is the one that does.
    pub written_prompt_tokens: Option<u64>,
}

/// Port for LLM chat completion.
///
/// Each infra provider (`OpenAIClient`, `CopilotClient`, `AnthropicClient`)
/// implements this trait. The `application` layer depends only on this port.
#[async_trait]
pub trait LlmPort: Send + Sync {
    async fn chat_completion(
        &self,
        model: &str,
        messages: &[Message],
        tools: Option<&[Value]>,
        extra_body: &HashMap<String, Value>,
    ) -> Result<LlmResponse>;
}

/// Why a completion stopped.
///
/// The vocabulary used to exist only as the arms of a `match` in the runner, with each
/// adapter expected to produce one of those strings and nothing checking that it had.
/// Two consequences, both reachable:
///
/// - the Anthropic adapter passed unmapped stop reasons through as-is, so an agent with
///   `extra_body: stop_sequences: [...]` — which that adapter does not reserve — got a
///   perfectly good completion turned into "LLM returned unexpected finish_reason:
///   stop_sequence", and the text was discarded;
/// - the Responses adapter collapsed every `incomplete` reason except
///   `max_output_tokens` to `stop`, so filtered output arrived as an empty `stop` and was
///   reported as "LLM returned empty response … 3 times in a row" after two paid retries,
///   naming the wrong cause.
///
/// As an enum, an adapter cannot invent a fifth value and the runner's match is checked
/// by the compiler. What each dialect calls these stays in that dialect's adapter, which
/// is the only place that knows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// The model finished of its own accord.
    Stop,
    /// The output hit a token ceiling.
    Length,
    /// The provider refused to return what the model produced.
    ContentFilter,
    /// The model asked for tools.
    ToolCalls,
}

impl FinishReason {
    /// Read the OpenAI chat-completions spelling, which is the canonical one.
    ///
    /// `None` for anything else, so a provider inventing a value is visible rather than
    /// silently becoming `Stop`.
    pub fn from_openai(raw: &str) -> Option<Self> {
        match raw {
            "stop" => Some(Self::Stop),
            "length" => Some(Self::Length),
            "content_filter" => Some(Self::ContentFilter),
            "tool_calls" => Some(Self::ToolCalls),
            _ => None,
        }
    }

    /// The name to put in a message a person reads.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
            Self::ContentFilter => "content_filter",
            Self::ToolCalls => "tool_calls",
        }
    }
}

impl std::fmt::Display for FinishReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── Tool port ─────────────────────────────────────────────────────────────────

/// Result from a single MCP tool call.
#[derive(Debug, Default)]
pub struct ToolCallResult {
    pub content: String,
    /// Image blocks the tool returned, in MCP's own wire shape
    /// (`{"type":"image","data":"<base64>","mimeType":"image/png"}`).
    ///
    /// Kept beside `content` rather than folded into it because every consumer
    /// of a tool result reads text — logs, hooks, the built-in skill tool — and
    /// only the message the model receives cares about pictures. MCP's shape is
    /// stored as-is so nothing here has to pick a provider's; each LLM adapter
    /// maps it to its own, which is where that knowledge already lives.
    pub images: Vec<Value>,
    pub session_ends: bool,
}

/// Unified port for tools visible to the LLM.
///
/// Implementations may be external MCP servers or Atoma built-in tools.
#[async_trait]
pub trait ToolPort: Send {
    fn tool_definitions(&self) -> Vec<Value>;

    async fn call_tool(
        &mut self,
        agent_name: &str,
        name: &str,
        arguments: &Value,
    ) -> Result<ToolCallResult>;
}

// ── Persistence ports ─────────────────────────────────────────────────────────

/// Port for loading and saving agent sessions.
pub trait SessionPort: Send + Sync {
    fn load(&self, path: &Path) -> Result<Session>;
    fn save(&self, session: &Session, path: &Path) -> Result<()>;
}

/// Port for parsing agent definition files.
pub trait AgentDefPort: Send + Sync {
    fn parse(&self, path: &Path) -> Result<ParsedAgentDef>;
}

/// Port for loading tool definition files.
pub trait ToolDefPort: Send + Sync {
    fn load(&self, path: &Path) -> Result<HashMap<String, ToolDef>>;
}

/// Port for loading and validating a skill catalog.
pub trait SkillPort: Send + Sync {
    fn load(&self, root: &Path) -> Result<SkillCatalog>;
}

// ── Template port ─────────────────────────────────────────────────────────────

/// Port for rendering the system prompt.
///
/// Every other dependency the runner has arrives through `RunDeps`; this one was reached
/// for directly as `crate::infra::template`, the only `infra` import in `application`
/// outside tests. That is not a crash waiting to happen, it is a hole in the arrangement
/// the rest of the file keeps: with a port, a test can render its own prompt, and nothing
/// in `application` knows how the built-in template is stored.
pub trait TemplatePort: Send + Sync {
    fn build_system_prompt(&self, context: &PromptContext<'_>) -> String;
}

/// Everything the prompt is built from.
///
/// A struct because it was six positional parameters, four of them strings or slices of
/// strings — an order a caller can get wrong silently.
pub struct PromptContext<'a> {
    pub agent: &'a ParsedAgentDef,
    pub tool_descriptions: &'a [String],
    /// Overrides the built-in template entirely when present.
    pub custom_template: Option<&'a str>,
    pub working_dir: &'a str,
    pub colleagues: &'a [(String, String)],
    pub skills: &'a [SkillMetadata],
}

// ── MCP factory port ──────────────────────────────────────────────────────────

/// Port for constructing an MCP registry from a list of tool definitions.
#[async_trait]
pub trait McpFactory: Send + Sync {
    async fn build(&self, tool_defs: &[ToolDef]) -> Result<Box<dyn ToolPort + Send>>;
}

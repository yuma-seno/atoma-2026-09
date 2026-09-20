use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub function: ToolCallFunction,
}

/// A single message in a conversation.
///
/// `atoma_metadata` is an internal field stripped before sending to the LLM API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Atoma-internal metadata (e.g. GitHub comment ID).
    /// Stored in session.json but stripped before sending to the LLM API.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub atoma_metadata: Option<Value>,
    /// What the provider said that only the provider understands, kept to hand back.
    ///
    /// The opposite of `atoma_metadata`, which is stripped before sending. These are
    /// items a dialect requires in the NEXT request, and dropping them is not a lost
    /// nicety but a failed call: the Responses API in thinking mode answers
    /// `400 The reasoning_text in the thinking mode must be passed back to the API`
    /// once the conversation carries one. Measured on atomaton #766 -- an engineer run
    /// died that way after 49 minutes and 122 tool calls, and lost all of it.
    ///
    /// Opaque on purpose. Nothing here reads them and nothing should: they are one
    /// dialect's own items, and the only correct thing to do is return them in the
    /// order they arrived. Every other dialect ignores the field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_items: Option<Vec<Value>>,
}

impl Message {
    pub fn system(content: &str) -> Self {
        Self {
            role: "system".to_string(),
            content: Some(Value::String(content.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            atoma_metadata: None,
            provider_items: None,
        }
    }

    pub fn user(content: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: Some(Value::String(content.to_string())),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            atoma_metadata: None,
            provider_items: None,
        }
    }

    pub fn assistant(content: Option<&str>, tool_calls: Option<Vec<ToolCall>>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.map(|c| Value::String(c.to_string())),
            tool_calls,
            tool_call_id: None,
            name: None,
            atoma_metadata: None,
            provider_items: None,
        }
    }

    pub fn tool(tool_call_id: &str, content: &str) -> Self {
        Self {
            role: "tool".to_string(),
            content: Some(Value::String(content.to_string())),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.to_string()),
            name: None,
            atoma_metadata: None,
            provider_items: None,
        }
    }

    /// A tool result that carries pictures as well as text.
    ///
    /// Stored in MCP's own block shape rather than any provider's, because the
    /// session outlives the choice of provider — a run resumed against a
    /// different one must not find the previous one's wire format baked in.
    /// `infra/llm/*` maps it on the way out.
    pub fn tool_blocks(tool_call_id: &str, text: &str, images: &[Value]) -> Self {
        let mut blocks = vec![serde_json::json!({ "type": "text", "text": text })];
        blocks.extend(images.iter().cloned());
        Self {
            role: "tool".to_string(),
            content: Some(Value::Array(blocks)),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.to_string()),
            name: None,
            atoma_metadata: None,
            provider_items: None,
        }
    }

    /// The call this tool result answers.
    ///
    /// An error rather than a placeholder. Both constructors above take a
    /// `tool_call_id` and neither can be called without one, so a `role: "tool"`
    /// message without it did not come from this program -- a hand-edited session, or
    /// one written by something else. The adapters used to send `"unknown"` in its
    /// place, which is not a call id any provider has ever issued: Anthropic matches
    /// `tool_use_id` against a preceding `tool_use` and rejects the request, so the
    /// fabrication bought a 400 with a made-up id in it instead of a sentence naming
    /// the real problem.
    pub fn tool_call_id_for_result(&self) -> anyhow::Result<&str> {
        self.tool_call_id.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "a tool result in this session names no tool call, so there is nothing \
                 for the provider to match it to. The session is not one this program \
                 wrote; start a fresh one rather than resuming it."
            )
        })
    }

    /// Return a `serde_json::Value` with `atoma_metadata` stripped, suitable
    /// for sending to the LLM API.
    pub fn to_llm_value(&self) -> Value {
        let mut val = serde_json::to_value(self).expect("Message serialization is infallible");
        if let Some(obj) = val.as_object_mut() {
            obj.remove("atoma_metadata");
        }
        val
    }
}

/// What a tool call gets when the run ended before it could be answered.
///
/// Not a guess at what the tool would have returned. It says the run ended, which is
/// true and is what a resumed agent needs to know: the call did not happen, so
/// whatever it was for is still undone.
pub const TOOL_CALL_UNANSWERED: &str = "Error: the run ended before this tool call completed.";

/// Give every tool call still waiting for a result one that says why it never got one.
///
/// # The rule this restores
///
/// A `tool` message answers an assistant message's `tool_calls`, and **every provider
/// rejects a conversation carrying one without the other**. A session with an
/// unanswered call is not merely untidy: it cannot be resumed at all, and the error
/// says nothing about the missing pair.
///
/// # Why this exists rather than "we fixed the path that caused it"
///
/// One path did cause it: aborting a broken run part-way through a batch of parallel
/// tool calls. That is fixed where it happens, and this still exists, because a fix
/// closes one path and an invariant closes the shape. Nothing noticed the hole for as
/// long as it did only because a broken run's session was thrown away; now that every
/// session is kept, an unanswered call would be written to disk and every later run on
/// that issue would fail on it.
///
/// # Order
///
/// A synthetic result is appended to the end of its own call's result block, not to the
/// end of the session. A result that follows the next assistant turn answers nothing.
pub fn answer_unanswered_tool_calls(session: &mut Session, reason: &str) -> usize {
    let mut out: Vec<Message> = Vec::with_capacity(session.messages.len());
    let mut answered = 0usize;
    let mut i = 0usize;

    while i < session.messages.len() {
        let message = session.messages[i].clone();
        let calls = message.tool_calls.clone();
        out.push(message);
        i += 1;

        let Some(calls) = calls else { continue };

        // The run of tool messages that answers this turn. Copied first, so what was
        // already there keeps its order and the synthetic ones land after it.
        let mut seen: Vec<String> = Vec::new();
        while i < session.messages.len() && session.messages[i].role == "tool" {
            if let Some(id) = &session.messages[i].tool_call_id {
                seen.push(id.clone());
            }
            out.push(session.messages[i].clone());
            i += 1;
        }

        for call in &calls {
            if !seen.iter().any(|id| id == &call.id) {
                out.push(Message::tool(&call.id, reason));
                answered += 1;
            }
        }
    }

    if answered > 0 {
        session.messages = out;
    }
    answered
}

/// An in-memory conversation session.
///
/// Persistence (loading from / saving to disk) is handled by
/// `crate::infra::persistence::session`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Session {
    pub messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
    #[serde(flatten)]
    pub extra: std::collections::HashMap<String, Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(id: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            type_: "function".to_string(),
            function: ToolCallFunction {
                name: "shell_execute".to_string(),
                arguments: "{}".to_string(),
            },
        }
    }

    fn ids_of_tool_messages(session: &Session) -> Vec<String> {
        session
            .messages
            .iter()
            .filter(|m| m.role == "tool")
            .filter_map(|m| m.tool_call_id.clone())
            .collect()
    }

    /// The shape that cannot be resumed: a batch abandoned part-way.
    #[test]
    fn an_unanswered_call_is_given_a_result() {
        let mut session = Session {
            messages: vec![
                Message::assistant(None, Some(vec![call("a"), call("b"), call("c")])),
                Message::tool("a", "ok"),
                Message::tool("b", "Error: no"),
            ],
            ..Default::default()
        };

        assert_eq!(answer_unanswered_tool_calls(&mut session, "gone"), 1);
        assert_eq!(ids_of_tool_messages(&session), vec!["a", "b", "c"]);
    }

    /// A result that follows the NEXT assistant turn answers nothing, so the
    /// synthetic one has to land inside its own block rather than at the end.
    #[test]
    fn a_synthetic_result_lands_in_its_own_block() {
        let mut session = Session {
            messages: vec![
                Message::assistant(None, Some(vec![call("a"), call("b")])),
                Message::tool("a", "ok"),
                Message::assistant(Some("done"), None),
            ],
            ..Default::default()
        };

        assert_eq!(answer_unanswered_tool_calls(&mut session, "gone"), 1);
        let roles: Vec<&str> = session.messages.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["assistant", "tool", "tool", "assistant"]);
        assert_eq!(session.messages[2].tool_call_id.as_deref(), Some("b"));
    }

    /// The common case, and the one that must not be touched: every call answered.
    #[test]
    fn a_whole_conversation_is_left_alone() {
        let before = vec![
            Message::assistant(None, Some(vec![call("a")])),
            Message::tool("a", "ok"),
            Message::assistant(Some("done"), None),
        ];
        let mut session = Session {
            messages: before.clone(),
            ..Default::default()
        };

        assert_eq!(answer_unanswered_tool_calls(&mut session, "gone"), 0);
        assert_eq!(session.messages.len(), before.len());
        assert_eq!(ids_of_tool_messages(&session), vec!["a"]);
    }

    #[test]
    fn several_turns_are_each_repaired_in_place() {
        let mut session = Session {
            messages: vec![
                Message::assistant(None, Some(vec![call("a"), call("b")])),
                Message::tool("a", "ok"),
                Message::assistant(None, Some(vec![call("c")])),
            ],
            ..Default::default()
        };

        assert_eq!(answer_unanswered_tool_calls(&mut session, "gone"), 2);
        assert_eq!(ids_of_tool_messages(&session), vec!["a", "b", "c"]);
    }

    #[test]
    fn a_conversation_with_no_tool_calls_is_untouched() {
        let mut session = Session {
            messages: vec![Message::user("hello"), Message::assistant(Some("hi"), None)],
            ..Default::default()
        };
        assert_eq!(answer_unanswered_tool_calls(&mut session, "gone"), 0);
        assert_eq!(session.messages.len(), 2);
    }

    #[test]
    fn test_message_system() {
        let m = Message::system("You are helpful.");
        assert_eq!(m.role, "system");
        assert_eq!(m.content, Some(Value::String("You are helpful.".into())));
        assert!(m.tool_calls.is_none());
        assert!(m.atoma_metadata.is_none());
    }

    #[test]
    fn test_message_user() {
        let m = Message::user("Hello!");
        assert_eq!(m.role, "user");
        assert_eq!(m.content, Some(Value::String("Hello!".into())));
    }

    #[test]
    fn test_message_assistant_text() {
        let m = Message::assistant(Some("I'm here."), None);
        assert_eq!(m.role, "assistant");
        assert_eq!(m.content, Some(Value::String("I'm here.".into())));
        assert!(m.tool_calls.is_none());
    }

    #[test]
    fn test_message_tool() {
        let m = Message::tool("call-1", "result_value");
        assert_eq!(m.role, "tool");
        assert_eq!(m.tool_call_id.as_deref(), Some("call-1"));
        assert_eq!(m.content, Some(Value::String("result_value".into())));
    }

    #[test]
    fn test_to_llm_value_strips_metadata() {
        let m: Message = serde_json::from_value(serde_json::json!({
            "role": "user",
            "content": "Hi",
            "atoma_metadata": { "github_context": { "event_type": "issue_comment" } }
        }))
        .unwrap();
        let val = m.to_llm_value();
        let obj = val.as_object().unwrap();
        assert!(
            !obj.contains_key("atoma_metadata"),
            "atoma_metadata should be stripped"
        );
        assert_eq!(obj["role"].as_str(), Some("user"));
    }

    #[test]
    fn test_to_llm_value_preserves_content() {
        let m = Message::system("sys");
        let val = m.to_llm_value();
        assert_eq!(val["content"].as_str(), Some("sys"));
    }
}

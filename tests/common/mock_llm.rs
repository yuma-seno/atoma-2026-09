use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use atoma::domain::ports::{FinishReason, LlmChoice, LlmPort, LlmResponse, LlmUsage};
use atoma::domain::session::{Message, ToolCall, ToolCallFunction};

/// A mock LLM client that returns pre-queued responses in order.
pub struct MockLlmClient {
    queue: Mutex<VecDeque<LlmResponse>>,
}

impl MockLlmClient {
    pub fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
        }
    }

    /// Enqueue a simple text response with finish_reason "stop".
    pub fn enqueue_text(self, text: &str) -> Self {
        let msg = Message::assistant(Some(text), None);
        let response = LlmResponse {
            choices: vec![LlmChoice {
                message: msg,
                finish_reason: Some(FinishReason::Stop),
            }],
            usage: Some(LlmUsage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
                cached_prompt_tokens: None,
                written_prompt_tokens: None,
            }),
            request_id: None,
        };
        self.queue.lock().unwrap().push_back(response);
        self
    }

    /// Enqueue a tool_calls response.
    pub fn enqueue_tool_calls(self, tool_calls: Vec<ToolCall>) -> Self {
        let msg = Message {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(tool_calls),
            tool_call_id: None,
            name: None,
            atoma_metadata: None,
            provider_items: None,
        };
        let response = LlmResponse {
            choices: vec![LlmChoice {
                message: msg,
                finish_reason: Some(FinishReason::ToolCalls),
            }],
            usage: None,
            request_id: None,
        };
        self.queue.lock().unwrap().push_back(response);
        self
    }
}

#[async_trait]
impl LlmPort for MockLlmClient {
    async fn chat_completion(
        &self,
        _model: &str,
        _messages: &[Message],
        _tools: Option<&[Value]>,
        _extra_body: &HashMap<String, Value>,
    ) -> Result<LlmResponse> {
        let mut queue = self.queue.lock().unwrap();
        queue
            .pop_front()
            .ok_or_else(|| anyhow::anyhow!("MockLlmClient: no more responses queued"))
    }
}

/// Construct a minimal tool call for use in tests.
pub fn make_tool_call(id: &str, name: &str, args: &str) -> ToolCall {
    ToolCall {
        id: id.to_string(),
        type_: "function".to_string(),
        function: ToolCallFunction {
            name: name.to_string(),
            arguments: args.to_string(),
        },
    }
}

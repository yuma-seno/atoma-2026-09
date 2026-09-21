//! Inference loop and tool execution logic.

use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::domain::ports::{FinishReason, LlmPort, LlmUsage, ToolCallResult, ToolPort};
use crate::domain::session::{Message, Session, ToolCall};
use crate::domain::tool_loop::{CallOutcome, LoopTracker};

/// How many consecutive degenerate (contentless) completions to re-request
/// before giving up.
///
/// A completion carrying neither text nor tool calls is a provider-side
/// misfire, not a decision the model made — there is nothing in it to append to
/// the session, so re-sending the identical payload is a genuine retry and gets
/// a fresh sample. Bounded because a provider stuck in this state would
/// otherwise silently consume the whole iteration budget.
const MAX_EMPTY_COMPLETION_RETRIES: u8 = 2;

/// Sentinel error returned when the inference loop runs out of iterations.
#[derive(Debug)]
pub struct MaxIterationsReached(pub u32);

impl std::fmt::Display for MaxIterationsReached {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Inference loop exceeded maximum iterations ({})", self.0)
    }
}

impl std::error::Error for MaxIterationsReached {}

/// Sentinel error returned when the inference loop runs out of time.
///
/// Separate from `MaxIterationsReached` because the two ceilings mean different
/// things. A count of iterations is a guess about how much work a task needs; a
/// runtime is the wall the caller actually has -- a CI job's own timeout -- moved a
/// little earlier so the run ends on its own terms.
///
/// That earlier ending is the whole point. Being killed by the job is not the same as
/// stopping: the steps that save the session and report never run, so the work is
/// gone rather than resumable.
#[derive(Debug)]
pub struct RunTimeExceeded(pub Duration);

impl std::fmt::Display for RunTimeExceeded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Inference loop exceeded its time limit ({}s)",
            self.0.as_secs()
        )
    }
}

impl std::error::Error for RunTimeExceeded {}

/// Sentinel error returned when something outside the run asked it to stop.
///
/// The third way a run ends on purpose, and the only one that is not a ceiling: a
/// person changed their mind. A run under an orchestrator is watched by someone who
/// can see it going the wrong way an hour before any budget would notice, and until
/// this existed the only thing they could do was kill the job.
///
/// Killing the job is not the same as stopping it, and the difference is the reason
/// this is a file rather than a signal. `atoma` writes the session once, at the end;
/// a run killed mid-request leaves the previous run's session on disk, so "pause"
/// would silently mean "discard". Checked at the top of an iteration -- beside the
/// two ceilings, for the same reason -- the conversation is whole and the session
/// that gets written is the one the work actually reached.
#[derive(Debug)]
pub struct StopRequested(pub PathBuf);

impl std::fmt::Display for StopRequested {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Stop requested via {}", self.0.display())
    }
}

impl std::error::Error for StopRequested {}

/// Whether an error is a stop the caller asked for rather than something going wrong.
///
/// All three sentinels mean the same thing downstream: the session is worth saving and
/// the exit status is the soft-stop one, not a failure. Two call sites -- the runner
/// and `main` -- ask this question, and each answered it with its own `downcast_ref`
/// for a single type. Adding a third way to stop to only one of them is exactly the
/// bug this function exists to make impossible.
pub fn is_soft_stop(error: &anyhow::Error) -> bool {
    error.downcast_ref::<MaxIterationsReached>().is_some()
        || error.downcast_ref::<RunTimeExceeded>().is_some()
        || error.downcast_ref::<StopRequested>().is_some()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompletionReason {
    Stop,
    Length,
}

/// Result of the inference loop.
pub enum InferenceResult {
    /// Normal completion with final text response.
    Completed {
        text: String,
        usage: LlmUsage,
        reason: CompletionReason,
    },
    /// A tool requested session suspension (session_ends: true).
    /// The session has been saved; caller should exit cleanly.
    SessionEnded,
}

/// Execute all tool calls from an LLM response, appending results to the session.
///
/// Returns `true` if any tool requested session suspension.
///
/// Tool calls are executed sequentially. Parallel execution requires ToolPort
/// to adopt `&self` with internal synchronization.
async fn execute_tool_calls(
    agent_name: &str,
    tool_calls: &[ToolCall],
    session: &mut Session,
    tools: &mut Box<dyn ToolPort + Send>,
    loop_tracker: &mut LoopTracker,
) -> Result<bool> {
    session
        .messages
        .push(Message::assistant(None, Some(tool_calls.to_vec())));

    let mut session_ends = false;
    // Set instead of returned, so the batch this turn asked for is finished before the
    // run is. Walking out of the middle of it left the assistant's `tool_calls` with
    // some of its results missing, which no provider will accept -- so the session that
    // recorded the most work was the one session that could not be resumed.
    //
    // The remaining calls are NOT executed. Once the run is over, taking further
    // actions is doing work that has already been decided against; they are answered
    // instead, which keeps the conversation whole without committing anything.
    let mut abort: Option<String> = None;

    for tool_call in tool_calls {
        if abort.is_some() {
            session.messages.push(Message::tool(
                &tool_call.id,
                "Error: not executed. The run was aborted before this call was reached.",
            ));
            continue;
        }
        let tool_name = &tool_call.function.name;

        let arguments = match serde_json::from_str::<Value>(&tool_call.function.arguments) {
            Ok(args) => args,
            Err(e) => {
                let msg = format!(
                    "Invalid JSON arguments for tool '{}': {}\nRaw: {}",
                    tool_name, e, tool_call.function.arguments,
                );
                tracing::error!("{}", msg);
                session.messages.push(Message::tool(&tool_call.id, &msg));
                let signature = format!("{}:{}", tool_name, tool_call.function.arguments);
                abort = loop_tracker.record(&signature, CallOutcome::Failed);
                continue;
            }
        };
        let signature = format!(
            "{}:{}",
            tool_name,
            serde_json::to_string(&arguments)
                .unwrap_or_else(|_| tool_call.function.arguments.clone())
        );

        tracing::info!("Executing tool: {} (id: {})", tool_name, tool_call.id);

        match tools.call_tool(agent_name, tool_name, &arguments).await {
            Ok(result) => {
                // The result's own text, because the rule that matters is whether the
                // answer changed -- see `domain::tool_loop`. A call carrying only
                // images or metadata answered with something, just not with words.
                abort = if result.content.is_empty() && !result.images.is_empty() {
                    loop_tracker.record_informative();
                    None
                } else {
                    loop_tracker.record(&signature, CallOutcome::Answered(&result.content))
                };
                tracing::debug!(
                    "Tool '{}' result ({} chars)",
                    tool_name,
                    result.content.len()
                );
                session
                    .messages
                    .push(tool_result_message(&tool_call.id, &result));
                if result.session_ends {
                    tracing::info!(
                        "Tool '{}' requested session suspension — will end after this iteration",
                        tool_name
                    );
                    session_ends = true;
                }
            }
            Err(e) => {
                let msg = format!("Error: {}", e);
                tracing::error!("Tool '{}' failed: {}", tool_name, e);
                session.messages.push(Message::tool(&tool_call.id, &msg));
                abort = loop_tracker
                    .record(&signature, CallOutcome::Failed)
                    .map(|reason| format!("{reason} Last error: {e}"));
            }
        }
    }

    // An explicit request beats an inference. A tool that asked for the session to end
    // said so on purpose; a loop verdict is this module's guess about a pattern.
    if session_ends {
        return Ok(true);
    }

    if let Some(reason) = abort {
        bail!(reason);
    }

    Ok(session_ends)
}

/// Text put in place of a picture the running model cannot read.
///
/// It names the setting rather than saying the tool failed, because the tool did
/// not fail: it produced an image nobody asked whether this agent could see. An
/// agent told this will report it, so the omission surfaces on the issue instead
/// of looking like the image was never made.
const IMAGE_WITHHELD: &str =
    "[image withheld: this agent's model is not configured to read images. \
     Set `vision: true` in its agent definition if the model supports them.]";

/// Build the session message for a tool result.
///
/// A text-only result keeps the plain-string form every session written so far holds,
/// so this is invisible to the runs that do not use pictures.
///
/// It does NOT consult `vision`, and that is the fix rather than an omission. It used
/// to: a `vision: false` agent had the picture replaced by `IMAGE_WITHHELD` *before the
/// message entered the session*, so what was written to `atoma-data` was not what
/// happened. Turn `vision: true` and resume that session and the picture is gone for
/// good, while the model reads a line asserting it cannot see images.
///
/// Withholding belongs where messages LEAVE — see `messages_for_provider` — and having
/// it in both places meant the earlier one only had side effects on disk.
fn tool_result_message(tool_call_id: &str, result: &ToolCallResult) -> Message {
    if result.images.is_empty() {
        return Message::tool(tool_call_id, &result.content);
    }
    Message::tool_blocks(tool_call_id, &result.content, &result.images)
}

/// The messages to send, with pictures withheld from a model that cannot read them.
///
/// The only place this decision is made. A picture reaches a session two ways — a tool
/// returned it, or a caller embedded one it was given, such as a screenshot a person
/// pasted on an issue — and the gate used to sit on the first path only, so the second
/// went to the provider unchanged and a text-only model answered with an API error that
/// lost the run.
///
/// Here it cannot be missed by either: this is the only call site of `chat_completion`,
/// so a new way of getting a picture in, and a new adapter, are both covered.
///
/// The session is left alone, and now genuinely is. What is stored is the record of what
/// happened; this is only what one provider is asked to read.
fn messages_for_provider(messages: &[Message], vision: bool) -> Cow<'_, [Message]> {
    if vision || !messages.iter().any(carries_image) {
        return Cow::Borrowed(messages);
    }
    Cow::Owned(messages.iter().map(withhold_images).collect())
}

/// The content blocks of a message, if it has blocks rather than plain text.
fn content_blocks(message: &Message) -> Option<&[Value]> {
    message
        .content
        .as_ref()
        .and_then(Value::as_array)
        .map(Vec::as_slice)
}

/// Whether a block is a picture, in either dialect's spelling.
///
/// `image_url` is what an OpenAI-compatible body calls one and `image` is what
/// Anthropic calls it. Both are checked here rather than per adapter, because the
/// session holds whichever the caller wrote and the gate runs before the adapter
/// sees it.
fn is_image_block(block: &Value) -> bool {
    matches!(
        block.get("type").and_then(Value::as_str),
        Some("image_url" | "image")
    )
}

fn carries_image(message: &Message) -> bool {
    content_blocks(message).is_some_and(|blocks| blocks.iter().any(is_image_block))
}

/// One message with each picture replaced by the text that explains its absence.
fn withhold_images(message: &Message) -> Message {
    let mut out = message.clone();
    if let Some(blocks) = content_blocks(message) {
        let rewritten: Vec<Value> = blocks
            .iter()
            .map(|block| {
                if is_image_block(block) {
                    serde_json::json!({ "type": "text", "text": IMAGE_WITHHELD })
                } else {
                    block.clone()
                }
            })
            .collect();
        out.content = Some(Value::Array(rewritten));
    }
    out
}

/// Run the inference loop: call LLM, handle tool calls or final response.
#[allow(clippy::too_many_arguments)]
pub async fn inference_loop(
    llm_client: &dyn LlmPort,
    agent_name: &str,
    model: &str,
    session: &mut Session,
    tool_definitions: Option<&[Value]>,
    extra_body: &HashMap<String, Value>,
    tools: &mut Box<dyn ToolPort + Send>,
    max_iterations: Option<u32>,
    max_runtime: Option<Duration>,
    stop_file: Option<&Path>,
    vision: bool,
) -> Result<InferenceResult> {
    let mut total_usage = LlmUsage::default();
    let mut loop_tracker = LoopTracker::default();
    let mut consecutive_empty: u8 = 0;
    let started = Instant::now();

    let mut iteration: u32 = 0;
    loop {
        iteration += 1;

        // All three stops are checked here, at the top, and that placement is
        // load-bearing: the previous iteration appended its tool results before coming
        // back here, so the conversation is whole. Stopping between exchanges is what
        // leaves a session another run can resume from -- across 18 cut-off sessions,
        // every one ended on a tool result with no unanswered call.
        if let Some(path) = stop_file {
            if path.exists() {
                tracing::warn!(
                    "Stop requested ({}) after {} iterations",
                    path.display(),
                    iteration - 1,
                );
                return Err(anyhow::Error::new(StopRequested(path.to_path_buf())));
            }
        }
        if let Some(limit) = max_iterations {
            if iteration > limit {
                return Err(anyhow::Error::new(MaxIterationsReached(limit)));
            }
        }
        if let Some(limit) = max_runtime {
            let spent = started.elapsed();
            if spent >= limit {
                tracing::warn!(
                    "Time limit reached after {}s over {} iterations",
                    spent.as_secs(),
                    iteration - 1,
                );
                return Err(anyhow::Error::new(RunTimeExceeded(limit)));
            }
        }

        match max_iterations {
            Some(limit) => tracing::info!(
                "Inference iteration {}/{} ({} messages in session)",
                iteration,
                limit,
                session.messages.len()
            ),
            None => tracing::info!(
                "Inference iteration {} ({} messages in session)",
                iteration,
                session.messages.len()
            ),
        }

        let outgoing = messages_for_provider(&session.messages, vision);
        let response = llm_client
            .chat_completion(model, &outgoing, tool_definitions, extra_body)
            .await?;

        if let Some(u) = response.usage {
            // Per inference, not only per run. The run total cannot answer how the
            // prompt grows turn by turn, and that is the whole cost of resending the
            // conversation: a result read early is paid again on every turn after it.
            // Answering it once meant parsing a workflow log for iteration timestamps
            // and guessing at a characters-per-token ratio, which was wrong by three
            // times and was believed for an hour.
            //
            // `request=` is the provider's own id for this one call, and it is here
            // because this line is already the per-inference record -- a second line
            // saying the same thing about the same call would have to be joined back
            // up by iteration number. It says `unknown` when the provider returned no
            // id, on the same rule as `cached=`: an empty value would read as an
            // identifier that no support conversation could find. What it buys is the
            // step from "this iteration was billed as fresh" to a question the
            // provider can answer about a specific request of theirs.
            tracing::info!(
                "ATOMA_INFERENCE_USAGE: iteration={} prompt={} completion={} cached={} written={} request={}",
                iteration,
                u.prompt_tokens,
                u.completion_tokens,
                u.cached_prompt_tokens
                    .map_or_else(|| "unknown".to_string(), |n| n.to_string()),
                u.written_prompt_tokens
                    .map_or_else(|| "unknown".to_string(), |n| n.to_string()),
                response.request_id.as_deref().unwrap_or("unknown"),
            );
            total_usage.prompt_tokens += u.prompt_tokens;
            total_usage.completion_tokens += u.completion_tokens;
            total_usage.total_tokens += u.total_tokens;
            // Summed only over the inferences that reported one, and left as `None`
            // when not one did. Starting the accumulator at zero would turn a run
            // against a provider that says nothing about its cache into a run whose
            // cache did nothing, and the two look identical afterwards.
            if let Some(cached) = u.cached_prompt_tokens {
                total_usage.cached_prompt_tokens =
                    Some(total_usage.cached_prompt_tokens.unwrap_or(0) + cached);
            }
            if let Some(written) = u.written_prompt_tokens {
                total_usage.written_prompt_tokens =
                    Some(total_usage.written_prompt_tokens.unwrap_or(0) + written);
            }
        }

        let choice = response
            .choices
            .into_iter()
            .next()
            .context("No choices returned from LLM")?;

        // A provider that stated no reason is taken as a normal finish, as before. What
        // changed is that a reason it DID state and no adapter recognised no longer
        // arrives here as an unknown string: each adapter maps its own dialect, so this
        // is either one of four values or absent.
        let finish_reason = choice.finish_reason.unwrap_or(FinishReason::Stop);
        let tool_calls = choice.message.tool_calls;
        let content = choice.message.content;

        if let Some(calls) = tool_calls {
            if calls.is_empty() {
                consecutive_empty += 1;
                if consecutive_empty > MAX_EMPTY_COMPLETION_RETRIES {
                    bail!(
                        "LLM returned an empty tool_calls array {} times in a row",
                        consecutive_empty
                    );
                }
                tracing::warn!(
                    "LLM returned empty tool_calls array — re-requesting ({}/{})",
                    consecutive_empty,
                    MAX_EMPTY_COMPLETION_RETRIES
                );
                continue;
            }
            consecutive_empty = 0;
            if finish_reason != FinishReason::ToolCalls {
                tracing::warn!(
                    "LLM returned finish_reason '{}' with tool_calls — processing anyway",
                    finish_reason
                );
            }

            tracing::info!("LLM requested {} tool call(s)", calls.len());
            let session_ends =
                execute_tool_calls(agent_name, &calls, session, tools, &mut loop_tracker).await?;

            if session_ends {
                tracing::info!("Tool requested session suspension; ending inference loop");
                return Ok(InferenceResult::SessionEnded);
            }
        } else {
            match finish_reason {
                // `end_turn` used to be an arm here and was dead: the Anthropic adapter
                // has always translated it before this point.
                FinishReason::Stop => {
                    let text = content
                        .as_ref()
                        .and_then(|c| c.as_str())
                        .unwrap_or("")
                        .to_owned();
                    if text.is_empty() {
                        consecutive_empty += 1;
                        if consecutive_empty > MAX_EMPTY_COMPLETION_RETRIES {
                            bail!(
                                "LLM returned empty response (finish_reason: {}) {} times in a row",
                                finish_reason,
                                consecutive_empty
                            );
                        }
                        tracing::warn!(
                            "LLM returned empty response (finish_reason: {}) — re-requesting ({}/{})",
                            finish_reason,
                            consecutive_empty,
                            MAX_EMPTY_COMPLETION_RETRIES
                        );
                        continue;
                    }
                    // No counter reset here: this arm returns, so only the
                    // tool_calls branch needs to clear it before looping again.
                    session.messages.push(Message::assistant(Some(&text), None));
                    tracing::info!("LLM returned final response ({} chars)", text.len());
                    return Ok(InferenceResult::Completed {
                        text,
                        usage: total_usage,
                        reason: CompletionReason::Stop,
                    });
                }
                FinishReason::Length => {
                    let text = content
                        .as_ref()
                        .and_then(|c| c.as_str())
                        .unwrap_or("")
                        .to_owned();
                    session.messages.push(Message::assistant(Some(&text), None));
                    tracing::warn!(
                        "LLM response truncated (finish_reason: length, {} chars)",
                        text.len()
                    );
                    return Ok(InferenceResult::Completed {
                        text,
                        usage: total_usage,
                        reason: CompletionReason::Length,
                    });
                }
                FinishReason::ContentFilter => {
                    bail!("LLM response was blocked by content filter")
                }
                FinishReason::ToolCalls => {
                    bail!("LLM returned finish_reason 'tool_calls' but no tool_calls in message")
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The three deliberate stops, and only those. Something going wrong must not be
    /// mistaken for one: that would save a session, exit 2 and read as a pause, which
    /// is a failure reported as a hand-back.
    #[test]
    fn a_deliberate_stop_is_one_and_a_failure_is_not() {
        assert!(is_soft_stop(&anyhow::Error::new(MaxIterationsReached(50))));
        assert!(is_soft_stop(&anyhow::Error::new(RunTimeExceeded(
            Duration::from_secs(60)
        ))));
        assert!(is_soft_stop(&anyhow::Error::new(StopRequested(
            PathBuf::from("/tmp/stop")
        ))));
        assert!(!is_soft_stop(&anyhow::anyhow!("the provider hung up")));
    }

    fn user_with_image(url: &str) -> Message {
        Message {
            role: "user".to_string(),
            content: Some(serde_json::json!([
                { "type": "text", "text": "look at this" },
                { "type": "image_url", "image_url": { "url": url } },
            ])),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            atoma_metadata: None,
            provider_items: None,
        }
    }

    /// The hole this closes: a picture that a CALLER put in the session, which
    /// `tool_result_message` never sees. A text-only model answered one of these
    /// with an API error and the run was lost.
    #[test]
    fn a_pasted_image_does_not_reach_a_model_that_cannot_read_it() {
        let messages = vec![user_with_image("https://example.com/x.png")];
        let sent = messages_for_provider(&messages, false);

        let blocks = content_blocks(&sent[0]).expect("blocks");
        assert_eq!(blocks.len(), 2, "the text block stays");
        assert_eq!(blocks[0]["text"], "look at this");
        assert_eq!(blocks[1]["type"], "text");
        assert_eq!(blocks[1]["text"], IMAGE_WITHHELD);
    }

    #[test]
    fn the_session_itself_is_left_alone() {
        let messages = vec![user_with_image("https://example.com/x.png")];
        let _ = messages_for_provider(&messages, false);
        let blocks = content_blocks(&messages[0]).expect("blocks");
        assert_eq!(
            blocks[1]["type"], "image_url",
            "what happened is still recorded"
        );
    }

    /// The half this test used to leave out, and where the claim was false.
    ///
    /// A tool result went through a second gate that replaced the picture BEFORE the
    /// message entered the session, so the record on `atoma-data` said an image was
    /// withheld where an image had been. Resuming that session with `vision: true` could
    /// never get it back.
    #[test]
    fn a_tool_result_is_stored_as_it_arrived_whatever_the_agent_can_read() {
        let result = ToolCallResult {
            content: "Here is the screen:".to_string(),
            images: vec![
                serde_json::json!({"type": "image", "data": "AAAA", "mimeType": "image/png"}),
            ],
            ..Default::default()
        };

        let stored = tool_result_message("call_1", &result);
        let blocks = content_blocks(&stored).expect("the images are still blocks");
        assert_eq!(
            blocks[1]["type"], "image",
            "the session records the picture"
        );

        // And the agent that cannot read it still does not receive it.
        let stored_messages = [stored];
        let sent = messages_for_provider(&stored_messages, false);
        assert_eq!(
            content_blocks(&sent[0]).expect("blocks")[1]["text"],
            IMAGE_WITHHELD
        );
    }

    #[test]
    fn a_model_that_can_read_images_gets_them() {
        let messages = vec![user_with_image("https://example.com/x.png")];
        let sent = messages_for_provider(&messages, true);
        assert!(
            matches!(sent, Cow::Borrowed(_)),
            "nothing to rewrite, nothing copied"
        );
        assert_eq!(
            content_blocks(&sent[0]).expect("blocks")[1]["type"],
            "image_url"
        );
    }

    #[test]
    fn a_run_without_pictures_copies_nothing() {
        let messages = vec![Message::user("plain text")];
        let sent = messages_for_provider(&messages, false);
        assert!(matches!(sent, Cow::Borrowed(_)));
    }

    /// Anthropic spells it `image`, an OpenAI-compatible body spells it
    /// `image_url`, and the session holds whichever the caller wrote.
    #[test]
    fn both_dialects_of_picture_are_recognised() {
        let anthropic = Message {
            role: "user".to_string(),
            content: Some(serde_json::json!([
                { "type": "image", "source": { "type": "base64", "data": "..." } },
            ])),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            atoma_metadata: None,
            provider_items: None,
        };
        let messages = [anthropic];
        let sent = messages_for_provider(&messages, false);
        assert_eq!(
            content_blocks(&sent[0]).expect("blocks")[0]["text"],
            IMAGE_WITHHELD
        );
    }
}

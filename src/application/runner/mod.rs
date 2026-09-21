//! Agent runner — orchestrates agent definition loading, session management,
//! MCP connection, and the inference loop.

mod execution;

use anyhow::{Context, Result};
use std::io::IsTerminal;
use std::path::PathBuf;
use std::time::Duration;

use crate::application::tools::RuntimeTools;
use crate::domain::ports::{
    AgentDefPort, LlmPort, LlmUsage, McpFactory, PromptContext, SessionPort, SkillPort,
    TemplatePort, ToolDefPort, ToolPort,
};
use crate::domain::session::{
    answer_unanswered_tool_calls, Message, Session, TOOL_CALL_UNANSWERED,
};
use crate::domain::skill::SkillCatalog;
use crate::domain::tool::unknown_server_message;

// The three sentinel types stay unexported on purpose. `is_soft_stop` is the whole
// question anyone outside asks about them, and exporting the types invites each caller
// to answer it again with its own `downcast_ref` -- which is how one of the two callers
// came to know about only one ceiling.
use serde_json::Value;

// The three sentinels `ending_of` names. Imported rather than re-exported: naming an
// ending is this module's own business, and exporting them again would invite a caller
// to downcast a second time -- which is the duplication `is_soft_stop` exists to end.
use execution::{MaxIterationsReached, RunTimeExceeded, StopRequested};

pub use execution::{inference_loop, is_soft_stop, CompletionReason, InferenceResult};

// ── Bundled parameter structs ────────────────────────────────────────────────

/// User-supplied settings for a single `run` invocation.
pub struct RunSettings {
    pub agent_def_path: PathBuf,
    pub in_session: Option<PathBuf>,
    pub prompt_file: Option<PathBuf>,
    pub out_session: Option<PathBuf>,
    pub template_path: Option<PathBuf>,
    pub tools_file: Option<PathBuf>,
    pub skills_dir: Option<PathBuf>,
    /// A ceiling on turns, if the caller asked for one. `None` is unbounded.
    pub max_iterations: Option<u32>,
    /// A ceiling on wall-clock time, if the caller asked for one. `None` is unbounded.
    pub max_runtime: Option<Duration>,
    /// A path whose existence means "stop at the next iteration", if the caller wants
    /// to be able to say so. `None` is a run nothing outside it can interrupt.
    ///
    /// Not read from `atoma.toml`, unlike the two ceilings: a path that is fixed in
    /// configuration is a path that might already exist when a run starts, which would
    /// stop every run immediately. It belongs to one invocation.
    pub stop_file: Option<PathBuf>,
}

/// Observable outcome of a completed run. Presentation belongs to the caller.
#[derive(Debug)]
pub enum RunOutcome {
    Completed {
        text: String,
        usage: LlmUsage,
        reason: CompletionReason,
        session_path: Option<PathBuf>,
    },
    SessionEnded,
}

/// External dependencies (ports) required by the runner.
pub struct RunDeps<'a> {
    pub llm: &'a dyn LlmPort,
    pub agent_def: &'a dyn AgentDefPort,
    pub session: &'a dyn SessionPort,
    pub tool_def: &'a dyn ToolDefPort,
    pub skill: &'a dyn SkillPort,
    pub mcp_factory: &'a dyn McpFactory,
    pub template: &'a dyn TemplatePort,
}

/// Write the session out, whatever ended the run.
///
/// Repaired first. A tool call with no result is a conversation every provider refuses,
/// so writing one would produce a session that cannot be resumed and an error, later,
/// that says nothing about why. `execute_tool_calls` no longer leaves one; this is here
/// because a path we have not thought of would otherwise reach the disk.
///
/// Nothing here fails the run. It is called on a path that is already reporting
/// something, and a failure to save is a second problem rather than a replacement for
/// the first.
const RUNS_KEY: &str = "atoma_runs";

/// What a run leaves behind about itself, appended to the session it saved.
///
/// A session records what was said and nothing about the saying of it. That gap is why a
/// delivery template measuring its own agents can count tools and skills but not how long
/// anything took or why it stopped -- and the second is the more useful, because every
/// ending except a finished one is a mechanism giving up.
///
/// Appended rather than replaced: a session is resumed, so one file holds several runs
/// and the interesting questions are about the sequence. The whole array is rewritten on
/// each save, which is what delta compression is good at.
///
/// Beside `metadata` rather than inside it, because `metadata` is whatever the caller put
/// there and this is atoma's own.
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct RunRecord {
    /// RFC 3339, UTC. When the run began, not when the job did -- atoma cannot see the
    /// queue it waited in, and a number that silently mixed the two would be worse than
    /// no number at all.
    pub started: String,
    pub ended: String,
    /// Whole seconds. Derivable from the two above and written anyway: every reader wants
    /// it, and every reader computing it is a place to get it wrong.
    pub seconds: u64,
    /// How the run ended, in the terms the runner decides them.
    ///
    /// `completed` is the only one that is not a mechanism giving up. `iterations`,
    /// `runtime` and `stopped` are the three soft stops; `failed` is everything else,
    /// including a provider hanging up and a broken loop being cut short.
    ///
    /// Two endings are `completed`: the agent returning text, and a tool ending the
    /// session — which is how a run that opened a pull request ends. The second used to
    /// be recorded as nothing at all, so this list held the failures and the
    /// interruptions and left out the successes, and every rate taken over it was taken
    /// over the wrong population.
    pub ended_because: String,
    /// Messages in the session when it was saved: the cheapest proxy for how much work
    /// the run did.
    pub messages: usize,
    /// Inferences this run made -- round trips to the model, which is what a run waits
    /// on and what it is billed for.
    ///
    /// `messages` was the only size here and is a poor stand-in: it counts tool results
    /// too, so a run that asks for two tools per turn looks larger than one that asks
    /// for one, having waited the same number of times. Answering `how long does a
    /// round trip take here` meant parsing `Inference iteration N` out of a workflow
    /// log, which is not a thing a report should need.
    ///
    /// Counted by the inference loop as each round trip returns and carried down to this
    /// record. It used to be derived here instead, from the previous run's `messages`
    /// value as written on disk, which recorded zero -- silently -- for any embedder
    /// that compacts or prunes a session's history between runs. A round trip that came
    /// back empty and was re-requested counts: it was waited on and it was billed.
    ///
    /// `#[serde(default)]` because sessions written before this field exists are read
    /// back, and zero is the honest answer for a run that never recorded it.
    #[serde(default)]
    pub iterations: usize,
}

/// Now, as RFC 3339 in UTC.
///
/// Formatted from a Unix timestamp rather than by adding a date crate for two calls. The
/// civil-date arithmetic is Howard Hinnant's `civil_from_days`, exact for every date this
/// will see.
fn now_rfc3339() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let time = secs.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year,
        m,
        d,
        time / 3_600,
        (time % 3_600) / 60,
        time % 60
    )
}

/// Whole seconds between two stamps this module produced, or 0.
///
/// Only ever given its own output, so it parses by position. Zero rather than an error
/// for anything unexpected: a duration nobody can compute is not a reason to lose a
/// session.
fn seconds_between(started: &str, ended: &str) -> u64 {
    fn epoch(s: &str) -> Option<i64> {
        if s.len() < 20 {
            return None;
        }
        let num = |from: usize, to: usize| s.get(from..to)?.parse::<i64>().ok();
        let (y, m, d) = (num(0, 4)?, num(5, 7)?, num(8, 10)?);
        let (hh, mm, ss) = (num(11, 13)?, num(14, 16)?, num(17, 19)?);
        let y2 = if m <= 2 { y - 1 } else { y };
        let era = y2.div_euclid(400);
        let yoe = y2 - era * 400;
        let mp = if m > 2 { m - 3 } else { m + 9 };
        let doy = (153 * mp + 2) / 5 + d - 1;
        let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
        Some((era * 146_097 + doe - 719_468) * 86_400 + hh * 3_600 + mm * 60 + ss)
    }
    match (epoch(started), epoch(ended)) {
        (Some(a), Some(b)) if b >= a => (b - a) as u64,
        _ => 0,
    }
}

/// Append this run to the session's own record of its runs.
///
/// Never fails the save. A session that could not be described is still a session worth
/// keeping, and this runs immediately before a run that may already be failing writes
/// whatever it reached.
fn record_run(session: &mut Session, started: &str, ended_because: &str, inferences: usize) {
    let ended = now_rfc3339();

    let mut runs: Vec<Value> = session
        .extra
        .get(RUNS_KEY)
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default();

    // `iterations` arrives as a parameter, counted by the loop that made the round
    // trips. It used to be worked out here: the previous run record's `messages` value
    // was read back off disk, used as the first index of this run's slice, and the
    // assistant messages past it were counted. That rested on an assumption atoma does
    // not enforce and cannot check -- that a session's `messages` array is only ever
    // appended to between runs.
    //
    // An embedder that compacts or prunes the history breaks the assumption, and breaks
    // it silently: the stored number then exceeds `session.messages.len()`, `skip`
    // yields nothing, `count` is zero, and `atoma_runs` gains a record of a run that
    // apparently never called the model. Nothing errors and nothing warns, so the only
    // signal is a report that says a run which cost real money did no inference.
    //
    // The old comment here defended the read as the better trade: threading a second
    // parameter through two functions to reach one line, against reading a number that
    // is already written down. The parameter is the cheaper of the two after all, since
    // the number written down is written by whoever embeds atoma and was believed
    // without a check.
    let record = RunRecord {
        seconds: seconds_between(started, &ended),
        started: started.to_string(),
        ended,
        ended_because: ended_because.to_string(),
        messages: session.messages.len(),
        iterations: inferences,
    };
    match serde_json::to_value(&record) {
        Ok(value) => runs.push(value),
        Err(e) => {
            tracing::warn!("could not describe this run: {}", e);
            return;
        }
    }
    session
        .extra
        .insert(RUNS_KEY.to_string(), Value::Array(runs));
}

/// Which word describes an ending, from the error that produced it.
///
/// The three soft stops are somebody's decision -- a ceiling that was configured, or a
/// person asking. Everything else is `failed`, which includes the loop guards: a run cut
/// short for repeating itself did not complete, and a report that said it had would hide
/// the thing most worth seeing.
fn ending_of(error: &anyhow::Error) -> &'static str {
    if error.downcast_ref::<MaxIterationsReached>().is_some() {
        "iterations"
    } else if error.downcast_ref::<RunTimeExceeded>().is_some() {
        "runtime"
    } else if error.downcast_ref::<StopRequested>().is_some() {
        "stopped"
    } else {
        "failed"
    }
}

fn save_whatever_was_reached(
    session: &mut Session,
    out_path: Option<&std::path::Path>,
    port: &dyn SessionPort,
    started: &str,
    ended_because: &str,
    inferences: usize,
) {
    let Some(path) = out_path else { return };

    record_run(session, started, ended_because, inferences);

    let repaired = answer_unanswered_tool_calls(session, TOOL_CALL_UNANSWERED);
    if repaired > 0 {
        tracing::warn!(
            "{} tool call(s) had no result; answered them so the session can be resumed",
            repaired
        );
    }

    match port.save(session, path) {
        Ok(()) => tracing::info!("Session saved to: {:?}", path),
        Err(e) => tracing::error!("Failed to save session to {:?}: {}", path, e),
    }
}

/// Run the agent: parse agent def, load session, connect MCP, run inference loop, save session.
pub async fn run(settings: RunSettings, deps: RunDeps<'_>) -> Result<RunOutcome> {
    let RunSettings {
        agent_def_path,
        in_session,
        prompt_file,
        out_session,
        template_path,
        tools_file,
        skills_dir,
        max_iterations,
        max_runtime,
        stop_file,
    } = settings;

    // Before anything else, so the duration covers what the run actually spent --
    // including parsing an agent definition and starting every tool server, which is
    // fixed overhead anybody looking at a slow run wants counted.
    let started = now_rfc3339();

    // 1. Parse agent definition
    let parsed_agent = deps
        .agent_def
        .parse(&agent_def_path)
        .context("Failed to parse agent definition")?;
    let agent = &parsed_agent.frontmatter;

    tracing::info!("Loaded agent: {} (model: {})", agent.name, agent.model);

    // 2. Read or create session
    let mut session = match in_session.as_ref() {
        Some(path) => deps.session.load(path)?,
        None => Session::default(),
    };

    tracing::info!("Session has {} messages", session.messages.len());

    // 3. Connect to MCP servers and discover tools
    let external_tools: Option<Box<dyn ToolPort + Send>> = if agent.mcp_servers.is_empty() {
        None
    } else {
        let tools_path = tools_file
            .as_ref()
            .context("Agent has mcp_servers configured but --tools-file was not specified")?;
        let tools_map = deps.tool_def.load(tools_path)?;

        let tool_defs: Vec<_> = agent
            .mcp_servers
            .iter()
            .map(|name| {
                tools_map.get(name).cloned().with_context(|| {
                    format!(
                        "{} (tools file: {:?})",
                        unknown_server_message(name, tools_map.keys().map(String::as_str)),
                        tools_path
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let reg = deps.mcp_factory.build(&tool_defs).await?;
        tracing::info!(
            "Connected to {} MCP server(s), discovered {} tool(s)",
            tool_defs.len(),
            reg.tool_definitions().len()
        );
        Some(reg)
    };

    let skill_catalog = match skills_dir.as_ref() {
        Some(path) => deps.skill.load(path)?,
        None => SkillCatalog::default(),
    };
    let skill_metadata = skill_catalog.metadata();
    let mut runtime_tools: Box<dyn ToolPort + Send> =
        Box::new(RuntimeTools::new(skill_catalog, external_tools)?);

    let tool_definitions = runtime_tools.tool_definitions();

    let tool_descriptions: Vec<String> = tool_definitions
        .iter()
        .map(|t| {
            let name = t
                .pointer("/function/name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            format!("- `{}`", name)
        })
        .collect();

    // 4. Build system prompt
    let custom_template: Option<String> = if let Some(ref path) = template_path {
        let t = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read template file: {:?}", path))?;
        Some(t)
    } else {
        None
    };
    let working_dir = std::env::current_dir()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| ".".to_string());

    let agent_def_dir = agent_def_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    let colleagues: Vec<(String, String)> = agent
        .knows_about
        .iter()
        .map(|name| {
            let candidate = agent_def_dir.join(format!("{}.md", name));
            let parsed = deps.agent_def.parse(&candidate).with_context(|| {
                format!(
                    "Agent '{}' listed in knows_about but its definition was not found at: {:?}",
                    name, candidate
                )
            })?;
            Ok((name.clone(), parsed.frontmatter.description))
        })
        .collect::<Result<Vec<_>>>()?;

    let system_prompt = deps.template.build_system_prompt(&PromptContext {
        agent: &parsed_agent,
        tool_descriptions: &tool_descriptions,
        custom_template: custom_template.as_deref(),
        working_dir: &working_dir,
        colleagues: &colleagues,
        skills: &skill_metadata,
    });
    tracing::debug!("System prompt:\n{}", system_prompt);

    // 5. Replace system message
    session.messages.retain(|m| m.role != "system");
    session.messages.insert(0, Message::system(&system_prompt));

    // 6. Resolve user prompt
    let prompt_text: Option<String> = if let Some(ref prompt_path) = prompt_file {
        let text = std::fs::read_to_string(prompt_path)
            .with_context(|| format!("Failed to read prompt file: {:?}", prompt_path))?;
        tracing::info!("Read prompt from file: {:?}", prompt_path);
        Some(text)
    } else if !std::io::stdin().is_terminal() {
        use std::io::Read;
        let mut text = String::new();
        std::io::stdin()
            .read_to_string(&mut text)
            .context("Failed to read prompt from stdin")?;
        if text.trim().is_empty() {
            None
        } else {
            tracing::info!("Read prompt from stdin ({} bytes)", text.len());
            Some(text)
        }
    } else {
        tracing::debug!("No prompt provided; running with existing session only");
        None
    };

    if let Some(ref text) = prompt_text {
        session.messages.push(Message::user(text));
    }

    // 7. Run inference loop
    let tools: Option<Vec<serde_json::Value>> = if tool_definitions.is_empty() {
        None
    } else {
        Some(tool_definitions)
    };

    let out_path = out_session.or(in_session);

    // Written by the loop as it goes, so that it is still right on the paths that never
    // return an `InferenceResult`: a ceiling reached, a stop file, a provider hanging
    // up. Zero here is the honest starting value -- a run that fails before its first
    // response really did make no round trip.
    let mut inferences: usize = 0;

    let inference_result = inference_loop(
        deps.llm,
        &agent.name,
        &agent.model,
        &mut session,
        tools.as_deref(),
        &agent.extra_body,
        &mut runtime_tools,
        max_iterations,
        max_runtime,
        stop_file.as_deref(),
        agent.vision,
        &mut inferences,
    )
    .await;

    let (response_text, total_usage, completion_reason) = match inference_result {
        Ok(InferenceResult::Completed {
            text,
            usage,
            reason,
        }) => (text, usage, reason),
        Ok(InferenceResult::SessionEnded) => {
            tracing::info!("Session suspended by tool request");
            // Through the same door as every other ending, which it was not before:
            // this arm saved the session itself and returned, so it never reached
            // `record_run`. A tool ending the session is how a run that FINISHED ends --
            // `create_pr` does it -- so the runs missing from `atoma_runs` were the
            // successful ones, and every rate computed over that list was computed over
            // the failures alone. Measured: a session with 426 messages carried one run
            // record, for the run that had been interrupted.
            //
            // `completed`, because nothing gave up here. The mechanism did not stop this
            // run; the agent reached an outcome and said so with a tool.
            save_whatever_was_reached(
                &mut session,
                out_path.as_deref(),
                deps.session,
                &started,
                "completed",
                inferences,
            );
            return Ok(RunOutcome::SessionEnded);
        }
        Err(e) => {
            // Saved whichever it was. Two questions used to be one, and answering
            // them together was throwing work away: whether this was the ending
            // somebody asked for decides the exit status and what gets said, and
            // whether the conversation is whole decides whether it is worth keeping.
            //
            // A provider that hangs up three times used to take the whole run's
            // history with it. The next run started from nothing, on an issue where
            // the work had already been done once.
            //
            // Discarding was never the machinery's decision to make, either: the
            // person has `--session-mode recover`, which archives the session and
            // starts fresh. Keeping it leaves them both options; discarding takes
            // one away, silently, and cannot be undone.
            if is_soft_stop(&e) {
                tracing::warn!("{}", e);
            } else {
                tracing::error!("Run failed: {}", e);
            }
            save_whatever_was_reached(
                &mut session,
                out_path.as_deref(),
                deps.session,
                &started,
                ending_of(&e),
                inferences,
            );
            return Err(e);
        }
    };

    // `cached=` says `unknown` rather than `0` when no inference reported one. The
    // reader of this line is a delivery script and then a person, and `0` is a claim
    // about the cache while `unknown` is a claim about the measurement.
    tracing::info!(
        "ATOMA_TOKEN_USAGE: prompt={} completion={} total={} cached={} written={}",
        total_usage.prompt_tokens,
        total_usage.completion_tokens,
        total_usage.total_tokens,
        total_usage
            .cached_prompt_tokens
            .map_or_else(|| "unknown".to_string(), |n| n.to_string()),
        total_usage
            .written_prompt_tokens
            .map_or_else(|| "unknown".to_string(), |n| n.to_string()),
    );

    // 8. Save session, through the helper the failing path already used -- which is what
    // puts every ending in `atoma_runs` rather than only the unhappy ones.
    save_whatever_was_reached(
        &mut session,
        out_path.as_deref(),
        deps.session,
        &started,
        "completed",
        inferences,
    );

    Ok(RunOutcome::Completed {
        text: response_text,
        usage: total_usage,
        reason: completion_reason,
        session_path: out_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iterations_of(session: &Session, index: usize) -> Option<u64> {
        let runs = session
            .extra
            .get(RUNS_KEY)
            .and_then(|v| v.as_array().cloned())
            .unwrap_or_default();
        runs.get(index)?.get("iterations").and_then(Value::as_u64)
    }

    /// The case that used to record zero. The previous run wrote `messages: 400`, then
    /// whoever embeds atoma compacted the history down to one message before this run
    /// started. The old derivation skipped 400 messages of a one-message session,
    /// counted nothing, and wrote a run that reads as one that never called the model.
    #[test]
    fn a_compacted_history_does_not_erase_this_runs_inferences() {
        let mut session = Session::default();
        session.messages.push(Message::assistant(Some("x"), None));
        let previous = serde_json::json!([{ "messages": 400 }]);
        session.extra.insert(RUNS_KEY.to_string(), previous);

        record_run(&mut session, "2026-01-01T00:00:00Z", "completed", 7);

        assert_eq!(iterations_of(&session, 1), Some(7));
    }

    /// The ordinary case, kept beside it: a session with no record of earlier runs
    /// reports what the loop counted, not what its messages imply.
    #[test]
    fn a_first_run_reports_the_count_it_was_given() {
        let mut session = Session::default();

        record_run(&mut session, "2026-01-01T00:00:00Z", "completed", 3);

        assert_eq!(iterations_of(&session, 0), Some(3));
    }
}

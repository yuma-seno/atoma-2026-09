//! What a run says about itself, and the single place that shape is decided.
//!
//! A run used to describe itself in four different places, each of which knew something
//! the others did not: the final text on stdout, an `ATOMA_TOKEN_USAGE` line on stderr
//! that carried two cache counts the JSON did not, an `atoma_runs` record that carried
//! the ending word and the duration and was written only when a session file had been
//! asked for, and an exit status that collapsed all three deliberate stops into `2` -- a
//! status clap also uses for its own parse errors. A caller that wanted to know why a
//! run ended had to read a log line, a session file it may not have asked for, and an
//! ambiguous number.
//!
//! So the facts are gathered in one struct that every ending fills in, and shaped into
//! one envelope that every ending prints. `atoma_runs` is still written -- it is the
//! per-run history, and a resumed session holds several of them -- but it is built from
//! the same `RunFacts`, so the word the session records and the word the envelope
//! reports cannot drift apart.

use serde_json::Value;
use std::path::PathBuf;

use crate::domain::ports::LlmUsage;

use super::CompletionReason;

/// Observable outcome of a run that reached an ending of its own. Presentation belongs
/// to the caller.
///
/// It carries what only this variant has -- the assistant's text and why the model
/// stopped -- and nothing that every ending has. Usage and the session path used to be
/// fields here, which is exactly how `SessionEnded` came to report no tokens at all: a
/// measurement attached to one variant is a measurement the other variants lose, and
/// the two endings that leave through `Err` have no variant to attach it to. Those live
/// in `RunFacts` now, which accompanies every exit path including those two.
#[derive(Debug)]
pub enum RunOutcome {
    Completed {
        text: String,
        reason: CompletionReason,
    },
    SessionEnded,
}

/// Everything a run knows about itself, whichever way it ended.
///
/// An out parameter of `run`, filled in as the run goes, for the reason `inference_loop`
/// gives about its own counter: every ending that is not a plain completion leaves
/// through `Err`, and those are exactly the runs whose numbers are worth having. A value
/// returned from `run` would describe the runs that finished and nothing else, while the
/// caller has to report all of them.
#[derive(Debug, Clone)]
pub struct RunFacts {
    /// RFC 3339, UTC. When the run began -- not when the job did.
    pub started: String,
    /// RFC 3339, UTC. Stamped by `conclude`, once, at whichever ending was reached.
    pub ended: String,
    /// How the run ended, in the terms the runner decides them: `completed`,
    /// `iterations`, `runtime`, `stopped` or `failed`.
    ///
    /// The same word `atoma_runs` records, because it is the same value: both readers
    /// are served from here.
    pub ended_because: &'static str,
    /// Whole seconds from `started` to `ended`.
    pub seconds: u64,
    /// Round trips to the model this run made. What a run waits on and is billed for.
    pub iterations: usize,
    /// Tokens this run spent, summed over the inferences that reported any. Partial on
    /// an ending that is not a completion, which is the honest answer: a run that failed
    /// on its fourth turn still spent what its first three cost.
    pub usage: LlmUsage,
    /// Where the session was written, if the caller asked for it to be.
    pub session_path: Option<PathBuf>,
}

impl Default for RunFacts {
    /// `failed` rather than an empty word, because these facts are printed on paths that
    /// may not have reached `conclude` at all. A run that fell over before it could name
    /// its own ending did not complete, and the default has to be the word that says so
    /// rather than one that reads as a successful run with a blank label.
    fn default() -> Self {
        Self {
            started: String::new(),
            ended: String::new(),
            ended_because: "failed",
            seconds: 0,
            iterations: 0,
            usage: LlmUsage::default(),
            session_path: None,
        }
    }
}

impl RunFacts {
    /// Start the clock.
    pub fn start(&mut self) {
        self.started = now_rfc3339();
    }

    /// Stamp the ending: which word it was, and the wall clock it took.
    ///
    /// The duration is computed here rather than by whoever writes the session, because
    /// the session is written only when `--out-session` was passed and the envelope is
    /// printed either way. Computing it at the save site is what made `seconds` a number
    /// that existed only for the callers who had already asked for a file.
    pub fn conclude(&mut self, because: &'static str) {
        self.ended = now_rfc3339();
        self.seconds = seconds_between(&self.started, &self.ended);
        self.ended_because = because;
    }
}

/// The `--output json` envelope for a run, whichever way it ended.
///
/// `outcome` is `None` for the endings that have no outcome: the three soft stops and
/// every failure. Those still fill in every key -- `response` and `finish_reason` as
/// `null` -- because a stable key set is what makes this machine-readable. A caller
/// reading `.ended_because` should not first have to work out which shape it was handed,
/// and `SessionEnded` printing nothing whatsoever was the extreme case of that: a run
/// that had genuinely finished was indistinguishable, on stdout, from one that produced
/// no output.
///
/// The four keys that existed before -- `response`, `usage`, `finish_reason`,
/// `session_path` -- keep their names and their meaning. `usage` gains the two cache
/// counts, which until now were only in the `ATOMA_TOKEN_USAGE` log line: the
/// human-readable channel carried more than the machine-readable one, and the counts it
/// carried are the ones that decide what a run cost, since a run here is almost entirely
/// prompt and a cached prompt token is billed at a fraction of a fresh one.
pub fn envelope(facts: &RunFacts, outcome: Option<&RunOutcome>) -> Value {
    let completed = match outcome {
        Some(RunOutcome::Completed { text, reason }) => Some((text, reason)),
        _ => None,
    };
    // `null` rather than an absent key, and `null` rather than `""`: an empty string is
    // a response the agent might actually have given.
    let response = completed.map(|(text, _)| text.as_str());
    let finish_reason = completed.map(|(_, reason)| word_for(*reason));
    let session_path = facts
        .session_path
        .as_ref()
        .map(|p| p.to_string_lossy().to_string());

    serde_json::json!({
        "response": response,
        "usage": {
            "prompt_tokens": facts.usage.prompt_tokens,
            "completion_tokens": facts.usage.completion_tokens,
            "total_tokens": facts.usage.total_tokens,
            // `null`, never `0`, when no inference reported one. The spelling of the
            // same distinction `ATOMA_TOKEN_USAGE` makes with `unknown`: zero is a
            // claim about the cache, absence is a claim about the measurement, and a
            // provider that reports no tokens at all -- GitHub Copilot bills per
            // request -- must not read afterwards as a cache that did nothing.
            "cached_prompt_tokens": facts.usage.cached_prompt_tokens,
            "written_prompt_tokens": facts.usage.written_prompt_tokens,
        },
        "finish_reason": finish_reason,
        "session_path": session_path,
        "ended_because": facts.ended_because,
        "seconds": facts.seconds,
        "iterations": facts.iterations,
    })
}

/// The wire word for a completion reason.
fn word_for(reason: CompletionReason) -> &'static str {
    match reason {
        CompletionReason::Stop => "stop",
        CompletionReason::Length => "length",
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_facts() -> RunFacts {
        RunFacts {
            started: "2026-01-01T00:00:00Z".to_string(),
            ended: "2026-01-01T00:00:09Z".to_string(),
            ended_because: "completed",
            seconds: 9,
            iterations: 4,
            usage: LlmUsage {
                prompt_tokens: 100,
                completion_tokens: 20,
                total_tokens: 120,
                cached_prompt_tokens: Some(80),
                written_prompt_tokens: Some(10),
            },
            session_path: Some(PathBuf::from("/tmp/session.json")),
        }
    }

    fn keys(value: &Value) -> Vec<String> {
        let mut names: Vec<String> = value
            .as_object()
            .expect("the envelope is a JSON object")
            .keys()
            .cloned()
            .collect();
        names.sort();
        names
    }

    /// The contract is the key set, not what any one path happens to print. A caller
    /// reads `.ended_because` without knowing in advance which way the run went, so
    /// every ending has to answer the same questions -- and a key that is missing on
    /// one path is a caller that crashes on that path, months later, on the day the
    /// ceiling is finally hit.
    #[test]
    fn every_exit_path_answers_the_same_questions() {
        let completed = RunOutcome::Completed {
            text: "done".to_string(),
            reason: CompletionReason::Stop,
        };

        let after_completion = envelope(&sample_facts(), Some(&completed));
        let after_session_end = envelope(&sample_facts(), Some(&RunOutcome::SessionEnded));
        let after_soft_stop = envelope(&sample_facts(), None);

        assert_eq!(keys(&after_completion), keys(&after_session_end));
        assert_eq!(keys(&after_completion), keys(&after_soft_stop));
        for name in [
            "response",
            "usage",
            "finish_reason",
            "session_path",
            "ended_because",
            "seconds",
            "iterations",
        ] {
            assert!(after_soft_stop.get(name).is_some(), "missing {name}");
        }
    }

    /// The keys a caller already reads keep meaning what they meant. Everything added
    /// here is additive, because the callers that parse this envelope today were written
    /// against those four names.
    #[test]
    fn a_finished_run_still_answers_what_it_always_answered() {
        let outcome = RunOutcome::Completed {
            text: "done".to_string(),
            reason: CompletionReason::Length,
        };

        let value = envelope(&sample_facts(), Some(&outcome));

        assert_eq!(value["response"], "done");
        assert_eq!(value["finish_reason"], "length");
        assert_eq!(value["session_path"], "/tmp/session.json");
        assert_eq!(value["usage"]["prompt_tokens"], 100);
    }

    /// A run a tool ended with `_meta.session_ends` is a run that FINISHED -- that is
    /// how a run which opened a pull request ends -- and it used to print nothing at all
    /// and exit 0, carrying no usage anywhere. Having no assistant text is a `null`
    /// response, not an absent envelope.
    #[test]
    fn a_session_a_tool_ended_still_reports_what_it_spent() {
        let value = envelope(&sample_facts(), Some(&RunOutcome::SessionEnded));

        assert_eq!(value["response"], Value::Null);
        assert_eq!(value["finish_reason"], Value::Null);
        assert_eq!(value["ended_because"], "completed");
        assert_eq!(value["usage"]["total_tokens"], 120);
        assert_eq!(value["iterations"], 4);
        assert_eq!(value["seconds"], 9);
    }

    /// The three deliberate stops all exit `2`, and so does clap's own parse error, so
    /// the status cannot say which ceiling was hit or whether a ceiling was hit at all.
    /// The word is the answer; it has to be in the envelope for the soft stops, which
    /// are the endings that print no response.
    #[test]
    fn a_soft_stop_says_which_ending_it_was() {
        let mut facts = sample_facts();
        facts.ended_because = "runtime";

        let value = envelope(&facts, None);

        assert_eq!(value["ended_because"], "runtime");
        assert_eq!(value["response"], Value::Null);
        assert_eq!(value["iterations"], 4);
        assert_eq!(value["seconds"], 9);
    }

    /// Absence is not zero. Zero reads as "the cache is doing nothing", which is the one
    /// conclusion an unmeasured cache must not be allowed to support -- and it is the
    /// conclusion a caller would draw about every run against a provider that reports no
    /// tokens at all.
    #[test]
    fn a_cache_nobody_measured_is_null_and_not_zero() {
        let mut facts = sample_facts();
        facts.usage.cached_prompt_tokens = None;
        facts.usage.written_prompt_tokens = None;

        let value = envelope(&facts, None);

        assert_eq!(value["usage"]["cached_prompt_tokens"], Value::Null);
        assert_eq!(value["usage"]["written_prompt_tokens"], Value::Null);
    }

    /// `conclude` is the only thing that names an ending, so the word in the envelope is
    /// the word in `atoma_runs` by construction rather than by two call sites agreeing.
    #[test]
    fn concluding_stamps_the_word_and_the_clock() {
        let mut facts = RunFacts::default();
        facts.start();
        facts.conclude("stopped");

        assert_eq!(facts.ended_because, "stopped");
        assert!(!facts.started.is_empty());
        assert!(!facts.ended.is_empty());
        assert_eq!(envelope(&facts, None)["ended_because"], "stopped");
    }

    /// Facts nobody managed to fill in describe a run that did not get to its own
    /// ending. `completed` as a default would report every such run as a success.
    #[test]
    fn facts_that_reached_no_ending_do_not_claim_one() {
        let value = envelope(&RunFacts::default(), None);

        assert_eq!(value["ended_because"], "failed");
    }
}

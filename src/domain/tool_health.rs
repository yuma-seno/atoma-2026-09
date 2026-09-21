//! What a tool server said about its own trouble, and how the agent hears it.
//!
//! # The problem
//!
//! A tool server can degrade without anyone finding out. The delivery template's
//! search server loads a reranker at startup; a permission change made its cache
//! unwritable, the load failed, and it fell back to a first-stage ranking. Every
//! search still answered. Every search answered worse. Two releases shipped that
//! way, and it was found because somebody grepped a log for an unrelated reason.
//!
//! Nothing could have noticed. No error was returned; the results looked like
//! results.
//!
//! # The shape
//!
//! Not a logging feature. A tool result today implicitly claims to be the whole
//! answer, and has no way to say "this is the best I could manage":
//!
//! > **A tool's answer includes how well it could answer.**
//!
//! So a warning is attached to that server's next tool result. Which means:
//!
//! - nothing to read, and nowhere to look. It arrives where it is used
//! - nothing at all on a healthy run
//! - no asking a model to sense a degradation it cannot see
//! - a warning raised at startup rides on the first call that follows. If no call
//!   follows, no tool was used and nothing was affected
//!
//! Chosen over a log file the agent reads because a file gives a model three
//! chances to go wrong -- deciding to read, reading the right part, connecting it
//! to the tool it used -- and an annotation gives it none. Information a model has
//! to go and fetch is the shape it is most willing to invent.
//!
//! # Two sources, one form
//!
//! This is the normalisation point, and that is the argument for it:
//!
//! | source | severity |
//! |---|---|
//! | `notifications/message` (MCP's `logging` capability) | the `level` field |
//! | the server's stderr | inferred from the text |
//!
//! The first is primary: structured, and it works whatever transport a server
//! speaks. The second is the fallback, and today it carries everything -- no server
//! implements `logging`, including atoma's own. Whichever it came from, the agent
//! sees one line attached to a result.

/// How bad a server said something was.
///
/// The syslog levels MCP uses, reduced to the question this asks: does the agent
/// need to know? `Warning` and worse are surfaced; the rest are for a log.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// debug, info, notice -- ordinary operation.
    Routine,
    /// warning -- it answered, and worse than it should have.
    Warning,
    /// error, critical, alert, emergency -- it could not do what was asked.
    Error,
}

/// The severity of an MCP `notifications/message` level string.
///
/// Unknown levels count as `Routine`. A server sending a level this does not
/// recognise is more likely to be sending something new and harmless than
/// something urgent, and treating the unknown as urgent would put noise into every
/// result -- which is how a channel gets ignored.
pub fn severity_of_level(level: &str) -> Severity {
    match level.trim().to_ascii_lowercase().as_str() {
        "warning" => Severity::Warning,
        "error" | "critical" | "alert" | "emergency" => Severity::Error,
        _ => Severity::Routine,
    }
}

/// The words that make a stderr line a report of trouble.
///
/// A list because the alternative is a rule, and every rule that reaches the
/// family (`warn`, `warning`, `warnings`) also reaches something that is not a
/// report -- a left-boundary-only match on `error` claims `errorless`. Naming the
/// words instead keeps the false positives to ones somebody chose.
const ERROR_WORDS: [&str; 4] = ["error", "errors", "fatal", "panic"];
const WARNING_WORDS: [&str; 3] = ["warn", "warning", "warnings"];

/// The severity of a line a server wrote to stderr.
///
/// A text match, and it is the fallback's weakness: a server that says "warning"
/// inside a sentence about something else is misread, and one that reports trouble
/// in words that are not on the list is missed. That is the cost of a channel with
/// no severity field, and the reason `notifications/message` is the primary one.
///
/// Anchored on word boundaries so `forward` and `errorless` do not match. Case
/// insensitive, because `WARN`, `warn` and `Warning` are all in use across the
/// servers this project ships.
pub fn severity_of_stderr(line: &str) -> Severity {
    let lower = line.to_ascii_lowercase();
    if ERROR_WORDS.iter().any(|&word| contains_word(&lower, word)) {
        return Severity::Error;
    }
    if WARNING_WORDS
        .iter()
        .any(|&word| contains_word(&lower, word))
    {
        return Severity::Warning;
    }
    Severity::Routine
}

/// The severity to record for a line a server wrote to a log channel.
///
/// `guess_from_words` is the whole of the difference, and it is off unless that
/// server's own entry in the tools file asked for it. The word list above was
/// calibrated on the output of servers somebody here had read; atoma ships no MCP
/// server at all, so a run that only wires up somebody else's inherits the
/// calibration without the servers it was calibrated on.
///
/// The measured case: `npx -y @modelcontextprotocol/server-filesystem` prints npm's
/// deprecation notice to stderr before the server starts, so that run's first tool
/// result carried "1 problem reported by the 'filesystem' server" describing npm --
/// and the person who connected the server had nothing to turn off. A build tool
/// that ends with "0 errors" is the same mistake pointed the other way.
///
/// Off means routine, not silent. The line is still logged where every other line
/// from that server is logged; what stops is a guess being attached to a tool result
/// as though the server had reported it.
///
/// `notifications/message` does not come through here. There the level is a field the
/// server filled in, nothing is inferred, and that is why that channel stays on by
/// default while this one has to be asked for.
pub fn severity_of_output(line: &str, guess_from_words: bool) -> Severity {
    if !guess_from_words {
        return Severity::Routine;
    }
    severity_of_stderr(line)
}

/// Whether `needle` appears in `haystack` bounded by non-alphanumerics.
///
/// `haystack` is expected lowercase already; this does not lowercase again.
fn contains_word(haystack: &str, needle: &str) -> bool {
    let bytes = haystack.as_bytes();
    let mut from = 0;
    while let Some(found) = haystack[from..].find(needle) {
        let start = from + found;
        let end = start + needle.len();
        let before_ok = start == 0 || !bytes[start - 1].is_ascii_alphanumeric();
        let after_ok = end == bytes.len() || !bytes[end].is_ascii_alphanumeric();
        if before_ok && after_ok {
            return true;
        }
        from = end;
    }
    false
}

/// What a server reported, waiting to be attached to its next result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HealthNote {
    pub severity: Severity,
    pub message: String,
}

/// How many distinct reports one result will carry.
///
/// Dedup stops one repeated sentence but not a server producing a fresh one every
/// second. Twenty is past the point where a person or a model would keep reading,
/// and the count that follows says how many did not fit -- which is the part that
/// matters once it is this many.
const MAX_NOTES: usize = 20;

/// Everything one server has reported this run, in order, without repeats.
///
/// Deduplicated because a server that warns on every call would otherwise append
/// the same sentence to every result -- and a run with fifty search calls would
/// carry it fifty times. The first time is information; the fiftieth is noise that
/// crowds out the answer.
#[derive(Debug, Default)]
pub struct HealthLog {
    notes: Vec<HealthNote>,
    dropped: usize,
}

impl HealthLog {
    /// Record a note if it is worth the agent's attention and is not already held.
    ///
    /// Returns whether it was kept, which is what lets a caller log the decision
    /// rather than guess at it.
    pub fn record(&mut self, severity: Severity, message: &str) -> bool {
        if severity == Severity::Routine {
            return false;
        }
        let message = message.trim();
        if message.is_empty() {
            return false;
        }
        if self.notes.iter().any(|note| note.message == message) {
            return false;
        }
        if self.notes.len() >= MAX_NOTES {
            self.dropped += 1;
            return false;
        }
        self.notes.push(HealthNote {
            severity,
            message: message.to_string(),
        });
        true
    }

    /// Take everything held, leaving it empty.
    ///
    /// Taken rather than read, so one note reaches one result. Leaving it in place
    /// would attach a startup warning to every call for the rest of the run, which
    /// is the repetition `record` already refuses within a single result.
    ///
    /// Anything the cap turned away is not silently gone: it becomes a last note
    /// saying how much. A truncated list that does not admit it is truncated reads
    /// as the whole story.
    pub fn drain(&mut self) -> Vec<HealthNote> {
        let mut notes = std::mem::take(&mut self.notes);
        let dropped = std::mem::take(&mut self.dropped);
        if dropped > 0 {
            notes.push(HealthNote {
                severity: Severity::Warning,
                message: format!("{dropped} further reports from this server were not shown"),
            });
        }
        notes
    }
}

/// The line appended to a tool result, or `None` when there is nothing to say.
///
/// Named as coming from the server, because the agent has to be able to tell this
/// apart from the tool's own answer -- and, downstream, from its own mistake. An
/// agent asked whether something went badly reaches for its own conduct first, so
/// the attribution has to be in the text.
pub fn annotation(server: &str, notes: &[HealthNote]) -> Option<String> {
    if notes.is_empty() {
        return None;
    }
    let mut lines = Vec::with_capacity(notes.len() + 1);
    lines.push(format!(
        "--- {} {} reported by the '{}' server, not part of the answer above ---",
        notes.len(),
        if notes.len() == 1 {
            "problem"
        } else {
            "problems"
        },
        server,
    ));
    for note in notes {
        let label = match note.severity {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Routine => "note",
        };
        lines.push(format!("{label}: {}", note.message));
    }
    Some(lines.join("\n"))
}

/// A tool's answer with what the server reported appended.
///
/// Appended rather than prepended: the answer is what was asked for, and a result
/// that opens with an aside about the tool buries it.
pub fn with_annotation(content: String, annotation: Option<String>) -> String {
    match annotation {
        None => content,
        Some(note) if content.trim().is_empty() => note,
        Some(note) => format!("{content}\n\n{note}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_levels_map_to_what_the_agent_needs_to_know() {
        assert_eq!(severity_of_level("warning"), Severity::Warning);
        assert_eq!(severity_of_level("ERROR"), Severity::Error);
        assert_eq!(severity_of_level("emergency"), Severity::Error);
        for routine in ["debug", "info", "notice", " "] {
            assert_eq!(severity_of_level(routine), Severity::Routine, "{routine}");
        }
    }

    /// A level this does not recognise is treated as routine. Guessing "urgent"
    /// would put noise into every result, which is how a channel comes to be
    /// ignored.
    #[test]
    fn an_unknown_level_is_routine() {
        assert_eq!(severity_of_level("verbose"), Severity::Routine);
        assert_eq!(severity_of_level("trace"), Severity::Routine);
    }

    /// The real line from the run this exists because of.
    #[test]
    fn the_reranker_line_is_a_warning() {
        let line = "[atoma-search] WARN could not preload the reranker (EACCES: permission denied)";
        assert_eq!(severity_of_stderr(line), Severity::Warning);
    }

    #[test]
    fn stderr_severity_reads_the_words_it_knows() {
        assert_eq!(
            severity_of_stderr("Error: could not connect"),
            Severity::Error
        );
        assert_eq!(
            severity_of_stderr("fatal: not a git repository"),
            Severity::Error
        );
        assert_eq!(
            severity_of_stderr("warning: falling back"),
            Severity::Warning
        );
        assert_eq!(
            severity_of_stderr("Secure MCP Filesystem Server running on stdio"),
            Severity::Routine,
        );
    }

    /// The plural is the same report. Named in the list rather than reached by a
    /// looser rule, because the looser rule also claims `errorless`.
    #[test]
    fn the_plurals_are_on_the_list() {
        assert_eq!(
            severity_of_stderr("3 warnings during startup"),
            Severity::Warning
        );
        assert_eq!(
            severity_of_stderr("2 errors while indexing"),
            Severity::Error
        );
    }

    /// Word boundaries, or ordinary output becomes a warning. `forward` and
    /// `errorless` both contain a keyword and neither is a report of trouble.
    #[test]
    fn a_keyword_inside_a_word_is_not_a_report() {
        assert_eq!(
            severity_of_stderr("forwarding to port 8080"),
            Severity::Routine
        );
        assert_eq!(
            severity_of_stderr("errorless parse completed"),
            Severity::Routine
        );
        assert_eq!(severity_of_stderr("warning"), Severity::Warning);
        assert_eq!(severity_of_stderr("[WARN] x"), Severity::Warning);
    }

    /// The line that made this opt-in. npm prints it before the server it is about
    /// to start has said anything, and it was reaching the model as a problem the
    /// `filesystem` server had reported about itself.
    #[test]
    fn the_guess_is_not_made_unless_it_was_asked_for() {
        let line = "npm warn deprecated glob@7.2.3: this package is no longer supported";
        assert_eq!(severity_of_output(line, false), Severity::Routine);
        assert_eq!(severity_of_output(line, true), Severity::Warning);
    }

    /// Asking for it gets the existing word list unchanged -- the flag gates the
    /// reading, it does not narrow it. A server whose operator calibrated the words
    /// against their own output is exactly who this is for.
    #[test]
    fn asking_for_the_guess_gets_the_whole_of_it() {
        assert_eq!(
            severity_of_output("fatal: not a git repository", true),
            Severity::Error
        );
        assert_eq!(
            severity_of_output("running on stdio", true),
            Severity::Routine
        );
    }

    #[test]
    fn routine_is_not_recorded() {
        let mut log = HealthLog::default();
        assert!(!log.record(Severity::Routine, "listening on stdio"));
        assert!(log.drain().is_empty());
    }

    /// Fifty search calls must not carry the same sentence fifty times. The first
    /// is information; the rest crowd out the answer.
    #[test]
    fn the_same_message_is_recorded_once() {
        let mut log = HealthLog::default();
        assert!(log.record(Severity::Warning, "reranker unavailable"));
        assert!(!log.record(Severity::Warning, "reranker unavailable"));
        assert!(!log.record(Severity::Warning, "  reranker unavailable  "));
        assert_eq!(log.drain().len(), 1);
    }

    #[test]
    fn draining_leaves_nothing_behind() {
        let mut log = HealthLog::default();
        log.record(Severity::Warning, "one");
        log.record(Severity::Error, "two");
        assert_eq!(log.drain().len(), 2);
        assert!(
            log.drain().is_empty(),
            "a note reaches one result, not every result after it"
        );
    }

    /// A server producing a fresh sentence every second defeats dedup, so there is a
    /// cap -- and the cap says how much it turned away, because a truncated list
    /// that does not admit it is truncated reads as the whole story.
    #[test]
    fn past_the_cap_the_count_is_kept_instead() {
        let mut log = HealthLog::default();
        for n in 0..MAX_NOTES + 7 {
            log.record(Severity::Warning, &format!("problem {n}"));
        }
        let notes = log.drain();
        assert_eq!(
            notes.len(),
            MAX_NOTES + 1,
            "the capped notes plus one count"
        );
        assert_eq!(
            notes.last().unwrap().message,
            "7 further reports from this server were not shown",
        );
        assert!(
            log.drain().is_empty(),
            "the count is reported once, not on every later result"
        );
    }

    #[test]
    fn under_the_cap_nothing_is_counted() {
        let mut log = HealthLog::default();
        log.record(Severity::Warning, "one");
        let notes = log.drain();
        assert_eq!(notes.len(), 1, "no trailing count when nothing was dropped");
    }

    #[test]
    fn an_empty_message_is_not_a_report() {
        let mut log = HealthLog::default();
        assert!(!log.record(Severity::Warning, "   "));
        assert!(log.drain().is_empty());
    }

    #[test]
    fn nothing_held_means_no_annotation() {
        assert_eq!(annotation("search", &[]), None);
    }

    /// The agent has to be able to tell this from the tool's own answer, and from
    /// its own mistake. So the server is named and the boundary is explicit.
    #[test]
    fn the_annotation_names_the_server_and_the_severity() {
        let notes = vec![HealthNote {
            severity: Severity::Warning,
            message: "reranker unavailable, results are first-stage ordered".to_string(),
        }];
        let text = annotation("search", &notes).expect("a note produces an annotation");
        assert!(text.contains("'search' server"), "{text}");
        assert!(text.contains("warning:"), "{text}");
        assert!(text.contains("not part of the answer above"), "{text}");
        assert!(text.contains("1 problem"), "{text}");
    }

    #[test]
    fn several_notes_are_counted_in_the_plural() {
        let notes = vec![
            HealthNote {
                severity: Severity::Warning,
                message: "a".into(),
            },
            HealthNote {
                severity: Severity::Error,
                message: "b".into(),
            },
        ];
        let text = annotation("shell", &notes).unwrap();
        assert!(text.contains("2 problems"), "{text}");
        assert!(text.contains("warning: a"), "{text}");
        assert!(text.contains("error: b"), "{text}");
    }

    /// Appended, not prepended: a result that opens with an aside about the tool
    /// buries the thing that was asked for.
    #[test]
    fn the_answer_comes_first() {
        let combined = with_annotation("the results".into(), Some("--- note ---".into()));
        assert!(combined.starts_with("the results"), "{combined}");
        assert!(combined.ends_with("--- note ---"), "{combined}");
    }

    #[test]
    fn no_annotation_leaves_the_answer_untouched() {
        assert_eq!(with_annotation("the results".into(), None), "the results");
    }

    /// A tool that returned nothing and warned should not open with a blank line.
    #[test]
    fn an_empty_answer_is_replaced_rather_than_padded() {
        assert_eq!(with_annotation("  ".into(), Some("note".into())), "note");
    }
}

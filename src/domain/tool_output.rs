//! What a tool is allowed to return.
//!
//! # The failure
//!
//! Nothing bounded a tool result. Whatever a server handed back became a message in
//! the session and a message in every request after it, at whatever size it was.
//!
//! Measured in one run of the delivery template (its #399, 352 tool calls):
//!
//! ```text
//!    230k chars  shell__shell_execute        capped by that server at 50,000
//!    141k chars  filesystem__read_text_file  capped by nobody
//!
//!    largest single result: 72,141 chars -- one file read, about 18k tokens,
//!    a seventh of a 128k window in a single message
//! ```
//!
//! The pattern is the point. `shell` is a server that repository wrote, and it caps
//! itself. `filesystem` is a third-party server, and it caps nothing — so the one
//! server nobody could change returned the largest results. **A limit on the client
//! is the only kind that covers a server somebody else wrote.**
//!
//! # Head and tail, not head
//!
//! A command's exit status, a stack trace's origin and a test summary are all at the
//! end. Keeping only the beginning of a build log throws away the part that says what
//! failed. A quarter at the front is enough for the echo of the command and the first
//! failing line; the rest goes to the end.
//!
//! This mirrors `domain/tool-output.ts` in the delivery template, deliberately: two
//! caps in two languages that behaved differently would be two things to learn.
//!
//! # Characters, not bytes
//!
//! A byte budget delivers about a third of its nominal size for Japanese text, and
//! that repository's issues are part Japanese. A limit whose real size depends on the
//! language of the content is a limit nobody can reason about.

/// How much of one tool result reaches the model, for a server that does not say
/// otherwise.
///
/// 50,000 characters, about 12.5k tokens: a tenth of the smallest context window
/// worth designing for. A tool call that wants more than a tenth of the window is a
/// call that should have returned less.
///
/// The same number the delivery template chose for the servers it writes. Adopting
/// it rather than picking another keeps one answer in two places.
pub const DEFAULT_MAX_OUTPUT_CHARS: usize = 50_000;

/// A capped result, and how much of it was dropped.
#[derive(Debug, PartialEq, Eq)]
pub struct Capped {
    pub text: String,
    /// Characters removed. Zero when nothing was.
    pub dropped: usize,
}

/// Keep the head and the tail of `text`, up to `limit` characters.
///
/// Counted in `char`s rather than bytes, so the cut never lands inside a multi-byte
/// character — which would be a panic on a `&str` slice, and a run that dies because
/// a tool returned Japanese.
///
/// The marker names the number of characters dropped, where from, and WHO dropped
/// them. A truncated result that does not say it is truncated is worse than a short
/// one: a `grep` that matched everything and a file that contains nothing look the
/// same.
///
/// The last of those is not decoration. A tool server may cap its own output before
/// returning it -- one that knows what it produced can often project rather than cut,
/// which loses nothing -- and this cap may then cut the result again. Two notes sit in
/// one tool result with two sets of numbers, and a person reading the session cannot
/// tell which of them describes the loss in front of them. Naming atoma costs seven
/// characters and answers that.
pub fn cap(text: &str, limit: usize) -> Capped {
    let total = text.chars().count();
    if total <= limit {
        return Capped {
            text: text.to_string(),
            dropped: 0,
        };
    }

    let dropped = total - limit;
    // A quarter at the front, the rest at the back. See the module comment for why
    // the tail is the bigger share.
    let head_len = limit / 4;
    let tail_len = limit - head_len;

    let head: String = text.chars().take(head_len).collect();
    let tail: String = text.chars().skip(total - tail_len).collect();

    Capped {
        text: format!(
            "{head}\n\n[atoma: {dropped} characters dropped from the middle; {limit} shown]\n\n{tail}"
        ),
        dropped,
    }
}

/// The limit for a server, from its configuration or the default.
///
/// `None` and `Some(0)` both mean the default, which is the rule every other limit
/// in this crate follows -- `infra::timeouts` made it one after three call sites took
/// a configured `0` literally and turned a stall detector into an immediate failure.
/// A server that genuinely wants no cap would have to say so in a way that reads as
/// saying so, and none does yet.
pub fn resolve_limit(configured: Option<usize>) -> usize {
    match configured {
        Some(n) if n > 0 => n,
        _ => DEFAULT_MAX_OUTPUT_CHARS,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_result_within_the_limit_is_untouched() {
        let capped = cap("short", 100);
        assert_eq!(capped.text, "short");
        assert_eq!(capped.dropped, 0);
    }

    #[test]
    fn exactly_the_limit_is_within_it() {
        let text = "x".repeat(50);
        assert_eq!(cap(&text, 50).dropped, 0);
    }

    /// The end is where the exit status, the stack trace's origin and the test
    /// summary are. Keeping only the beginning throws away what failed.
    #[test]
    fn both_ends_survive_and_the_tail_gets_the_larger_share() {
        let text = format!("{}{}", "A".repeat(500), "Z".repeat(500));
        let capped = cap(&text, 100);
        assert!(capped.text.starts_with("AAAA"), "{}", capped.text);
        assert!(capped.text.ends_with("ZZZZ"), "{}", capped.text);
        assert_eq!(capped.dropped, 900);
        // 25 from the head, 75 from the tail.
        assert_eq!(capped.text.matches('A').count(), 25);
        assert_eq!(capped.text.matches('Z').count(), 75);
    }

    /// A truncated result that does not say so is worse than a short one: a grep
    /// that matched everything and a file that contains nothing read the same.
    #[test]
    fn the_marker_says_how_much_went_and_from_where() {
        let capped = cap(&"x".repeat(1000), 100);
        assert!(
            capped
                .text
                .contains("900 characters dropped from the middle"),
            "{}",
            capped.text
        );
        assert!(capped.text.contains("100 shown"), "{}", capped.text);
    }

    /// Bytes would cut inside a character here, which on a `&str` slice is a panic --
    /// a run that dies because a tool returned Japanese.
    #[test]
    fn a_cut_never_lands_inside_a_character() {
        let text = "設計判断はコメントに書く。".repeat(100);
        let capped = cap(&text, 40);
        assert_eq!(capped.text.chars().filter(|c| *c != '\n').count() > 0, true);
        assert!(capped.dropped > 0);
        // The real assertion is that the two lines above did not panic.
        assert!(capped.text.contains("dropped from the middle"));
    }

    #[test]
    fn zero_and_absent_mean_the_default() {
        assert_eq!(resolve_limit(None), DEFAULT_MAX_OUTPUT_CHARS);
        assert_eq!(resolve_limit(Some(0)), DEFAULT_MAX_OUTPUT_CHARS);
    }

    #[test]
    fn a_configured_limit_is_used() {
        assert_eq!(resolve_limit(Some(1234)), 1234);
    }
}

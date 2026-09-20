use serde::Deserialize;
use std::collections::BTreeMap;

/// The name of the built-in tool that loads a skill.
///
/// Here rather than in `application`, because the system prompt names it too and
/// `infra::template` cannot reach an application constant without inverting the
/// dependency. It was a literal in that template, so renaming this would have told every
/// model, in every prompt, to call a tool that no longer exists -- and it would have
/// called it, received "Unknown tool", and loaded no skill, with nothing failing at
/// build time.
pub const LOAD_SKILL_TOOL: &str = "atoma_builtin__load_skill";

/// The name of that tool's one argument.
///
/// Here rather than in `application` for the same reason as the tool name above, and
/// shared for a sharper one: it is spelled in two places that cannot see each other --
/// the schema `application::tools` declares, and the corrected call
/// [`skill_called_as_tool_message`] writes out for a model to copy. They drifted. The
/// message went on teaching `{"name": ...}` after the schema became `skill_name` and
/// every other spelling stopped being accepted, so following the message exactly was a
/// refusal.
///
/// Nothing caught it: the argument key is a JSON string built on one side and a JSON
/// string built on the other, and the tests below assert the message names the tool and
/// the skill, never the key. One constant is what makes the two agree.
pub const LOAD_SKILL_ARGUMENT: &str = "skill_name";

/// What to say when a tool call names a skill instead of a tool.
///
/// A reviewer called `engineering/environment` as a tool and was told only "Invalid tool
/// name format (expected server__tool)". It did not call the loader afterwards -- it
/// reported to the pull request that it HAD run the skill, which was read by a person.
///
/// It reached for that name because the prompt hands it one: skills are listed as
/// ``- `engineering/environment`: ...`` and the prose says "Load `engineering/environment`",
/// in the same backticked shape a tool name takes. Presenting them so they cannot be
/// confused is the better fix and belongs in the prompt; this is the message for when
/// the confusion happens anyway, and it names the call that works rather than the rule
/// that was broken -- measured three times in this project as the difference between a
/// refusal that is followed and one that is not.
///
/// Only for a name shaped like a skill path. Anything else keeps the format error,
/// which is the honest answer for a tool name that is simply wrong.
pub fn skill_called_as_tool_message(name: &str) -> Option<String> {
    let (head, tail) = name.split_once('/')?;
    if head.is_empty() || tail.is_empty() || tail.contains('/') {
        return None;
    }
    Some(format!(
        "'{name}' is a skill, not a tool. A skill is loaded, not called: use {LOAD_SKILL_TOOL} \
         with {{\"{LOAD_SKILL_ARGUMENT}\": \"{name}\"}}, then follow what it returns. \
         Nothing has run yet."
    ))
}

/// Metadata exposed in the system prompt before a skill is loaded.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct SkillMetadata {
    pub name: String,
    pub description: String,
}

/// A validated skill available to the current run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    pub metadata: SkillMetadata,
    pub instructions: String,
}

/// Immutable, name-indexed set of skills validated at run startup.
#[derive(Debug, Clone, Default)]
pub struct SkillCatalog {
    skills: BTreeMap<String, Skill>,
}

impl SkillCatalog {
    pub fn new(skills: Vec<Skill>) -> anyhow::Result<Self> {
        let mut indexed = BTreeMap::new();
        for skill in skills {
            let name = skill.metadata.name.clone();
            if indexed.insert(name.clone(), skill).is_some() {
                anyhow::bail!("Duplicate skill name '{}'", name);
            }
        }
        Ok(Self { skills: indexed })
    }

    pub fn metadata(&self) -> Vec<SkillMetadata> {
        self.skills
            .values()
            .map(|skill| skill.metadata.clone())
            .collect()
    }

    pub fn get(&self, name: &str) -> Option<&Skill> {
        self.skills.get(name)
    }
}

#[cfg(test)]
mod skill_called_as_tool_tests {
    use super::{skill_called_as_tool_message, LOAD_SKILL_ARGUMENT, LOAD_SKILL_TOOL};

    /// The property is that the message names the call that works. A reviewer that saw
    /// only "expected server__tool" went on to tell a pull request it had run the skill.
    #[test]
    fn a_skill_path_is_told_which_tool_loads_it() {
        let message =
            skill_called_as_tool_message("engineering/environment").expect("a skill path");
        assert!(message.contains(LOAD_SKILL_TOOL));
        assert!(message.contains("engineering/environment"));
    }

    /// The corrected call is one a model copies verbatim, so the argument's name has to
    /// be the one the tool takes.
    ///
    /// It was not. This message went on writing `{"name": ...}` after the schema became
    /// `skill_name` and every other spelling stopped being accepted, which made following
    /// it exactly a refusal. Nothing caught it: the key was a JSON string built here and
    /// a JSON string built in `application::tools`, and the assertions above name the
    /// tool and the skill but never the key.
    #[test]
    fn the_corrected_call_names_the_argument_the_tool_takes() {
        let message = skill_called_as_tool_message("engineering/tdd").expect("a skill path");
        assert!(
            message.contains(&format!(r#"{{"{LOAD_SKILL_ARGUMENT}": "engineering/tdd"}}"#)),
            "{message}"
        );
    }

    /// It has to say that nothing happened, because the failure this came from was an
    /// agent reporting the skill as having run.
    #[test]
    fn the_message_says_nothing_has_run() {
        let message =
            skill_called_as_tool_message("review/quick-quality-gate").expect("a skill path");
        assert!(message.to_lowercase().contains("nothing has run"));
    }

    /// A tool name that is simply wrong keeps the format error. Widening this to every
    /// unrecognised name would answer "did you mean a skill?" to a hallucinated tool,
    /// which reads as confirmation that the skill exists.
    #[test]
    fn a_name_that_is_not_a_skill_path_is_left_alone() {
        for name in ["github", "create_pr", "a/b/c", "/leading", "trailing/"] {
            assert!(
                skill_called_as_tool_message(name).is_none(),
                "{name} should keep the format error"
            );
        }
    }
}

/// The built-in template, as a port implementation.
///
/// A unit struct with no state: what it renders is this module's own constant, and the
/// point of the port is that `application` does not need to know that.
pub struct FileTemplateAdapter;

impl crate::domain::ports::TemplatePort for FileTemplateAdapter {
    fn build_system_prompt(&self, context: &crate::domain::ports::PromptContext<'_>) -> String {
        build_system_prompt(
            context.agent,
            context.tool_descriptions,
            context.custom_template,
            context.working_dir,
            context.colleagues,
            context.skills,
        )
    }
}

#[cfg(test)]
use crate::domain::agent::AgentDef;
use crate::domain::agent::ParsedAgentDef;
use crate::domain::skill::SkillMetadata;

/// The system prompt a run gets when the caller passes no `--template`.
///
/// It says what atoma provides, and nothing else. It used to also teach a handover
/// convention -- "include a `/agent-name` command in your output text" -- and claim
/// that agents share memory. atoma implements neither: nothing here reads a model's
/// output text looking for a command, and there is no shared memory. The only
/// implementation of that convention lives in an embedder, which passes its own
/// `--template` and so never saw these lines. Everyone else was the reader.
///
/// The example was wrong as well. The one parser that exists matches a whole line
/// against `^/([a-z][a-z0-9-]*)$`, and `/ReviewAgent Please review...` fails it twice --
/// for the capitals and for the text after the name.
///
/// `{{COLLEAGUES_LIST}}` stays. Who else works here is a fact about the environment;
/// how work reaches them is not, so an embedder that implements a handover says so in
/// its own template.
static DEFAULT_TEMPLATE: &str = r#"# Identity & Purpose
You are "{{AGENT_NAME}}".

{{AGENT_ROLE_PROMPT}}

# Available Colleagues
Other agents work in this environment. Below is who they are and what each one does.

{{COLLEAGUES_LIST}}

# Environment & Tools
Working directory: `{{WORKING_DIRECTORY}}`

You interact with the environment through the Model Context Protocol (MCP). Do not guess code or environment state; always execute tools to verify facts.

Each tool runs as its own process and receives only the credentials its own configuration declares. A credential you cannot see from one tool is confined, not missing: a shell that reports nothing for an API token is behaving as intended, and the tool that needs that token has it. Do not hardcode a value, hunt for it in other places, or conclude the setup is broken because a token is absent from where you looked. If a tool genuinely fails to authenticate, report which tool and what it said.

{{AVAILABLE_TOOLS}}

# Available Skills
A skill is a set of instructions this project has written for a particular kind of work. When the work in front of you is of a kind a skill below covers, call `{{LOAD_SKILL_TOOL}}` and follow it in place of your own approach -- it is what this project has decided, not advice to weigh.

The list carries names and descriptions only. A description is not the instructions, so load the skill rather than reconstructing it from the line or reading the file yourself; loading counts toward no limit. Check the list again whenever the work changes shape, because a skill that was irrelevant when the run started becomes relevant the moment the work reaches it.

{{AVAILABLE_SKILLS}}

# Thought Process & Execution
Before taking action or generating final output, always use the `<thought>` tag to develop a rigorous thought process following the steps below:

<thought>
1. [Analyze]: Analyze the current context, requirements, and environment state.
2. [Plan]: Plan the next steps to execute based on your role and available tools.
3. [Act & Verify]: Execute tools and verify results. If errors or unexpected results occur, analyze the cause and re-execute. Do not proceed based on assumptions.
4. [Communicate]: Determine task completion, blockage status, and what text to output.
</thought>

# Strict Rules
- [Tone] Eliminate all greetings, unnecessary apologies, and verbose explanations. Communicate in a technical and concise manner.
- [Tool Trustworthiness] Do not fabricate (hallucinate) file contents or execution results.
- [Autonomy & Coordination] Do not repeatedly call yourself or other agents without purpose (no infinite loops).
"#;

/// Everything a template may say, as one list.
///
/// The vocabulary was written out three times -- in `DEFAULT_TEMPLATE`, in a doc
/// comment above this function, and in the `replace` calls that did the work -- and
/// nothing held them together. An eighth placeholder added to two of the three would
/// have looked right in review.
///
/// An enum rather than an array of strings, because it makes the substitution below
/// an exhaustive match: a variant added here does not compile until it has a value.
/// That is the property `atoma validate` now depends on. Without it, validation
/// would be checking a template against a list that is only believed to be current.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placeholder {
    AgentName,
    AgentRolePrompt,
    ColleaguesList,
    AvailableTools,
    AvailableSkills,
    LoadSkillTool,
    WorkingDirectory,
}

impl Placeholder {
    /// Every one of them. The order is the order they are substituted in, which does
    /// not matter -- no value here contains another's token.
    pub const ALL: [Placeholder; 7] = [
        Placeholder::AgentName,
        Placeholder::AgentRolePrompt,
        Placeholder::ColleaguesList,
        Placeholder::AvailableTools,
        Placeholder::AvailableSkills,
        Placeholder::LoadSkillTool,
        Placeholder::WorkingDirectory,
    ];

    /// What it looks like in a template.
    pub fn token(self) -> &'static str {
        match self {
            Placeholder::AgentName => "{{AGENT_NAME}}",
            Placeholder::AgentRolePrompt => "{{AGENT_ROLE_PROMPT}}",
            Placeholder::ColleaguesList => "{{COLLEAGUES_LIST}}",
            Placeholder::AvailableTools => "{{AVAILABLE_TOOLS}}",
            Placeholder::AvailableSkills => "{{AVAILABLE_SKILLS}}",
            Placeholder::LoadSkillTool => "{{LOAD_SKILL_TOOL}}",
            Placeholder::WorkingDirectory => "{{WORKING_DIRECTORY}}",
        }
    }
}

/// A `{{...}}` in a template that nothing will substitute, in the order it appears.
///
/// Worth reporting rather than tolerating, because the failure is silent and looks
/// like an instruction: an unsubstituted `{{AVAILABLE_TOOL}}` renders literally into
/// the system prompt, and a model reads it as text it was given on purpose.
///
/// Only `{{NAME}}` shapes are considered. A template is prose, and something like
/// `{{ see the docs }}` is a sentence rather than a mistyped placeholder -- so the
/// name has to look like one: capitals, digits and underscores.
pub fn unknown_placeholders(template: &str) -> Vec<String> {
    let known: Vec<&str> = Placeholder::ALL.iter().map(|p| p.token()).collect();
    let mut unknown = Vec::new();
    let mut rest = template;

    while let Some(start) = rest.find("{{") {
        let after = &rest[start + 2..];
        let Some(end) = after.find("}}") else { break };
        let name = &after[..end];
        rest = &after[end + 2..];

        let looks_like_one = !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
        if !looks_like_one {
            continue;
        }
        let token = format!("{{{{{name}}}}}");
        if !known.contains(&token.as_str()) && !unknown.contains(&token) {
            unknown.push(token);
        }
    }
    unknown
}

/// Build the system prompt by substituting template variables.
///
/// The vocabulary is `Placeholder`; see there for why it is not listed here too.
///
/// Pass `custom_template` to override the built-in template entirely.
pub fn build_system_prompt(
    agent: &ParsedAgentDef,
    tool_descriptions: &[String],
    custom_template: Option<&str>,
    working_dir: &str,
    colleagues: &[(String, String)],
    skills: &[SkillMetadata],
) -> String {
    let template = custom_template.unwrap_or(DEFAULT_TEMPLATE);
    let mut prompt = template.to_string();

    for placeholder in Placeholder::ALL {
        // Exhaustive on purpose: a placeholder added to the enum does not compile
        // until it has a value here.
        let value: String = match placeholder {
            Placeholder::AgentName => agent.frontmatter.name.clone(),
            Placeholder::AgentRolePrompt => agent
                .body
                .as_deref()
                .unwrap_or(&agent.frontmatter.description)
                .to_string(),
            Placeholder::ColleaguesList => {
                if colleagues.is_empty() {
                    "No agents currently available for collaboration.".to_string()
                } else {
                    colleagues
                        .iter()
                        .map(|(name, desc)| format!("- `{}`: {}", name, desc))
                        .collect::<Vec<_>>()
                        .join("\n")
                }
            }
            Placeholder::AvailableTools => {
                if tool_descriptions.is_empty() {
                    "No tools currently available.".to_string()
                } else {
                    tool_descriptions.join("\n")
                }
            }
            Placeholder::AvailableSkills => {
                if skills.is_empty() {
                    "No skills currently available.".to_string()
                } else {
                    skills
                        .iter()
                        .map(|skill| format!("- `{}`: {}", skill.name, skill.description))
                        .collect::<Vec<_>>()
                        .join("\n")
                }
            }
            // The one place the tool's name enters the prompt. A custom template written
            // before this placeholder existed simply keeps whatever it says, which is the
            // same tolerance every other placeholder here has.
            Placeholder::LoadSkillTool => crate::domain::skill::LOAD_SKILL_TOOL.to_string(),
            Placeholder::WorkingDirectory => working_dir.to_string(),
        };
        prompt = prompt.replace(placeholder.token(), &value);
    }

    prompt
}

#[cfg(test)]
mod placeholder_tests {
    use super::{unknown_placeholders, Placeholder, DEFAULT_TEMPLATE};

    /// The invariant that makes the vocabulary trustworthy: the template shipped in
    /// this binary uses all of it and nothing else. A token added to the enum and
    /// forgotten in the template, or the reverse, shows up here.
    #[test]
    fn the_built_in_template_uses_the_whole_vocabulary_and_nothing_else() {
        for placeholder in Placeholder::ALL {
            assert!(
                DEFAULT_TEMPLATE.contains(placeholder.token()),
                "the built-in template never uses {}",
                placeholder.token(),
            );
        }
        assert_eq!(unknown_placeholders(DEFAULT_TEMPLATE), Vec::<String>::new());
    }

    /// The failure this reports is silent: an unsubstituted placeholder renders
    /// literally into the system prompt, and a model reads it as text it was handed
    /// on purpose.
    #[test]
    fn a_placeholder_nothing_substitutes_is_reported() {
        let unknown = unknown_placeholders("Hello {{AGENT_NAME}}, use {{AVAILABLE_TOOL}}.");
        assert_eq!(unknown, vec!["{{AVAILABLE_TOOL}}".to_string()]);
    }

    #[test]
    fn each_unknown_is_reported_once_in_the_order_it_appears() {
        let unknown = unknown_placeholders("{{B_ONE}} {{A_TWO}} {{B_ONE}}");
        assert_eq!(
            unknown,
            vec!["{{B_ONE}}".to_string(), "{{A_TWO}}".to_string()],
        );
    }

    /// A template is prose. Braces around a sentence are a sentence, and reporting
    /// them would teach whoever reads this to ignore it.
    #[test]
    fn something_that_is_not_shaped_like_a_placeholder_is_left_alone() {
        for text in [
            "see {{ the docs }} for more",
            "a JSON example: {{\"a\": 1}}",
            "{{lowercase}} is prose",
            "an unclosed {{THING",
        ] {
            assert_eq!(unknown_placeholders(text), Vec::<String>::new(), "{text}");
        }
    }

    #[test]
    fn a_template_with_no_placeholders_at_all_is_fine() {
        assert_eq!(unknown_placeholders("just words"), Vec::<String>::new());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_test_agent(body: Option<String>) -> ParsedAgentDef {
        ParsedAgentDef {
            frontmatter: AgentDef {
                name: "TestAgent".to_string(),
                description: "A test agent for unit testing".to_string(),
                model: "openrouter/anthropic/claude-3.5-sonnet".to_string(),
                provider: None,
                vision: false,
                knows_about: vec!["ReviewAgent".to_string()],
                mcp_servers: vec![],
                extra_body: HashMap::default(),
                extra_headers: HashMap::default(),
            },
            body,
        }
    }

    fn default_colleagues() -> Vec<(String, String)> {
        vec![(
            "ReviewAgent".to_string(),
            "Agent responsible for reviews".to_string(),
        )]
    }

    #[test]
    fn test_custom_body_injected_into_template() {
        let agent = make_test_agent(Some("Custom role description".to_string()));
        let result = build_system_prompt(&agent, &[], None, "/repo", &default_colleagues(), &[]);
        assert!(result.contains("TestAgent"));
        assert!(result.contains("Custom role description"));
        assert!(result.contains("Strict Rules"));
    }

    #[test]
    fn test_description_fallback_when_no_body() {
        let agent = make_test_agent(None);
        let result = build_system_prompt(&agent, &[], None, "/repo", &default_colleagues(), &[]);
        assert!(result.contains("A test agent for unit testing"));
    }

    #[test]
    fn test_template_substitution() {
        let agent = make_test_agent(None);
        let result = build_system_prompt(
            &agent,
            &["- `read_file`".to_string()],
            None,
            "/repo",
            &default_colleagues(),
            &[],
        );
        assert!(result.contains("TestAgent"));
        assert!(result.contains("A test agent for unit testing"));
        assert!(result.contains("ReviewAgent"));
        assert!(result.contains("read_file"));
        assert!(result.contains("/repo"));
    }

    #[test]
    fn test_custom_template() {
        let agent = make_test_agent(None);
        let custom = "Hello {{AGENT_NAME}}! Role: {{AGENT_ROLE_PROMPT}}";
        let result = build_system_prompt(&agent, &[], Some(custom), "/repo", &[], &[]);
        assert_eq!(
            result,
            "Hello TestAgent! Role: A test agent for unit testing"
        );
    }

    #[test]
    fn test_working_dir_substitution() {
        let agent = make_test_agent(None);
        let result = build_system_prompt(
            &agent,
            &[],
            None,
            "/home/runner/work/myrepo",
            &default_colleagues(),
            &[],
        );
        assert!(result.contains("/home/runner/work/myrepo"));
    }

    #[test]
    fn test_colleague_with_description() {
        let agent = make_test_agent(None);
        let colleagues = vec![
            (
                "engineer".to_string(),
                "Agent responsible for implementation".to_string(),
            ),
            (
                "reviewer".to_string(),
                "Agent responsible for reviews".to_string(),
            ),
        ];
        let result = build_system_prompt(&agent, &[], None, "/repo", &colleagues, &[]);
        assert!(result.contains("`engineer`: Agent responsible for implementation"));
        assert!(result.contains("`reviewer`: Agent responsible for reviews"));
    }

    #[test]
    fn test_skill_catalog_exposes_metadata_without_instructions() {
        let agent = make_test_agent(None);
        let skills = vec![SkillMetadata {
            name: "engineering/tdd".to_string(),
            description: "Test first.".to_string(),
        }];
        let result = build_system_prompt(&agent, &[], None, "/repo", &[], &skills);
        assert!(result.contains("`engineering/tdd`: Test first."));
        assert!(!result.contains("red-green-refactor"));
    }
}

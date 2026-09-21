use anyhow::{bail, Result};
use std::path::{Path, PathBuf};

use crate::domain::ports::{AgentDefPort, ToolDefPort};
use crate::domain::tool::{unknown_server_message, ToolDef};
use crate::infra::llm::{check_credentials, check_provider_name};
use crate::infra::template::unknown_placeholders;

/// Validate an agent definition file and optional tools file.
///
/// Checks:
///   1. Agent definition parses without error (YAML, required fields).
///   2. Each `knows_about` entry has a corresponding `<name>.md` in the same
///      directory, and it parses. That is the whole check: naming an agent in
///      `knows_about` IS saying it may be delegated to, so there is no second
///      declaration to agree with. `callable_by` used to be that second
///      declaration and is gone.
///   3. `extra_body` does not override the reserved keys `model` or `messages`.
///   4. If a tools file is provided:
///      a. The tools file parses without error.
///      b. Each `mcp_servers` entry is present in the tools file.
///   5. `provider`, when named, is one this build has.
///   8. If a template is provided: every `{{...}}` in it is one that gets
///      substituted.
///
/// The last two are here rather than in a caller because they are facts only this
/// crate holds. `atoma-autonomous-delivery` runs this against every agent
/// definition a pull request would merge; checking them there would mean keeping a
/// copy of the provider list and of the template vocabulary in another repository,
/// in another language.
///
/// Findings come in two kinds, and `strict` is the caller's answer about the second.
/// An error is a definition that cannot do what it says: a `knows_about` target with
/// no file behind it, a provider this build does not have. A warning is a
/// configuration that is defined and merely unusual, or a check this run could not
/// make. Without `strict` a warning is printed and the command still succeeds; with
/// it, every warning is an error.
///
/// The flag exists because the alternative was this function deciding. A server
/// setting both `tool_allowlist` and `tool_denylist` was fatal HERE while
/// `describe_hooks` warned about the same configuration and let the run proceed,
/// `docs/tools-and-skills.md` said both may be set, and the field doc on
/// `domain::tool::Hooks` gave the precedence -- three places describing a
/// well-defined configuration and a fourth refusing it, leaving a reader no way to
/// know which one the program obeyed. The reason recorded for refusing was that a
/// delivery runs this over every pull request, which is that delivery's policy and
/// not a fact about the file. So the default is now the answer a run gives, and a
/// caller who wants the old one asks for it.
pub fn validate(
    agent_def_path: PathBuf,
    tools_file: Option<PathBuf>,
    template_file: Option<PathBuf>,
    strict: bool,
    agent_def_port: &dyn AgentDefPort,
    tool_def_port: &dyn ToolDefPort,
) -> Result<()> {
    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    let parsed_agent = match agent_def_port.parse(&agent_def_path) {
        Ok(a) => {
            println!("✓ Agent definition parsed: {}", a.frontmatter.name);
            Some(a)
        }
        Err(e) => {
            errors.push(format!("Agent definition parse error: {}", e));
            None
        }
    };

    if let Some(ref parsed) = parsed_agent {
        let agent = &parsed.frontmatter;
        let agent_def_dir = agent_def_path
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));

        for name in &agent.knows_about {
            let candidate = agent_def_dir.join(format!("{}.md", name));
            if candidate.exists() {
                println!("  ✓ knows_about '{}' → {:?}", name, candidate);
                match agent_def_port.parse(&candidate) {
                    // Parsing it is the check. A `knows_about` entry naming a file that
                    // exists and reads as an agent definition is a delegation that can
                    // happen; there is no second declaration to agree with.
                    Ok(_) => {}
                    Err(e) => {
                        errors.push(format!(
                            "knows_about '{}': failed to parse target definition: {}",
                            name, e
                        ));
                    }
                }
            } else {
                errors.push(format!(
                    "knows_about '{}': definition file not found at {:?}",
                    name, candidate
                ));
            }
        }

        // The adapters' own list, not a copy of it. This was `["model", "messages"]`
        // written out here, so `validate` passed an `extra_body.input` that the Responses
        // adapter drops and an `extra_body.system` that Anthropic's does -- reporting a
        // configuration as sound while two of the three dialects quietly ignored part of
        // it.
        for key in crate::infra::llm::shared::RESERVED_KEYS {
            if agent.extra_body.contains_key(key) {
                errors.push(format!(
                    "extra_body contains reserved key '{}' which would be silently ignored",
                    key
                ));
            }
        }

        // The same rule for headers, and from the same source: the providers
        // themselves, not a list written here a second time.
        //
        // Refused rather than dropped. A header Atoma sets carries authentication, the
        // shape of the body, the API version, or -- for Copilot -- which models the
        // account may reach. An agent that quietly lost one of those would not see a
        // rejected setting; it would see the provider behaving strangely.
        let reserved = crate::infra::llm::reserved_header_names();
        for name in agent.extra_headers.keys() {
            if reserved.iter().any(|r| r == &name.to_lowercase()) {
                errors.push(format!(
                    "extra_headers contains '{}', which Atoma sets itself",
                    name
                ));
            }
        }

        // The name only. A provider that does not exist is a defect in the definition;
        // a credential that is not set is not, and a validation run has none -- so
        // conflating them would fail every run of this command.
        if let Some(ref provider) = agent.provider {
            let name = provider.trim();
            if name.is_empty() {
                errors.push("provider is present but empty; remove it or name one".to_string());
            } else {
                match check_provider_name(name) {
                    Ok(()) => println!("  ✓ provider '{}' is known", name),
                    Err(e) => errors.push(format!("{}", e)),
                }
            }
        }

        // A `tools` that is not an array is refused HERE, where the person who wrote it
        // is reading, rather than at the first request that uses the agent.
        //
        // This check was here once, was removed, and the removal is the clearest example
        // of what a tolerance costs. `reconcile_tools` had grown an arm that warned and
        // kept the runtime definitions instead of refusing, and the comment that replaced
        // this check said so: "being stricter than the code it describes is its own kind
        // of wrong answer". Correct, and backwards -- the code was wrong, and making the
        // validator agree with it removed the last thing that could have said so.
        if let Some(tools) = agent.extra_body.get("tools") {
            if !tools.is_array() {
                errors.push(
                    "extra_body.tools must be an array of tool definitions; \
                     it cannot be added to this run's tools as written"
                        .to_string(),
                );
            }
        }

        if let Some(ref tools_path) = tools_file {
            match tool_def_port.load(tools_path) {
                Ok(tools_map) => {
                    println!("✓ Tools file parsed: {} server(s) defined", tools_map.len());
                    // Both lists has a defined meaning: the denylist is checked
                    // first, so a tool named in both is blocked. `check_access` acts
                    // on that order, the field doc on `domain::tool::Hooks` states
                    // it, a test in `infra::hooks` pins it, and `describe_hooks`
                    // warns rather than refusing. This used to call the same file
                    // fatal, which left a reader four descriptions of one
                    // configuration and no way to tell which the program obeyed.
                    //
                    // Still said here, because validation is where it is cheap to
                    // hear: the runtime warning arrives only after every server has
                    // been spawned. Said at the volume the run uses, and a caller
                    // whose policy is stricter than the runtime's says so with
                    // `--strict`.
                    let mut both: Vec<&str> = tools_map
                        .iter()
                        .filter(|(_, def)| {
                            !def.hooks.tool_allowlist.is_empty()
                                && !def.hooks.tool_denylist.is_empty()
                        })
                        .map(|(name, _)| name.as_str())
                        .collect();
                    both.sort_unstable();
                    for server in both {
                        warnings.push(format!(
                            "Server '{}' sets both tool_allowlist and tool_denylist. \
                             The denylist is checked first, so a tool matching both is \
                             blocked.",
                            server
                        ));
                    }
                    for server in &agent.mcp_servers {
                        if tools_map.contains_key(server.as_str()) {
                            println!("  ✓ mcp_servers '{}' found in tools file", server);
                        } else {
                            errors.push(format!(
                                "{} (tools file: {:?})",
                                unknown_server_message(
                                    server,
                                    tools_map.keys().map(String::as_str)
                                ),
                                tools_path
                            ));
                        }
                    }
                }
                Err(e) => {
                    errors.push(format!("Tools file parse error: {}", e));
                }
            }
        } else if !agent.mcp_servers.is_empty() {
            // A warning rather than a line of prose in the middle of the output.
            // Every server check above needed the file that was not passed, so this
            // run answered less than its "Validation passed." suggests -- and a
            // caller that asked for `--strict` is exactly the one for whom a check
            // that did not happen should count.
            warnings.push(
                "mcp_servers is non-empty but --tools-file was not provided; \
                 the servers it names were not checked"
                    .to_string(),
            );
        }
    }

    // Independent of the agent definition, so it runs even when that failed to parse:
    // a template is wrong or right on its own terms, and reporting both problems at
    // once beats reporting them one run apart.
    if let Some(ref template_path) = template_file {
        match std::fs::read_to_string(template_path) {
            Ok(template) => {
                let unknown = unknown_placeholders(&template);
                if unknown.is_empty() {
                    println!("✓ Template checked: every placeholder is one atoma substitutes");
                } else {
                    // Not a warning. An unsubstituted placeholder renders literally into
                    // the system prompt, where a model reads it as text it was given on
                    // purpose -- which is worse than an obviously missing section.
                    errors.push(format!(
                        "template {:?} uses {} placeholder(s) nothing substitutes: {}",
                        template_path,
                        unknown.len(),
                        unknown.join(", "),
                    ));
                }
            }
            Err(e) => errors.push(format!(
                "Failed to read template {:?}: {}",
                template_path, e
            )),
        }
    }

    // `strict` is the caller's policy about warnings and not a second opinion about
    // the configuration: what each check found is unchanged, and only the exit status
    // moves. Applied here rather than at each check so that a warning added later
    // cannot forget to honour it.
    if strict {
        errors.append(&mut warnings);
    }
    for warning in &warnings {
        println!("  ⚠ {}", warning);
    }

    if errors.is_empty() {
        println!("\nValidation passed.");
        Ok(())
    } else {
        eprintln!("\nValidation failed with {} error(s):", errors.len());
        for e in &errors {
            eprintln!("  ✗ {}", e);
        }
        bail!("Validation failed")
    }
}

/// Whether the environment about to run this definition holds the credential it needs.
///
/// Separate from `validate` rather than a flag inside it, because it is a different
/// question and the two must not be confused. `validate` asks whether the definition
/// is correct, and runs where no credential exists -- a pull request checking a file
/// -- which is why `a_known_provider_passes_without_its_credential` exists. This asks
/// whether the runner is equipped, and only a caller that says so gets it.
///
/// Reported by `atomaton` #766: switching provider without adding the matching secret
/// is silent until the run has checked out, installed dependencies and started tool
/// servers. The information needed to say so earlier is all in this crate, and this is
/// the door to it.
pub fn validate_credentials(
    agent_def_path: &Path,
    present: &[String],
    agent_def_port: &dyn AgentDefPort,
) -> Result<()> {
    let parsed = agent_def_port.parse(agent_def_path)?;
    // The same precedence a run uses: an empty `provider:` is no hint rather than a
    // hint to nothing, which is what `validate` already reports as its own error.
    let hint = parsed
        .frontmatter
        .provider
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty());
    println!("  ✓ {}", check_credentials(hint, present)?);
    Ok(())
}

/// What starting this definition's servers says about the tools file.
///
/// Opt-in for the same reason as the credential check: it needs more than a pull
/// request has. The servers must be installed and startable, and the default check
/// deliberately requires neither -- it reads a file, and this runs a program.
///
/// Worth the cost because the default check cannot see a tool name at all. An
/// allowlist is a list of strings to it, so `filesystem__reed_text_file` passes and
/// then refuses every read at run time. This repository has already paid that once,
/// in the other direction: a list written as five names refused 64 calls that were
/// all reads, and nothing in validation could have known.
pub async fn validate_live_tools(
    agent_def_path: &Path,
    tools_file: &Path,
    agent_def_port: &dyn AgentDefPort,
    tool_def_port: &dyn ToolDefPort,
) -> Result<()> {
    let parsed = agent_def_port.parse(agent_def_path)?;
    let tools_map = tool_def_port.load(tools_file)?;

    // Only the servers this definition declares. A tools file may describe others,
    // and one this agent never reaches is not this agent's problem. A name that is
    // not in the file is already an error from `validate`; a second one here would
    // say the same thing twice.
    let defs: Vec<ToolDef> = parsed
        .frontmatter
        .mcp_servers
        .iter()
        .filter_map(|name| tools_map.get(name.as_str()).cloned())
        .collect();

    let found = crate::infra::mcp::inspect(&defs).await?;
    if found.is_empty() {
        println!(
            "  ✓ {} server(s) started and answered; nothing wrong with their tools",
            defs.len()
        );
        return Ok(());
    }
    for finding in &found {
        eprintln!("  ✗ {}", finding.message);
    }
    bail!(
        "{} problem(s) found by starting the tool servers",
        found.len()
    )
}

/// The credential names a caller says are set, from one comma-separated argument.
///
/// Empty entries are dropped rather than kept as a name, so a caller reporting that
/// none are set -- an empty string -- produces a list of none rather than a list of
/// one thing called nothing. That is a real state: a repository that has added no
/// credential at all looks exactly like it.
pub fn credentials_from_arg(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .collect()
}
#[cfg(test)]
mod tests {
    /// What a caller means by an empty list.
    ///
    /// `--credentials-present ""` is a runner saying none are set, which is exactly
    /// the state this check exists to catch. Read as one nameless credential it would
    /// match nothing and report the wrong reason; dropped, it reaches the provider
    /// resolution as the empty set and gets the run's own answer.
    #[test]
    fn an_empty_argument_is_no_credentials_rather_than_one_without_a_name() {
        assert!(super::credentials_from_arg("").is_empty());
        assert!(super::credentials_from_arg("  ").is_empty());
        assert!(super::credentials_from_arg(",,").is_empty());
    }

    /// Whitespace is the shell's, not the caller's intent.
    #[test]
    fn credential_names_survive_the_spacing_a_shell_leaves() {
        assert_eq!(
            super::credentials_from_arg(" OPENAI_API_KEY , ANTHROPIC_API_KEY "),
            vec![
                "OPENAI_API_KEY".to_string(),
                "ANTHROPIC_API_KEY".to_string()
            ]
        );
    }
    use super::*;
    use crate::infra::persistence::agent_def::FileAgentDefAdapter;
    use crate::infra::persistence::tool_def::FileToolDefAdapter;
    use std::fs;

    fn write_agent(dir: &std::path::Path, name: &str, extra_frontmatter: &str) -> PathBuf {
        let path = dir.join(format!("{}.md", name));
        fs::write(
            &path,
            format!(
                "---\nname: {name}\ndescription: test\nmodel: test-model\n{extra}\n---\n",
                name = name,
                extra = extra_frontmatter
            ),
        )
        .unwrap();
        path
    }

    fn write_tools(dir: &std::path::Path, body: &str) -> PathBuf {
        let path = dir.join("tools.yaml");
        fs::write(&path, body).unwrap();
        path
    }

    /// Both lists on one server: unusual, defined, and not this command's call.
    ///
    /// It was fatal here while `describe_hooks` warned about the same configuration
    /// and let the run go on, and while two documents and a test in `infra::hooks`
    /// described the precedence it obeys. The precedence is real, so the file is not
    /// wrong -- and a validator calling it fatal left a reader four descriptions of
    /// one configuration with no way to tell which the program acts on.
    ///
    /// `validate` reports its findings to stderr and returns only that it failed, so
    /// this cannot read the message. What isolates the rule is the PAIR with
    /// `a_server_setting_both_lists_is_fatal_under_strict`: the same file, the same
    /// call, and only the caller's policy different. The first version of this test
    /// wrote `command: true`, which is a boolean to YAML and a string to `ToolDef`
    /// -- so it passed on the parse error and never reached the rule at all.
    #[test]
    fn a_server_setting_both_lists_is_a_warning_rather_than_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let agent = write_agent(dir.path(), "solo", "mcp_servers: [fs]\n");
        let tools = write_tools(
            dir.path(),
            "fs:\n  command: \"/bin/true\"\n  hooks:\n    tool_allowlist: [\"fs__read\"]\n    tool_denylist: [\"fs__write\"]\n",
        );
        let result = validate(
            agent,
            Some(tools),
            None,
            false,
            &FileAgentDefAdapter,
            &FileToolDefAdapter::default(),
        );
        assert!(result.is_ok(), "{:?}", result.err());
    }

    /// The same file under the policy that used to be built in.
    ///
    /// `--strict` is the whole of what a caller that wanted the refusal has to do:
    /// a delivery gating pull requests on this configuration keeps its answer by
    /// asking for it, and nothing else in the codebase has to agree with that
    /// choice.
    #[test]
    fn a_server_setting_both_lists_is_fatal_under_strict() {
        let dir = tempfile::tempdir().unwrap();
        let agent = write_agent(dir.path(), "solo", "mcp_servers: [fs]\n");
        let tools = write_tools(
            dir.path(),
            "fs:\n  command: \"/bin/true\"\n  hooks:\n    tool_allowlist: [\"fs__read\"]\n    tool_denylist: [\"fs__write\"]\n",
        );
        let result = validate(
            agent,
            Some(tools),
            None,
            true,
            &FileAgentDefAdapter,
            &FileToolDefAdapter::default(),
        );
        assert!(result.is_err());
    }

    /// The other warning, and the reason the category is not just the one rule: a
    /// definition naming servers with no tools file to check them against is a check
    /// that did not happen. Without `--strict` it passes, because `run` resolves the
    /// tools file from its own arguments or from `atoma.toml` and validating the
    /// definition alone is a legitimate thing to ask for.
    #[test]
    fn mcp_servers_without_a_tools_file_is_a_warning_that_strict_makes_fatal() {
        let dir = tempfile::tempdir().unwrap();
        let agent = write_agent(dir.path(), "solo", "mcp_servers: [fs]\n");
        assert!(validate(
            agent.clone(),
            None,
            None,
            false,
            &FileAgentDefAdapter,
            &FileToolDefAdapter::default()
        )
        .is_ok());
        assert!(validate(
            agent,
            None,
            None,
            true,
            &FileAgentDefAdapter,
            &FileToolDefAdapter::default()
        )
        .is_err());
    }

    /// One list is the ordinary case and stays ordinary.
    #[test]
    fn a_server_setting_one_list_passes() {
        let dir = tempfile::tempdir().unwrap();
        let agent = write_agent(dir.path(), "solo", "mcp_servers: [fs]\n");
        let tools = write_tools(
            dir.path(),
            "fs:\n  command: \"/bin/true\"\n  hooks:\n    tool_allowlist: [\"fs__read\"]\n",
        );
        let result = validate(
            agent,
            Some(tools),
            None,
            false,
            &FileAgentDefAdapter,
            &FileToolDefAdapter::default(),
        );
        assert!(result.is_ok(), "{:?}", result.err());
    }
    /// The error the run would have produced, produced earlier. A definition naming
    /// a provider that does not exist used to fail at `build_llm_client` -- after the
    /// tool servers had started and the prompt had been assembled.
    #[test]
    fn an_unknown_provider_is_a_validation_error_that_names_the_alternatives() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_agent(dir.path(), "solo", "provider: openai-responsez\n");
        let result = validate(
            path,
            None,
            None,
            false,
            &FileAgentDefAdapter,
            &FileToolDefAdapter::default(),
        );
        assert!(result.is_err());
    }

    /// The distinction the check rests on: the name being unknown and the credential
    /// being absent are different facts, and only the first is a defect in the
    /// definition. No credentials are set here, which is the state every validation
    /// run is in.
    #[test]
    fn a_known_provider_passes_without_its_credential() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_agent(dir.path(), "solo", "provider: openai\n");
        let result = validate(
            path,
            None,
            None,
            false,
            &FileAgentDefAdapter,
            &FileToolDefAdapter::default(),
        );
        assert!(result.is_ok(), "{:?}", result.err());
    }

    #[test]
    fn a_definition_that_names_no_provider_is_not_checked_for_one() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_agent(dir.path(), "solo", "");
        assert!(validate(
            path,
            None,
            None,
            false,
            &FileAgentDefAdapter,
            &FileToolDefAdapter::default()
        )
        .is_ok());
    }

    #[test]
    fn a_template_placeholder_nothing_substitutes_fails_validation() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_agent(dir.path(), "solo", "");
        let template = dir.path().join("prompt.md");
        fs::write(&template, "You are {{AGENT_NAME}}. Use {{AVAILABLE_TOOL}}.").unwrap();
        let result = validate(
            path,
            None,
            Some(template),
            false,
            &FileAgentDefAdapter,
            &FileToolDefAdapter::default(),
        );
        assert!(result.is_err());
    }

    #[test]
    fn a_template_that_only_uses_known_placeholders_passes() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_agent(dir.path(), "solo", "");
        let template = dir.path().join("prompt.md");
        fs::write(
            &template,
            "You are {{AGENT_NAME}} in {{WORKING_DIRECTORY}}.",
        )
        .unwrap();
        let result = validate(
            path,
            None,
            Some(template),
            false,
            &FileAgentDefAdapter,
            &FileToolDefAdapter::default(),
        );
        assert!(result.is_ok(), "{:?}", result.err());
    }

    /// A definition that still carries `callable_by` is accepted and the value
    /// ignored, rather than refused. Nothing enforced it when it was read, so an
    /// adopter who kept it in a file has not asked for anything they are not getting.
    #[test]
    fn a_leftover_callable_by_is_ignored_rather_than_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_agent(dir.path(), "solo", "callable_by:\n  - human\n");
        let result = validate(
            path,
            None,
            None,
            false,
            &FileAgentDefAdapter,
            &FileToolDefAdapter::default(),
        );
        assert!(result.is_ok(), "{:?}", result.err());
    }

    /// This test asserted the opposite for a while, and the reason is worth keeping:
    /// `reconcile_tools` had been relaxed to warn instead of refuse, and the validator
    /// was then relaxed to agree with it. Two places accepted a declaration neither
    /// could act on, and the agent's tools reached no request either way.
    #[test]
    fn a_non_array_tools_fails_validation() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_agent(dir.path(), "solo", "extra_body:\n  tools: web_search\n");
        assert!(
            validate(
                path,
                None,
                None,
                false,
                &FileAgentDefAdapter,
                &FileToolDefAdapter::default()
            )
            .is_err(),
            "a string is not a list of tool definitions"
        );
    }

    /// Every key any adapter assembles itself. `input` and `system` are the two that
    /// used to pass validation and then be silently dropped by the Responses and
    /// Anthropic adapters respectively.
    #[test]
    fn a_reserved_key_fails_validation_including_the_ones_only_one_dialect_owns() {
        for key in ["model", "messages", "input", "system", "store"] {
            let dir = tempfile::tempdir().unwrap();
            let path = write_agent(
                dir.path(),
                "solo",
                &format!("extra_body:\n  {key}: something\n"),
            );
            assert!(
                validate(
                    path,
                    None,
                    None,
                    false,
                    &FileAgentDefAdapter,
                    &FileToolDefAdapter::default()
                )
                .is_err(),
                "{key} should be refused"
            );
        }
    }

    /// Every header Atoma sets, whoever sets it -- the fixed ones each adapter writes
    /// at the call, and every provider's own. Refused rather than dropped: an agent
    /// that lost one would see the provider behaving strangely rather than a rejected
    /// setting, and for Copilot's `Copilot-Integration-Id` it would see models
    /// disappear from the catalogue.
    #[test]
    fn a_header_atoma_sets_itself_cannot_be_set_by_an_agent() {
        for name in crate::infra::llm::reserved_header_names() {
            let dir = tempfile::tempdir().unwrap();
            let path = write_agent(
                dir.path(),
                "solo",
                &format!("extra_headers:\n  {name}: something\n"),
            );
            assert!(
                validate(
                    path,
                    None,
                    None,
                    false,
                    &FileAgentDefAdapter,
                    &FileToolDefAdapter::default()
                )
                .is_err(),
                "{name} should be refused"
            );
        }
    }

    /// Case is not part of a header's identity, and a definition writing one in a
    /// different case is making the same mistake.
    #[test]
    fn a_reserved_header_is_refused_whatever_case_it_is_written_in() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_agent(
            dir.path(),
            "solo",
            "extra_headers:\n  AUTHORIZATION: bearer x\n",
        );
        assert!(validate(
            path,
            None,
            None,
            false,
            &FileAgentDefAdapter,
            &FileToolDefAdapter::default()
        )
        .is_err());
    }

    /// The case the field exists for: a name Atoma does not set goes through.
    #[test]
    fn a_header_atoma_does_not_set_is_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_agent(
            dir.path(),
            "solo",
            "extra_headers:\n  X-OrcaRouter-Session-Id: atomaton-solo\n",
        );
        assert!(validate(
            path,
            None,
            None,
            false,
            &FileAgentDefAdapter,
            &FileToolDefAdapter::default()
        )
        .is_ok());
    }

    #[test]
    fn extra_body_tools_as_an_array_is_accepted() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_agent(
            dir.path(),
            "solo",
            "extra_body:\n  tools:\n    - type: openrouter:web_search\n",
        );
        assert!(validate(
            path,
            None,
            None,
            false,
            &FileAgentDefAdapter,
            &FileToolDefAdapter::default()
        )
        .is_ok());
    }

    /// Naming a target is the whole declaration. These two used to assert that a
    /// `knows_about` target had separately written `callable_by: ["agent"]`, which
    /// tested only that two declarations of one fact agreed.
    #[test]
    fn knows_about_passes_when_the_target_exists_and_parses() {
        let dir = tempfile::tempdir().unwrap();
        write_agent(dir.path(), "helper", "");
        let caller = write_agent(dir.path(), "caller", "knows_about:\n  - helper\n");
        let result = validate(
            caller,
            None,
            None,
            false,
            &FileAgentDefAdapter,
            &FileToolDefAdapter::default(),
        );
        assert!(result.is_ok(), "{:?}", result.err());
    }

    /// The check that still means something: a name with no file behind it is a
    /// delegation that cannot happen.
    #[test]
    fn knows_about_fails_when_the_target_is_missing() {
        let dir = tempfile::tempdir().unwrap();
        let caller = write_agent(dir.path(), "caller", "knows_about:\n  - absent\n");
        let result = validate(
            caller,
            None,
            None,
            false,
            &FileAgentDefAdapter,
            &FileToolDefAdapter::default(),
        );
        assert!(result.is_err());
    }
}

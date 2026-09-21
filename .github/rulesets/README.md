# Rulesets

`main.json` is what protects the default branch. **It is a file, not a setting** —
nothing here is in force until someone imports it through GitHub's web interface.
Checking it in is what makes the protection reviewable, diffable and restorable;
applying it is a separate, manual act.

## Applying it

Settings → Rules → Rulesets → New ruleset → **Import a ruleset** → upload
`main.json`.

Import it as **Disabled** the first time. Then open a throwaway pull request, look
at the check GitHub actually reports on it, and confirm the name matches the
`context` below. A ruleset that requires a check nobody reports blocks every pull
request forever, and the only way out is the settings page. Once the name agrees,
switch the ruleset to Active.

## What each rule is for

| Rule | Why |
| --- | --- |
| `deletion` | The default branch cannot be deleted. |
| `non_fast_forward` | No force-push. History on `main` is append-only, so a commit that exists stays reachable. |
| `pull_request` | Every change arrives as a pull request. This is the rule that actually changes how work happens here. |
| `required_status_checks` | `Test` must pass. That job runs `cargo fmt --check`, `cargo clippy -- -D warnings` and `cargo test --locked`. |

`required_approving_review_count` is **0** on purpose. The protection here is that a
change is a pull request with green checks, not that a second account clicked
approve — this repository has one developer, and a rule that cannot be satisfied is
a rule that gets bypassed. Raise it the day there is somebody to do the approving.

`strict_required_status_checks_policy` is **false**: a branch does not have to be
rebased onto the tip before merging. `main` moves rarely enough that requiring it
would cost more than it catches.

## `bypass_actors` is empty, and the release still works

Nothing is exempt, including GitHub Actions. That is safe here, and it was checked
rather than assumed:

`release.yml` runs after CI succeeds on `main` and pushes **a tag** —
`git push origin "${TAG}"` — never a commit to a branch. This ruleset targets
`~DEFAULT_BRANCH`, so it does not see tags at all. The release is unaffected.

What does change: bumping the version in `Cargo.toml` is a commit to `main`, so it
goes through a pull request like anything else. CI then runs on `main` after the
merge, and `release.yml` triggers off that run exactly as it does today.

## What this costs

Direct commits to `main` stop working — for people and for agents. That is the
point, and it is worth saying out loud because this repository has been developed by
committing straight to `main`.

## Keeping the file and the setting in agreement

GitHub does not tell you when the two drift. Whoever changes the ruleset in the
interface exports it again and commits the result here, in the same way a schema
change is checked in. A file that no longer describes the repository is worse than
no file, because it is read as though it does.

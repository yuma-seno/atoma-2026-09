# Rulesets

`main.json` protects the default branch. It is a file, not a setting: nothing in it
is in force until someone imports it through Settings → Rules → Rulesets → Import.

**Import it as Disabled first.** Then open a throwaway pull request, read the name
of the check GitHub actually reports on it, and confirm it matches the `context` in
the file. A ruleset that requires a check nobody reports blocks every pull request
forever, and the only way out is the settings page.

`bypass_actors` is empty, Actions included, and the release still works:
`release.yml` pushes a tag, never a commit to a branch, and this ruleset targets
`~DEFAULT_BRANCH`.

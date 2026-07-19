# Engineering Conventions

Shared conventions for every project.
`CLAUDE.md` imports this file and adds the project's own map, invariants, and gotchas.
When this file and an enforcing hook disagree, the hook wins — fix whichever is wrong in the same commit.

## First principles

1. Comments and docstrings earn their place: write them only when they add what the code cannot say — the why, a non-obvious constraint — and keep them concise, dense, and evergreen.
2. Never use emojis in generated code.
3. Do not generate markdown files unless instructed to do so.
4. After making changes, run the gate: `make check`.
5. Prose in Markdown is one sentence per line — never hard-wrap.
   Single newlines render as spaces, so this is invisible to readers while keeping diffs and blame line-precise.
   (Tables, code blocks, and list items stay one line each.)

## The gate

- `make check` is the gate: lint + format check + typecheck + tests.
  Run all of it before declaring work done.
- `make fmt` auto-fixes what the gate would flag.
- `make dev` once on a fresh clone: installs dependencies and wires the git hooks via `core.hooksPath`; the hooks are dormant until then.
- The pre-commit hook runs the secret scan and the gate, and caches the result by tree hash — identical content never pays for the gate twice; bypassing it (`--no-verify`) is for emergencies, and CI still runs the gate.
- The gate is incremental: every tool that can cache, caches (ruff and mypy caches, `eslint --cache`, `prettier --cache`, `tsc --incremental`, `cspell --cache`).
  A warm no-change gate should cost seconds; treat a slow warm gate as a defect, not a tax.
- Where CI exists, it runs exactly `make check` — local and CI cannot drift.

## Commit messages

- Format: `<type>(<scope>): Title case ending with period.`
  Some repos drop the scope; `.githooks/commit-msg` is the source of truth for this repo's exact types and scopes.
- Base types: feat, fix, chore, clean, revert.
  Repos may extend (e.g. docs, refactor, exec).
- Subject line only — no body, no Co-Authored-By or other trailers.
- Backtick code identifiers and filenames in the subject.
- One logical change per commit (one fix, one feature, one phase), with the full gate green before each; a snapshot or doc truth-keeping edit rides the commit that caused it.
- For a bug fix, land the failing test with the fix.
- Commit locally and stop: pushing is the operator's review step.

# Engineering Conventions

Shared conventions for every project. `CLAUDE.md` imports this file and adds the
project's own map, invariants, and gotchas. When this file and an enforcing hook
disagree, the hook wins — fix whichever is wrong in the same commit.

## First principles

1. Avoid comments or docstrings unless absolutely necessary.
2. Never use emojis in generated code.
3. Do not generate markdown files unless instructed to do so.
4. After making changes, run the gate: `make check`.

## The gate

- `make check` is the gate: lint + format check + typecheck + tests. Run all of it before declaring work done.
- `make fmt` auto-fixes what the gate would flag.
- `make dev` once on a fresh clone: installs dependencies and wires the commit hook via `core.hooksPath`; the hook is dormant until then.
- Where CI exists, it runs exactly `make check` — local and CI cannot drift.

## Commit messages

- Format: `<type>(<scope>): Title case ending with period.` Some repos drop the scope; `.githooks/commit-msg` is the source of truth for this repo's exact types and scopes.
- Base types: feat, fix, chore, clean, revert. Repos may extend (e.g. docs, refactor, exec).
- Subject line only — no body, no Co-Authored-By or other trailers.
- Backtick code identifiers and filenames in the subject.
- One logical change per commit (one fix, one feature, one phase), with the full gate green before each; a snapshot or doc truth-keeping edit rides the commit that caused it.
- For a bug fix, land the failing test with the fix.
- Commit locally and stop: pushing is the operator's review step.

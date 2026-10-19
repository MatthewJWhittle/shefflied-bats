# AGENTS.md

General principles for work on this project. Setup, CLI preference, and PR checks: [CONTRIBUTING.md](CONTRIBUTING.md). Concrete numbers, paths, and step-by-step rules live in issues, config, code, and [docs/](docs/).

## Principles

- Understand the existing code and docs before changing anything.
- Prefer libraries we already use, or the standard library, over hand-rolled code. When you don't, say why.
- Keep changes small and focused, with tests.
- Never skip or bypass CI.
- Follow the issue's brief; say so in the PR when you depart from it.
- State root causes. Prefer shared fixes over one-off patches.
- Back performance claims with measurements.
- For modelling: prefer the ecologically plausible approach; change one thing at a time with the reason written down; don't let the model see information it shouldn't, and record any data processing where it's done. Never let evaluation data leak into anything the model learns.
- This repo is public: no secrets or sensitive location data. Licensed or private data stays out of git.
- Never overwrite existing production data or outputs.
- In PRs, commits, and docs, write under a role heading. No bot or agent names, and no @-mentions.

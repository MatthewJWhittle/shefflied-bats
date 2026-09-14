# Contributing

Thanks for helping improve the HSM toolchain.

## Setup

```bash
git clone https://github.com/MatthewJWhittle/shefflied-bats.git
cd shefflied-bats
uv sync
uv run pytest
```

## Guidelines

- Prefer the **`sdm` CLI** and `config.yml` over one-off scripts for repeatable workflows.
- Keep the **visualiser boundary** at artefact files (`model.pkl`, `package.json`, GeoTIFF COGs) — avoid coupling to a specific web app or API in core library code.
- Update [docs/model-package-contract.md](docs/model-package-contract.md) when changing publish formats or `package.json` schema.
- Match existing code style; imports at module top, exhaustive switches on discriminated unions where applicable.

## Pull requests

- Target `main` with a focused description of what changed and why.
- Ensure CI passes (`uv run pytest` and flake8 as in `.github/workflows/python-tests.yml`).

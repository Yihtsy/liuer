# AGENTS.md

This project is `liuer`, a Python package whose purpose is to mimic familiar Python package APIs and add practical missing utilities.

## Product Direction

- Keep the package idea centered on "六耳": imitate well-known Python package/module shapes, then extend them with small, useful helpers.
- Prefer namespaced modules that mirror their inspiration:
  - `liuer.scipy.stats` for statistics helpers inspired by `scipy.stats`.
  - `liuer.matplotlib.pyplot` for plotting helpers built on `matplotlib.pyplot`.
  - `liuer.itertools` for iterator helpers inspired by the standard library `itertools`.
- Do not shadow the real top-level standard-library or third-party modules. Users should import through `liuer.*`.

## Code Rules

- Code comments and docstrings should be English.
- Existing Chinese comments should be translated to English or removed.
- Keep public APIs small, documented, and easy to import.
- Prefer compatibility with the original package's style before inventing a new style.
- Add tests for new public functions when behavior is not trivial.
- Avoid large dependencies unless the target namespace already implies them.

## Packaging Rules

- Use `pyproject.toml` as the source of packaging metadata.
- Keep generated metadata such as `*.egg-info`, `build/`, and `dist/` out of git.
- Update `README.md` for user-facing API changes.
- Update `CHANGELOG.md` for every release or meaningful unreleased change.
- Keep `src/liuer/__init__.py` version aligned with `pyproject.toml`.

## Release Checklist

1. Update version in `pyproject.toml` and `src/liuer/__init__.py`.
2. Update `CHANGELOG.md` and replace `Unreleased` with the release date.
3. Run `python -m pytest`.
4. Run `python -m ruff check .`.
5. Run `python -m build`.
6. Run `python -m twine check dist/*`.
7. Create a git tag, push to GitHub, then publish to PyPI.

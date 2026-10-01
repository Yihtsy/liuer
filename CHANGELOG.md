# Changelog

All notable changes to `liuer` are recorded here.

## 0.0.2 - Unreleased

### Added

- Added a project-level `AGENTS.md` with development and release rules.
- Added modern packaging metadata for GitHub and PyPI publishing.
- Added `liuer.itertools` with standard-library re-exports and additional iterator helpers.
- Added `liuer.matplotlib.pyplot` and `liuer.plot` as Matplotlib-based visualization extension modules.
- Migrated local `handytool.stats` helpers into `liuer.scipy.stats`.
- Migrated local `handytool.itertools` combination helpers into `liuer.itertools`.
- Migrated local `handytool.plot` jittered faceted boxplot helper into `liuer.matplotlib.pyplot`.
- Added a GitHub Actions publishing workflow for PyPI Trusted Publishing.

### Changed

- Cleaned Chinese comments from existing `liuer` source files.
- Improved `liuer.scipy.stats.compare_pos_neg_groups` with English output labels, configurable formatting, NaN handling, and safer statistical fallbacks.

## 0.0.1 - 2026-02-12

### Added

- Initial PyPI release.
- Added early extensions for `pandas`, `scipy.stats`, and `sklearn`.

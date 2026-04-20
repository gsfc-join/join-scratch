## ADDED Requirements

### Requirement: Benchmark infrastructure module
`src/join_scratch/utils/benchmark.py` SHALL consolidate `BenchmarkResult`, `_rss_mib()`, `_time_call()`, and `render_report()` from the three currently triplicated benchmark files.

#### Scenario: Single source for benchmark infrastructure
- **WHEN** searching the codebase for `class BenchmarkResult`
- **THEN** it is defined exactly once, in `utils/benchmark.py`

### Requirement: Duplicate constants eliminated
`LIS_RELPATH` SHALL NOT appear in any `src/join_scratch` library module. Any reference to the LIS input file path SHALL live exclusively in `workflows/swe/`.

#### Scenario: LIS_RELPATH absent from library source
- **WHEN** searching `src/join_scratch/` for the string `"lis_input_NMP_1000m_missouri.nc"`
- **THEN** no matches are found

### Requirement: Unused imports removed
Unused imports identified in the codebase (`import io` in `storage.py`, `field` in benchmark files, `import tempfile` in `_scratch/swe_naive.py`) SHALL be removed.

#### Scenario: No flagged unused imports remain
- **WHEN** a linter is run on refactored source files
- **THEN** no F401 warnings for the items identified in this change

# Plan: Production-Ready Notebook & Doc Cleanup

**TL;DR** — Overhaul all 21 Jupyter Book notebooks, fix the broken TOC, standardize on `ds.pyradtran.run(config_path=...)` API, adopt `~/.pyradtran/config.yaml` for portable paths, unify tone to friendly-but-polished, fix all bugs and typos, flesh out the README and book pages (especially "Check your input file!"), and add basic GitHub Actions CI for unit tests. This gets pyRadtran to release quality.

---

## Phase 1: Fix structural issues (TOC, paths, versions)

1. **Fix `book/_toc.yml`:** Remove the two broken refs (`notebooks/quickstart`, `notebooks/cloud`). Not all 11 orphan notebooks in a sensible order — group by difficulty (Quickstart → Advanced → Processing). Proposed structure:
   - *Quickstart:* `albedo_test`, `thermal`, `thermal_imager`, `solar_spectral`, `era5_atmosphere`
   - *Advanced:* `get_radiosonde`, `water_cloud`, `mixed_phase_cloud`, `arctic_cloud_experiment`, `shupe_and_intrieri_plot`, `thermal_imager_2`, `radiosonde`, `halo-ac3_bbr_all_aircraft`, `radiosonde_thermal`, `radiosonde_solar_spectral`
   - *Processing/Reference:* `sea_ice_era5`, `sea_ice_era5_plotting`, 
   DON'T INCLUDE: `nya_rad`, `carra_atmosphere`, `stn2024`

2. **Unify Python version to `>= 3.9`** across `README.md`, `book/installation.md`, and `pyproject.toml`.

3. **Make libRadtran paths version-agnostic** in all YAML configs under `config/` and `book/notebooks/config/`: change `/opt/libRadtran-2.0.6/` and `/opt/libradtran/2.0.4/` → a generic placeholder documented via the master config system. Update configs to use `/opt/libradtran/` or similar.

4. **Document `~/.pyradtran/config.yaml`** as the recommended setup in `book/installation.md` and `book/usage.md` — explain that users create this once to set their local `libradtran_path`, `data_path`, etc. Add a template snippet.

---

## Phase 2: Fix notebook bugs (code-breaking issues)

5. **`book/notebooks/era5_atmosphere.ipynb`:** Remove reference to undefined `ds_era5_local`. Fix the simulation cell to use the correctly subset dataset. Add `pip install gcsfs` guidance.

6. **`book/notebooks/radiosonde.ipynb`:** Replace `xr.open_dataset()` on CSV with `pd.read_csv()` + conversion to xarray. Remove unused imports (`asdict`, `os`, `yaml`). Remove stored error output.

7. **`book/notebooks/solar_spectral.ipynb`:** Fix undefined `X`, `Y`, `z` variables in the 3D plot cell. Fix fragile `../config/` relative path.

8. **`book/notebooks/thermal.ipynb`:** Change `ds.pyradtran.run_uvspec()` → `ds.pyradtran.run()`. Fix `../config/` relative path.

9. **`book/notebooks/radiosonde_thermal.ipynb`:** Update from old API (`run_uvspec`) to `ds.pyradtran.run()`. Remove hardcoded `median = -5` debug override. Replace `/projekt_agmwend/` paths with documented data-requirement admonition. Remove `ipywidgets` interactive slider (won't work in static book build) — replace with static plot.

10. **`book/notebooks/radiosonde_solar_spectral.ipynb`:** Same API update as above. Add data-requirement admonition for private paths.

12. **`book/notebooks/era5_seasonal_sea_ice_profiles.ipynb`:** Add a title markdown cell. Fix the missing config reference (`arctic_cloud_experiment_solar2.yaml`).

---

## Phase 3: Standardize API usage across all notebooks

13. Every notebook that calls `ds.pyradtran.run_uvspec()` → change to `ds.pyradtran.run()`. Every `config=` (Python object) → standardize to `config_path=` where YAML exists. This affects: `thermal`, `radiosonde_thermal`, `radiosonde_solar_spectral`, `stn2024`, `albedo_test`.

14. Replace the boilerplate `import logging; logging.getLogger('pyradtran').setLevel(logging.CRITICAL)` pattern — keep it but add a one-line comment explaining *why* (suppresses verbose solver output in book builds). Make it consistent across all notebooks.

15. Remove the `# This should register the accessor automatically` comment from every notebook, or reword to a user-friendly note: *"Importing pyradtran registers the `.pyradtran` xarray accessor."*

---

## Phase 4: Polish notebook content (tone, structure, clarity)

For **each of the 21 notebooks**, apply these transformations:

16. **Add/fix titles** — every notebook gets a properly capitalized `# Title` as its first cell using the pattern: `# {Difficulty}: {Topic}` (e.g., `# Quickstart: Surface Albedo`).

17. **Add introductory paragraph** after the title: 1–3 sentences explaining what the notebook demonstrates and what the reader will learn. Use friendly first-person plural ("In this notebook, we'll…").

18. **Add section headings** (`## Setup`, `## Configuration`, `## Running the Simulation`, `## Results`, `## Summary`) to break up long notebooks. Not every notebook needs all sections — scale to complexity.

19. **Fix all typos:** "basik" → "basic", "minum" → "minimum", "exptected" → "expected", "dimensiopnal" → "dimensional", `#s_sel`, truncated `# this is ` comment.

20. **Remove dev language:** "NEW IO System" → remove entirely; "REFACTORED VERSION" → remove; "juhu!" → "The simulation completed successfully." or similar.

21. **Add result interpretation cells** — after plots, add 1–2 sentence markdown explaining what the reader should observe (e.g., "Notice how the downwelling irradiance increases with surface albedo due to enhanced multiple scattering.").

22. **For private-data notebooks** (`carra_atmosphere`, `radiosonde_thermal`, `radiosonde_solar_spectral`, `stn2024`): add an admonition box at the top:
    > **Note:** This notebook uses campaign data from the HALO-(AC)³ field experiment. The datasets are not publicly available. You can adapt the workflow to your own data by replacing the file paths.

23. **Clean up `book/notebooks/carra_atmosphere.ipynb`:** Remove all commented-out code blocks. Fix the misleading title (it's CARRA, not ERA5). Add explanatory markdown.

24. **Clean up `book/notebooks/nya_rad.ipynb`:** Add a proper title (not just "Advanced:"). Flesh out markdown cells.

25. **Fix title duplicates:** `radiosonde_thermal` and `radiosonde_solar_spectral` both say "Constructing realistic clouds" — differentiate them (e.g., "…clouds (thermal)" vs "…clouds (solar spectral)").

26. **Split overly long code cells** (65+ lines in `halo-ac3_bbr_all_aircraft`, 72-line cloud-top calculation in `radiosonde_thermal`) — break into logical chunks with markdown annotations between them.

---

## Phase 5: Book pages & README

27. **Fill `book/intro.md` "Check your input file!" section** — write guidance on:
    - Setting `cleanup_temp_files: false` in the YAML config to inspect generated `.inp` files
    - Where to find them (`pyradtran_work/` directory)
    - How to read a `.inp` file (key lines: `source solar`, `atmosphere_file`, `albedo`, etc.)
    - Common mistakes (wrong path, missing atmosphere file, spectral range mismatch)
    - How to enable `debug_mode` for verbose output

28. **Overhaul `README.md`:**
    - Fix "exptected" typo
    - Unify Python version to 3.9+
    - Fix the example code's config path (use a portable path or inline config)
    - Fix "Jupyter Notebooks" link → correct path
    - Add a "Getting Started" section pointing to the book
    - Add a "Configuration" section briefly explaining `~/.pyradtran/config.yaml`
    - Clean up Acknowledgments hierarchy
    - Add a badge for CI status (once GitHub Actions is set up)

29. **Expand `book/usage.md`:**
    - Add sections on batch processing, cloud configuration, ERA5 integration
    - Document the `parameter_overrides` mechanism (shown in `water_cloud` notebook)
    - Cross-reference relevant notebooks

30. **Fix `book/installation.md`:**
    - Python version → 3.9+
    - Add HTTPS clone option alongside SSH
    - Add `~/.pyradtran/config.yaml` setup instruction
    - Mention libRadtran installation (link to their docs, explain the path the user must set)

31. **Expand `book/contributing.md`:**
    - Add code style expectations (black, ruff, or whatever is used)
    - Add test running instructions (`python run_tests.py --unit`)
    - Add notebook contribution guidelines (clear titles, markdown cells, no hardcoded paths)

32. **Update `book/changelog.md`:** Add a "Version 0.2.0" entry (or whatever the upcoming version is) with actual changes: notebook overhaul, config system docs, CI, API standardization.

---

## Phase 6: GitHub Actions CI

33. **Create `.github/workflows/tests.yml`:**
    - Trigger on push to `main` and PRs
    - Python 3.9, 3.10, 3.11 matrix
    - Install package with `pip install -e '.[test]'`
    - Run `pytest -m unit` (unit tests only — no libRadtran needed)
    - The `unit` marker is already defined in `pytest.ini`

34. **Audit `tests/conftest.py`:** Ensure `unit`-marked tests don't depend on `/opt/libradtran/` — add `pytest.skipif` guards for tests that need a real libRadtran installation. This keeps CI green without libRadtran.

35. **Add a Jupyter Book build check** (optional lightweight step): `jupyter-book build book/ --builder html` to catch broken refs. Can run only on `main` to save CI minutes.

---

## Phase 7: Final cleanup

36. **Delete duplicate notebooks in `notebooks/`** (the root-level folder) that are superseded by the book versions, or add a note in the folder explaining they're development copies. The book notebooks are the canonical versions.

37. **Review all YAML configs in `book/notebooks/config/`:** ensure paths are version-agnostic and consistent. Remove any that are unused/orphaned.

38. **Clear all cell outputs** from notebooks before committing (clean state for the book build).

---

## Verification Checklist

- [ ] `jupyter-book build book/` completes without errors (no broken TOC refs, no missing files)
- [ ] Every notebook runs top-to-bottom on a machine with libRadtran installed and `~/.pyradtran/config.yaml` configured (already verified on two machines)
- [ ] `pytest -m unit` passes locally and in GitHub Actions
- [ ] `grep -r "NEW IO\|juhu\|basik\|minum\|exptected\|REFACTORED" book/` returns zero hits
- [ ] All notebook titles are unique and properly capitalized
- [ ] The "Check your input file!" section in intro.md has real content

---

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| TOC grouping | Quickstart → Advanced → Processing | Natural difficulty progression |
| API canon | `ds.pyradtran.run(config_path=...)` | Simpler; `run_uvspec` still works but isn't shown in docs |
| Private-data notebooks | Included with admonition boxes | Show real-world workflows; users adapt to their own data |
| `stn2024.ipynb` | Included | Active development notebook, presumably important |
| Root `notebooks/` folder | Kept, marked as development | Book versions are canonical |
| CI scope | Unit tests on push/PR; book build optional on main | Incremental; doesn't require libRadtran in CI |
| Paths | `~/.pyradtran/config.yaml` master config | Portable; already supported by config loader |
| Python version | `>= 3.9` everywhere | Matches `pyproject.toml` |
| libRadtran version | Version-agnostic (`/opt/libradtran/`) | Users install whatever version they have |
| Tone | Friendly + polished, first-person plural | Keeps the author's warm style without dev slang |

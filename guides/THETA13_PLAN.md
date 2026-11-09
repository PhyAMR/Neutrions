# θ₁₃ Plan: from IBD selection to sin²2θ₁₃

This is the working plan, shared across sessions. Claude explains and reviews, and the team writes all the code. Tick the boxes as you go and keep the status block up to date.

## Status / next step

| | |
|---|---|
| **Current phase** | Part A: fix the preprocessing |
| **Next action** | **Objectives first; refactoring is parked** (the user's call, 2026-11-09; see "Parked for later"). Order: (1) get the teammate's changes onto `main`. PR #1 was closed and resubmitted as PR #2, which was merged into the branch **`Tests`**, not into `main` (2026-11-09; `origin/main` is still `86b071d`). Next comes a PR `Tests` → `main`, opened by the teammate and approved by the user. It also fixes the import in `src/data_loader.py` (the local one-line fix was already thrown away). (2) **A.12**: regenerate with `--force` and run the checks; the old files have no `event_id`, no `total_labels.parquet` and no `parameters.npz`. (3) The rest of **A.11**: the trainer raises on missing features, and the MLP takes `N_FEATURES` from the data. (4) **A.10**: a time-ordered split, starting from the teammate's notebook (cell 8). Done and committed (latest `86b071d`): A.1–A.7, A.9, cleanup items 1 and 3. **A.8 is deferred** until test data arrives. Don't train until A.10–A.12 are done. |
| **Open PR** | None yet. **#2** (= the closed #1; a teammate, 2026-11-09) was merged into `Tests`: `event_id` → `event_id_1/2` in the events labels, a stable sort in `_load_raw`, the `_ensure_processed` import fix plus a missing-output check, `_LABEL_COLS` refreshed, and a notebook (in-memory pairing, a time-ordered split, on/off plots). Bring it to `main` as is; the review notes are under "Parked for later". |
| **Blocked on** | Nothing for Part A. Part C needs to know what the test data contains (see "Open questions"). |
| **Last updated** | 2026-11-09 |

---

## Where we are and why

**What the data is.** It is a simulated detector, about the size of one Daya Bay detector, running in "nH mode" (neutron captured on hydrogen):
- **Geometry:** a cylinder with r ≤ 1.5 m and |z| ≤ 1.55 m.
- **Events:** 25,043,297 events over about 143 days, so about 2.03 Hz of radioactivity singles.
- **Signal:** 100,000 IBD pairs. The neutron capture time is τ ≈ 200 µs and the delayed energy peaks at 2.18 MeV.
- **Background:** only accidentals. There are no ⁹Li or fast neutrons.

**How the classifier fits in.** It is the *IBD selection* step. θ₁₃ comes from a counting step on top of it:

```
N_selected = ε · N_expected(no oscillation) · P_ee + B

R = (N_selected − B) / (ε · N_expected)
sin²2θ13 ≈ (1 − R) / ⟨sin²(1.267 Δm²31 L / E)⟩     (E_ν ≈ E_prompt + 0.78 MeV)
```

So, besides the classifier, you need:
- the efficiency **ε**, from the simulation truth;
- the accidental background **B**, from the off‑window pairs;
- **N_expected**, from a near/far ratio or an absolute prediction.

**What was wrong with the data the model was trained on:**
- The signal label was wrong.
- The model saw only 9 Δ‑features: no individual energies and no positions.
- The off‑window pairs couldn't be scored, because their Δt is never shifted back.
- The files in `data/processed/` were written by an older script.

---

## Part A: Fix the preprocessing (you implement it)

Work in **`src/scripts/preprocessing.py`**; it is the canonical script. Keep its streaming machinery: the two‑pointer scan and the `ParquetWriter` flushes in `_find_forward_pairs`, `_find_false_pairs` and `_project_pairs_chunked`. Only the columns, labels and windows change.

**Rule: preprocessing builds pairs, features and labels. It applies no physics cuts.** The only cut is the 2 ms coincidence window, which already keeps 99.996% of the IBDs (the largest Δt is 2.28 ms). Selection is the model's job. That way the efficiency ε is defined relative to all IBD pairs in the window.

### Problems found (why each step exists)

| # | Problem | Evidence |
|---|---|---|
| A1 | IBD_p and IBD_d are merged into "detected" **before** pairing (`_relabel`). The target is then `label_name_2 == "detected"` (`training.yaml:14`). | 796 accidental radio→IBD pairs are labelled as signal. |
| A2 | The files in `data/processed/` hold only 9 `delta_*` columns, written by the old `src/preprocessing.py`. | No E_prompt, no E_delayed, no positions. |
| A3 | The 28‑column schema contains unphysical or redundant features. | The absolute times `t_1`/`t_2`. `dphi`, which jumps at 0/2π. `dr`, `dr2`, `dD` and `x`/`y`, which repeat ΔR and the positions. |
| A4 | Distance to the wall is missing. | 78% of radio events are within 20 cm of the wall, versus about 32% of IBDs. |
| A5 | There is no multiplicity information, so counting would double‑count. | 436 prompts have 2 or more later events within 2 ms; 398 have an earlier one. |
| A6 | Off‑window pairs have Δt in (2, 4] ms and it is never shifted back. There is only one off‑window, and its labels are mixed. | `_find_false_pairs` (`src/scripts/preprocessing.py:363`) |
| A7 | `sort_values("time")` is not a stable sort, and there is 1 duplicated timestamp. | One IBD pair has Δt = 0. |
| A8 | The truth file is required, and the paths are hard‑coded to `data/raw/training` → `data/processed`. | You can't preprocess the test set, or doing so overwrites the training outputs. |
| A9 | Two preprocessing modules. A broken import at `src/data_loader.py:133` (`from preprocess import run`). `config.yaml` is empty. The trainer silently drops unknown feature names. | |
| A10 | Pairs that share an event are split at random between train and test. | Leakage. |

### Steps

- [x] **A.1 Keep the truth per event.** Remove the `_relabel` call. Carry `label_name_1`, `label_name_2`, `pair_id_1` and `pair_id_2` through the pair scan, and stop dropping `pair_id_2` in `_rename_pair_columns`.
  - Edits: drop `_relabel` (line 655) and the `MERGE_*` maps; keep `label` and `pair_id` out of `cols_to_delta` (lines 273 and 411); stop the rename and drop at lines 544–556; set `EVENTS_LABEL_COLS` (line 103) to `pair_id_1, pair_id_2, label_1, label_2` (+ names); use integer types in the empty‑file fallback (line 635). Truth goes in the labels file only.
  - Breaks until A.2: the `training.yaml` target (no "detected" anymore), `_LABEL_COLS` in `src/data_loader.py:113`, and label filters and plots that use `label_name_2`/`pair_id`. **Don't retrain between A.1 and A.2.**
  - Review 2026-10-09: edits 1, 3 and 4 done. Still to do: the `"labels"` typo in `cols_to_delta` (lines 236 and 374; the column is `label`), the integer types in the fallback (line 598), and stale comments (lines 98, 492, 507, 516, 541, 617). Data not regenerated yet; run the check at A.12.
  - Review 2026-11-02: the `"labels"` typo is fixed on lines 236 and 374. The fallback at line 598 now gives int64 to `pair_id_*`/`label_*` and string to `label_name_*` (schema checked). The stale comments are fixed, and the dead `_drop` lines are removed (`delta_label`/`delta_pair_id` are no longer created). Committed in `95a10bc` and `441da06`. **The A.1 code is done.** The row-count check below still waits for the regeneration at A.12.
  - Check: still **201,596** rows. (label_1 → label_2, same pair_id) gives: IBD_p→IBD_d same = **99,996**; IBD→IBD different = 5; IBD_p→radio 435; IBD_d→radio 419; radio→IBD_p 398; radio→IBD_d 398; radio→radio 99,945.
- [x] **A.2 Add one correct target column, `is_ibd`:**
  `is_ibd = (pair_id_1 == pair_id_2) and (pair_id_1 != -1) and (label_1 == IBD_p) and (label_2 == IBD_d)`.
  Also add a diagnostic `pair_type` column with three values: true IBD / accidental that includes an IBD event / radio‑radio.
  - Review 2026-11-02: `is_ibd` is done. It is computed in `_rename_pair_columns` after `dR`, added to `EVENTS_LABEL_COLS`, and comes out int64 in the fallback. It was tested on 5 toy cases (radio–radio and d→p with the same pair_id both give 0). `pair_type` is done too (`np.select` returning `ibd` / `acc_ibd` / `radio_radio`; tested on 6 toy cases) and added to `EVENTS_LABEL_COLS`. The fallback rule at line 598 now also gives `pair_type` a string type (the schema was checked for all 8 label columns). Comment typos fixed. **The A.2 code is done** and committed in `fae9798`. Leftovers: the blank line before `return`; in the docstring's `Labels :` line, `pair_id` should be `pair_id_1, pair_id_2`, and `label_name_1, label_name_2` is missing.
- [x] **A.3 Build the physics feature set** (checked against the code 2026-11-06: all 10 are written, and `training.yaml` `include` = `E_1, z_1, r2_1, E_2, z_2, r2_2, dt, dR, dwall_1, dwall_2`; the `d_wall2` typo is gone. Optional, not done: Δt in µs), about 10 columns, modelled on Gavrikov et al. 2025 (Part B, step 2):
  - `E_p` and `E_d`
  - `dt_us`, the time difference in µs
  - `dR`, the 3D distance
  - `r2_p`, `z_p`, `r2_d` and `z_d`
  - `dwall_p` and `dwall_d`, where `dwall = min(R − r, H/2 − |z|)`, with R = 1.5 m and H/2 = 1.55 m
  - Review 2026-11-02: the helper `_event_distance_to_wall(r, z, radius, half_height)` is written and tested on 5 hand‑checked points (cap, bottom cap, side, mixed, centre). The constants `DETECTOR_RADIUS_M` and `DETECTOR_HALF_HEIGHT_M` are defined next to `PAIR_THRESHOLD_NS`. Positions are confirmed to be in metres. `dwall_1`/`dwall_2` are now computed in `_rename_pair_columns` and added to `EVENTS_INDIV_COLS`. Tested end to end through `_rename_pair_columns` on a synthetic IBD pair: 30 feature columns are written, dwall = 0.05/0.10 m, `is_ibd` = 1, `pair_type` = ibd. `src/data_loader.py` `_FEAT_COLS` also knows `dwall_*`. Still to do: Δt in µs (optional, for readability); the docstring typo `d_wall2`. Naming: the code keeps `_1`/`_2` (event order) instead of `_p`/`_d`.
- [x] **A.4 Drop from the features:** `t_1`, `t_2`, `dphi`, `dr`, `dr2`, `dD`, signed `dx`/`dy`/`dz`, and absolute `x`/`y`. Keep `t_1` in the output only as metadata.
  - Physics reviewed 2026-11-02 (the user reasoned through each one): `t` is "when", not "what", and lies outside the training range on new data. `dphi` wraps at 0/2π, so close events get dphi ≈ 2π. `dr`/`dr2`/`dD` compare distances to the axis or centre, not between events (opposite sides: dr = 0, dR = 2 m). `x`/`y` and the signs of `dx`/`dy` break the rotational symmetry, so they add nothing beyond `r`.
  - Decision: keep the parquet wide (all columns stay for plots and diagnostics) and choose the features in the `training.yaml` `features.include` list. Final 10: `E_1, E_2, dt, dR, r2_1, z_1, r2_2, z_2, dwall_1, dwall_2`.
  - Found: `src/models/MLP_model.py:58–68` hard‑codes `FEATURE_NAMES` (28) and `N_FEATURES`, which are used for the input layer (line 111) and a shape check (line 250). With 10 features the MLP will raise. Take the size from the data or the config instead (do it with A.11). `src/test/pipelin_test.py` imports `FEATURE_NAMES` too.
  - Done 2026-11-02: `training.yaml` `include` has exactly the 10 (checked against the columns preprocessing writes; no typos). `dE` was dropped too, since it is E_2 − E_1. Also done early from A.11: `target_col: is_ibd`, `positive_label: 1` (`trainer.py:143` compares `==`, so this works).
- [x] **A.5 Add multiplicity columns per pair** (checked against the code 2026-11-06: `n_before`, `n_on`, `n_gap`, `n_in_window` are in `EVENTS_COUNTS_COLS`, and `is_dc` is computed in `_rename_pair_columns`; committed in `cf4cfc0` + `a476e42`):
  - `n_before`: events in [t₁ − T, t₁)
  - `n_in_window`: events in (t₁, t₁ + T]
  - `is_dc`: `n_in_window == 1 and n_before == 0`, the Daya Bay isolated double coincidence (nH paper, Sec. IV B 4)

  Keep **all** pairs for training. Use `is_dc` later, when counting.
  - Review 2026-11-02: `_find_forward_pairs` now records `n_in_window = j − i` and `n_before = i − b`. `b` is a trailing pointer: `while b < i and t_i − times[b] > threshold`. Tested on 6 synthetic events (A→B and A→C get n_in_window = 2; B→C gets n_before = 1; D→E is isolated) and on the edge (an event exactly T before counts, T + 1 ns doesn't). Decided: the counts are observables, so they go to the **features** file, but not into `include`.
  - **Don't wire them into the output lists yet.** `_project_pairs_chunked` and `_rename_pair_columns` also run on the off‑window pairs, and `_find_false_pairs` promises the same schema. Adding the counts to the written lists, or computing `is_dc` in `_rename_pair_columns`, will raise a KeyError on the off‑window run until `_find_false_pairs` produces them too. Open design question for A.6: what should `n_in_window` count for an off‑window pair so that `is_dc` means the same thing as on‑window?
  - Answer (the user, 2026-11-02): count events in the **shifted** window (t_i + kT, t_i + (k+1)T], which must be 1, and keep `n_before` as is, which must be 0. For Poisson singles this matches the on‑window probability. In `_find_false_pairs` the count can come from `j_lo`/`j_hi`. Still open: the **gap** (t_i, t_i + kT]. An IBD prompt anchor with its delayed in the gap and a random event in the shifted window becomes an off‑window "DC accidental" with no on‑window counterpart, and its anchor features look signal‑like. Should the gap be required empty? (If so, there is a known factor e^(−R_s·kT) to correct with the A.9 metadata.)
  - Decided (the user): **require the gap to be empty** (`n_gap == 0`). The off‑window `is_dc` is then n_before == 0, n_gap == 0, and n_in_shifted_window == 1. Consequence: off‑window DC accidentals are low by P(gap empty) = e^(−R_s·kT), so correct each window k by e^(+R_s·kT) when computing B (C.4). Numbers: R_s·T = 0.00406, so the loss is 0.4% at k = 1, 4.0% at k = 10, and ≈ 2.2% averaged over k = 1…10. Correct **per window before averaging**.
  - A.6 progress, 2026-11-03 (not committed; the script is mid‑rewrite): `mode` is removed from the signature and the caller; `n_off: int = 10` is added; `hi_bound = t_i + (n_off+1)·T`; the docstring states window k = (t_i + kT, t_i + (k+1)T], k = 1…n_off.
  - **A pre‑existing bug was found** (it was also in the committed `double_window` code): when the first event beyond T (`j_lo`) is already past `hi_bound`, `range(j_lo, j_hi + 1)` still emits it. Test on 6 synthetic events with window (2, 4] ms: there should be 0 pairs, but it emitted 5 pairs with Δt = 9–10 ms. At 2 Hz the window is almost always empty, so the old off‑window sample is mostly pairs at arbitrary Δt (on the order of one per event, not ~100k). Fix: skip the anchor when `times[j_lo] > hi_bound`. **Fixed by the user** (`if times[j_lo] > hi_bound: continue`). Retested: the 6 events give 0 pairs. Edges with n_off = 2 are correct: Δt exactly T is excluded, T + 1 ns is included, exactly 3T is included, 3T + 1 ns is excluded.
  - The log line is fixed (`n_off=%d`). Committed in `89bd981`. Still stale: the `j_hi` comment at lines 406–408 (it says `t_k+1 - t_i <= threshold`, but `j_hi` is the last index with t_k − t_i ≤ (n_off+1)·T, and it still mentions double_window/all_beyond).
  - Next in A.6: per pair, `k_window` (which window Δt falls in) and `dt_shifted = Δt − k·T`; then the counts `n_before`, `n_gap` and `n_in_window` per window.
  - **k_window exercise (done 2026-11-03):** find k from Δt and T such that k·T < Δt ≤ (k+1)·T. With T = 2 ms:

    | Δt | k |
    |---|---|
    | 2.000001 ms | 1 |
    | 3.0 ms | 1 |
    | 4.0 ms (exactly 2T) | 1 |
    | 4.000001 ms | 2 |
    | 6.0 ms (exactly 3T) | 2 |
    | 7.5 ms | 3 |

    Steps: (1) fill in the two `?`. Done: 2 and 3, both correct. (2) Done: **k = ⌈Δt/T⌉ − 1** (the user derived it; floor fails on the closed upper end). Watch out in code: the partner loop variable is already called `k`, so name the window index `k_window`. (3) Done: `k_window = int(np.ceil(delta_t/threshold)) - 1` in the partner loop of `_find_false_pairs`, added to `out_col_names` (the user chose the name `k_window`). Edge test with n_off = 2: all 8 pairs match the formula, and the values stay in 1–2. (4) Done: `dt_shifted = delta_t - k_window * threshold` (the user checked 3 rows by hand). Tests: edge cases all in (0, T]. Poisson stress test (50k events at 2.03 Hz, n_off = 10, 2012 pairs): the k rule holds; k = 1…10 each have 192–219 pairs, against an expected 203 ± 14; `dt_shifted` is flat (499/525/508/480 in 4 bins). Committed in `bdeb727`. **Counts for off‑window pairs:**
    - `n_before` is done (the `b` pointer, reused from the forward scan). Hand test: an anchor with an event 1 ms before gets 1, isolated anchors get 0. Poisson test: matches a brute‑force count on all 1942 pairs, and the fraction with n_before = 0 is 0.9959 = e^(−R_s·T). Not committed yet. The `j_hi` comment at about line 412 is still wrong.
    - Next: `n_gap` (events in (t_i, t_i + k·T]) and `n_in_window` (events in window k). Idea: on‑window events = j_lo − 1 − i; then n_gap(k) = that plus the partners with k_window < k, and n_in_window(k) = the partners with k_window == k. So compute the k_window of all partners before emitting.
    - 2026-11-03: `n_before` and the comment fix are committed in `3d7ce78`. The user worked out n_on = j_lo − i − 1 and the example (n_on = 0, partners in windows 1, 1, 3: the window‑3 partner gets gap 2 and window 1; the window‑1 partners get gap 0 and window 2), and saw that this needs **two passes**. Now: pass 1 counts partners per window in a `counts` list (length n_off + 1 or more, reset for each anchor). Pass 2 (the existing emit loop): `n_gap = n_on + sum(counts[1:k_window])` and `n_in_window = counts[k_window]`.
    - **Done (2026-11-03):** both passes are written (pass 1 is a separate loop that counts; pass 2 emits). The new columns are `n_on`, `n_gap` and `n_in_window` (it uses `sum(counts[:k_window])`, which is fine because `counts[0]` is always 0). Hand test: partners in windows 1, 1, 3 give (gap, win) = (0, 2), (0, 2), (2, 1); an anchor with an on‑window event and a partner in window 2 gives n_on = 1, n_gap = 1. Poisson brute force: `n_gap` and `n_in_window` match on all 2000 pairs.
    - **Next: one shared schema, then wiring.** Both scans must write the same columns. The forward scan lacks `k_window`, `dt_shifted`, `n_gap` and `n_on`: for on‑window pairs these are 0, `dt`, 0 and `n_in_window`. Then `is_dc = (n_before == 0) & (n_gap == 0) & (n_in_window == 1)` works for both in `_rename_pair_columns`. Also force `is_ibd = 0` (plus a flag) for off‑window pairs. Then add the columns to the written lists (features file, not `include`) and update the fallback types.
    - **Steps 1–2 done (2026-11-03):** the forward scan now also writes `k_window = 0`, `dt_shifted = dt`, `n_gap = 0` and `n_on = n_in_window` (a first version had `n_gap = n_on` and missing appends; both fixed). `is_dc` is computed in `_rename_pair_columns`. Test on 6 events: only D→E gets is_dc = 1; A→B is a true IBD but not DC; B→C is rejected by n_before. The off‑window pairs also pass through `_rename_pair_columns` with correct hand‑checked counts. The two scans have the same column set; the order differs (harmless; optionally align it). **Not committed yet.**
    - **Step 3 (wiring) is next:** add the counts, `k_window`, `dt_shifted` and `is_dc` to the written lists (features file); update the fallback types; force `is_ibd = 0` for off‑window pairs plus a flag for real IBDs found there.
      - Committed `e1c7582` (shared schema + `is_dc`). Then a new list `EVENT_COUNTS_COLS` (7 columns), added to `REQUIRED_EVENT_COLS`, the features writer, the fallback `all_feat_cols` and the docstring. Full projection test (scan → rename → project) on both samples: 37 feature columns, all 7 present as int64, `is_dc` correct. **Not committed yet.**
      - Remaining in step 3: (a) the fallback type for `EVENT_COUNTS_COLS` → int64 (about line 664); (b) `src/data_loader.py` `_FEAT_COLS` must list the 7 new names; (c) force `is_ibd = 0` (plus a flag) for off‑window pairs. Also (pre‑existing): `dt` is int64 in the real files but float64 in the fallback.
      - **(a), (b) and (c) are all done (2026-11-03).** The fallback gives int64 to `EVENT_COUNTS_COLS`; the loader knows the 7 names (the duplicate list in the loader was removed). In `_rename_pair_columns`, in this order: `is_ibd` → `false_ibd = (is_ibd == 1) & (k_window > 0)` → `is_ibd = 0` where `k_window > 0` → `pair_type` → `is_dc`. `false_ibd` is in `EVENTS_LABEL_COLS` (int64 via the label fallback rule). Test: an off‑window IBD at 2.3 ms gives is_ibd 0, false_ibd 1, acc_ibd; the off‑window accidentals get false_ibd 0; an on‑window IBD is unchanged (is_ibd 1, ibd, is_dc 1). **The A.5/A.6 code is complete**, committed in `cf4cfc0` + `a476e42`.
      - Still for A.6/C.4: the per‑window correction e^(+R_s·k·T) is applied when computing B (Part C), not in preprocessing. The real‑data check comes at A.12 (about 4 rows with false_ibd = 1 in window 1; a flat `dt_shifted`; about 101.6k pairs per window). Remember the approach: signature/docstring first, small steps, test on synthetic events.
  - The comment above the `b` loop is fixed. Committed in `ae89778`.
- [x] **A.6 Rebuild the off‑window sample** (checked against the code 2026-11-06: `_find_false_pairs(..., n_off=10)`, `k_window = ⌈Δt/T⌉ − 1`, `dt_shifted`, the counts per window, and `false_ibd` + `is_ibd = 0` for k > 0; the `j_hi` comment is fixed. The per‑window e^(+R_s·k·T) correction belongs to C.4, not here):
  - Use N_off successive windows (kT, (k+1)T] for k = 1…N_off (start with N_off = 10).
  - Store `k_window` and `dt_shifted = dt − k·T`, so the model sees the same Δt range as in the signal window.
  - Use the same features and multiplicity columns as the signal pairs.
  - Force `is_ibd = 0`, but add a flag for the about 4 real IBDs with Δt > 2 ms that land in the first off‑window.
    - Note (2026-11-02): `_project_pairs_chunked` → `_rename_pair_columns` runs on both the on‑window and the off‑window pairs. If A.2 computes `is_ibd` there, those ~4 off‑window rows come out as 1. Override that here.
- [x] **A.7 Make the ordering deterministic:** a stable sort on (time, then IBD_p before IBD_d).
  - Done 2026-11-03: both scans use `sort_values(["time","label"], kind='stable')`. A first attempt (sort by time, then a separate stable sort by label) made label the primary key and produced negative Δt; it was fixed. Tests: a tie listed as (delayed, prompt) now gives is_ibd 1; a radio at 4.0 ms before an IBD pair gives correct positive Δt; shuffled random input gives forward 0 ≤ Δt ≤ T and off‑window Δt > T everywhere.
  - 2026-11-03: the user found that the default `quicksort` is not stable and that a tie in the order (delayed, prompt) gives is_ibd = 0, so a real IBD is lost. "Stable" alone is not enough, because the order before the sort (`_load_raw` sorts by `[pair_id, time]`) doesn't guarantee the prompt first. Decision: sort by `["time", "label"]` (1 < 2 < 3) in **both** scans (lines 239 and 389). Note for A.8: real data has no `label` column, so this sort must fall back to `time` only when the truth is absent.
- [ ] **A.8 Make I/O dataset‑aware:** ⏸ **Deferred 2026-11-04** until test data exists. Only the training data is used for now. The notes below stay valid for when it resumes.
  - **Update 2026-11-06 (status check against the code):** part of step 1 is done by cleanup item 1. `_validate_raw_inputs` now checks the column **names** of both files from the schema (missing → ValueError), and the positional rename in `_load_raw` is gone. So the notes below about "the 8‑name rename" and "renames by position" are **outdated**. Still to do when A.8 resumes: truth optional (`-> bool`), the sort fallback to `time` without `label`, the truth‑aware places (4)–(6), and `--dataset`.
  - Add a `--dataset {training,test,inference}` option that reads `data/raw/<dataset>/` and writes `data/processed/<dataset>/{singles,pairs,offwindow_pairs}.parquet`.
  - Make `truth.parquet` optional. ~~Label columns become null when it is absent.~~ **Decision 2026-11-03 (option b): without truth, the label columns don't exist and no labels file is written.** Null labels would make `dropna(subset=REQUIRED_EVENT_COLS)` drop every row (the user worked this out).
  - Places to make truth‑aware, in order: (1) `_validate_raw_inputs`: only `events.parquet` is required; (2) `_load_raw`: concat, the 8‑name rename and the sort by `pair_id`; (3) the scan sorts: `["time","label"]` only if `label` exists; (4) `_rename_pair_columns`: `is_ibd`, `false_ibd`, forcing, `pair_type`; (5) `_project_pairs_chunked`: `REQUIRED_EVENT_COLS` and the labels writer; (6) `_process`: `total_labels`; (7) `--dataset` paths. Approach: from (3) on, check `"label" in df.columns` locally; only (1) and (2) check whether the file exists.
  - **Resume here (A.8 step 1, 2026-11-03).** The user's idea: validate the raw files by their **contents**, not only by whether they exist, and log what was found. Agreed, with two refinements:
    - Check column **names**, not the number of columns. `_load_raw` renames by position (`df.columns = raw_col_names`), so a swapped column would silently relabel data. The current names are `events.parquet` = `time, energy, x, y, z` and `truth.parquet` = `label, pair_id, label_name`.
    - Read only the **schema**: `pq.read_schema(path).names` (no rows loaded; the events file has 25M rows). Row counts for the log come from `pq.ParquetFile(path).metadata.num_rows`.
    - Design: `_validate_raw_inputs` **returns a `bool`** (has truth), so `_load_raw` (step 2) uses it instead of checking the file again.
    - **Task when resuming:** first only the signature `-> bool` and the docstring: what is required (events and its columns) and what is optional (truth and its columns if present); what raises an error and what is only logged; what the returned bool means. Keep the old body until the docstring is reviewed. Then write the body, and Claude tests it on scratch folders (events only / both files / nothing / wrong column names).
    - Also fix the error message, which hard‑codes `data/raw/training/` and says truth is required.
    - Review 2026-11-04 (first draft of the docstring): the signature `-> bool` is in. To fix: (a) as written the function never returns `False`; the bool should mean "truth is present and valid"; (b) don't prompt the user for the column order (`input()` hangs non‑interactive runs); raise a `ValueError` that shows the expected and found names instead; (c) say that events is required and truth is optional; (d) add the order and row‑count rules; (e) use numpy format: `Parameters`/`Returns`/`Raises` underlined with `-`, not `_`. Still open: does the user want `_load_raw` to select columns by name (then order doesn't matter)?
- [x] **A.9 Write run metadata** (done for training 2026-11-06; per dataset once A.8 is done) for each dataset: the live time (t_max − t_min) and the singles rate R_s. Part C needs both for R_acc = R_s²·T·ε_m (nH paper, eq. 7).
  - Decisions 2026-11-04: N = **all events**, because data without truth can't separate IBD events from singles. The ~0.8% IBD share biases R_s² by about 1.6%; that's fine for the ε_m correction and the R_acc cross‑check, and B itself comes from the off‑window pairs. Compute it in `_process` before step 4 (`_find_forward_pairs`). The A.8 docstring draft was reverted with `git restore`.
  - Watch out: at that point `df_total` is sorted by `[pair_id, time]`, not by time, so use min/max, not the first and last row. Time is in ns, so T must be converted to s.
  - Update 2026-11-04: the user keeps **ns** throughout (`time` is int64 ns, consistent with `PAIR_THRESHOLD_NS`), so no conversion. R_s is then in **ns⁻¹** (≈ 2.03e‑9), which is not nHz, so the key or variable name must say `per_ns`. R_s·k·T is dimensionless either way. The user is leaning towards `.npy` (fixed order, no names); Claude pointed out that the file then carries no units either. The user's call.
  - Decision 2026-11-06: **`.npz`** (named keys, units in the key names). Location: **`data/processed/`** (confirmed by the user; it is git‑ignored and per dataset later). Design: **computing is separate from saving.** A pure function returns the numbers, and `_process` saves them. Next: the signature and docstring of the compute function (input: times only, or the df? return: tuple or dict? what happens with 0 or 1 events, where T = 0?), plus whether the path goes in `PROCESSED_FILES`.
  - Review 2026-11-06 (first draft, `compute_contants`, line 683): good: the input is times only, it returns a dict, and the sections are in numpy format. To fix: the typo, and "constants" is the wrong word (these are measured quantities of the run); a leading `_` like the other helpers? `-> dict()` should be `-> dict[str, ...]`; the parameter `time` shadows `import time` (line 26); the type should be `np.ndarray or pd.Series` and the docstring should say "doesn't need to be sorted"; Returns must list each key with its type and unit (and N?); the definition of T (t_max − t_min) is missing; Raises (T = 0) and Notes (N includes IBD) are missing; the summary should be one line.
  - Review 2 (2026-11-06): the signature is fixed: `_compute_parameters(events_times: np.ndarray | pd.Series) -> dict[str, float | int]`. **`pass` at line 698 has 3 spaces, which is an IndentationError that breaks the whole module.** Docstring points 4–9 are still open: the Parameters name and type don't match the signature; Returns isn't in numpy format and lists no real key names; the T definition, "unsorted", Raises and Notes are all missing.
  - Review 3 (2026-11-06): it compiles; the summary is one line; "Rate" is used; **N is included**. Still open: `events_time` vs `events_times`; "unsorted"; Raises is empty, underlined with `_` and comes before Returns; the Returns keys are not real names with units and not indented; T is still not defined as t_max − t_min.
  - Review 4 (2026-11-06): **the docstring is accepted, apart from two one‑line fixes**: define `time_total` as t_max − t_min, and use the same name in Raises (it still says "T"). The keys are `R_s`, `time_total`, `N_total`; the units are only in the docstring, not in the key names, which is the user's choice (Claude noted that the docstring isn't saved in the `.npz`). **Next: the body.** Edge case to test: an empty `pd.Series` gives `.max()` = NaN, `NaN <= 0` is False, so the check passes and R_s = NaN. An empty `np.ndarray` makes `np.max` raise a ValueError by itself. Decision on the checks (from the user): **N ≥ 2, then max ≠ min**, checking N before calling max/min. Next: Raises with both conditions, then the body, then tests on 5 toy cases × (array, Series).
  - Review 5 (2026-11-06): **the body of `_compute_parameters` is correct.** 10/10 toy cases pass (normal, unsorted, 1 event, `[5,5,5]`, empty × array/Series). On the real training data: N = 25,043,297, `time_total` = 1.2344e16 ns = 142.87 d, `R_s` = 2.0288e‑9 ns⁻¹ = 2.029 Hz, ε_m = e^(−2·R_s·T) = 0.99192, which matches the 0.9919 check. Leftovers in the docstring: `(t_N - t_1)` → `t_max − t_min`; Raises still says "If T ≤ 0" (list both conditions); the typos in the error messages, which also don't show the values received. **Next: the saving in `_process`** (`np.savez(path, **params)` before step 4, a log line, and the decision on `PROCESSED_FILES`).
  - Review 6 (2026-11-06): `"parameters": PROCESSED_DIR / "parameters.npz"` is added to `PROCESSED_FILES` (decision made). The docstring is fine. **Two bugs, both reproduced on toys:** (1) `np.savez(path, params)` without `**` saves a single pickled object `arr_0`, which `np.load` refuses (allow_pickle=False), and the names are lost; (2) `ValueError("... %d", N)` doesn't format, so the message becomes a tuple. Also: `log.info` with an f‑string and the typo "corretly" sit inside the compute function; suggested moving it to `_process` as `"Saved %-45s"`, like the other outputs. Then regenerate or test the file end to end.
  - Review 7 (2026-11-06): **both bugs are fixed; A.9 is done.** Round trip tested (compute → `np.savez(**params)` → `np.load`): the keys are `N_total`, `time_total`, `R_s` (0‑d int64/int64/float64, read with `.item()` or `float()`). Real data: R_s = 2.0288e‑9 ns⁻¹, time_total = 1.2344e16 ns. The log stays inside the compute function (the user's choice). Cosmetic: "corretly"; `{x=}` prints `np.float64(...)` in the log. The old `data/processed/` lacks `parameters.npz`, so the next run regenerates everything (fine, A.12).
- [ ] **Cleanup of hard‑coded name lists** (raised by the user 2026-11-04; do it after A.9, before or together with A.11). **Status 2026-11-06 (checked against the code):** item 1 ✓ (column check + no rename) · item 2 → the config step · item 3 ✓ (`EVENTS_FEATURE_COLS`) · item 4 resolved: no YAML, see "Input data contract"; named constants optional · item 5 ✗ (`"label_name" in c or c == "pair_type"`, l. 697) · item 6 half done: `_project_pairs_chunked` ✓, but `_rename_pair_columns` still lists columns by hand (l. 577–581, stale: no counts, no `false_ibd`/`is_dc`). Pass 2 (across modules) is open. Principle: what matters is not hard‑coding but **duplication**, so each name should be defined once. Findings to verify: the raw files already have exactly the names `COLUMN_NAMES[:8]`, so the positional rename in `_load_raw` is a no‑op. The 4 engineered names `r, r2, phi, distance` are written in 3 places (`COLUMN_NAMES`, the exclusion tuple in `_load_raw`, the assignments). `_RAW_TO_PHYS` is mostly the identity (only time→t, energy→E, distance→D). In `RAW_FILES` only `.name` is used.
  - Started 2026-11-06. **Pass 1, inside `preprocessing.py`, one item per commit:** (1) `COLUMN_NAMES` + the positional rename (l. 59, 182–184); (2) the file names, written in `RAW_FILES` (l. 43) *and* in `_load_raw` (l. 177); (3) `EVENTS_INDIV_COLS + EVENTS_DELTA_COLS + EVENT_COUNTS_COLS`, written 3× (l. 617, 650, 675); (4) the magic label numbers 1/2/3/−1 (l. 583, 592; verified: 1 = IBD_p, 2 = IBD_d, 3 = radio, and the A.7 sort relies on 1 < 2 < 3); (5) the fallback type rule `"label_name" in c or c == "pair_type"` (l. 677); (6) docstrings that list columns by hand (l. 550–560, 614–619), already stale. Leave alone: `_RAW_TO_PHYS`, `EVENTS_LABEL_COLS`, `EVENT_COUNTS_COLS` (single definitions), and `PAIR_THRESHOLD_NS`/`DETECTOR_*` (they go to config in A.11).
  - **Pass 2, across modules (later, together with A.11):** `src/data_loader.py` copies `_RAW_DIR`, `_PROCESSED_FILES`, `_FEAT_COLS`, and `_LABEL_COLS` (which is already stale: `pair_id, label_name_1, label_name_2`); `plots.py` has `EVENTS_FEATURE_COLS`; the MLP has `FEATURE_NAMES` (28); `training.yaml` has `features.include` (10). This is where the 9/10/28 mismatch comes from. To decide: one source of truth per kind of name.
  - Rule agreed 2026-11-06: "would you change it without changing code?" Yes → YAML (paths, file names, window, geometry; model features in `training.yaml`, and the MLP should take N_FEATURES from it). No → a Python constant, defined once and imported (names of created columns, label codes). The user wants **dirs + file names in `config.yaml`**: agreed. Evidence: `data_loader._PROCESSED_FILES` already lacks `parameters`, one day after A.9. Open question: a shared YAML reader that both modules import (`trainer.load_config` already exists, with the wrong default path). The user also suggested **moving `_load_raw` to `data_loader`**: Claude advised against it, because of the cycle (data_loader → preprocessing at l. 137, so preprocessing → data_loader would close it) and because most of `_load_raw` is feature engineering, not I/O. The user's decision is pending. Item 1 (delete the rename + `COLUMN_NAMES`) comes first either way.
  - **Item 1 revised 2026-11-06.** The user's concern: teammates' files might have other column names. So don't delete the expected names; **keep them as two constants (events, truth) and replace the rename with a check** in `_validate_raw_inputs` (`pq.read_schema`, raising a ValueError that lists the missing/found names; both files still required, since A.8's optional truth is still deferred). The code reads columns by name, so the order in the file no longer matters, and a different name fails loudly instead of silently relabelling. Delete only the rename and the engineered names in `COLUMN_NAMES`. Tip for the team: compare `sha256sum` of the raw files. The team's files are identical (the user confirmed). For outside users with other names: (A) now, a clear ValueError (expected vs found); (B) optionally, during the config step, a mapping `raw_columns: {code_name: file_name}` in `config.yaml`, with a rename **by name**. B also needs A's check, so A is never wasted work.
  - Review 2026-11-06 (item 1, first try): the rename is deleted (good; the columns still come out in the file's order). **Bugs:** `LABEL_COLS_TOTAL` was renamed to `TRUTH_COLS_TOTAL`, but l. 743 still uses the old name, so `_process` raises a NameError (py_compile doesn't catch it); `log.info("Saved %-45s  %d", path, params)` with a dict gives "--- Logging error --- TypeError: %d format". **Not as agreed:** `EVENT_COLS_TOTAL` still holds all 12 names (it should be the 5 from events.parquet), and its comment is stale; the Raises went into `_load_raw` instead of `_validate_raw_inputs` (which runs first and reads only the schema); the Raises entries aren't in numpy format; "combinned"; "for the training".
  - Review 2 (item 1): `EVENT_COLS_TOTAL` now holds the 5 events names ✓; the log uses `%s` ✓. **`_load_raw` HEAD vs new on toy files with the real schema: same columns, same order, identical DataFrame** ✓. **Still a NameError:** the constant is now called `LABELS_COLS_TOTAL` (l. 63), but l. 748 uses `LABEL_COLS_TOTAL`. Also open: the Raises is duplicated in `_load_raw`, not in numpy format, and the summary of `_validate_raw_inputs` mentions only FileNotFoundError; the comment above `EVENT_COLS_TOTAL` is stale; the names are inconsistent (EVENT/LABELS, "TOTAL"). Next: the check body (a dict file → expected cols, used in the existing loop).
  - 2026-11-06: the user wants the raw file names to be configurable too. Agreed; it is the config step (cleanup item 2 + A.11). Split: the **keys** `"events"`/`"truth"` are roles in the code (they stay in Python and the check uses them); the **values** (file names, dirs) go to `config.yaml`. Watch out: `config.yaml` is committed, so personal values cause git conflicts. Keep shared defaults in `config.yaml`; personal overrides go through the CLI (`--raw-dir` exists) or an untracked `config.local.yaml` in `.gitignore`. Order: finish the column check first; it doesn't depend on where the names come from.
  - Review of the loop (2026-11-06), with the test script `test_validate.py` (4 toy folders): **all 4 cases raise, the correct files included.** Causes: `required = EVENT_COLS_TOTAL + LABEL_COLS_TOTAL` (8 names) is compared with each file (5 or 3 names); `!=` on lists also fails on order (`LABEL_COLS_TOTAL` is `pair_id, label, …`, the file has `label, pair_id, …`) and on extra columns; the ValueError again uses `"…", list`, so the message is a tuple, and the `\` inside the string adds 20 spaces. The FileNotFoundError message says "Ensure <file> contains events.parquet". To fix: a dict keyed like `RAW_FILES`; the condition "missing is non‑empty"; an f‑string with the file, missing and found.
  - Review 2 of the loop: ok ✓, missing truth ✓, X ✓ (by coincidence). **The set difference goes the wrong way** (`file − expected` = extra columns): extra column → raises with `[]`, and **"missing z" passes**, which is exactly the case the check exists for. The user used `if name == "events"/"truth"` branches instead of the dict, so the bug is written twice. The `\` inside the string is still there, "mismtach", the found names are missing, and the schema is read twice. A "missing z" case was added to `test_validate.py`.
  - Review 3 of the loop: **the condition is correct now** (dict `required[name]`, expected − found); the 5 cases ok/missing truth/X/extra/missing z all behave as they should. **A remaining bug in the message:** it uses `EVENT_COLS_TOTAL` instead of `required[name]`, so "truth no pair_id" (a 6th test case) reports `['time','energy','x','y','z']` instead of `['pair_id']`. Also open: compute once (schema + missing), the found names in the message, "expectedstructure" (missing space), "mismtach", `contains {full}` in the FileNotFoundError, and Raises in numpy format.
  - **Item 1 done in function (2026-11-06): 6/6 test cases correct** (ok, missing truth, X, extra column, missing z, truth no pair_id → `['pair_id']`). Optional polish left (the user's call): compute once, the found names in the message, the typos, Raises in numpy format, the stale comment above `EVENT_COLS_TOTAL`, and `_load_raw`'s duplicated Raises. Next in pass 1: item 3 (the feature list written 3×), or go straight to the config step (items 2 + A.11).
  - Polish attempt (2026-11-06): the typos are fixed, but **the message now computes found − expected (extras)**, so "missing z" and "truth no pair_id" report `[]` and X reports `['X']`. Raising/passing is still correct in 6/6 cases. Cause: still computed twice. Not committed yet.
  - **Final (2026-11-06): 6/6 correct, messages included** (`missing` is right in every case; X shows found `['X']` and missing `['x']`). Small leftovers: the variable `found` actually holds the *unexpected* columns, but the message calls them "found"; the schema is read twice; the docstrings and comment from the earlier list. **Item 1 done.** Next: item 3. The expression `EVENTS_INDIV_COLS + EVENTS_DELTA_COLS + EVENT_COUNTS_COLS` appears at l. 638 (docstring), 671 and 696, and `REQUIRED_EVENT_COLS` (l. 107) repeats the parts.
  - Item 3, 2026-11-06: the user proposed one constant as a **set**. A list is needed instead: `df[set]` raises a TypeError in pandas 2.3.3, and a set has no fixed order, which the file's column order needs (and `training.yaml` relies on it). For `REQUIRED_EVENT_COLS` (only `dropna(subset=…)`) the order doesn't matter (the user answered correctly).
  - **Item 3 done (2026-11-06):** `EVENTS_FEATURE_COLS` (l. 106) is used at l. 672 and 697 and in the docstring; `REQUIRED_EVENT_COLS = EVENTS_LABEL_COLS + EVENTS_FEATURE_COLS`; `EVENT_COUNTS_COLS` was renamed to `EVENTS_COUNTS_COLS` (no use of the old name is left in `src/`). Checked against HEAD: the features have the same 37 names in the same order; required has the same set of 46 with no duplicates; features ∩ labels = ∅ (no leakage). Leftover: the stale hand‑written lists in the docstring at l. 640 and 642 (item 6). Next: item 4 (label codes).
  - Item 4, 2026-11-06: the user wants the label codes in YAML. Accepted, since they describe how the input is encoded (like the `raw_columns` mapping), **on one condition**: the sort `["time","label"]` (l. 262, 412) assumes prompt < delayed, so the config must be **validated** when loaded (raise if prompt ≥ delayed), because a comment in the YAML stops nobody. Order: now, named Python constants (e.g. prompt/delayed/radio/no‑pair) at l. 605 and 614; in the config step their values come from YAML and the names in the code stay the same. Careful: `is_ibd == 1` (l. 608) and `n_in_window == 1` (l. 619) are **not** label codes.
  - **Revised the same day:** no YAML for the codes. They are set by the dataset, so they belong to the "Input data contract" section (below the A.12 checks), together with an adapter if the format ever changes. Named Python constants are still optional, for readability only (the user's call).
- [ ] **A.10 Add a time‑block id** (e.g. per day), so the trainer can split by time block instead of at random.
- [ ] **A.11 Plumbing** (status checked against the code 2026-11-06):
  - [x] Delete the stale `src/preprocessing.py` (gone; commit `0195351`).
  - [ ] Fix the import in `src/data_loader.py` to point at the real module (still `from preprocess import run` at l. 137). Fixed in PR #1; tick it once merged.
  - [ ] Move T (2 ms), N_off, R, H/2 and the paths into `config.yaml` (still empty, 0 bytes). **Parked 2026-11-09, see "Parked for later".**
    - Plan (2026-11-06): (1) the user drafts `config.yaml`: raw dir + the 2 raw file names, processed dir + the 7 output file names (the keys `events`/`truth`/`total_features`/… stay in code, the values go in YAML), `coincidence window` in ns, `n_off`, R and H/2 in m, `flush_every`; units in the key names. (2) A shared reader in its **own small module** (not `trainer.load_config`: importing trainer pulls in optuna/sklearn/models), with signature + docstring first; the path is resolved from the repo root (`Path(__file__)`), not from the CWD. (3) `preprocessing.py` keeps its constant **names** (`RAW_DIR`, `PAIR_THRESHOLD_NS`, …) but takes their values from the config **at import**, because they are used as default arguments (l. 228–229, 381–382), which are evaluated when `def` runs. (4) Then `data_loader.py` (l. 88–122) does the same. Then `trainer.py` (pass 2).
    - 2026-11-07: (1) done, and `config.yaml` loads (5 groups, 16 values). The user's decisions for (2): find the files with `Path(__file__)`; build the `{key: Path}` dicts in the reader's module; raise `ValueError` for bad content. Also decided: the data dirs are anchored to the repo root too, and the required keys (group → keys) are written once, in the reader's module. Next: signatures + numpy docstrings, reviewed before any body.
    - Review 1 of `src/config.py` (2026-11-07). It works: `parents[1]` finds the root even when run from `/tmp`, `YAMLError` becomes `ValueError`, and there are checks for a non-dict root and missing groups. Not yet matching the decisions: (a) the Path loop (l. 43–53) changes nothing, because it tests top-level keys for `_dir`/`_path`, while `dir` sits one level down; tested, it returns exactly the raw YAML with `dir` still a str. (b) `REQUIRED_KEYS` lists groups only, so a missing `events_labels` passes. (c) No int check on the window. (d) The return type is wrong (the values are dicts), and the docstring is not numpy style. Minor: l. 23–24 duplicate what `open()` already does; l. 58 loads at import (decide this together with step 3). Next: `REQUIRED_KEYS` → group → keys, and the path builder as its own function, signature + docstring first.
    - Review 2 (2026-11-07). `REQUIRED_KEYS` is now group → keys ✓. A missing key gives a clear `ValueError` ✓, and `dir` becomes an absolute Path ✓ (both tested). Tested on toy dicts: an empty group (`detector: None`) gives an `AttributeError`, not a `ValueError` (l. 41; the dict check only sits in the recursive branch, which never runs, because the values are sets). A window of `"2e6"` passes. The file names are not joined to `dir` (`events_labels` is still a bare str). The new `validate_config` and the nested `process_paths` are generic recursions (deeper nesting, lists) for a 2-level config. The path builder with its signature + docstring first was not written. Asked the user to write that step themselves. The user then removed the generated functions, so `src/config.py` now holds only `REQUIRED_KEYS` and `get_repo_root`. Next: stubs (signature + numpy docstring, body `...`) for the reader and the path builder.
    - PyYAML gotchas (tested): `2e6` is read as a **string**; `2.0e+6` is a float; `2_000_000` is an int; `1.55` is a float.
  - [ ] Make `src/trainer.py` raise an error when a feature listed in `training.yaml` is missing (still silently dropped at l. 131: `[c for c in include if c in X_df.columns]`). Also its default `config/training.yaml` (l. 98) doesn't exist.
  - [x] In `training.yaml`, set `target_col: is_ibd` and update `features.include` to the new names (done early, with A.4).
  - [ ] The MLP takes `N_FEATURES` from the config or data instead of hard‑coding 28 (`MLP_model.py:58–68`; see A.4).
  - **Docs are out of date** (review 2026-11-05, for teammates who are just starting): the README has no physics, no "how to run" and points to the stale `src/preprocessing.py`. `STRUCTURE_GUIDE` says `scripts/` and `notebooks/` are at the top level. `MODELS.md` lists 9 `delta_*` features while the code has 28 and `training.yaml` says 10. `EVALUATION.md` has wrong file names (`calibration.py`, `config/training.yaml`) and 4 wrong statements (PR‑AUC isn't trapezoidal; the G‑mean "chance" claim; G‑mean ≠ Youden; the prevalence factor in the cost threshold). ~~CONNECT / GIT_WORKFLOW~~ **rewritten 2026-11-05**: each person has their own CloudVeneto VM (no shared VM anymore); ~~everyone commits to `main`~~ **since 2026-11-09, every change goes through a PR** (branch → push → review → merge; GIT_WORKFLOW rewritten. since 2026-11-09 `main` is protected (a PR with 1 approval, admins included, no force-push or deletion) and the teammate who opened PR #1 is invited as a collaborator with write access); `data/raw/training/` lives inside each clone. ~~This plan file is untracked.~~ Tracked since `163ecf1`. README, STRUCTURE_GUIDE, MODELS.md and EVALUATION.md haven't changed since `35b8887` (still to do).
- [ ] **A.12 Regenerate** the processed data with `--force`, then run the checks below. (2026-11-06: `data/processed/` is still from an old run: no `parameters.npz`, no `total_labels.parquet`.)

### Checks after regenerating (training set)

- [ ] `is_ibd == 1`: **99,996** pairs, i.e. 100,000 minus the 4 with Δt > 2 ms, and exactly one per `pair_id`.
- [ ] On‑window accidentals: about **101.6 k**, which is 99,945 + 796 + 854.
- [ ] Off‑window pairs **per window**: about **101.6 k ± 320**, consistent with R_s²·T·live time. About 4 `is_ibd`‑like rows in window 1.
- [ ] True pairs: mean Δt ≈ 200 µs (exponential), median ΔR ≈ 0.44 m, E_d peaks at 2.18 MeV.
- [ ] `is_dc` among true IBDs: about **99.2%**. This should match ε_m ≈ e^(−2·R_s·T) = 0.9919.
- [ ] `dt_shifted` in the off‑window covers the same range as `dt_us` on‑window, and its distribution is flat.
- [ ] No `t_1`/`t_2` among the model features and no NaNs. A misspelled feature name makes the trainer fail.

### Input data contract (what changes if the data format changes)

Decision 2026-11-06 (the user): the label codes, units and so on are **set by the dataset**, not by us, so they don't go in YAML. If a dataset ever comes in a different format, **one adapter function** at the input converts it to the format below ("canonical"). The rest of the code always assumes this format. Today the adapter would do nothing, so it isn't written; write it only when a second format exists. `_validate_raw_inputs` checks the canonical format.

| What the code assumes | Today | Where it is relied on | Checked? |
|---|---|---|---|
| Two files, events + truth, **row‑aligned** (row i ↔ row i) | same row count (25,043,297) | `pd.concat(axis=1)` in `_load_raw` | ✗ only the names, not the row count |
| events columns | `time, energy, x, y, z` | `_load_raw`, scans | ✓ `_validate_raw_inputs` |
| truth columns | `label, pair_id, label_name` | `_load_raw`, scans, `_rename_pair_columns` | ✓ `_validate_raw_inputs` |
| Time unit and type | ns, int64; unsorted allowed; ties possible | `PAIR_THRESHOLD_NS`, `_compute_parameters`, scans | ✗ |
| Position unit and origin | m, centred on the detector (z ∈ [−1.55, 1.55], r ≤ 1.5) | `DETECTOR_*`, `_event_distance_to_wall` | ✗ |
| Energy unit | MeV (E_d peak at 2.18) | A.12 checks, physics cuts | ✗ |
| Label codes | 1 = IBD_p, 2 = IBD_d, 3 = radio | `_rename_pair_columns` (`is_ibd`, `pair_type`) | ✗ |
| **prompt < delayed** in the codes | 1 < 2 | sorts `["time","label"]` (scans, l. 262 and 412): the tie‑break | ✗ |
| Non‑pair events | `pair_id = −1` | `is_ibd` | ✗ |
| An IBD pair shares its `pair_id` | yes | `is_ibd` | ✗ |
| Truth present | yes (training) | everything with labels; optional in A.8 (deferred) | ✓ the file must exist |

Cheap checks that could go into `_validate_raw_inputs` if ever needed: equal row counts (from the metadata, without loading), and the `time` type is int64.

---

## Parked for later

Parked 2026-11-09 (the user's call): structure and cleanup work that does not block the θ₁₃ objectives. Resume each one from its "resume at" line.

### The config step (A.11, cleanup item 2)

- State: `config.yaml` (5 groups, 16 values; it loads and matches the constants) and `src/config.py` (`REQUIRED_KEYS` group → keys, `get_repo_root`) are **not on `main`**. Keep them on a branch (e.g. `alvaro/config-wip`). Nothing reads `config.yaml` yet, so editing it changes nothing.
- Decisions already made are in the A.11 notes (Path(__file__); paths anchored to the repo root; `{key: Path}` built in the reader's module; `ValueError` for bad content; required keys written once).
- Resume at: stubs (signature + numpy docstring, body `...`) for the reader and the path builder.
- Unpark when the test data arrives: A.8 needs paths that aren't hard-coded.
- Cost of waiting: paths and constants stay duplicated (`data_loader` copies `_PROCESSED_FILES`, without `parameters`).

### PR #1/#2 follow-ups

Taken as is (the user's call). Review notes for later:
- `event_id` lands in `total_features.parquet` (`feature_cols` = everything not in `LABEL_COLS_TOTAL`). This is harmless as long as nobody trains on the total files.
- The 25M-row sort runs 3× (`_load_raw` + both scans); `importlib` isn't needed; `_LABEL_COLS` is a hand copy of `EVENTS_LABEL_COLS` (pass 2 of the name cleanup).
- Notebook: `build_pairs` is a third pairing implementation (row-by-row `iloc`), so it should load the processed files instead. It reintroduces a positional column rename and hard-coded constants, and it is committed with its outputs (366 KB).
- `event_id` is numbered after the sort in the notebook but before it in the script. The two agree today only because the raw file is already in (time, label) order (checked: 0 rows move).
- Keep and use: the time-ordered split with a guard (cell 8, for A.10); the on vs off/n_off plots (a check of B); the counts (201,596 on-time, 99,996 `is_ibd`, 1,016,812 off-time, 4 `false_ibd`) as a cross-check for A.12. Clipping negative bins in the "pure signal" plots hides fluctuations, so don't count B from them.

### Reusable joint plot (`src/plots.py`)

**Parked 2026-11-07** (the user's call). Resume at P.1. Raised by the user 2026-11-07. Generalize `plot_joint_threshold_split` (l. 447) so that the caller passes any row conditions and any column conditions, and the plot can be reused in other analyses. It is not on the θ₁₃ critical path and does not block Part A.

- [ ] **P.1 Design: the signature + a numpy docstring first.** What the code shows:
  - Internally the function already works with conditions. `_build_row_conditions` and `_build_col_conditions` (l. 188, 204) return `(mask_fn, title)` pairs. Only the public API is narrow: rows take one column + one threshold, columns take one column + bin edges.
  - Claude's recommendation: pass rows and columns as `{title: condition}` dicts, where each condition is a `DataFrame.eval` string and the dict order sets the grid order. Keep the threshold case and the bin-edge case as small public builders that return such a dict.
  - To solve: (a) `df[needed_cols]` (l. 494) is taken before the conditions run, so a condition on any other column would fail. The order project → sample → split has to change. (b) `row_col` only names the output folder (`_output_path`, l. 531). `main.py` passes `row_col="pair_id_2"` with no threshold just for that. Something like `tag` (as in `plot_distributions`) could replace it. (c) `labels` is used only in a warning (l. 256), and after the groupby sample its order doesn't have to match `unique_cats`. (d) The docstring says `sample_size` is "per series per cell", but the sample is taken once, before the split.
  - Stale along the way: `LABEL_DICT` and `plots.EVENTS_FEATURE_COLS` use the old `delta_*` names, and the default is `cat="label_name"`. `main.py` calls `plot_all_combos` on 37 columns, which makes C(37,2) = 666 PDFs.
- [ ] **P.2 Implement it**, then switch `plot_all_combos` to the new API.
- [ ] **P.3 Check:** remake one old plot (a threshold row split) with the new API and compare.

---

## Part B: Where the theory is, step by step

**Readings in `readings/`:**
- **nH:** Daya Bay, *New measurement of θ₁₃ via neutron capture on hydrogen*, PRD 93, 072011 (2016), arXiv:1603.03549. Page numbers below are **PDF pages**.
- **JUNO:** *Measurement of Neutrino Oscillation Parameters at JUNO and Indication of the Neutrino Mass Ordering*, arXiv:2609.37895 (2026). It is a letter with no numbered sections, so it is cited by PDF page, table and equation.

> **Note:** JUNO does **not** measure θ₁₃. At 52.5 km it targets θ₁₂, Δm²₂₁, Δm²₃₁ and the mass ordering, and it takes θ₁₃ from Daya Bay as an external constraint, the `χ²_pull(sin²θ13)` term on p. 8. Use it for its selection, background and fit techniques, not for the θ₁₃ logic. **The nH paper is your template.**

### Step 1: Oscillation theory
- [ ] **nH:** Sec. III, eq. (3), p. 4 (the full P_ee); Sec. VIII A–B, Figs. 22–23, p. 22–23 (what the deficit looks like).
- [ ] **JUNO:** Intro, p. 4; the Δm²₃₁ relation, p. 10.
- [ ] **Recent:** PDG 2024 review *Neutrino Masses, Mixing, and Oscillations*, Phys. Rev. D 110, 030001 (2024), reactor section. For current parameter values: **NuFIT 6.0**, JHEP 12 (2024) 216, arXiv:2410.05380.
- **Take away:** at L ~ 1–2 km, P_ee ≈ 1 − sin²2θ₁₃·sin²Δ₃₁ (the solar term is about 0.1%), and E_ν ≈ E_prompt + 0.78 MeV.

### Step 2: IBD selection (what your classifier replaces)
- [ ] **nH:** Sec. IV B and Table I, p. 5–7, especially IV B 4 (coincidence time, multiplicity) and IV B 5 (distance < 50 cm removes 98% of accidentals); Fig. 2.
- [ ] **JUNO:** Table I, p. 6. The **likelihood‑ratio cut** on p. 7, built from a signal‑like sample and an off‑window accidental sample, is the classical version of your MLP: about 40× fewer accidentals for about 1% signal loss.
- [ ] **Recent:** Gavrikov et al. (JUNO), *Interpretable machine learning approach for electron antineutrino selection in a large liquid scintillator detector*, Phys. Lett. B 860 (2025) 139141, arXiv:2406.12901. It covers the feature set (E_p, E_d, R³, cosθ, φ, ΔR, Δt), balanced training, threshold choice and SHAP interpretation.

### Step 3: Accidental background B
- [ ] **nH:** Sec. V, p. 7–9: eqs. (6)–(7), R_acc = R_s²·T_c·ε_m, and the accidental background sample (ABS) built by pairing singles. Sec. IV B 4, eq. (5), for ε_m.
- [ ] **JUNO:** the off‑window accidental sample and likelihood‑ratio cut, p. 7; Table II, p. 7.
- [ ] **Recent:** Daya Bay, *Measurement of Electron Antineutrino Oscillation Amplitude and Frequency via Neutron Capture on Hydrogen*, PRL 133, 151801 (2024), arXiv:2406.01007: synthetic random pairing and its validation.
- [ ] **Recent, focused:** J. Li, *Exact time‑correlated coincidence modeling with reset boundaries*, arXiv:2608.18728 (Aug 2026, preprint, not yet refereed): exact accidental rates and multiplicity efficiency.

### Step 4: Selection efficiency ε
- [ ] **nH:** Sec. VII, p. 14–21:
  - VII A: prompt energy.
  - **VII A 1: efficiency varies with baseline, which requires the iterative correction.** Read this one carefully: if your model uses E_p, the same effect applies to you.
  - VII B: time window, eq. (14).
  - VII C: delayed energy.
  - VII D: distance.
  - VII H: summary and Table III.
- [ ] **JUNO:** Table III and the efficiency paragraph, p. 7–8.
- [ ] **Recent:** Daya Bay 2024, arXiv:2406.01007: the combined distance–time cut "DT", efficiency from a relaxed cut minus synthetic accidentals, and the delayed‑energy efficiency (Table 2).

### Step 5: Expected signal without oscillation (flux × cross‑section × target protons)
- [ ] **nH:** Sec. III, eqs. (1), (2), (4), p. 4; Sec. VII F (target protons).
- [ ] **JUNO:** the reactor flux paragraph, p. 8 (TAO and Daya Bay near detectors constrain the unoscillated spectrum).
- [ ] **Recent:** Daya Bay, *Comprehensive Measurement of the Reactor Antineutrino Spectrum and Flux at Daya Bay*, PRL 134, 201802 (2025), arXiv:2501.00746.
- [ ] **Recent, cross‑section:** Ricciardi, Vignaroli, Vissani, *An accurate evaluation of electron (anti‑)neutrino scattering on nucleons*, JHEP 08 (2022) 212, arXiv:2206.05567.

### Step 6: The fit for sin²2θ₁₃
- [ ] **nH:** Sec. VIII A: the near/far weights α and β, eqs. (24)–(27), which give R = 0.950. Sec. VIII B: the χ² with pull terms, eq. (28), and the error budget in Table IV, p. 22–23.
- [ ] **JUNO:** the binned χ² with CNP variances and pull terms, p. 8; Feldman–Cousins intervals, p. 9.
- [ ] **Recent:** Daya Bay, *Precision Measurement of Reactor Antineutrino Oscillation at Kilometer‑Scale Baselines*, PRL 130, 161802 (2023), arXiv:2211.14988: the reference near/far spectral fit.
- [ ] **Recent, statistics:** Ji, Gu, Qian, Wei, Zhang, *Combined Neyman–Pearson chi‑square*, NIM A 961 (2020) 163677, arXiv:1903.07185.
- [ ] **Recent, far predicted from near:** "Analysis B" in arXiv:2406.01007.

### What you can skip
- nH Sec. VI (⁹Li/⁸He, fast neutrons, Am‑C): not in your simulation.
- nH Sec. IV A (calibration): your energies are already reconstructed.
- JUNO's geoneutrino, world‑reactor, mass‑ordering and Bayesian parts.

---

## Part C: Classifier → θ₁₃ (later; details once Part A is done)

- [ ] C.1 Retrain on the new features and the `is_ibd` target, with a time‑block split.
- [ ] C.2 Pick a **fixed cut on the raw score**. Don't use a threshold on calibrated probabilities: they depend on the signal fraction, which differs between datasets. Choose the cut to minimise σ(N_IBD), not with the FN = 10×FP cost rule.
- [ ] C.3 ε = (true IBDs passing the cut and `is_dc`) / (all true IBDs), from the simulation, as a function of E_p (Step 4, nH VII A 1).
- [ ] C.4 B = (off‑window pairs passing the cut and `is_dc`) / N_off.
- [ ] C.5 N_IBD = (N_selected − B) / ε, with its uncertainty.
- [ ] C.6 R and the fit for sin²2θ₁₃ (Step 6). The method depends on the open question below.

## Open questions

- [ ] **What is in `data/raw/test/` and `data/inference/`?** Both are empty now. If you get near and far samples, use the near/far ratio (nH Sec. VIII A). If you get a single baseline, you need an absolute prediction (Step 5) and the baseline. Ask the supervisor.
- [ ] Should E_p be a model feature? It helps the selection but makes ε energy‑dependent (nH VII A 1). Decide during C.1 by comparing ε(E_p) with and without it.

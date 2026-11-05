# θ₁₃ Plan: from IBD selection to sin²2θ₁₃

This is the working plan, shared across sessions. Claude explains and reviews, and the team writes all the code. Tick the boxes as you go and keep the status block up to date.

## Status / next step

| | |
|---|---|
| **Current phase** | Part A: fix the preprocessing |
| **Next action** | **A.9** (run metadata: live time and R_s, needed for the e^(R_s·k·T) correction). **A.8 is deferred** (decision 2026-11-04): for now only the training data, which has truth, is used, so the current "both files must exist" check is enough. Come back to A.8 when test data arrives (`data/raw/test/` is empty). A.1–A.7 are done and committed (latest `6cc58cc`). Don't train until A.11/A.12; the MLP hard‑codes 28 features. |
| **Blocked on** | Nothing for Part A. Part C needs to know what the test data contains (see "Open questions"). |
| **Last updated** | 2026-11-03 |

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
- [ ] **A.3 Build the physics feature set**, about 10 columns, modelled on Gavrikov et al. 2025 (Part B, step 2):
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
- [ ] **A.5 Add multiplicity columns per pair:**
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
- [ ] **A.6 Rebuild the off‑window sample:**
  - Use N_off successive windows (kT, (k+1)T] for k = 1…N_off (start with N_off = 10).
  - Store `k_window` and `dt_shifted = dt − k·T`, so the model sees the same Δt range as in the signal window.
  - Use the same features and multiplicity columns as the signal pairs.
  - Force `is_ibd = 0`, but add a flag for the about 4 real IBDs with Δt > 2 ms that land in the first off‑window.
    - Note (2026-11-02): `_project_pairs_chunked` → `_rename_pair_columns` runs on both the on‑window and the off‑window pairs. If A.2 computes `is_ibd` there, those ~4 off‑window rows come out as 1. Override that here.
- [x] **A.7 Make the ordering deterministic:** a stable sort on (time, then IBD_p before IBD_d).
  - Done 2026-11-03: both scans use `sort_values(["time","label"], kind='stable')`. A first attempt (sort by time, then a separate stable sort by label) made label the primary key and produced negative Δt; it was fixed. Tests: a tie listed as (delayed, prompt) now gives is_ibd 1; a radio at 4.0 ms before an IBD pair gives correct positive Δt; shuffled random input gives forward 0 ≤ Δt ≤ T and off‑window Δt > T everywhere.
  - 2026-11-03: the user found that the default `quicksort` is not stable and that a tie in the order (delayed, prompt) gives is_ibd = 0, so a real IBD is lost. "Stable" alone is not enough, because the order before the sort (`_load_raw` sorts by `[pair_id, time]`) doesn't guarantee the prompt first. Decision: sort by `["time", "label"]` (1 < 2 < 3) in **both** scans (lines 239 and 389). Note for A.8: real data has no `label` column, so this sort must fall back to `time` only when the truth is absent.
- [ ] **A.8 Make I/O dataset‑aware:** ⏸ **Deferred 2026-11-04** until test data exists. Only the training data is used for now. The notes below stay valid for when it resumes.
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
- [ ] **A.9 Write run metadata** for each dataset: the live time (t_max − t_min) and the singles rate R_s. Part C needs both for R_acc = R_s²·T·ε_m (nH paper, eq. 7).
  - Decisions 2026-11-04: N = **all events**, because data without truth can't separate IBD events from singles. The ~0.8% IBD share biases R_s² by about 1.6%; that's fine for the ε_m correction and the R_acc cross‑check, and B itself comes from the off‑window pairs. Compute it in `_process` before step 4 (`_find_forward_pairs`). The A.8 docstring draft was reverted with `git restore`.
  - Watch out: at that point `df_total` is sorted by `[pair_id, time]`, not by time, so use min/max, not the first and last row. Time is in ns, so T must be converted to s.
  - Update 2026-11-04: the user keeps **ns** throughout (`time` is int64 ns, consistent with `PAIR_THRESHOLD_NS`), so no conversion. R_s is then in **ns⁻¹** (≈ 2.03e‑9), which is not nHz, so the key or variable name must say `per_ns`. R_s·k·T is dimensionless either way. The user is leaning towards `.npy` (fixed order, no names); Claude pointed out that the file then carries no units either. The user's call.
- [ ] **Cleanup of hard‑coded name lists** (raised by the user 2026-11-04; do it after A.9, before or together with A.11). Principle: what matters is not hard‑coding but **duplication**, so each name should be defined once. Findings to verify: the raw files already have exactly the names `COLUMN_NAMES[:8]`, so the positional rename in `_load_raw` is a no‑op. The 4 engineered names `r, r2, phi, distance` are written in 3 places (`COLUMN_NAMES`, the exclusion tuple in `_load_raw`, the assignments). `_RAW_TO_PHYS` is mostly the identity (only time→t, energy→E, distance→D). In `RAW_FILES` only `.name` is used.
- [ ] **A.10 Add a time‑block id** (e.g. per day), so the trainer can split by time block instead of at random.
- [ ] **A.11 Plumbing:**
  - Delete the stale `src/preprocessing.py`.
  - Fix the import in `src/data_loader.py` to point at the real module.
  - Move T (2 ms), N_off, R, H/2 and the paths into `config.yaml`.
  - Make `src/trainer.py` raise an error when a feature listed in `training.yaml` is missing; it currently drops it silently around line 131.
  - In `training.yaml`, set `target_col: is_ibd` and update `features.include` to the new names.
  - **Docs are out of date** (review 2026-11-05, for teammates who are just starting): the README has no physics, no "how to run" and points to the stale `src/preprocessing.py`. `STRUCTURE_GUIDE` says `scripts/` and `notebooks/` are at the top level. `MODELS.md` lists 9 `delta_*` features while the code has 28 and `training.yaml` says 10. `EVALUATION.md` has wrong file names (`calibration.py`, `config/training.yaml`) and 4 wrong statements (PR‑AUC isn't trapezoidal; the G‑mean "chance" claim; G‑mean ≠ Youden; the prevalence factor in the cost threshold). ~~CONNECT / GIT_WORKFLOW~~ **rewritten 2026-11-05**: each person has their own CloudVeneto VM (no shared VM anymore); everyone commits to `main` (pull → commit → pull → push); `data/raw/training/` lives inside each clone. This plan file is untracked.
- [ ] **A.12 Regenerate** the processed data with `--force`, then run the checks below.

### Checks after regenerating (training set)

- [ ] `is_ibd == 1`: **99,996** pairs, i.e. 100,000 minus the 4 with Δt > 2 ms, and exactly one per `pair_id`.
- [ ] On‑window accidentals: about **101.6 k**, which is 99,945 + 796 + 854.
- [ ] Off‑window pairs **per window**: about **101.6 k ± 320**, consistent with R_s²·T·live time. About 4 `is_ibd`‑like rows in window 1.
- [ ] True pairs: mean Δt ≈ 200 µs (exponential), median ΔR ≈ 0.44 m, E_d peaks at 2.18 MeV.
- [ ] `is_dc` among true IBDs: about **99.2%**. This should match ε_m ≈ e^(−2·R_s·T) = 0.9919.
- [ ] `dt_shifted` in the off‑window covers the same range as `dt_us` on‑window, and its distribution is flat.
- [ ] No `t_1`/`t_2` among the model features and no NaNs. A misspelled feature name makes the trainer fail.

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

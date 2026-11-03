# results/earlier_workspace_tests/

Files in this folder come from an **earlier workspace of the author** (the folder `graduation project 2`,
runs of 2026-04-29). That workspace is marked do-not-use for Table 2 constants by the author, because it
predates the 2026-06-01 paper-constant correction; its two side experiments are copied here only so that the
percentages quoted in `REPORT.md` Section 5.4 can be recomputed. Nothing was re-run.

| File | What it is |
|---|---|
| `test1_summary_20260429_135108.csv` | Test 1 result: one row per (dataset, method), 39 rows, protocol A and B means and differences to the paper values embedded in the script. 100 folds. This is the full run (11.7 minutes); a shorter earlier run of the same script in that folder was not copied. |
| `test1_fold_results_20260429_135108.csv` | 3,900 saved fold records: 100 distinct fold IDs for each of 39 dataset-method pairs. The verifier checks that their means reproduce the summary to 1e-10. |
| `test2_raw_20260429_142018.csv` | Test 2 result: one row per (variation, dataset, method). 20 folds (5 splits x 4 repeats). |
| `test2_summary_20260429_142018.csv` | Test 2 per-variation summary as produced by the script. |
| `test1_uniform_protocol_test.py`, `test2_comprehensive_protocol_test.py` | The scripts, for the definitions only. They read data from a folder of the earlier workspace and are not runnable from this repository as they stand. |
| `verify_tests.py` | Recomputes the figures below from the CSVs (pandas only). |

## What the figures are

| Figure (as stated) | What it actually measures | Recomputed (as run) | Against the repository's corrected reference |
|---|---|---|---|
| Test 1: 14.9 percent and 25.1 percent | Shares of 195 metric comparisons (5 datasets WINE, GLASS, CB, SEED, BCW; 8 non-GAN methods; 39 rows x 5 metrics; 100 folds) in which protocol A (probability-based AUC/AP) or protocol B (binary-prediction AUC/AP) is **closer to the paper value**. | A 29 (14.9 %), B 49 (25.1 %), ties 117 (60.0 %). Reproduced exactly. | A 28 (14.4 %), B 50 (25.6 %), ties 117 |
| Test 1: "60.0 percent ties" | Structural: ACC, F1 and G-Mean do not depend on the protocol, so 3 of the 5 metrics always tie (3 x 39 = 117). Only AUC and AP can differ. Among those 78 comparisons B is closer in 62.8 % and A in 37.2 %. | as above | A 35.9 %, B 64.1 % |
| Test 1: mean differences 0.7213 and 0.7159 | Mean absolute difference to the paper over all 195 metric values with ACC in percentage points and the others on the 0-1 scale. Dominated by ACC. | Reproduced exactly. | not recomputed |
| Test 2: 37.9 percent (baseline), 43.6 percent (Platt or isotonic calibration) | Share of 195 metric cells (39 rows x 5) within tolerance per metric (ACC within 3.0 points, AUC, F1, AP and G-Mean within 0.03), 20 folds. | 74/195 = 37.9 %, Platt and isotonic 85/195 = 43.6 %. Reproduced exactly. | baseline 69/195 = 35.4 %, Platt 78/195 = 40.0 %, isotonic 75/195 = 38.5 %, depth_full 77/195 = 39.5 % |

## What they are not

* They are **not** "the share of cells that match the paper within tolerance" for Test 1: 14.9 and 25.1 percent are
  win shares between two protocols. In the same data, protocol A has 72 of 195 metric cells within the per-metric
  tolerance (36.9 percent) and protocol B 75 (38.5 percent) as run; 4 and 8 of 39 rows have an average error below 3
  points (the row criterion of the main tables).
* They are not comparable with 42/143 or 143/143, which count (dataset, algorithm) rows by their five-metric average error.
  A per-metric cell tolerance (Test 2) is a different criterion from a row average (main tables).
* Test 2 used 20 folds, not 100 (stated in its script and in the author's earlier README), so it is the one place where
  fewer evaluations were used; the configured main replication protocol is 100 evaluations; this does not certify completion of every main-track fold.

## Paper values used by the tests differ from Table 2

The scripts embed their own copy of the paper values. In 95 of the 195 Test 1 cells (all in WINE, GLASS and CB; SEED and BCW
agree) those embedded values differ from the repository's corrected reference (for example WINE CSRBoost accuracy 99.81 in the script and 98.86
in the current reference), so the as-run percentages were measured against an older transcription. Against the repository's corrected reference the Test 2
figures become 35.4 percent (baseline) and 40.0 percent (Platt, best), a smaller gain (4.6 points instead of 5.7) with the
same ranking at the top; Test 1 becomes 28 / 50 / 117 and, for the tolerance share, A 34.4 percent and B 32.8 percent.
None of the tested variations reproduced all evaluated entries. Other uniform protocols are not ruled out. Calibration leads the tested per-metric tolerance counts under each reference, but Platt and isotonic tie only under the original reference. Both sets of numbers are printed by `verify_tests.py`.

## Independent checks

The final thesis text (Section 5.3) and the author's report copies state the same two percentages. The files do not
contradict the recomputation but use the word "matched" for Test 1, which the data support only as "was closer to the paper".
The uniform-protocol engine of the author (`../../exact_replication/`, 100 folds, different implementation) gives, on the
same 39 rows, 76 of 195 metric cells within the per-metric tolerance (39.0 percent) and 10 of 39 rows below the 3 point
row average, which is of the same size as the Test 1 and Test 2 baselines (36.9, 38.5 and 37.9 percent).

## MD5 of the copies (first 12 characters)

d71fb8ff30de test1_summary_20260429_135108.csv, 269b818fedd1 test2_raw_20260429_142018.csv,
bc16940ec9f2 test2_summary_20260429_142018.csv, d42c04badfd1 test1_uniform_protocol_test.py,
a52035c14ffd test2_comprehensive_protocol_test.py (identical to the author's files).

## Scope of the correction

The corrected reference is the repository's current `PAPER_TABLE2`, not a fresh independent transcription of every paper entry. Test 1's tolerance counts against it are A 67/195 (34.4 percent), B 64/195 (32.8 percent). For Test 2, original versus corrected counts are baseline 74/195 versus 69/195; Platt 85/195 versus 78/195; isotonic 85/195 versus 75/195. Under the corrected row-average criterion the Test 2 baseline passes 2/39 rows, Platt and isotonic 4/39 each. These units differ from the main table's 143 rows.

Saved workspace documentation records Test 1 as 11.7 minutes and Test 2 as 1,209 seconds (20.2 minutes). These do not establish the wider thesis project's configuration count or compute-hours.

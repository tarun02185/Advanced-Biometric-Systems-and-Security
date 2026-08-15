# Assignment 1 — User Authentication using Biometric Features

Performance evaluation of a biometric verification system on `biomet_data.csv`
(100 users × 10 samples × 144 features), using **Euclidean distance** and **cosine similarity**
as the matching criteria.

---

## 1. Files

| File | Contents |
|---|---|
| `biometric_eval.ipynb` | All code. |
| `results/fig1_distributions_euclidean.png` | **A + B** — genuine & imposter distributions, Euclidean |
| `results/fig2_distributions_cosine.png` | **A + B** — genuine & imposter distributions, cosine |
| `results/fig3_far_frr_euclidean.png` | **C + D** — FAR plot, FRR plot, and their crossing (Euclidean) |
| `results/fig4_far_frr_cosine.png` | **C + D** — FAR plot, FRR plot, and their crossing (cosine) |
| `results/fig5_roc.png` | **E** — ROC for both matchers (linear and log-FAR) |
| `results/fig6_det.png` | **E (extra)** — DET curve |
| `results/fig7_summary.png` | **F** — EER and d′ across every configuration |
| `results/metrics_summary.txt` / `.csv` | **F** — all numbers in text and machine-readable form |

**To run:** open `biometric_eval.ipynb` with `biomet_data.csv` in the same folder and
*Run All*. Requires only `numpy` and `matplotlib`. 

---

## 2. Reading the data file, and how the split was verified

The file is 144 rows × 1000 whitespace-separated columns, i.e. **one feature per row and one
sample per column**. Transposing gives the usual *(sample × feature)* matrix, reshaped to
`(100 users, 10 samples, 144 features)`.

The column ordering is not documented, so it was **tested rather than assumed** (notebook §1):

| Hypothesis | Mean within-user similarity | Mean between-user similarity |
|---|---|---|
| **user-major** (`u1s1…u1s10, u2s1…`) | **0.9886** | 0.9565 |
| sample-major (`u1s1…u100s1, u1s2…`) | 0.9607 | 0.9568 |

Only the user-major reading produces the within-class/between-class gap a biometric dataset must
have. A second check confirms the enrollment/test boundary sits between sample 5 and 6:

| Mean cosine similarity inside one user | |
|---|---|
| within the enrollment session (samples 1–5) | 0.98997 |
| within the later session (samples 6–10) | 0.98905 |
| **across the two sessions** | **0.98556** |

Similarity drops across the session boundary — exactly the template-ageing effect a 6-month gap
should cause. So `data[:, :5]` is the gallery (training/enrollment) and `data[:, 5:]` is the test
set, as the assignment specifies. No test sample is used for enrollment, and no feature
statistic is fitted on test data.

---

## 3. Matching and score generation

$$d_{\text{euc}}(x,y)=\lVert x-y\rVert_2 \qquad
  s_{\cos}(x,y)=\frac{x\cdot y}{\lVert x\rVert\,\lVert y\rVert}$$

Euclidean is a **distance** (small = same user), cosine a **similarity** (large = same user), so
every function carries a `higher_is_better` flag that flips the accept rule. Both are computed as
full matrix products (`probes @ gallery.T`), so the whole 500 × 500 score matrix is one BLAS call.

Each of the 500 test probes is scored against each of the 500 enrolled templates. A comparison is
**genuine** if probe and template belong to the same user, **imposter** otherwise.

**Primary protocol — `all-pairs`** (used for every figure): one score per *(probe, template)* pair.

* genuine = 100 × 5 × 5 = **2 500** scores
* imposter = 100 × 5 × 99 × 5 = **247 500** scores

**Secondary protocol — `best-of-5`**: the five template scores of a claimed user are fused by
*minimum distance* / *maximum similarity*, giving one score per *(probe, identity claim)* — which
is how a deployed system holding five templates per user would actually decide.
Genuine = 500, imposter = 49 500. Reported in the summary table for comparison.

---

## 4. How the metrics are computed

**Decision rule.** Similarity: accept iff `score ≥ t`. Distance: accept iff `score ≤ t`.

$$\mathrm{FAR}(t)=\frac{\#\{\text{imposter scores accepted at }t\}}{\#\text{imposter}}\qquad
  \mathrm{FRR}(t)=\frac{\#\{\text{genuine scores rejected at }t\}}{\#\text{genuine}}$$

* **FAR / FRR** are evaluated **exactly at every distinct score present in the data**, using
  `np.searchsorted` on the sorted score arrays. Nothing is histogram-binned, so the curves and the
  EER are not quantised by an arbitrary bin width. (Histograms appear only in the *plots* of
  A and B, never in the arithmetic.)
* **ROC** plots GAR = 1 − FRR against FAR; AUC by the trapezoid rule.
* **EER** is the rate where FAR(t) = FRR(t), obtained by linearly interpolating the sign change
  of FAR − FRR rather than by picking the nearest sampled point.
* **Decidability index** $d' = \dfrac{|\mu_{\text{gen}} - \mu_{\text{imp}}|}{\sqrt{(\sigma^2_{\text{gen}}+\sigma^2_{\text{imp}})/2}}$ — sample variances use `ddof=1`.

---

## 5. Results (F)

Primary protocol — raw features, all-pairs:

| | **Euclidean distance** | **Cosine similarity** |
|---|---|---|
| genuine mean ± sd | 346.86 ± 173.26 | 0.98556 ± 0.01306 |
| imposter mean ± sd | 732.05 ± 312.22 | 0.95652 ± 0.01937 |
| **EER** | **15.64 %** | **11.32 %** |
| EER threshold | 475.95 | 0.97496 |
| **Decidability d′** | **1.526** | **1.758** |
| AUC | 0.9190 | 0.9473 |
| GAR @ FAR = 1 % | 57.40 % | 74.08 % |
| GAR @ FAR = 0.1 % | 43.60 % | 63.76 % |

All eight configurations (2 matchers × 2 fusion rules × raw/z-scored features):

| features | protocol | matcher | EER % | d′ | AUC | GAR@FAR=1% | GAR@FAR=0.1% |
|---|---|---|---|---|---|---|---|
| raw | all-pairs | Euclidean | 15.640 | 1.526 | 0.91903 | 57.40 % | 43.60 % |
| raw | all-pairs | Cosine | 11.320 | 1.758 | 0.94730 | 74.08 % | 63.76 % |
| raw | best-of-5 | Euclidean | 9.400 | 1.838 | 0.96259 | 74.40 % | 63.40 % |
| raw | best-of-5 | Cosine | **5.600** | 2.025 | **0.97826** | **89.80 %** | **84.40 %** |
| z-scored | all-pairs | Euclidean | 15.840 | 1.533 | 0.91850 | 57.32 % | 43.68 % |
| z-scored | all-pairs | Cosine | 11.633 | 2.425 | 0.95078 | 52.28 % | 28.90 % |
| z-scored | best-of-5 | Euclidean | 9.200 | 1.858 | 0.96247 | 73.80 % | 63.00 % |
| z-scored | best-of-5 | Cosine | 7.566 | **2.650** | 0.97350 | 65.40 % | 41.20 % |

*(The z-scored rows standardise each feature using enrollment statistics only. They are not
required by the assignment — they are included because they explain why the two matchers differ;
see §6.)*

---

## 6. Which matching criterion performs best?

**Cosine similarity**, on every measure computed here: lower EER (11.32 % vs 15.64 %), higher d′
(1.76 vs 1.53), higher AUC, and — most importantly — a ROC that **dominates** the Euclidean ROC
across the entire FAR range, so the advantage is not an artefact of one operating point. At a
realistic FAR of 0.1 % cosine accepts 63.8 % of genuine attempts where Euclidean accepts only
43.6 %; that is a 20-point difference in usability at equal security. The ranking is unchanged
under the `best-of-5` fusion (5.60 % vs 9.40 % EER).

**Why it wins.** The 144 features are unnormalised magnitudes spanning roughly 16 to 682, so
Euclidean distance is dominated by the *overall scale* of the feature vector — and overall scale
is exactly what drifts between the enrollment session and the session six months later
(illumination, sensor gain, image scale). Two samples of the same user can differ mostly in
magnitude while keeping the same *pattern* across features. Cosine similarity divides that
magnitude out and compares only the direction of the vector, discarding the session-dependent
nuisance factor while keeping the identity-bearing information. That is also visible in the raw
statistics: the genuine and imposter Euclidean distributions have standard deviations of 173 and
312 — huge spread caused by scale variation — whereas the cosine distributions are tight.

**A caution about d′.** Standardising the features raises cosine's d′ from 1.76 to 2.43, yet its
low-FAR performance gets *worse* (GAR @ FAR = 1 % falls from 74 % to 52 %). d′ is a two-moment
summary that only ranks systems faithfully when the scores are roughly Gaussian, and these score
distributions are visibly skewed with heavy tails (see figures 1 and 2). **EER, the ROC and fixed
operating points are the trustworthy comparison; d′ is reported because the assignment asks for
it, not because it should settle the question.** Fortunately all of them agree on the main
verdict: cosine beats Euclidean.

**Limits of these estimates.** With 2 500 genuine scores the standard error on an 11 % EER is
about ±0.6 %, so the ~4-point gap between the two matchers is comfortably real, while the
difference between the raw and z-scored variants of the *same* matcher sometimes is not. FAR
comes from 247 500 imposter scores, so the ROC is well determined down to about FAR = 10⁻⁴; below
that it rests on a handful of scores and should not be over-read. Also note that the 2 500 genuine
scores are not independent — they come from only 500 probes — so the effective sample size, and
hence the true uncertainty, is somewhat larger than the naive figure.

---


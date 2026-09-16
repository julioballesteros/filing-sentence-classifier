# Validation error analysis

The initial MLP and selected TF-IDF reference achieve similar aggregate scores but make different mistakes on **83 of 519 validation sentences** where exactly one model is correct. They share **92 errors**. A qualitative review of 30 MLP errors exposes difficulties separating current statements from forecasts, and company-specific outlooks from generic language. Several published labels also require context or clearer annotation conventions.

The evidence supports testing regularization first, alongside the previously observed overfitting. Vocabulary coverage and truncation warrant monitoring, but this review does not establish them as the main causes of error.

## Scope and sampling

The analysis compares the restored epoch-8 predictions from `mean-pool-mlp-v1` with `tfidf-bigram-c10-v1`, using the same frozen validation IDs and published labels. Prediction, encoder, and history checksums were verified, and both models' metrics were recomputed with the shared evaluator and matched their saved reports. Training text was used only to count whether unknown tokens had appeared during training. No fitting, label changes, or test access occurred; the configuration budget remains five of ten.

Before displaying sentences, the review sample was fixed at **five errors from each of the six gold-to-MLP prediction directions**. Within each direction, examples were sorted by the hexadecimal SHA-256 digest of UTF-8 `validation-errors-v1|2026|<sample_id>`, taking the first five. Selection did not use sentence text, TF-IDF predictions, length, or unknown-token rate. This covers every error direction but deliberately overrepresents smaller groups; qualitative counts are not population estimates.

The [machine-readable evidence](validation-errors.json) records the sampling rule, all 30 sample IDs, source-row positions, annotations, diagnostics, and input hashes. Review judgments were produced by one assistant using the [documented class definitions](../data/README.md#labels); they are not independent label adjudications. Source sentences and the working analysis are retained locally under `artifacts/analysis/validation-errors-v1/`. The observations below paraphrase their content.

## Full-partition comparison

| Paired outcome | Sentences |
| --- | ---: |
| Both correct | 344 |
| Only TF-IDF correct | 45 |
| Only MLP correct | 38 |
| Both wrong | 92 |

The MLP makes **137 errors** and TF-IDF makes **130**. Of the shared errors, 85 have the same wrong prediction and seven have different wrong predictions. Agreement between models does not establish that an annotation is right or wrong.

For the MLP, **100 errors cross the FLS/non-FLS boundary**, while **37 confuse specific and non-specific FLS**. Compared with TF-IDF, it recovers ten more specific-FLS sentences (50 versus 40), but predicts specific FLS for 20 non-FLS sentences versus only two for TF-IDF. The [reference comparison](mean-pool-mlp-v1/README.md#validation-results) records the resulting precision/recall tradeoff.

## Findings from the review

- **Future vocabulary is insufficient to identify a forecast.** Historical actions, current estimates, and accounting definitions contain future-oriented terms (R11, R14, R16–R18). Conversely, both models miss explicit company goals and intentions (R01–R03).
- **Specificity depends on what is being predicted.** First-person wording and numeric detail can accompany generic market conditions or rules (R21, R25). Company forecasts can also use conditional or risk language (R08–R10). These cues do not determine the class individually.
- **Clause structure and negation matter.** R19 denies that a risk has been reported; the negation survives encoding. R13 and R15 combine past facts with explicit future commitments. Mean pooling discards token order, making loss of local structure a plausible limitation; this review does not attribute individual predictions to particular tokens or prove that another architecture would fix them.
- **Some distinctions require annotation context.** Ten reviewed cases are flagged for context or label-policy review, including explicit future clauses labeled non-FLS and current policies labeled generic FLS. These are triage judgments within the selected sample, not ten verified labeling mistakes. All original labels remain in the metrics.
- **Residual extraction noise is visible in one case.** R03 retains a page-number fragment and an unfinished quotation. It also has complete vocabulary coverage and no truncation. Its contribution to the prediction has not been tested, so it does not justify a new cleaning rule by itself.

### Length and vocabulary diagnostics

These figures use all 519 validation sentences, rather than the selected review sample. UNK rates count unknown occurrences among retained MLP tokens; TF-IDF uses its own representation.

| MLP outcome | Sentences | Median original tokens | Retained UNK rate | Truncated sentences |
| --- | ---: | ---: | ---: | ---: |
| Correct | 382 | 29 | 5.90% | 1 |
| Incorrect | 137 | 32 | 6.40% | 2 |

The aggregate differences are modest and depend on class composition. For specific FLS, incorrectly classified sentences actually have lower UNK occurrence rates than correct ones (7.85% versus 8.74%). **31 MLP errors contain no UNK tokens**, including explicit forecasts and accounting statements. Unknown words can remove useful meaning in individual cases, such as R05 and R30, without explaining every failure.

Of **1,047 retained UNK occurrences**, **690 (65.9%) never appeared in training** and **357 (34.1%) were training singletons excluded by `min_frequency=2`**. Retaining singletons could recover the latter occurrences, but would add embeddings learned from only one occurrence and would not cover the unseen words. These counts measure coverage, not the accuracy gain such a change would produce.

Only three validation sentences are truncated; two are MLP errors. R18 is the one included in the review sample. Thus, truncation directly affects at most two of the 137 observed errors. The length-bin error rates are not monotonic, and the longest bin contains only 24 examples. This is weak support for spending the next experiment on a larger sequence limit.

## Reviewed cases

Class IDs: **0 = specific FLS**, **1 = non-FLS**, **2 = non-specific FLS**. Every row is an MLP error against the published label. `Source row` is the zero-based position in the pinned published training split. A dagger (†) marks a context/annotation-policy review flag. Full diagnostics and longer notes are in the JSON evidence.

| Review / source row | Gold → MLP | TF-IDF | Observation |
| --- | --- | --- | --- |
| R01 / 1620 | 0 → 1 | 1 | A concrete company income/margin goal is missed by both models, with almost complete vocabulary coverage. |
| R02 / 591 | 0 → 1 | 1 | An explicit plan to expand vendor use is missed; the planning cue is preserved. |
| R03 / 1350 | 0 → 1 | 1 | Expected recovery of company benefit costs is missed despite zero UNK tokens. Extraction noise remains visible. |
| R04 / 949 | 0 → 1 | 1 | Optimism about the brands' future under another owner is treated as non-FLS; four words are unknown. |
| R05 / 1012 † | 0 → 1 | 1 | A regional crop projection has eight unknown tokens. Its future meaning is clear; its company-versus-market scope needs context. |
| R06 / 1758 † | 0 → 2 | 2 | Possible dividend-policy changes combine a company subject with reusable boilerplate. |
| R07 / 516 † | 0 → 2 | 2 | Uncertain company funding needs are expressed in generic cautionary language. |
| R08 / 1576 | 0 → 2 | 2 | Planned use of excess cash names a corporate group and year, yet both models choose generic FLS. |
| R09 / 1857 | 0 → 2 | 2 | A forecast of company and segment margin pressure is read as generic risk language. |
| R10 / 2377 | 0 → 2 | 2 | A company's future revenue expectation remains specific even though it depends on renewals. |
| R11 / 782 | 1 → 0 | 1 | A completed inventory build has a later sales purpose. The MLP predicts a forecast; TF-IDF follows the historical label. |
| R12 / 226 | 1 → 0 | 1 | Historical quarterly averages are treated as a company outlook by the MLP. |
| R13 / 117 † | 1 → 0 | 1 | A past balance is followed by an explicit next-year portfolio expectation, making the non-FLS label debatable. |
| R14 / 472 | 1 → 0 | 1 | Future receipts occur within an explanation of an existing hedging mechanism. |
| R15 / 569 † | 1 → 0 | 0 | A past contract agreement includes a future staffing commitment. Both models choose specific FLS. |
| R16 / 1559 | 1 → 2 | 2 | A current tax assessment contains uncertainty and predictive language, with zero UNK tokens. |
| R17 / 2039 | 1 → 2 | 2 | A present explanation of price effects refers to future reserves; both models choose generic FLS. |
| R18 / 283 | 1 → 2 | 1 | An earnings definition loses 100 of 228 tokens. The retained opening still identifies a definition; causation is unproven. |
| R19 / 1374 | 1 → 2 | 2 | No bank has reported possible nonperformance. The preserved negation changes the scope of the risk statement. |
| R20 / 248 † | 1 → 2 | 2 | Current inability to estimate a future regulatory impact overlaps with generic risk-disclaimer language. |
| R21 / 934 | 2 → 0 | 2 | A numeric statutory-tax forecast concerns a general rule, illustrating that numbers do not imply specificity. |
| R22 / 1207 † | 2 → 0 | 2 | A stated future company interest expense is labeled generic; the accounting context may determine the distinction. |
| R23 / 2471 | 2 → 0 | 0 | A broad capital-allocation intention is mistaken for a specific forecast by both models. |
| R24 / 608 | 2 → 0 | 0 | Continued industry regulation is described using a company subject; both models choose specific FLS. |
| R25 / 1104 | 2 → 0 | 2 | A first-person expectation concerns general market liquidity. The key liquidity descriptor is unknown to the MLP. |
| R26 / 2182 | 2 → 1 | 1 | Possible future use of derivatives is read as a description of current practice by both models. |
| R27 / 2373 † | 2 → 1 | 2 | A current capitalization policy includes conditions about future benefits; the temporal boundary needs review. |
| R28 / 1600 † | 2 → 1 | 1 | An apparently current revenue-recognition policy is labeled generic FLS; both models choose non-FLS. |
| R29 / 910 † | 2 → 1 | 1 | Forecasts are inputs to a current assessment, making the generic-FLS label questionable without context. |
| R30 / 1168 | 2 → 1 | 1 | A prospective currency downside is missed. The MLP loses several words describing the adverse mechanism to UNK. |

TF-IDF matches the published label on nine of these 30 examples. Its successes and failures here describe this balanced error sample, not its overall performance.

## Implications for the next experiments

1. **Prioritize regularization.** Compare dropout and weight decay through separate, declared changes to the initial reference. The rising validation loss alongside falling training loss provides stronger evidence of overfitting than this review provides for a particular vocabulary or length change. Monitor specific-FLS recall together with false positives and macro-F1.
2. **Treat vocabulary expansion as a bounded secondary hypothesis.** A train-only singleton-retaining vocabulary would test whether recovering excluded terms helps enough to offset its extra capacity. Most current UNK occurrences would remain unseen, so improved coverage alone would not establish success.
3. **Use semantic cases to interpret results.** Keep the reviewed IDs fixed when comparing later runs. Improvements on these exposed examples are development observations, not new evaluation evidence. Persistent negation and clause-scope problems could motivate a future order-sensitive representation after the current MLP study.

The integrity and metric checks passed. This review did not identify a reproducible implementation defect requiring a pipeline change. It establishes hypotheses and annotation questions; concrete configurations and their selection rule remain the next step.

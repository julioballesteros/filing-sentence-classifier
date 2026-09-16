# MLP selection study

**Status: all five planned runs completed and verified; formal configuration selection is pending.** With seed 17, combined dropout and weight decay achieves the highest validation macro-F1, **0.7024**, compared with **0.6919** for the initial MLP and **0.6924** for selected TF-IDF. These are development results from one partition and one seed.

The immutable [registration](plan.json) predates the new runs. [Execution evidence](execution.json) records the commands, order, timestamps, exit codes, and manifest hashes. [Comparison data](comparison.json) contains unrounded scores, full per-class metrics and confusion matrices, numerical learning curves, preprocessing statistics, artifact identities, MLflow run IDs, and verification results.

## Validation results

Each row uses predictions from its restored best macro-F1 checkpoint. Changes are relative to the original recipe, not cumulative changes to the preceding row. All neural runs use the same 2,074 training and 519 validation sentences.

| Candidate | Macro-F1 | Δ from reference | Accuracy | Best / completed epochs | Parameters | Training seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Reference](../../configs/experiments/mean-pool-mlp-v1.toml) | 0.691861 | — | 0.7360 | 8 / 13 | 388,227 | 2.186 |
| [Dropout 0.3](../../configs/experiments/mean-pool-mlp-dropout-v1.toml) | 0.692259 | +0.000398 | 0.7360 | 8 / 13 | 388,227 | 2.318 |
| [Weight decay 0.1](../../configs/experiments/mean-pool-mlp-weight-decay-v1.toml) | 0.691641 | −0.000220 | 0.7360 | 8 / 13 | 388,227 | 2.134 |
| [Dropout + weight decay](../../configs/experiments/mean-pool-mlp-regularized-v1.toml) | **0.702429** | **+0.010568** | **0.7437** | 8 / 13 | 388,227 | 2.138 |
| [64-dimensional embeddings](../../configs/experiments/mean-pool-mlp-small-v1.toml) | 0.688045 | −0.003816 | 0.7380 | 8 / 13 | 194,243 | 1.782 |
| [Expanded vocabulary](../../configs/experiments/mean-pool-mlp-vocab-min1-v1.toml) | 0.678778 | −0.013082 | 0.7360 | 5 / 10 | 720,515 | 2.123 |

Training durations include synchronous per-epoch MLflow logging. The reference time comes from its earlier tracked run on the same recorded environment; these single observations are not a controlled speed benchmark. Checkpoint sizes are 1,556,057 bytes for the reference and regularization variants, 780,121 for smaller embeddings, and 2,885,209 for expanded vocabulary.

## What changed

**Regularization:** the reference, dropout, decay, and combination form the registered 2 × 2 comparison. Dropout alone changes macro-F1 by only +0.0004; decay alone changes one prediction from one wrong class to another. Combining them improves macro-F1 by +0.0106 over the reference, +0.0102 over dropout, and +0.0108 over decay. It corrects seven reference errors and loses three previously correct predictions, leaving 133 errors instead of 137. The combination helps at these strengths and this seed; this does not establish a reliable interaction across runs.

**Capacity:** reducing embedding dimension from 128 to 64 roughly halves parameters, but lowers macro-F1 by 0.0038. Accuracy slightly increases because more correct non-specific-FLS predictions offset lower specific-FLS recall. Changing tensor dimensions also changes initialization and the optimization trajectory even with the same seed.

**Vocabulary:** retaining training singletons expands the vocabulary from 2,967 to 5,563 IDs, including PAD/UNK. Validation UNK occurrences after truncation fall from **1,047 to 690 of 17,324 tokens**, or **6.04% to 3.98%**. Nevertheless, macro-F1 falls by 0.0131 and specific-FLS recall drops to 0.4419. Coverage and model capacity change together, so this does not isolate the effect of rare-word coverage. The 690 remaining occurrences were unseen in train.

| Candidate | Specific FLS precision | Specific FLS recall | Specific FLS F1 | Not-FLS F1 | Non-specific FLS F1 | Not-FLS → specific FLS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference | 0.5747 | 0.5814 | 0.5780 | 0.8182 | 0.6794 | 20 |
| Dropout | 0.5556 | 0.5814 | 0.5682 | 0.8170 | 0.6916 | 22 |
| Weight decay | 0.5747 | 0.5814 | 0.5780 | 0.8197 | 0.6772 | 20 |
| Combined regularization | 0.5930 | 0.5930 | 0.5930 | 0.8205 | 0.6938 | 19 |
| Smaller embeddings | 0.5402 | 0.5465 | 0.5434 | 0.8227 | 0.6981 | 19 |
| Expanded vocabulary | 0.6667 | 0.4419 | 0.5315 | 0.8287 | 0.6761 | 7 |
| Selected TF-IDF | 0.7143 | 0.4651 | 0.5634 | 0.8410 | 0.6727 | 2 |

Class supports are 86 specific FLS, 273 not-FLS, and 160 non-specific FLS. The final column counts only false specific-FLS predictions on true not-FLS sentences, not all specific-FLS false positives. Full confusion matrices and every class's precision and recall are in the comparison JSON.

Combined regularization exceeds TF-IDF's macro-F1 by **0.0101**, with higher specific-FLS recall (51/86 versus 40/86), but lower accuracy (**0.7437 versus 0.7495**) and many more not-FLS sentences predicted as specific FLS (19 versus 2). It has no established overall generalization advantage from this validation comparison alone.

## Learning curves and reviewed cases

![Training and validation cross-entropy for the six fixed neural configurations](learning-curves.png)

All runs show training loss continuing to fall while validation loss eventually rises. Regularization moderates this divergence without removing it: at epoch 13, validation loss is **0.7226** for the combination versus **0.8137** for the reference. Their selected epoch-8 losses are **0.6193** and **0.6356**. Training loss includes dropout when enabled; validation is evaluated with dropout disabled. Dashed lines identify the best macro-F1 epoch, which need not minimize loss. Every run stops after five epochs without a strict macro-F1 improvement.

The comparison also revisits exactly the same 30 IDs from the [earlier error review](../validation-errors.md), keeping the published labels unchanged:

| Candidate | Reference errors corrected, all 519 rows | Reference correct predictions lost | Corrected among the 30 reviewed errors |
| --- | ---: | ---: | ---: |
| Dropout | 5 | 5 | 2 |
| Weight decay | 0 | 0 | 0 |
| Combined regularization | 7 | 3 | 2 |
| Smaller embeddings | 48 | 47 | 10 |
| Expanded vocabulary | 29 | 29 | 4 |

Dropout and combined regularization both correct **R11**, a completed inventory action with a future purpose, and **R27**, a standing accounting policy with prospective conditions. R27 remains flagged as context-sensitive in the earlier review; agreement with its label does not resolve that ambiguity. Neither run corrects any of the ten reviewed specific-FLS false negatives.

The smaller model fixes more of these reviewed errors while scoring lower on the complete partition. This illustrates why a sample chosen entirely from reference errors cannot measure net improvement: it omits the reference's correct predictions that a new model may lose. These previously exposed, direction-balanced cases remain qualitative development evidence.

## Execution and verification

The runs followed the declared order: **dropout → weight decay → combined → smaller embeddings → expanded vocabulary**, with no retries or failed attempts. They used the existing training CLI and local MLflow experiment `filing-sentence-classifier`. The plan's data, config, vocabulary, encoder, source, lockfile, and environment identities were verified before execution.

Fixed settings were seed **17**, CPU float32, one computation thread, zero workers, batch size 32, hidden dimension 64, learning rate `0.001`, at most 30 epochs, patience 5, and `min_delta=0.0`. Tokenization, prefix length 128, label mapping, and unweighted cross-entropy were unchanged. Only the expanded-vocabulary run used `artifacts/preprocessing/vocabulary-min1-v1/`; the others used `vocabulary-v1/`. Its training TOML intentionally matches the reference's effective settings, so its vocabulary and encoder hashes are essential to its identity.

The reference reuses `mean-pool-mlp-v1-tracked`, as registered. Its historical manifest retains the uncommitted-source status from that earlier run; the saved source bytes match the committed training code pinned in the plan. All five new runs used clean training source at execution commit `029c484b360df0a2f2383c43d1c77ea10fc25228`, with the same package hashes and environment.

Verification passed for all six neural runs: every manifest file hash and size, registered inputs and recipe, earliest best-score epoch, and early-stopping history. A separate process restored each encoder and checkpoint without fitting and reproduced its validation predictions, metrics, and loss exactly. The shared evaluator also recomputed saved-prediction metrics for all neural runs and the TF-IDF reference. All six MLflow runs are `FINISHED`, with matching parameters, epoch metrics, summaries, and byte-identical artifact mirrors.

Complete checkpoints, encoders, predictions, source snapshots, and histories remain under `artifacts/runs/<run_id>/` and in local MLflow, outside Git. The comparison JSON publishes aggregate evidence and their hashes; execution commands use paths relative to the repository root. Reproduction requires the frozen inputs and environment and a new output directory.

## Selection rule and remaining work

The registered rule maximizes **unrounded validation macro-F1** among the six eligible candidates. Exact ties prefer, in order: smaller embeddings, reference, weight decay, dropout, combined regularization, expanded vocabulary. Secondary diagnostics do not override that rule. Applying it and recording the selected configuration is the next step; this report records observed results without freezing a delivery model.

All **ten** initial configuration slots have now been evaluated: four TF-IDF and six MLP recipes. Later repetitions of the chosen neural configuration use seeds **17, 29, and 43**, reusing the verified seed-17 run when its identity matches. The predeclared neural delivery seed is 17. Test remains reserved, and there is no refit on train plus validation.

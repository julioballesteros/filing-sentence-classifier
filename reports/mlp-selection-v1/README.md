# MLP selection study

**Status: planned.** Five new configurations are registered in [plan.json](plan.json); their training runs have not been executed. The study builds on the [initial neural reference](../mean-pool-mlp-v1/README.md) and [validation error review](../validation-errors.md). Its hypotheses therefore use previously observed development evidence.

## Candidates and hypotheses

Every change below is relative to the initial reference, including the small-model and vocabulary probes. They are not successive changes to whichever candidate performs best.

| Candidate | Change from reference | Parameters | Question |
| --- | --- | ---: | --- |
| [Reference](../../configs/experiments/mean-pool-mlp-v1.toml) | Existing unregularized model | 388,227 | Does any candidate improve on the initial result? |
| [Dropout](../../configs/experiments/mean-pool-mlp-dropout-v1.toml) | Hidden-layer dropout `0.3` | 388,227 | Does dropout improve validation performance under the observed overfitting? |
| [Weight decay](../../configs/experiments/mean-pool-mlp-weight-decay-v1.toml) | AdamW `weight_decay=0.1` | 388,227 | Does decay of all trainable parameters, including embeddings, help? |
| [Combined regularization](../../configs/experiments/mean-pool-mlp-regularized-v1.toml) | Both settings above | 388,227 | Do the two regularizers complement each other, or constrain fitting too much? |
| [Smaller embeddings](../../configs/experiments/mean-pool-mlp-small-v1.toml) | `embedding_dim=64` | 194,243 | Can reduced capacity retain useful distinctions with less overfitting? |
| [Expanded vocabulary](../../configs/experiments/mean-pool-mlp-vocab-min1-v1.toml) | Training vocabulary `min_frequency=1` | 720,515 | Does retaining rare words compensate for the additional, sparsely trained embeddings? |

The reference and three regularization candidates form a **2 × 2 comparison**: dropout in `{0.0, 0.3}` and weight decay in `{0.0, 0.1}`. Compare each single regularizer with the reference, and the combination with both single-regularizer runs as well as the reference. The strengths are fixed probes, not previously optimized settings. Weight decay `0.1` provides a substantive decay probe at the fixed learning rate of `0.001`; the resulting validation scores will determine whether it helps.

The vocabulary probe increases the number of IDs from **2,967 to 5,563**, including PAD/UNK. The earlier review found that only 357 of 1,047 validation UNK occurrences were excluded training singletons; the other 690 were unseen in train. Expanded coverage therefore has a limited scope, and its increased model capacity is part of this experiment rather than an independently controlled factor.

## Fixed inputs and execution

All candidates use the saved **2,074 train / 519 validation** partition, seed **17**, CPU float32, one computation thread, zero workers, batch size 32, hidden dimension 64, learning rate `0.001`, maximum 30 epochs, patience 5, and `min_delta=0.0`. Tokenization, prefix length 128, label mapping, and unweighted cross-entropy follow the existing implementation. Each run selects its highest unrounded validation macro-F1 checkpoint, keeping the earliest epoch on an exact tie.

The plan pins configuration bytes, data hashes, vocabulary manifests, expected encoder hashes, package source hashes, the training-code commit, lockfile, and environment. The new `artifacts/preprocessing/vocabulary-min1-v1/` artifact was built only from train during registration; no model was trained to define the study.

The existing `train` CLI receives configuration and vocabulary paths separately. **The expanded-vocabulary candidate must use `vocabulary-min1-v1`; all other candidates use `vocabulary-v1`.** Its TOML intentionally has the same effective model/training settings as the reference. The complete experiment identity combines that TOML with the vocabulary and encoding settings recorded in the plan.

The baseline configuration will reuse `mean-pool-mlp-v1-tracked`, which has the same source bytes, lockfile, training settings, encoder, data, and recorded environment as the current study setup. It reproduces the original reference's validation macro-F1 **0.6918605265426075**. Its historical manifest remains unchanged, including its original uncommitted-source status; those saved sources now match the committed training code pinned by this plan.

The new execution order is **dropout → weight decay → combined → smaller embeddings → expanded vocabulary**. Runs use the existing local MLflow experiment `filing-sentence-classifier`, with new output directories identified in the plan. The order does not authorize adapting later settings after seeing earlier scores. The plan documents inputs and rules; it does not add a new execution command.

## Selection rule

Rank the six eligible candidates, including the reference, by **unrounded validation macro-F1 from the restored checkpoint's saved predictions**, recomputed with the shared evaluator. On an exact score tie, use this fixed preference order:

1. Smaller embeddings.
2. Initial reference.
3. Weight decay only.
4. Dropout only.
5. Combined regularization.
6. Expanded vocabulary.

This prefers fewer parameters, then the simpler reference recipe, one regularizer, and two regularizers. Weight decay precedes dropout as a declared preference between otherwise comparable single-regularizer recipes. Accuracy, loss, per-class metrics, and timing do not break ties or override the primary metric.

Report those secondary diagnostics to interpret the result, particularly specific-FLS recall and false positives. Learning curves and the same 30 reviewed IDs help explain observed changes. Training durations include synchronous epoch tracking and are comparable only within the recorded environment; parameter counts measure model size, not inference speed. A small validation gain from one seed is not evidence of a reliable generalization advantage.

## Budget and completion

Five of the ten configuration slots have already been evaluated: four TF-IDF recipes and the initial MLP. This plan reserves the remaining **five**, bringing the total to ten when they have all been attempted. Reusing the reference and later seed repetitions do not consume additional slots.

Record failures and stop for diagnosis rather than replacing candidates or changing hyperparameters silently. Infrastructure retries retain the same recipe in a new run directory. Changes to code or methodology require a versioned amendment identifying affected results. Once new training starts, preserve this registration and record execution and selection separately.

Executing the fixed candidates is the next step; applying the selection rule follows it. The chosen neural configuration will later be checked with seeds **17, 29, and 43** on the same partitions. The predeclared neural delivery seed remains **17**. Test remains reserved, and this study does not refit on train plus validation.

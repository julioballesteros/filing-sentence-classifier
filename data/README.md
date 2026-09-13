# Forward-Looking Statements Dataset

This project uses [FinanceMTEB/FLS](https://huggingface.co/datasets/FinanceMTEB/FLS) for sentence-level classification of forward-looking statements in English financial reports.

**Review status:** Source documentation, the [training audit](../notebooks/01_data_exploration.ipynb), cleaning, and development splitting are complete. The published test has been accessed only for automated text checks; its examples and labels have not been explored.

## Published dataset

The selected source revision is `39b6719f1d7197df4498fea9fce20d4ad782a083`, also pinned by the [FinMTEB task implementation](https://github.com/yixuantt/FinMTEB/blob/main/finance_mteb/tasks/Classification/eng/FLSClassification.py).

| Published split | Sentences |
| --- | ---: |
| Train | 2,600 |
| Test | 1,000 |
| **Total** | **3,600** |

Files are published as Parquet under the `default` configuration. Each row contains `text` (string), `label` (integer), and `label_text` (string). These figures and fields come from the [publisher's metadata](https://huggingface.co/api/datasets/FinanceMTEB/FLS). Both row counts and the training schema have also been verified locally; test labels have not been loaded.

## Labels

| ID | Published label | Meaning |
| --- | --- | --- |
| 0 | `specific fls` | A statement about the future of the particular company. |
| 1 | `not-fls` | A statement that is not about the future. |
| 2 | `non-specific fls` | A future-oriented statement that could apply to any company, such as generic cautionary language or risk disclosure. |

The IDs follow the [published dataset](https://huggingface.co/datasets/FinanceMTEB/FLS); the definitions follow the [authors' FLS description](https://www.allenhuang.org/uploads/2/6/5/5/26555246/fls_description.pdf). Specificity does not require numbers or dates, and conditional wording alone does not determine the class.

## Provenance and sampling

The [FinMTEB paper](https://aclanthology.org/2025.emnlp-main.179.pdf) attributes FLS to the InvestLM work. The [authors' FLS description](https://www.allenhuang.org/uploads/2/6/5/5/26555246/fls_description.pdf) reports 3,600 manually labeled sentences from the Management Discussion and Analysis (MD&A) sections of Russell 3000 companies' 10-K filings, covering 1994–2019.

Sampling deliberately selected 75% of sentences with a forward-looking keyword and 25% without one. Consequently, the sample should not be assumed to reflect the natural distribution of sentences in complete filings. Row-level correspondence between the described corpus and the FinanceMTEB release has not been verified.

## Limitations and unresolved questions

- **Version history:** The authors' description uses a 360-sentence test set, while FinanceMTEB publishes 1,000 test sentences. The [FinBERT-FLS model card](https://huggingface.co/yiyanghkust/finbert-fls) also mentions 3,500 annotated sentences. The reviewed sources do not explain these differences or how the current split was constructed.
- **Annotation process:** Class definitions are available, but the reviewed documentation does not specify the number of annotators, inter-annotator agreement, or how disagreements were resolved.
- **Evaluation scope:** The published schema contains no company, document, or date identifiers. It does not support verifying separation by company or period. Development splitting removes exact text overlap after the declared normalization. Near duplicates and paraphrases can still cross partitions.
- **Dataset license:** The [repository metadata and file listing](https://huggingface.co/api/datasets/FinanceMTEB/FLS) contain neither a declared dataset license nor a license file. Conditions for use and redistribution remain unconfirmed.

## Data storage and evaluation policy

Only this description is versioned in `data/`; downloaded and generated datasets are excluded from Git. Source files are treated as immutable, with revisions, file hashes, and split manifests providing provenance. The published test split is reserved for final evaluation.

## Download

From the repository root:

```bash
uv sync --locked --extra data
uv run --locked --extra data filing-sentence-classifier download-data
```

The command saves the published README and both original Parquet files under `data/raw/<revision>/`, retaining their source paths. `manifest.json` records the repository, revision, file sizes, and SHA-256 checksums. The pinned `FLS_SOURCE` specification lives in [data/fls.py](../src/filing_sentence_classifier/data/fls.py). It is an immutable `DatasetSource` passed by the CLI to `download_dataset`; the downloader does not import the FLS definition. Parquet hashes come from the [source file metadata](https://huggingface.co/api/datasets/FinanceMTEB/FLS/revision/39b6719f1d7197df4498fea9fce20d4ad782a083?blobs=true).

Downloads are staged and verified before the snapshot becomes available. Repeating the command verifies the existing snapshot without network access or overwriting files. If verification fails, move the snapshot aside or use `--output-dir PATH` to select another storage root. Relative paths are resolved from the current working directory.

The download step verifies file integrity only. It does not parse sentences, validate row counts, create development partitions, or inspect test labels.

## Training audit

[01_data_exploration.ipynb](../notebooks/01_data_exploration.ipynb) records the source revision, checksum, environment versions, and sampling seed. Install its dependencies with `uv sync --locked --extra data --group notebooks`, select the `.venv` kernel, and run all cells. Reusable checks live in [data/audit.py](../src/filing_sentence_classifier/data/audit.py); they inspect records without modifying them.

The 2,600 training rows pass structural and label-mapping checks. Eight exact duplicate groups contain nine repeated occurrences, including one three-row group with conflicting labels. There are 135 rows with suspicious Unicode characters. The notebook documents these findings, class balance, length distributions, similar-text candidates, and annotation review examples. The cleaning and splitting policies below follow this audit.

## Cleaning policy (v1)

The fixed, versioned rules live in [data/cleaning.py](../src/filing_sentence_classifier/data/cleaning.py). They use only the published training split and apply in this order:

1. Exclude records with invalid fields or inconsistent label mappings, recording each reason without coercing values.
2. Repair the six observed C1 characters using their Windows-1252 punctuation equivalents: `U+0092 → U+2019`, `U+0093 → U+201C`, `U+0094 → U+201D`, `U+0095 → U+2022`, `U+0096 → U+2013`, and `U+0097 → U+2014`. Training examples place these characters in possessives, quotations, bullets, and dashes. This is a targeted interpretation of the observed corruption, not proof of the original encoding. Other C1 characters and `U+FFFD` cause exclusion for review, before whitespace normalization can conceal them.
3. Normalize Unicode to NFC, trim surrounding whitespace, and collapse whitespace runs to a single space. Preserve case, punctuation, numbers, negation, and word order. No tokenization, length cutoff, or label correction is applied.
4. Group by the cleaned text. Quarantine **every member** of a group with conflicting labels. Retain same-label duplicates with a shared `group_id`; future development partitions must keep each group together. Similar-text candidates are not automatically merged. Qualitatively ambiguous annotations remain unchanged pending stronger evidence.

From the repository root:

```bash
uv sync --locked --extra data
uv run --locked --extra data filing-sentence-classifier prepare-data
```

The command runs offline against the downloaded training file, verifies its size and SHA-256 before parsing, and writes `data/interim/<revision>/clean-v1/`. Use `--raw-dir PATH` or `--output-dir PATH` to change storage roots; paths are relative to the working directory. PyArrow belongs to the `data` extra so preparation does not require notebook dependencies.

| Output | Contents |
| --- | --- |
| `records.jsonl` | Eligible cleaned rows, original labels, zero-based `source_row`, stable `sample_id`, and `group_id`. |
| `excluded.jsonl` | Original row identities and exclusion reasons; conflicting groups retain their group ID. Source texts can be recovered from the pinned input. |
| `changes.jsonl` | Row identities and the operations that changed their text, including any subsequently quarantined rows. |
| `manifest.json` | Exact source, effective policy, label mapping, code hashes, environment versions, counts, and output checksums. |

`sample_id` hashes the source identity and original row position, distinguishing duplicate occurrences. `group_id` hashes the cleaned UTF-8 text, so it also serves as a text checksum. Outputs are ordered by source row. Writes are staged; repeated runs verify identical output bytes and leave existing files untouched. Changes to the recipe, code, environment, or existing output are reported as a mismatch: bump the policy version for a new policy, or select another output root to preserve previous artifacts.

**Applied result:** 2,597 eligible rows and 3 quarantined rows (source positions 618, 1136, and 2557). Text changed in 223 rows: 135 punctuation repairs and 94 whitespace normalizations, with 6 rows receiving both. There are 2,590 distinct cleaned-text groups; seven same-label duplicate pairs remain. Retained class counts are 426 `specific fls`, 1,369 `not-fls`, and 802 `non-specific fls`. A post-clean audit reports no structural, label-mapping, or suspicious-Unicode findings and no conflicting groups. A repeated preparation reproduced all artifact bytes.

These cleaning outputs are intermediate development candidates. The splitting step below checks overlap and freezes train/validation assignments. Cleaning itself neither opens nor modifies the published test, and raw training bytes remain unchanged.

## Development partitions (v1)

```bash
uv sync --locked --extra data
uv run --locked --extra data filing-sentence-classifier split-data
```

The command verifies the prepared manifest and files, then reads only `text` from the pinned test Parquet. Comparison keys apply the six existing C1 repairs, NFC, and whitespace normalization. Every development group whose key matches test is excluded before splitting; no test rows are removed. Comparison does not apply training eligibility filters: one reserved row contains `U+0099`, which remains uncorrected in its key and is counted as a warning. No test examples or labels were inspected to add encoding repairs. Empty or non-string test text stops splitting.

Eligible rows use [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) with five folds, shuffle enabled, seed `2026`, and **fold 0 fixed as validation**. Inputs are sorted by original row position. Groups remain disjoint, and the target validation fraction is 20%. Each class needs at least five eligible groups; the selected fold must retain every class in both partitions and stay within five percentage points of the target overall and per class. Failure stops the command without trying another seed or fold. These rules live in [data/split.py](../src/filing_sentence_classifier/data/split.py); input verification and artifact writing live in [data/partition.py](../src/filing_sentence_classifier/data/partition.py).

| Partition | Specific FLS | Not-FLS | Non-specific FLS | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 340 | 1,093 | 641 | **2,074** |
| Validation | 86 | 273 | 160 | **519** |

Of the 2,597 cleaned candidates, four rows in four groups overlap test and are excluded. The remaining 2,593 rows have no shared groups between train and validation or with test under this comparison rule. The 1,000 published test rows remain unchanged. This controls exact matches after normalization, not semantic similarity, company overlap, or temporal overlap.

Outputs live in `data/processed/<revision>/split-v1/`:

- `train.jsonl` and `val.jsonl` preserve the cleaned text, original labels, IDs, and source positions.
- `assignments.jsonl` records the split for every eligible sample and group; `excluded.jsonl` records development IDs excluded for test overlap. Earlier cleaning exclusions remain in the referenced parent artifact.
- `manifest.json` pins the parent manifest hash, input/output hashes, recipes, seed, fold, counts, code hashes, dependency versions, and aggregate overlap findings.

Use `--prepared-dir PATH` to select a cleaned artifact, `--raw-dir PATH` for source snapshots, and `--output-dir PATH` for split artifacts. Inputs are preserved, writes are staged, and repeated execution verifies existing output bytes without replacing them. Training and evaluation should consume these saved assignments rather than rerun splitting. Fit vocabulary, IDF, length limits, and other learned transforms on development train only.

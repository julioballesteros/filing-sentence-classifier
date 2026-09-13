# Forward-Looking Statements Dataset

This project uses [FinanceMTEB/FLS](https://huggingface.co/datasets/FinanceMTEB/FLS) for sentence-level classification of forward-looking statements in English financial reports.

**Review status:** Source documentation and repository metadata have been reviewed. Local data quality checks and verification of the published counts are pending.

## Published dataset

The selected source revision is `39b6719f1d7197df4498fea9fce20d4ad782a083`, also pinned by the [FinMTEB task implementation](https://github.com/yixuantt/FinMTEB/blob/main/finance_mteb/tasks/Classification/eng/FLSClassification.py).

| Published split | Sentences |
| --- | ---: |
| Train | 2,600 |
| Test | 1,000 |
| **Total** | **3,600** |

Files are published as Parquet under the `default` configuration. Each row contains `text` (string), `label` (integer), and `label_text` (string). These figures and fields come from the [publisher's metadata](https://huggingface.co/api/datasets/FinanceMTEB/FLS), rather than a local audit.

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
- **Evaluation scope:** The published schema contains no company, document, or date identifiers. It does not support verifying separation by company or period. Duplicate texts, conflicting labels, and invalid rows remain subject to local inspection.
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

This step verifies file integrity only. It does not parse sentences, validate row counts, create development partitions, or inspect test labels. Local data quality checks remain pending.

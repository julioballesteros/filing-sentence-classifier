"""Pinned source specification for FinanceMTEB/FLS."""

from filing_sentence_classifier.data.spec import DatasetSource, SourceFile

LABEL_NAMES = {0: "specific fls", 1: "not-fls", 2: "non-specific fls"}

# Parquet hashes come from the pinned revision's Hub LFS metadata. The README
# hash was computed from its raw bytes at that same revision.
FLS_SOURCE = DatasetSource(
    repo_id="FinanceMTEB/FLS",
    revision="39b6719f1d7197df4498fea9fce20d4ad782a083",
    files=(
        SourceFile(
            "README.md",
            441,
            "37c28808134de90e28d1fbd2c8d305a6264e914807fc40be5bf31926b80b54c6",
        ),
        SourceFile(
            "data/train-00000-of-00001.parquet",
            292230,
            "b8eace45f565497fb57740a15d8b72eeee00cf03d878f2caccef526bb0ddc44c",
        ),
        SourceFile(
            "data/test-00000-of-00001.parquet",
            109784,
            "9b1c472161fb8a0bbcb2f65f84afe552e33cd7bba5abad0beeb658b42dbb32ec",
        ),
    ),
)

"""
convert_mdpl2_catalog.py
------------------------
Runnable script: convert the MDPL2 halo catalog CSVs to the Parquet files the
pipeline reads.

Thin driver only: `convert_catalog_to_parquet` and `ConversionReport` live in
`dvcorr.pipeline.catalog_conversion`, the single source of truth. This script
resolves paths and prints; it owns no conversion logic.

Run once per catalog. It is cache-and-skip (CLAUDE.md hard rule 7), so
re-running is cheap and safe -- an up-to-date Parquet file is reported, not
rewritten. `--force` reconverts anyway.

Usage
-----
    .venv/bin/python -m scripts.convert_mdpl2_catalog                 # both
    .venv/bin/python -m scripts.convert_mdpl2_catalog mvir12         # one
    .venv/bin/python -m scripts.convert_mdpl2_catalog full --force

Converting `full` reads 11.2 GB and takes several minutes; `mvir12` takes
seconds. Memory stays bounded regardless (see the module docstring of
`dvcorr.pipeline.catalog_conversion`).
"""

from __future__ import annotations

import argparse

from dvcorr.config import VALID_CATALOGS, PathsConfig
from dvcorr.pipeline.catalog_conversion import convert_catalog_to_parquet


def main() -> None:
    """Convert the requested catalogs and print each report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    parser.add_argument(
        "catalogs",
        nargs="*",
        choices=sorted(VALID_CATALOGS),
        default=sorted(VALID_CATALOGS),
        help="Which catalogs to convert. Default: all of them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reconvert even if an up-to-date Parquet file is already present.",
    )
    args = parser.parse_args()

    paths = PathsConfig()
    for name in args.catalogs:
        report = convert_catalog_to_parquet(
            paths.halo_catalog(name, parquet=False),
            paths.halo_catalog(name),
            force=args.force,
        )
        print(report.summary())
        print()


if __name__ == "__main__":
    main()

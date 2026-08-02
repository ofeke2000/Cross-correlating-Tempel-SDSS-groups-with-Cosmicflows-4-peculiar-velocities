"""
catalog_conversion.py
---------------------
One-time CSV -> Parquet conversion of the MDPL2 halo catalogs.

The CSVs stay the source of truth; this module writes the derived file the
pipeline actually reads (`dvcorr.pipeline.velocity_centered._load_all_halos`).
Driven by `scripts/convert_mdpl2_catalog.py`, which is orchestration only.

Why convert
-----------
The full catalog is 11.2 GB of text. `pandas.read_csv` on it costs ~10 minutes
and, at float64, roughly 12 GB of peak RAM -- per run, every run. The same data
as float32 Parquet is ~3.5 GB on disk and loads in seconds, because Parquet
stores columns separately (so `columns=` skips the ones a run does not want
without parsing them) and records per-row-group statistics (so a `mass_min`
filter skips whole row groups without decoding them).

Streaming, not slurping
-----------------------
`convert_catalog_to_parquet` reads the CSV in chunks and writes each chunk as
it goes, so peak memory is set by `chunk_rows`, not by the file size. This is
what makes an 11.2 GB input tractable on a machine that could not hold it.

One schema for both catalogs
----------------------------
Both catalogs are written with the SAME columns and dtypes:

    x, y, z, vx, vy, vz, mvir : float32
    is_distinct               : bool     (pid == -1, collapsed at write time)

`CATALOG_MVIR12`'s four legacy derived columns (`delta_5`, `delta_3`,
`near_virgo`, `bulkflow_3`) are left behind -- nothing in this project reads
them. A single schema is what lets the loader have exactly one code path and
makes the two catalogs genuinely interchangeable rather than two special cases.

Why float32 is not a loss
-------------------------
The CSVs record positions to 4 decimal places and velocities to 2. float32
carries ~7 significant digits, which at box coordinates up to
`conventions.BOX_SIZE` resolves ~6e-5 h^-1 Mpc -- finer than the text it is
read from. The cast therefore discards no information that was in the file.

For `mvir` the relevant question is not precision but whether the cast can move
a halo across a mass cut. It cannot: `mvir` is quantized in multiples of
`conventions.PARTICLE_MASS` (~1.5e9), while float32's absolute resolution near
the largest masses in the catalog is ~2e8 -- smaller than the gap between
adjacent allowed masses, and far smaller than the distance from any allowed
mass to a round threshold like `MVIR12_CATALOG_FLOOR`. Row counts either side
of a cut are unchanged by the cast, which is why
`tests/test_catalog_equivalence.py` can assert an exact figure.

Positions are folded into the box
---------------------------------
Rockstar emits a small number of halos with a coordinate at exactly
`conventions.BOX_SIZE` (196 rows in the full catalog, 0 in the pre-cut one)
rather than at `0.0`. In a periodic box those are the SAME point, so this is a
choice of representative, not a disagreement about where the halo is -- but it
meant the two catalogs stored some shared halos differently, and it meant box
coordinates could not be assumed to lie in the half-open interval
[0, `BOX_SIZE`).

Conversion folds every position with `np.mod(pos, BOX_SIZE)`, adopting the
representative the pre-cut catalog already used. The fold is exact and a
no-op for every interior coordinate: at float32, `mod` returns 1000.0 -> 0.0
and leaves everything in [0, 1000) bit-identical. Afterwards both catalogs
satisfy the half-open convention, so downstream code may rely on it.

This is a normalization of the coordinate's REPRESENTATION, not a minimum-image
reduction of a separation, and so is not the thing CLAUDE.md hard rule 3
forbids doing inside geometry -- it happens once, at ingest, before any
separation exists.

Row order is preserved
----------------------
`CATALOG_FULL` arrives sorted ascending by `mvir`. Writing rows in input order
means each row group covers a narrow, contiguous mass range, so its recorded
min/max statistics let a `mass_min` filter skip nearly the whole file. Sorting
or shuffling here would silently cost that.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dvcorr import conventions

#: Rows per `read_csv` chunk. Sets peak conversion memory (a chunk is ~8
#: columns x this many rows), not the output layout.
_DEFAULT_CHUNK_ROWS: int = 5_000_000

#: Rows per Parquet row group -- the granularity at which a `mass_min` filter
#: can skip data unread. Smaller than `_DEFAULT_CHUNK_ROWS` on purpose: chunk
#: size is a memory knob, row-group size is a query knob, and tying them
#: together would make one unavailable whenever the other was tuned.
_DEFAULT_ROW_GROUP_ROWS: int = 1_000_000

#: Column written in place of `pid`. The pipeline only ever asks "is this a
#: distinct halo", never "which halo is its parent", so the parent id is
#: collapsed to a bool at write time rather than carried at 8 bytes a row.
IS_DISTINCT_COLUMN: str = "is_distinct"

#: Sentinel value of `pid` marking a distinct (non-sub) halo in Rockstar.
_DISTINCT_PARENT_ID: int = -1

#: The converted schema: what every catalog Parquet file contains, in order.
PARQUET_COLUMNS: tuple[str, ...] = (
    *conventions.POSITION_COLUMNS,
    *conventions.VELOCITY_COLUMNS,
    conventions.HALO_COLUMNS["mass"],
    IS_DISTINCT_COLUMN,
)

#: File-level Parquet metadata keys. The converted file records what it was
#: made from, so `convert_catalog_to_parquet` can tell "already converted" from
#: "converted from a different or since-changed CSV" -- the existence check
#: CLAUDE.md hard rule 7 asks for, done on provenance rather than on the mere
#: presence of a filename.
_META_SOURCE_NAME: bytes = b"dvcorr.source_name"
_META_SOURCE_BYTES: bytes = b"dvcorr.source_bytes"
_META_N_DISTINCT: bytes = b"dvcorr.n_distinct"
_META_MVIR_MIN: bytes = b"dvcorr.mvir_min"
_META_MVIR_MAX: bytes = b"dvcorr.mvir_max"


@dataclass(frozen=True)
class ConversionReport:
    """What a conversion produced, or what an already-converted file contains.

    Returned by `convert_catalog_to_parquet` on both paths -- a fresh
    conversion and a cache hit -- so a caller gets the same summary either way
    and never has to branch on whether work was actually done.

    Attributes
    ----------
    csv_path : Path
        The source CSV.
    parquet_path : Path
        The file written (or found already present).
    n_rows : int
        Rows in the converted file.
    n_distinct : int
        Rows with `pid == -1`. `n_rows - n_distinct` is the subhalo count.
    mvir_min, mvir_max : float
        Mass range, h^-1 M_sun. `mvir_min / conventions.PARTICLE_MASS` is the
        catalog's floor in particles -- the number that says whether the
        smallest objects in it are resolved.
    n_nonfinite : int
        Rows containing a NaN or inf in any position, velocity, or mass
        column. Reported rather than silently dropped: a missing velocity is
        missing data (CLAUDE.md hard rule 5) and the decision of what to do
        about it belongs to the selection stage, not to a format conversion.
    n_folded : int
        Rows that had a coordinate at or beyond `conventions.BOX_SIZE` and
        were folded into [0, BOX_SIZE). Reported because it is a fact about
        the SOURCE file: it should be a handful, and a large or newly-changed
        value means the upstream catalog changed coordinate convention.
    skipped : bool
        True if an existing, provenance-matching Parquet file was reused and
        nothing was rewritten.
    """

    csv_path: Path
    parquet_path: Path
    n_rows: int
    n_distinct: int
    mvir_min: float
    mvir_max: float
    n_nonfinite: int
    n_folded: int
    skipped: bool

    @property
    def n_subhalos(self) -> int:
        """Rows with `pid != -1`."""
        return self.n_rows - self.n_distinct

    @property
    def min_particle_count(self) -> float:
        """`mvir_min` in units of `conventions.PARTICLE_MASS`."""
        return self.mvir_min / conventions.PARTICLE_MASS

    def summary(self) -> str:
        """Multi-line human-readable summary, printed by the driver script."""
        verb = "already converted" if self.skipped else "converted"
        return (
            f"{verb}: {self.csv_path.name} -> {self.parquet_path.name}\n"
            f"  rows          {self.n_rows:,}\n"
            f"  distinct      {self.n_distinct:,} "
            f"({100.0 * self.n_distinct / max(self.n_rows, 1):.2f}%)\n"
            f"  subhalos      {self.n_subhalos:,} "
            f"({100.0 * self.n_subhalos / max(self.n_rows, 1):.2f}%)\n"
            f"  mvir range    {self.mvir_min:.4g} .. {self.mvir_max:.4g} h^-1 M_sun\n"
            f"  mass floor    {self.min_particle_count:.1f} particles\n"
            f"  non-finite    {self.n_nonfinite:,}\n"
            f"  folded to box {self.n_folded:,}\n"
            f"  size          {self.parquet_path.stat().st_size / 1e9:.2f} GB"
        )


def _read_report(csv_path: Path, parquet_path: Path) -> ConversionReport | None:
    """Return the report recorded in an existing Parquet file, or None.

    None means "convert": either the file is absent, or it carries no dvcorr
    provenance metadata, or the metadata names a source whose size no longer
    matches the CSV on disk. The size check is what distinguishes a genuine
    cache hit from a stale file left over from a different catalog -- a bare
    `path.exists()` would happily reuse the latter.

    Parameters
    ----------
    csv_path : Path
        The source the caller intends to convert.
    parquet_path : Path
        Candidate already-converted file.

    Returns
    -------
    ConversionReport | None
    """
    if not parquet_path.exists():
        return None
    try:
        metadata = pq.read_metadata(parquet_path)
    except (OSError, pa.ArrowInvalid):
        return None

    stored = (metadata.metadata or {})
    if _META_SOURCE_NAME not in stored:
        return None
    if stored[_META_SOURCE_NAME].decode() != csv_path.name:
        return None
    if not csv_path.exists():
        return None
    if int(stored[_META_SOURCE_BYTES]) != csv_path.stat().st_size:
        return None

    return ConversionReport(
        csv_path=csv_path,
        parquet_path=parquet_path,
        n_rows=metadata.num_rows,
        n_distinct=int(stored[_META_N_DISTINCT]),
        mvir_min=float(stored[_META_MVIR_MIN]),
        mvir_max=float(stored[_META_MVIR_MAX]),
        n_nonfinite=0,
        n_folded=0,
        skipped=True,
    )


def convert_catalog_to_parquet(
    csv_path: Path,
    parquet_path: Path,
    *,
    chunk_rows: int = _DEFAULT_CHUNK_ROWS,
    row_group_rows: int = _DEFAULT_ROW_GROUP_ROWS,
    force: bool = False,
) -> ConversionReport:
    """Stream a halo-catalog CSV into a float32 Parquet file.

    Reads only the eight columns the pipeline uses, casts them per this
    module's docstring, collapses `pid` to `IS_DISTINCT_COLUMN`, and writes
    row groups in input order. Memory is bounded by `chunk_rows` regardless of
    input size.

    Cache-and-skip (CLAUDE.md hard rule 7): if `parquet_path` already exists
    and records this same CSV at its current size, returns that file's report
    with `skipped = True` without rewriting. Pass `force=True` to reconvert
    anyway.

    Parameters
    ----------
    csv_path : Path
        Source CSV. Must carry the `dvcorr.conventions.HALO_COLUMNS` headers.
    parquet_path : Path
        Destination. Overwritten on a real conversion; its parent directory
        must already exist.
    chunk_rows : int, keyword-only
        Rows per read chunk -- the memory knob.
    row_group_rows : int, keyword-only
        Rows per Parquet row group -- the filter-granularity knob.
    force : bool, keyword-only
        Reconvert even if an up-to-date Parquet file is present.

    Returns
    -------
    ConversionReport

    Raises
    ------
    FileNotFoundError
        If `csv_path` does not exist.
    RuntimeError
        If the CSV yields zero rows -- an empty catalog is a broken download,
        not a valid conversion, and is far cheaper to catch here than as a
        confusing zero-center failure several stages downstream.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"catalog CSV not found: {csv_path}")

    if not force:
        cached = _read_report(csv_path, parquet_path)
        if cached is not None:
            return cached

    mass_col = conventions.HALO_COLUMNS["mass"]
    parent_col = conventions.HALO_COLUMNS["parent_id"]
    float_cols = [*conventions.POSITION_COLUMNS, *conventions.VELOCITY_COLUMNS, mass_col]
    usecols = [*float_cols, parent_col]

    schema = pa.schema(
        [(name, pa.float32()) for name in float_cols]
        + [(IS_DISTINCT_COLUMN, pa.bool_())]
    )

    n_position_columns = len(conventions.POSITION_COLUMNS)
    box_size = np.float32(conventions.BOX_SIZE)

    n_rows = 0
    n_distinct = 0
    n_nonfinite = 0
    n_folded = 0
    mvir_min = np.inf
    mvir_max = -np.inf

    print(f"converting {csv_path} -> {parquet_path} ...")
    writer = pq.ParquetWriter(parquet_path, schema)
    try:
        reader = pd.read_csv(
            csv_path,
            usecols=usecols,
            dtype={name: "float32" for name in float_cols} | {parent_col: "int64"},
            chunksize=chunk_rows,
        )
        for chunk in reader:
            is_distinct = (chunk[parent_col] == _DISTINCT_PARENT_ID).to_numpy()
            values = chunk[float_cols].to_numpy()

            # Fold positions into [0, BOX_SIZE) -- see this module's docstring.
            # Counted, not silent: the number of rows this touches is a fact
            # about the input file worth reporting, and a sudden change in it
            # means the upstream catalog changed convention.
            positions = values[:, :n_position_columns]
            n_folded += int(np.any(positions >= box_size, axis=1).sum())
            values[:, :n_position_columns] = np.mod(positions, box_size)

            n_rows += len(chunk)
            n_distinct += int(is_distinct.sum())
            n_nonfinite += int((~np.isfinite(values)).any(axis=1).sum())

            masses = chunk[mass_col].to_numpy()
            if masses.size:
                mvir_min = min(mvir_min, float(np.nanmin(masses)))
                mvir_max = max(mvir_max, float(np.nanmax(masses)))

            table = pa.Table.from_arrays(
                [pa.array(values[:, i], type=pa.float32()) for i in range(len(float_cols))]
                + [pa.array(is_distinct, type=pa.bool_())],
                schema=schema,
            )
            writer.write_table(table, row_group_size=row_group_rows)
            print(f"  {n_rows:,} rows written", end="\r", flush=True)

        if n_rows == 0:
            raise RuntimeError(f"{csv_path} yielded zero rows.")

        # Provenance goes on at close time, once the totals are known -- a file
        # that died mid-write therefore carries no metadata and is correctly
        # treated as "not converted" by `_read_report` on the next attempt,
        # rather than being reused as a silently truncated catalog.
        writer.add_key_value_metadata(
            {
                _META_SOURCE_NAME: csv_path.name.encode(),
                _META_SOURCE_BYTES: str(csv_path.stat().st_size).encode(),
                _META_N_DISTINCT: str(n_distinct).encode(),
                _META_MVIR_MIN: repr(mvir_min).encode(),
                _META_MVIR_MAX: repr(mvir_max).encode(),
            }
        )
    finally:
        writer.close()
    print()

    return ConversionReport(
        csv_path=csv_path,
        parquet_path=parquet_path,
        n_rows=n_rows,
        n_distinct=n_distinct,
        mvir_min=mvir_min,
        mvir_max=mvir_max,
        n_nonfinite=n_nonfinite,
        n_folded=n_folded,
        skipped=False,
    )

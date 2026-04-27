#!/usr/bin/env python3
"""Post-processing module for instaGRAAL.

Remaps original Hi-C read pairs to the new assembly coordinate system
produced by ``instagraal-polish``, then builds a fragment-level ``.cool``
file and zooms it into a multi-resolution ``.mcool``.

Overview
--------
``instagraal-pre`` aligned reads against the *original* reference and stored
contacts as absolute fragment indices.  After ``instagraal`` + ``instagraal-polish``
the genome has been rearranged: contigs are joined into new scaffolds, some
fragments are flipped (orientation ``-1``), and gap/junction bases may have
been inserted between fragments from different source contigs.

``new_info_frags.txt`` encodes this mapping::

    >new_scaffold_name
    init_contig   id_frag   orientation   start   end
    contig_001    0         1             0       32766
    contig_001    1         1             32766   52321
    ...

This module uses that file to lift every read pair from original-assembly
coordinates to new-assembly coordinates, then builds a contact map.

Inputs required
---------------
pairs
    Original 4DN-format Hi-C pairs file (plain or ``.gz``), exactly as fed
    to ``instagraal-pre``.  Reads must be aligned to the *original* FASTA.
new_info_frags
    ``new_info_frags.txt`` produced by ``instagraal-polish``.

Note: the original FASTA is **not** required — all coordinate information
is already encoded in ``new_info_frags.txt``.  Pass it via ``--fasta`` only
if you want GC-content annotation on the output bins (not yet implemented).

Outputs
-------
<name>.cool
    Fragment-level Hi-C contact map in new-assembly coordinates.
<name>.mcool
    Multi-resolution contact map (via ``cooler.zoomify_cooler``).
<name>.pairs
    Hi-C pairs file remapped to new-assembly coordinates.
<name>_hic_map.png
    Genome-wide Hi-C contact map visualisation.
"""

import gzip
import pathlib

import cooler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .parse_info_frags import parse_info_frags

# ---------------------------------------------------------------------------
# Build new fragment bins from new_info_frags
# ---------------------------------------------------------------------------


def _build_new_bins(
    new_scaffolds: dict,
    junction_len: int = 6,
) -> pd.DataFrame:
    """Convert the new_info_frags scaffold dictionary into a bins DataFrame.

    Each fragment entry in *new_scaffolds* becomes one bin in the new
    assembly.  Bins are placed sequentially within their scaffold.  A gap of
    *junction_len* bases is inserted whenever consecutive fragments originate
    from **different** source contigs (matching the logic used by
    ``write_fasta`` in :mod:`parse_info_frags`).

    Parameters
    ----------
    new_scaffolds:
        Output of :func:`parse_info_frags`.  Maps scaffold name →
        list of ``[init_contig, frag_id, orig_start, orig_end, orientation]``.
    junction_len:
        Length (bp) of the junction sequence inserted between fragments from
        different source contigs.  Must match what was used in
        ``instagraal-polish --junction``.  Default 6 (``NNNNNN``).

    Returns
    -------
    pd.DataFrame
        Columns: ``chrom``, ``start``, ``end``, ``_orig_chrom``,
        ``_orig_start``, ``_orig_end``, ``_orientation``.
        Coordinates are 0-based half-open intervals in new-assembly space.
        The ``_``-prefixed columns are used for liftover and stripped before
        writing to cooler.
    """
    rows: list[dict] = []
    for scaffold, fragments in new_scaffolds.items():
        cursor = 0
        prev_contig: str | None = None
        for entry in fragments:
            init_contig, _frag_id, orig_start, orig_end, orientation = entry
            frag_len = orig_end - orig_start
            # Insert junction gap when source contig changes (same logic as
            # write_fasta: no gap before the very first fragment or when the
            # contig is the same as the previous one).
            if junction_len and prev_contig is not None and prev_contig != init_contig:
                cursor += junction_len
            new_start = cursor
            new_end = cursor + frag_len
            rows.append(
                {
                    "chrom": scaffold,
                    "start": new_start,
                    "end": new_end,
                    "_orig_chrom": init_contig,
                    "_orig_start": orig_start,
                    "_orig_end": orig_end,
                    "_orientation": orientation,
                }
            )
            cursor = new_end
            prev_contig = init_contig

    return pd.DataFrame(
        rows,
        columns=["chrom", "start", "end", "_orig_chrom", "_orig_start", "_orig_end", "_orientation"],
    )


# ---------------------------------------------------------------------------
# Liftover index
# ---------------------------------------------------------------------------


def _build_liftover_index(bins_extended: pd.DataFrame) -> dict:
    """Build a per-original-contig lookup for fast coordinate liftover.

    Parameters
    ----------
    bins_extended:
        Output of :func:`_build_new_bins` (must retain the ``_orig_*``
        columns).

    Returns
    -------
    dict
        Maps ``orig_chrom`` (str) → dict with numpy arrays:

        - ``orig_starts``   - sorted fragment start positions (0-based) on
          the original contig.
        - ``orig_ends``     - corresponding end positions.
        - ``bin_ids``       - 0-based row indices into *bins_extended* (i.e.
          into the cooler bins table).
        - ``new_chroms``    - new scaffold name for each fragment.
        - ``new_starts``    - fragment start in new-assembly coordinates.
        - ``new_ends``      - fragment end in new-assembly coordinates.
        - ``orientations``  - orientation (+1 or -1) of each fragment.
    """
    index: dict = {}
    for orig_chrom, grp in bins_extended.groupby("_orig_chrom", sort=False):
        grp_sorted = grp.sort_values("_orig_start")
        index[orig_chrom] = {
            "orig_starts": grp_sorted["_orig_start"].to_numpy(dtype=np.int64),
            "orig_ends": grp_sorted["_orig_end"].to_numpy(dtype=np.int64),
            "bin_ids": grp_sorted.index.to_numpy(dtype=np.int64),
            "new_chroms": grp_sorted["chrom"].to_numpy(),
            "new_starts": grp_sorted["start"].to_numpy(dtype=np.int64),
            "new_ends": grp_sorted["end"].to_numpy(dtype=np.int64),
            "orientations": grp_sorted["_orientation"].to_numpy(dtype=np.int8),
        }
    return index


def _pos_to_new_bin(orig_chrom: str, orig_pos_1based: int, index: dict) -> int | None:
    """Map a 1-based position on the original assembly to a new bin index.

    Returns ``None`` when the position is not covered by any fragment in
    *new_info_frags* (e.g. the contig was discarded during assembly).

    Parameters
    ----------
    orig_chrom:
        Chromosome / contig name in the original assembly.
    orig_pos_1based:
        1-based genomic position (as in the pairs file).
    index:
        Output of :func:`_build_liftover_index`.
    """
    if orig_chrom not in index:
        return None
    entry = index[orig_chrom]
    # Convert 1-based to 0-based before interval search, matching the
    # fragment-assignment convention used in pre.py (_pairs_to_pixels).
    pos0 = orig_pos_1based - 1
    i = int(np.searchsorted(entry["orig_starts"], pos0, side="right")) - 1
    if i < 0 or pos0 >= entry["orig_ends"][i]:
        return None
    return int(entry["bin_ids"][i])


def _pos_to_new_coords(
    orig_chrom: str,
    orig_pos_1based: int,
    index: dict,
) -> tuple[str, int] | None:
    """Map a 1-based position on the original assembly to new-assembly coords.

    Returns the new scaffold name and 1-based position, or ``None`` when the
    position cannot be lifted over (contig absent or position outside all
    fragments).

    Strand orientation is taken into account: when a fragment was placed in
    reverse complement (orientation ``-1``), positions are mirrored within
    the fragment.
    """
    if orig_chrom not in index:
        return None
    entry = index[orig_chrom]
    pos0 = orig_pos_1based - 1
    i = int(np.searchsorted(entry["orig_starts"], pos0, side="right")) - 1
    if i < 0 or pos0 >= int(entry["orig_ends"][i]):
        return None
    offset = pos0 - int(entry["orig_starts"][i])
    ori = int(entry["orientations"][i])
    new_chrom = str(entry["new_chroms"][i])
    new_frag_start = int(entry["new_starts"][i])
    if ori == 1:
        new_pos0 = new_frag_start + offset
    else:  # ori == -1: reverse complement — mirror position within fragment
        frag_len = int(entry["orig_ends"][i]) - int(entry["orig_starts"][i])
        new_pos0 = new_frag_start + (frag_len - 1 - offset)
    return new_chrom, new_pos0 + 1  # return 1-based


# ---------------------------------------------------------------------------
# Pairs → lifted pixels
# ---------------------------------------------------------------------------


def _pairs_to_lifted_pixels(
    pairs_path: pathlib.Path,
    index: dict,
) -> tuple[pd.DataFrame, int]:
    """Remap Hi-C pairs to new-assembly fragment bins and count contacts.

    Reads whose either end cannot be lifted over (unmapped contig, or
    position outside any fragment) are silently dropped.

    Parameters
    ----------
    pairs_path:
        Original pairs file (plain or ``.gz`` / ``.bgz``).
    index:
        Output of :func:`_build_liftover_index`.

    Returns
    -------
    pixels : pd.DataFrame
        Columns ``bin1_id``, ``bin2_id``, ``count``.  Upper-triangular
        (``bin1_id <= bin2_id``), sorted by ``(bin1_id, bin2_id)``.
    total_contacts : int
        Number of pairs successfully remapped (both ends).
    """
    # Column indices default to the standard 4DN order; overridden from the
    # ``#columns:`` header line when present.
    col_chr1, col_pos1, col_chr2, col_pos2 = 1, 2, 3, 4

    opener = gzip.open if str(pairs_path).endswith((".gz", ".bgz")) else open
    counts: dict[tuple[int, int], int] = {}
    total = 0

    with opener(pairs_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                if line.startswith("#columns:"):
                    cols = line.strip().split()[1:]
                    try:
                        col_chr1 = cols.index("chr1")
                        col_pos1 = cols.index("pos1")
                        col_chr2 = cols.index("chr2")
                        col_pos2 = cols.index("pos2")
                    except ValueError:
                        pass  # keep defaults
                continue
            parts = line.rstrip("\n").split("\t")
            try:
                chr1 = parts[col_chr1]
                pos1 = int(parts[col_pos1])
                chr2 = parts[col_chr2]
                pos2 = int(parts[col_pos2])
            except (IndexError, ValueError):
                continue

            b1 = _pos_to_new_bin(chr1, pos1, index)
            b2 = _pos_to_new_bin(chr2, pos2, index)
            if b1 is None or b2 is None:
                continue

            total += 1
            key = (min(b1, b2), max(b1, b2))
            counts[key] = counts.get(key, 0) + 1

    if not counts:
        pixels = pd.DataFrame(columns=["bin1_id", "bin2_id", "count"])
    else:
        keys = sorted(counts.keys())
        b1s, b2s = zip(*keys)
        pixels = pd.DataFrame(
            {
                "bin1_id": np.array(b1s, dtype=np.int32),
                "bin2_id": np.array(b2s, dtype=np.int32),
                "count": np.array([counts[k] for k in keys], dtype=np.int32),
            }
        )

    return pixels, total


# ---------------------------------------------------------------------------
# Lifted pairs file
# ---------------------------------------------------------------------------


def _write_lifted_pairs(
    pairs_path: pathlib.Path,
    index: dict,
    bins_extended: pd.DataFrame,
    output_path: pathlib.Path,
) -> tuple[int, int]:
    """Remap pairs to new-assembly coordinates and write a new pairs file.

    Header lines are updated: ``#chromsize`` / ``#chromosomes`` entries are
    replaced with new-assembly values, and ``#sorted:`` is set to ``none``
    because the output order follows the original pairs file, not new coords.

    Parameters
    ----------
    pairs_path:
        Original pairs file (plain or ``.gz`` / ``.bgz``).
    index:
        Extended liftover index (output of :func:`_build_liftover_index`).
    bins_extended:
        Output of :func:`_build_new_bins` (provides new chromsizes).
    output_path:
        Destination path for the remapped pairs file (plain text).

    Returns
    -------
    (total, remapped) : tuple[int, int]
        Total data lines read and number successfully remapped.
    """
    # Compute new chromsizes from bins table
    new_chromsizes: dict[str, int] = {}
    for row in bins_extended.itertuples(index=False):
        if row.end > new_chromsizes.get(row.chrom, 0):
            new_chromsizes[row.chrom] = row.end

    opener = gzip.open if str(pairs_path).endswith((".gz", ".bgz")) else open

    # ── Pass 1: collect header ────────────────────────────────────────────
    pairs_format_line = "## pairs format v1.0"
    other_meta: list[str] = []
    columns_line = "#columns: readID chr1 pos1 chr2 pos2 strand1 strand2"
    col_chr1, col_pos1, col_chr2, col_pos2 = 1, 2, 3, 4

    with opener(pairs_path, "rt") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            stripped = line.rstrip("\n")
            if stripped.startswith("## "):
                pairs_format_line = stripped
            elif stripped.startswith("#columns:"):
                columns_line = stripped
                cols = stripped.split()[1:]
                try:
                    col_chr1 = cols.index("chr1")
                    col_pos1 = cols.index("pos1")
                    col_chr2 = cols.index("chr2")
                    col_pos2 = cols.index("pos2")
                except ValueError:
                    pass
            elif stripped.startswith(("#chromsize:", "#chromosomes:", "#sorted:")):
                pass  # replaced below
            else:
                other_meta.append(stripped)

    # ── Pass 2: remap data and write output ───────────────────────────────
    total = 0
    remapped = 0

    with open(output_path, "w") as out_fh:
        out_fh.write(pairs_format_line + "\n")
        out_fh.write("#sorted: none\n")
        for meta in other_meta:
            out_fh.write(meta + "\n")
        out_fh.write(f"#chromosomes: {' '.join(new_chromsizes)}\n")
        for chrom, size in new_chromsizes.items():
            out_fh.write(f"#chromsize: {chrom} {size}\n")
        out_fh.write(columns_line + "\n")

        with opener(pairs_path, "rt") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                total += 1
                try:
                    chr1 = parts[col_chr1]
                    pos1 = int(parts[col_pos1])
                    chr2 = parts[col_chr2]
                    pos2 = int(parts[col_pos2])
                except (IndexError, ValueError):
                    continue
                coords1 = _pos_to_new_coords(chr1, pos1, index)
                coords2 = _pos_to_new_coords(chr2, pos2, index)
                if coords1 is None or coords2 is None:
                    continue
                new_chr1, new_pos1 = coords1
                new_chr2, new_pos2 = coords2
                parts[col_chr1] = new_chr1
                parts[col_pos1] = str(new_pos1)
                parts[col_chr2] = new_chr2
                parts[col_pos2] = str(new_pos2)
                remapped += 1
                out_fh.write("\t".join(parts) + "\n")

    return total, remapped


# ---------------------------------------------------------------------------
# P(s) curve visualisation
# ---------------------------------------------------------------------------

# Log-spaced genomic-distance bins (equivalent to PsBreaks() in R).
_PS_BREAK_POS = np.array(
    [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        16,
        17,
        19,
        21,
        23,
        26,
        28,
        31,
        34,
        37,
        41,
        45,
        50,
        55,
        60,
        66,
        73,
        80,
        88,
        97,
        107,
        117,
        129,
        142,
        156,
        172,
        189,
        208,
        229,
        252,
        277,
        304,
        335,
        368,
        405,
        446,
        490,
        539,
        593,
        653,
        718,
        790,
        869,
        956,
        1051,
        1156,
        1272,
        1399,
        1539,
        1693,
        1862,
        2048,
        2253,
        2479,
        2726,
        2999,
        3299,
        3629,
        3992,
        4391,
        4830,
        5313,
        5844,
        6429,
        7072,
        7779,
        8557,
        9412,
        10354,
        11389,
        12528,
        13781,
        15159,
        16675,
        18342,
        20176,
        22194,
        24413,
        26855,
        29540,
        32494,
        35743,
        39318,
        43249,
        47574,
        52332,
        57565,
        63322,
        69654,
        76619,
        84281,
        92709,
        101980,
        112178,
        123396,
        135735,
        149309,
        164240,
        180664,
        198730,
        218603,
        240463,
        264510,
        290961,
        320057,
        352063,
        387269,
        425996,
        468595,
        515455,
        567000,
        623700,
        686070,
        754677,
        830145,
        913160,
        1004475,
        1104923,
        1215415,
        1336957,
        1470653,
        1617718,
        1779490,
        1957439,
        2153182,
        2368501,
        2605351,
        2865886,
        3152474,
        3467722,
        3814494,
        4195943,
        4615538,
        5077092,
        5584801,
        6143281,
        6757609,
        7433370,
        8176707,
        8994377,
        9893815,
        10883197,
        11971516,
        13168668,
        14485535,
        15934088,
        17527497,
        19280247,
        21208271,
        23329099,
        25662008,
        28228209,
        31051030,
        34156133,
        37571747,
        41328921,
        45461813,
        50007995,
        55008794,
        60509674,
        66560641,
        73216705,
        80538375,
        88592213,
        97451434,
        107196578,
        117916236,
        129707859,
        142678645,
        156946509,
        172641160,
        189905276,
        208895804,
        229785385,
        252763923,
        278040315,
        305844347,
        336428781,
        370071660,
        407078826,
        447786708,
        492565379,
        541821917,
        596004109,
        655604519,
        721164971,
        793281468,
        872609615,
        959870577,
        1055857635,
        1161443398,
    ],
    dtype=np.int64,
)
_PS_BINWIDTH = np.array(
    [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        1,
        2,
        2,
        2,
        3,
        2,
        3,
        3,
        3,
        4,
        4,
        5,
        5,
        5,
        6,
        7,
        7,
        8,
        9,
        10,
        10,
        12,
        13,
        14,
        16,
        17,
        19,
        21,
        23,
        25,
        27,
        31,
        33,
        37,
        41,
        44,
        49,
        54,
        60,
        65,
        72,
        79,
        87,
        95,
        105,
        116,
        127,
        140,
        154,
        169,
        186,
        205,
        226,
        247,
        273,
        300,
        330,
        363,
        399,
        439,
        483,
        531,
        585,
        643,
        707,
        778,
        855,
        942,
        1035,
        1139,
        1253,
        1378,
        1516,
        1667,
        1834,
        2018,
        2219,
        2442,
        2685,
        2954,
        3249,
        3575,
        3931,
        4325,
        4758,
        5233,
        5757,
        6332,
        6965,
        7662,
        8428,
        9271,
        10198,
        11218,
        12339,
        13574,
        14931,
        16424,
        18066,
        19873,
        21860,
        24047,
        26451,
        29096,
        32006,
        35206,
        38727,
        42599,
        46860,
        51545,
        56700,
        62370,
        68607,
        75468,
        83015,
        91315,
        100448,
        110492,
        121542,
        133696,
        147065,
        161772,
        177949,
        195743,
        215319,
        236850,
        260535,
        286588,
        315248,
        346772,
        381449,
        419595,
        461554,
        507709,
        558480,
        614328,
        675761,
        743337,
        817670,
        899438,
        989382,
        1088319,
        1197152,
        1316867,
        1448553,
        1593409,
        1752750,
        1928024,
        2120828,
        2332909,
        2566201,
        2822821,
        3105103,
        3415614,
        3757174,
        4132892,
        4546182,
        5000799,
        5500880,
        6050967,
        6656064,
        7321670,
        8053838,
        8859221,
        9745144,
        10719658,
        11791623,
        12970786,
        14267864,
        15694651,
        17264116,
        18990528,
        20889581,
        22978538,
        25276392,
        27804032,
        30584434,
        33642879,
        37007166,
        40707882,
        44778671,
        49256538,
        54182192,
        59600410,
        65560452,
        72116497,
        79328147,
        87260962,
        95987058,
        105585763,
        116144340,
    ],
    dtype=np.int64,
)


def _compute_ps(pairs_path: pathlib.Path) -> pd.DataFrame:
    """Compute normalised P(s) curves from a 4DN pairs file.

    Only intra-chromosomal pairs (chr1 == chr2) are used.  Distances are
    binned with the log-spaced :data:`_PS_BREAK_POS` scheme, then
    normalised per strand combination and divided by bin width.

    Parameters
    ----------
    pairs_path:
        Hi-C pairs file (plain or ``.gz``/``.bgz``).

    Returns
    -------
    pd.DataFrame
        Columns: ``binned_distance``, ``strand_combo``, ``norm_p``.
    """
    opener = gzip.open if str(pairs_path).endswith((".gz", ".bgz")) else open

    # Default column indices for standard 4DN format
    col_chr1, col_pos1, col_chr2, col_pos2, col_s1, col_s2 = 1, 2, 3, 4, 5, 6

    distances: list[int] = []
    strands: list[str] = []

    with opener(pairs_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                if line.startswith("#columns:"):
                    cols = line.strip().split()[1:]
                    try:
                        col_chr1 = cols.index("chr1")
                        col_pos1 = cols.index("pos1")
                        col_chr2 = cols.index("chr2")
                        col_pos2 = cols.index("pos2")
                        col_s1 = cols.index("strand1")
                        col_s2 = cols.index("strand2")
                    except ValueError:
                        pass
                continue
            parts = line.rstrip("\n").split("\t")
            try:
                chr1 = parts[col_chr1]
                pos1 = int(parts[col_pos1])
                chr2 = parts[col_chr2]
                pos2 = int(parts[col_pos2])
                s1 = parts[col_s1]
                s2 = parts[col_s2]
            except (IndexError, ValueError):
                continue
            if chr1 != chr2:
                continue
            distances.append(abs(pos2 - pos1))
            strands.append(s1 + s2)

    if not distances:
        return pd.DataFrame(columns=["binned_distance", "strand_combo", "norm_p"])

    dist_arr = np.array(distances, dtype=np.int64)

    # findInterval(x, vec, all.inside=TRUE): clip to [0, n-2]
    idx = np.clip(
        np.searchsorted(_PS_BREAK_POS, dist_arr, side="right") - 1,
        0,
        len(_PS_BREAK_POS) - 2,
    )
    binned = _PS_BREAK_POS[idx]
    bw = _PS_BINWIDTH[idx]

    df = pd.DataFrame({"binned_distance": binned, "strand_combo": strands, "binwidth": bw})
    grouped = df.groupby(["strand_combo", "binned_distance", "binwidth"]).size().reset_index(name="ninter")
    total_per_strand = grouped.groupby("strand_combo")["ninter"].transform("sum")
    grouped["p"] = grouped["ninter"] / total_per_strand
    grouped["norm_p"] = grouped["p"] / grouped["binwidth"]

    return grouped[["binned_distance", "strand_combo", "norm_p"]]


def _plot_ps_curves(
    original_pairs: pathlib.Path,
    new_pairs: pathlib.Path,
    output_path: pathlib.Path,
) -> None:
    """Plot P(s) curves for original and new pairs side by side.

    Line plots are grouped by strand combination (``++``, ``--``, ``+-``,
    ``-+``) and drawn on log-log axes, mirroring the R ggplot code.

    Parameters
    ----------
    original_pairs:
        Hi-C pairs file aligned to the *original* reference.
    new_pairs:
        Hi-C pairs file remapped to the *new* assembly.
    output_path:
        Destination for the saved PNG.
    """
    _STRAND_COLORS = {"++": "C0", "--": "C1", "+-": "C2", "-+": "C3"}

    df_orig = _compute_ps(original_pairs)
    df_new = _compute_ps(new_pairs)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, df, title in zip(axes, [df_orig, df_new], ["Original", "New assembly"]):
        if df.empty:
            ax.set_title(f"{title} (no data)")
            continue
        for strand, grp in df.groupby("strand_combo"):
            grp_sorted = grp.sort_values("binned_distance")
            ax.plot(
                grp_sorted["binned_distance"],
                grp_sorted["norm_p"],
                label=strand,
                color=_STRAND_COLORS.get(strand),
                linewidth=1.5,
                alpha=0.85,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Genomic distance (bp)")
        ax.set_ylabel("P(s) / bp")
        ax.set_title(title)
        ax.legend(title="Strands", fontsize=8)

    fig.suptitle("P(s) — contact probability vs. genomic distance", y=1.01)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# HiC map visualisation
# ---------------------------------------------------------------------------


def _plot_hic_map(
    cool_path: pathlib.Path | str,  ## can be a Path or string
    output_path: pathlib.Path,
    title: str = "",
    max_display_bins: int = 1000,
) -> None:
    """Plot a genome-wide Hi-C contact map from a ``.cool`` file.

    Pixels are aggregated into a display matrix of at most *max_display_bins*
    x *max_display_bins* to keep memory usage bounded for fragment-level
    coolers.  The colour scale uses a log1p transform clipped at the 98th
    percentile of non-zero values.

    Parameters
    ----------
    cool_path:
        Fragment-level (or binned) ``.cool`` file.
    output_path:
        Path for the saved image (PNG).
    title:
        Figure title.  Defaults to the cool file stem.
    max_display_bins:
        Maximum number of rows/columns in the display matrix.
    """
    clr = cooler.Cooler(str(cool_path))
    n_bins = clr.info["nbins"]

    agg = max(1, (n_bins + max_display_bins - 1) // max_display_bins)
    display_n = (n_bins + agg - 1) // agg

    mat = np.zeros((display_n, display_n), dtype=np.float32)
    pixels = clr.pixels()[:]
    b1 = (pixels["bin1_id"].to_numpy() // agg).astype(np.int32)
    b2 = (pixels["bin2_id"].to_numpy() // agg).astype(np.int32)
    c = pixels["count"].to_numpy().astype(np.float32)
    np.add.at(mat, (b1, b2), c)
    off_diag = b1 != b2
    np.add.at(mat, (b2[off_diag], b1[off_diag]), c[off_diag])

    mat = np.log1p(mat)
    nonzero = mat[mat > 0]
    vmax = float(np.percentile(nonzero, 98)) if nonzero.size else 1.0

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.matshow(mat, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto", origin="upper")
    ax.set_title(title or cool_path.stem, pad=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# mcool generation
# ---------------------------------------------------------------------------


def _zoomify_cooler(
    cool_path: pathlib.Path,
    mcool_path: pathlib.Path,
    resolutions: list[int],
    balance: bool = True,
    balance_args: dict | None = None,
    chunksize: int = 10_000_000,
) -> None:
    """Zoom a ``.cool`` file into an ``.mcool`` using the cooler Python API.

    After zoomification, ICE balancing is applied at every zoom level when
    *balance* is ``True``.

    Parameters
    ----------
    cool_path:
        Input fragment-level ``.cool`` file.
    mcool_path:
        Output path for the ``.mcool`` file.
    resolutions:
        List of target bin sizes in base pairs.
    balance:
        Apply ICE balancing at each zoom level.
    balance_args:
        Extra keyword arguments forwarded to :func:`cooler.balance_cooler`
        (e.g. ``{"max_iters": 2000, "mad_max": 10}``).
    chunksize:
        Number of pixels to process at a time during zoomification.
    """
    cooler.zoomify_cooler(
        str(cool_path),
        str(mcool_path),
        resolutions,
        chunksize,
    )
    if balance:
        for res in resolutions:
            clr_res = cooler.Cooler(f"{mcool_path}::resolutions/{res}")
            cooler.balance_cooler(clr_res, store=True, **(balance_args or {}))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_post(
    pairs: pathlib.Path,
    new_info_frags: pathlib.Path,
    output_dir: pathlib.Path,
    resolutions: str | list[int],
    cool_name: str | None = None,
    junction_len: int = 6,
    balance: bool = True,
    balance_args: dict | None = None,
) -> None:
    """Remap Hi-C pairs to the polished assembly and write an mcool.

    Parameters
    ----------
    pairs:
        Original Hi-C pairs file (plain or gzip, 4DN format).
    new_info_frags:
        ``new_info_frags.txt`` from ``instagraal-polish``.
    output_dir:
        Directory where output files will be written (created if absent).
    resolutions:
        Target bin size(s) in bp for the ``.mcool``.  Either a
        comma-separated string (``"1000,5000,25000"``) or a list of ints.
    cool_name:
        Base name for output files.  Defaults to the stem of the pairs
        file name.
    junction_len:
        Length of the junction sequence used during polishing (bp).
        Must match ``--junction`` passed to ``instagraal-polish``.
        Default 6 (``NNNNNN``).
    balance:
        Apply ICE balancing at each zoom level.  Default ``True``.
    balance_args:
        Extra kwargs forwarded to :func:`cooler.balance_cooler` during
        balancing (e.g. ``{"max_iters": 2000, "mad_max": 10}``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse resolutions to list[int]
    if isinstance(resolutions, str):
        resolutions_list: list[int] = [int(r.strip()) for r in resolutions.split(",") if r.strip()]
    else:
        resolutions_list = list(resolutions)

    # Derive output name from pairs file stem when not provided
    if cool_name is None:
        stem = pairs.name
        for _ in range(2):
            stem, ext = pathlib.Path(stem).stem, pathlib.Path(stem).suffix
            if not ext:
                break
        cool_name = stem

    # ── Step 1: parse new_info_frags ──────────────────────────────────────
    print(f"[1/6] Parsing new_info_frags: {new_info_frags}")
    new_scaffolds = parse_info_frags(str(new_info_frags))
    n_scaffolds = len(new_scaffolds)
    n_fragments = sum(len(v) for v in new_scaffolds.values())
    print(f"      → {n_scaffolds:,} new scaffolds, {n_fragments:,} fragments")

    # ── Step 2: build new-assembly fragment bins ───────────────────────────
    print("[2/6] Building new-assembly fragment bins")
    bins_extended = _build_new_bins(new_scaffolds, junction_len)
    n_bins = len(bins_extended)
    print(f"      → {n_bins:,} bins (restriction fragments)")

    # ── Step 3: build liftover index ──────────────────────────────────────
    print("[3/6] Building liftover index")
    index = _build_liftover_index(bins_extended)
    n_orig_chroms = len(index)
    print(f"      → covering {n_orig_chroms:,} original contigs")

    # ── Step 4: remap pairs → pixels ──────────────────────────────────────
    print(f"[4/6] Remapping pairs: {pairs}")
    pixels, total = _pairs_to_lifted_pixels(pairs, index)
    print(f"      → {total:,} remapped contacts, {len(pixels):,} non-zero pixels")

    # ── Step 5: write cool, HiC map, lifted pairs ─────────────────────────
    bins_for_cooler = bins_extended[["chrom", "start", "end"]].copy()
    cool_path = output_dir / f"{cool_name}.cool"
    mcool_path = output_dir / f"{cool_name}.mcool"
    pairs_out_path = output_dir / f"{cool_name}.pairs"
    plot_path = output_dir / f"{cool_name}_hic_map.png"

    print(f"[5/6] Writing outputs to {output_dir}")
    print(f"      → {cool_path.name}  (fragment-level cool)")
    cooler.create_cooler(
        str(cool_path),
        bins=bins_for_cooler,
        pixels=pixels,
        dtypes={"count": np.int32},
        ordered=True,
        symmetric_upper=True,
    )

    print(f"      → {plot_path.name}  (genome-wide Hi-C map)")
    _plot_hic_map(cool_path, plot_path, title=f"{cool_name} — post-assembly Hi-C map")

    print(f"      → {pairs_out_path.name}  (remapped pairs)")
    total_pairs, remapped_pairs = _write_lifted_pairs(pairs, index, bins_extended, pairs_out_path)
    dropped = total_pairs - remapped_pairs
    print(f"         {remapped_pairs:,}/{total_pairs:,} pairs remapped ({dropped:,} dropped)")

    ps_plot_path = output_dir / f"{cool_name}_ps_curves.png"
    print(f"      → {ps_plot_path.name}  (P(s) curves, original vs new)")
    _plot_ps_curves(pairs, pairs_out_path, ps_plot_path)

    # ── Step 6: zoomify ───────────────────────────────────────────────────
    res_label = ",".join(str(r) for r in resolutions_list)
    print(f"      → {mcool_path.name}  (resolutions: {res_label})")
    _zoomify_cooler(cool_path, mcool_path, resolutions_list, balance=balance, balance_args=balance_args)

    # -- Step 7: plot binned Hi-C map
    plot_path = output_dir / f"{cool_name}_hic_map_binned.png"
    fpath = f"{mcool_path}::resolutions/{resolutions_list[0]}"
    print(plot_path)
    print(f"      → {plot_path.name}  (binned Hi-C map at {res_label} bp)")
    _plot_hic_map(fpath, plot_path, title=f"{cool_name} — post-assembly Hi-C map (binned)")

    print("Done.")

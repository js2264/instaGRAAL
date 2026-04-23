"""Click CLI for the ``instagraal-polish`` assembly polishing command."""

import pathlib

import click

from ..parse_info_frags import (
    DEFAULT_CRITERION,
    DEFAULT_CRITERION_2,
    DEFAULT_MIN_SCAFFOLD_SIZE,
    DEFAULT_NEW_GENOME_NAME,
    DEFAULT_NEW_INFO_FRAGS_NAME,
    correct_spurious_inversions,
    find_lost_dna,
    integrate_lost_dna,
    parse_info_frags,
    plot_info_frags,
    rearrange_intra_scaffolds,
    reorient_consecutive_blocks,
    remove_spurious_insertions,
    write_fasta,
    write_info_frags,
)

VALID_MODES = (
    "fasta",
    "singleton",
    "inversion",
    "inversion2",
    "rearrange",
    "reincorporation",
    "polishing",
    "plot",
)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-m",
    "--mode",
    required=True,
    type=click.Choice(VALID_MODES, case_sensitive=False),
    help="Processing mode.",
)
@click.option(
    "-i",
    "--input",
    "info_frags",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help="Input info_frags.txt file to process.",
)
@click.option(
    "-f",
    "--fasta",
    "init_fasta",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help="Reference FASTA file (required for fasta, reincorporation, and polishing modes).",
)
@click.option(
    "-o",
    "--output",
    default=None,
    type=click.Path(path_type=pathlib.Path),
    help="Output file path.",
)
@click.option(
    "-c",
    "--criterion",
    default=None,
    help="Block criterion stringency (used for inversion/inversion2 modes).",
)
@click.option(
    "-s",
    "--min-scaffold-size",
    default=DEFAULT_MIN_SCAFFOLD_SIZE,
    show_default=True,
    type=int,
    help="Minimum scaffold size in bins.",
)
@click.option(
    "-j",
    "--junction",
    default="",
    help="Junction sequence inserted between stitched bins.",
)
def main(
    mode: str,
    info_frags: pathlib.Path,
    init_fasta: pathlib.Path | None,
    output: pathlib.Path | None,
    criterion: str | None,
    min_scaffold_size: int,
    junction: str,
) -> None:
    """Polish and post-process instaGRAAL assemblies.

    Manipulates info_frags.txt files produced by the instaGRAAL scaffolder to
    correct artefact inversions, singletons, rearrangements, and reincorporate
    lost DNA.

    \b
    Available modes:
      fasta           Write a new genome FASTA from info_frags + reference.
      singleton       Remove spurious singleton insertions.
      inversion       Correct spurious inversions (criterion: colinear).
      inversion2      Correct spurious inversions (criterion: blocks).
      rearrange       Rearrange intra-scaffold blocks.
      reincorporation Reincorporate lost DNA from reference.
      polishing       Full polishing pipeline (rearrange + inversion2 + reincorporation + fasta).
      plot            Plot a visual summary of the scaffolds.
    """
    scaffolds = {
        name: scaffold
        for name, scaffold in parse_info_frags(str(info_frags)).items()
        if len(scaffold) > min_scaffold_size
    }

    if mode == "fasta":
        if init_fasta is None:
            raise click.UsageError(
                "A reference FASTA file must be provided (--fasta) for 'fasta' mode."
            )
        write_fasta(
            init_fasta=str(init_fasta),
            info_frags=str(info_frags),
            junction=junction,
            output=str(output) if output else None,
        )

    elif "singleton" in mode:
        new_scaffolds = remove_spurious_insertions(scaffolds)
        write_info_frags(new_scaffolds, output=str(output) if output else None)

    elif mode == "inversion":
        output_file = str(output) if output else DEFAULT_NEW_INFO_FRAGS_NAME
        effective_criterion = criterion or DEFAULT_CRITERION
        new_scaffolds = correct_spurious_inversions(
            scaffolds=scaffolds, criterion=effective_criterion
        )
        write_info_frags(new_scaffolds, output=output_file)

    elif mode == "inversion2":
        output_file = str(output) if output else DEFAULT_NEW_INFO_FRAGS_NAME
        effective_criterion = criterion or DEFAULT_CRITERION_2
        new_scaffolds = reorient_consecutive_blocks(
            scaffolds=scaffolds, mode=effective_criterion
        )
        write_info_frags(new_scaffolds, output=output_file)

    elif "rearrange" in mode:
        output_file = str(output) if output else DEFAULT_NEW_INFO_FRAGS_NAME
        new_scaffolds = rearrange_intra_scaffolds(scaffolds=scaffolds)
        write_info_frags(new_scaffolds, output=output_file)

    elif "reincorporation" in mode:
        if init_fasta is None:
            raise click.UsageError(
                "A reference FASTA file must be provided (--fasta) for 'reincorporation' mode."
            )
        output_file = str(output) if output else DEFAULT_NEW_INFO_FRAGS_NAME
        removed = find_lost_dna(init_fasta=str(init_fasta), scaffolds=scaffolds)
        new_scaffolds = integrate_lost_dna(
            scaffolds=scaffolds, lost_dna_positions=removed
        )
        write_info_frags(new_scaffolds, output=output_file)

    elif "polishing" in mode:
        if init_fasta is None:
            raise click.UsageError(
                "A reference FASTA file must be provided (--fasta) for 'polishing' mode."
            )
        output_file = str(output) if output else DEFAULT_NEW_GENOME_NAME
        arranged_scaffolds = rearrange_intra_scaffolds(scaffolds=scaffolds)
        reoriented_scaffolds = reorient_consecutive_blocks(arranged_scaffolds)
        removed = find_lost_dna(
            init_fasta=str(init_fasta), scaffolds=reoriented_scaffolds
        )
        new_scaffolds = integrate_lost_dna(
            scaffolds=reoriented_scaffolds, lost_dna_positions=removed
        )
        write_info_frags(new_scaffolds, output=DEFAULT_NEW_INFO_FRAGS_NAME)
        write_fasta(
            init_fasta=str(init_fasta),
            info_frags=DEFAULT_NEW_INFO_FRAGS_NAME,
            output=output_file,
            junction=junction,
        )

    elif mode == "plot":
        plot_info_frags(scaffolds)

"""Utilities for parsing mmCIF files and generating Boltz-compatible YAML inputs."""

from dataclasses import dataclass
from pathlib import Path

import yaml
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import is_aa, index_to_one, three_to_index


@dataclass
class ChainInfo:
    """Metadata for a single protein chain extracted from an mmCIF file."""

    chain_id: str
    sequence: str
    n_residues: int


def parse_cif_chains(cif_path: Path) -> list[ChainInfo]:
    """Extract protein chain sequences from an mmCIF file.

    Uses BioPython MMCIFParser with standard residue filtering.
    Returns only protein chains (standard amino acids).
    Skips HETATM, water, and non-standard residues.

    Args:
        cif_path: Path to an mmCIF (.cif) file.

    Returns:
        List of ChainInfo for each protein chain found.

    Raises:
        ValueError: If no protein chains are found in the file.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("structure", str(cif_path))
    model = structure[0]

    chains: list[ChainInfo] = []
    for chain in model:
        residues = []
        for residue in chain:
            if is_aa(residue, standard=True):
                try:
                    idx = three_to_index(residue.get_resname())
                    one_letter = index_to_one(idx)
                    residues.append(one_letter)
                except KeyError:
                    # Skip residues that cannot be converted
                    continue

        if residues:
            sequence = "".join(residues)
            chains.append(
                ChainInfo(
                    chain_id=chain.id,
                    sequence=sequence,
                    n_residues=len(residues),
                )
            )

    if not chains:
        raise ValueError(f"No protein chains found in {cif_path}")

    return chains


def chains_to_boltz_yaml(chains: list[ChainInfo], use_msa: bool = False) -> str:
    """Generate Boltz-compatible YAML content from chain info.

    Args:
        chains: List of ChainInfo objects.
        use_msa: If False (default), sets ``msa: empty`` for single-sequence
            mode. If True, omits the msa field so Boltz uses its default
            MSA search.

    Returns:
        YAML string matching Boltz's expected input format.
    """
    sequences = []
    for chain in chains:
        protein_entry: dict = {
            "id": chain.chain_id,
            "sequence": chain.sequence,
        }
        if not use_msa:
            protein_entry["msa"] = "empty"

        sequences.append({"protein": protein_entry})

    doc = {
        "version": 1,
        "sequences": sequences,
    }

    return yaml.dump(doc, default_flow_style=False, sort_keys=False)

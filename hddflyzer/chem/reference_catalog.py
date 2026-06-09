# SPDX-License-Identifier: LGPL-3.0-or-later

"""Reference molecule catalog and CLI selection helpers."""

from typing import Dict, Optional


REFERENCE_MOLECULES: Dict[str, str] = {
    "rosiglitazone": "CN(CCOC1=CC=C(C=C1)CC2C(=O)NC(=O)S2)C3=CC=CC=N3",
    "pioglitazone":  "CCC1=CN=C(C=C1)CCOC2=CC=C(C=C2)CC3C(=O)NC(=O)S3",
    "rimonabant":    "CC1=C(N(N=C1C(=O)NN2CCCCC2)C3=C(C=C(C=C3)Cl)Cl)C4=CC=C(C=C4)Cl",
    "otenabant":     "CCNC1(CCN(CC1)C2=NC=NC3=C2N=C(N3C4=CC=C(C=C4)Cl)C5=CC=CC=C5Cl)C(=O)N",
    "resveratrol":   "C1=CC(=CC=C1/C=C/C2=CC(=CC(=C2)O)O)O",
}

USER_REFERENCE_TOKEN = "user_reference"


def list_reference_molecules(show_examples: bool = True) -> None:
    """Print built-in reference molecules in a compact CLI-friendly format."""
    print("Available reference molecules or user reference:")
    for i, name in enumerate(REFERENCE_MOLECULES, start=1):
        print(f"  {i}. {name}")
    print(f"  U. {USER_REFERENCE_TOKEN}")
    if show_examples:
        print("\nUse:")
        print("  hddflyzer chem reference-features aocd --reference")


def resolve_reference_name(selection: str) -> Optional[str]:
    """Resolve a reference selection by 1-based number or molecule name."""
    if selection is None:
        return None

    selection = str(selection).strip()
    names = list(REFERENCE_MOLECULES)
    if selection.isdigit():
        idx = int(selection)
        if 1 <= idx <= len(names):
            return names[idx - 1]
        return None

    key = selection.lower()
    if key in ("u", "user", USER_REFERENCE_TOKEN):
        return USER_REFERENCE_TOKEN
    for name in names:
        if key == name.lower():
            return name
    return None


def prompt_reference_name() -> Optional[str]:
    """Ask the CLI user to choose one built-in reference molecule."""
    list_reference_molecules(show_examples=False)
    try:
        choice = input("\nSelect reference molecule (number, name, or U): ").strip()
    except EOFError:
        return None
    return resolve_reference_name(choice)


def prompt_reference_smiles(reference_name: str) -> Optional[str]:
    """Ask the CLI user for a SMILES string for a custom reference."""
    print(f"[INFO] Using custom reference: {reference_name}")
    try:
        smiles = input("Write reference molecule SMILES: ").strip()
    except EOFError:
        return None
    return smiles or None

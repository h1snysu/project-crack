"""Anonymized dataset (D_gen) reader."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


def _is_qi_value(s: str) -> bool:
    """A cell counts as a QI value iff it's an interval string or '*' (ARX
    suppression). Identifier/SA columns (UUIDs, free-text diagnoses, raw
    integers under min1) will never match."""
    s = s.strip().strip('"')
    return s == "*" or (len(s) > 0 and s[0] in "[(")


@dataclass
class Dataset:
    """Header + raw string rows for the QI columns only, paired with k.

    Non-QI columns (identifiers like PATIENT, sensitive attributes like
    Primary_Diagnosis) are dropped at parse time by inspecting cell shape.
    """

    qi_names: list[str]
    rows: list[tuple[str, ...]]
    k: int

    @classmethod
    def from_csv(
        cls,
        path: Path | str,
        k: int,
        qi_columns: list[str] | None = None,
    ) -> "Dataset":
        """Read an anonymised CSV.

        `qi_columns`: if given, restrict to those header names in that order.
        Otherwise auto-detect: a column is a QI iff every one of its values
        is an interval (`[a, b[`) or the suppression marker `*`.
        """
        path = Path(path)
        with path.open(newline="") as f:
            data = [row for row in csv.reader(f) if row]
        if not data:
            raise ValueError(f"Dataset {path} is empty")
        header = [c.strip() for c in data[0]]
        raw_rows = [tuple(c.strip() for c in r) for r in data[1:]]
        bad = [
            i for i, r in enumerate(raw_rows, start=2) if len(r) != len(header)
        ]
        if bad:
            raise ValueError(
                f"Dataset {path}: inconsistent row widths at line(s) "
                f"{bad[:5]}{'...' if len(bad) > 5 else ''}"
            )

        if qi_columns is None:
            qi_indices = [
                i for i in range(len(header))
                if all(_is_qi_value(r[i]) for r in raw_rows)
            ]
            if not qi_indices:
                raise ValueError(
                    f"Dataset {path}: no QI columns detected. Every column "
                    f"contained non-interval values. Pass qi_columns "
                    f"explicitly."
                )
        else:
            try:
                qi_indices = [header.index(c) for c in qi_columns]
            except ValueError:
                raise ValueError(
                    f"Dataset {path}: requested QI column not in header "
                    f"{header}"
                ) from None

        qi_names = [header[i] for i in qi_indices]
        rows = [tuple(r[i] for i in qi_indices) for r in raw_rows]
        return cls(qi_names=qi_names, rows=rows, k=k)

    @property
    def m(self) -> int:
        return len(self.qi_names)

    @property
    def n(self) -> int:
        return len(self.rows)

    def __repr__(self) -> str:
        return (
            f"Dataset(m={self.m}, n={self.n}, k={self.k}, "
            f"qis={self.qi_names!r})"
        )

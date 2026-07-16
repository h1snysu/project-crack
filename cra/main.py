"""Phase 1 smoke test.

Run from the cra/ directory, e.g.:

    python main.py \\
        --dataset ../dataset/min1/anonymized_2qi_k5.csv \\
        --hier-dir ../dataset/min1/hierarchies \\
        -k 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dataset import Dataset
from equivalence import extract_equivalence_classes, sort_equivalence_classes
from hierarchy import Hierarchy


# Map QI names (as they appear in D_gen headers) -> hierarchy CSV filenames.
QI_TO_HIER_FILE: dict[str, str] = {
    "Age": "hierarchy_age.csv",
    "Systolic Blood Pressure": "hierarchy_systolic-blood-pressure.csv",
    "Heart Rate": "hierarchy_heart-rate.csv",
    "Body Weight": "hierarchy_body-weight.csv",
}


def load_hierarchies(qi_names: list[str], hier_dir: Path) -> list[Hierarchy]:
    lookup = {k.lower(): v for k, v in QI_TO_HIER_FILE.items()}
    out: list[Hierarchy] = []
    for i, qi in enumerate(qi_names):
        fname = lookup.get(qi.lower())
        if fname is None:
            raise KeyError(
                f"No hierarchy mapping registered for QI {qi!r}. "
                f"Update QI_TO_HIER_FILE in main.py."
            )
        out.append(Hierarchy.from_csv(hier_dir / fname, qi_name=qi, qi_index=i))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="CRA Phase 1 smoke test")
    parser.add_argument(
        "--dataset", required=True, type=Path,
        help="Path to anonymized D_gen CSV",
    )
    parser.add_argument(
        "--hier-dir", required=True, type=Path,
        help="Directory containing hierarchy CSVs",
    )
    parser.add_argument(
        "-k", required=True, type=int,
        help="k-anonymity parameter",
    )
    parser.add_argument(
        "--show-rows", type=int, default=5,
        help="How many resolved D_gen rows to print (default: 5)",
    )
    parser.add_argument(
        "--show-eqs", type=int, default=10,
        help="How many sorted EQs to print (default: 10)",
    )
    args = parser.parse_args()

    ds = Dataset.from_csv(args.dataset, k=args.k)
    print(ds)

    hiers = load_hierarchies(ds.qi_names, args.hier_dir)
    print("\nHierarchies:")
    for h in hiers:
        assert h.root is not None
        print(
            f"  {h}  root={h.root.value!r}  "
            f"root_qloss={h.root.q_loss:.4f}"
        )
        # Layer-by-layer node counts give a quick feel for tree shape.
        sizes = ", ".join(f"L{l}={len(h.layers[l])}" for l in range(h.height + 1))
        print(f"    layer sizes: {sizes}")

    print(f"\nFirst {args.show_rows} D_gen rows resolved (value @ layer / qloss):")
    for row in ds.rows[: args.show_rows]:
        parts = []
        for cell, h in zip(row, hiers):
            n = h.resolve(cell)
            parts.append(
                f"{h.qi_name}={cell!r} @ L{n.layer} (qloss={n.q_loss:.3f})"
            )
        print("  " + " | ".join(parts))

    # ---- Phase 2: equivalence-class extraction and sorting --------------
    extraction = extract_equivalence_classes(ds, hiers)
    sorted_eqs = sort_equivalence_classes(extraction.eqs)
    outlier_count = extraction.outlier_count

    sizes = [eq.record_count for eq in sorted_eqs] if sorted_eqs else [0]
    losses = [eq.info_loss for eq in sorted_eqs] if sorted_eqs else [0.0]
    print(
        f"\nStandard EQs: {len(sorted_eqs)} unique groups, "
        f"size range [{min(sizes)}, {max(sizes)}], "
        f"loss range [{min(losses):.4f}, {max(losses):.4f}]"
    )
    print(
        f"Outliers (fully-suppressed records, state="
        f"{tuple(h.height for h in hiers)}): |O| = {outlier_count}"
    )
    print(
        f"State-metric cache size: {len(extraction.state_metrics)} unique "
        f"generalisation states (vs {len(sorted_eqs)} EQs)"
    )

    print(f"\nFirst {args.show_eqs} EQs in sorted order (ascending loss/c1/c2):")
    for i, eq in enumerate(sorted_eqs[: args.show_eqs]):
        layers = "/".join(str(l) for l in eq.qi_layers)
        print(
            f"  [{i:>3}] |EQ|={eq.record_count:>4}  loss={eq.info_loss:.4f}  "
            f"c1={eq.c1:>2}  c2={eq.c2:.4f}  layers={layers}  "
            f"values={eq.qi_values}"
        )


if __name__ == "__main__":
    main()

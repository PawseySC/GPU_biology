#!/usr/bin/env python3
"""Correctness checks for OpenFold3 container builds.

Deliberately dependency-light: numpy and the standard library, nothing else.
It runs on a login node without loading the container or an OST environment.

    ./score.py validity     results/correctness/hipblaslt0
    ./score.py determinism  results/determinism/<t>/run_a results/.../run_b
    ./score.py compare      results/correctness/hipblaslt0 results/correctness/hipblaslt1
    ./score.py freeze       results/correctness/hipblaslt0
    ./score.py golden       results/correctness/hipblaslt0

Why geometry and not LDDT/TM-score: the failure mode this suite is built to
catch - the hipBLASLt errata in the OF3-preview2 report - degrades *chemical
validity*, not global fold. A structure can sit at 0.9 LDDT with broken bond
lengths and inverted centres. Global similarity metrics will wave it through.
Bond geometry, clashes and chirality will not.

LDDT/TM-score against experimental structures are a separate job and want
OpenStructure in its own environment. See README.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ideal backbone geometry (Engh & Huber). Tolerance is deliberately loose -
# we are looking for a broken build, not refining a structure.
IDEAL_BONDS = {("N", "CA"): 1.458, ("CA", "C"): 1.525, ("C", "O"): 1.231}
PEPTIDE_CN = 1.329
BOND_TOL = 0.10          # angstroms from ideal before a bond is flagged
CLASH_CUTOFF = 2.0       # heavy-atom separation below this is a clash
DETERMINISM_TOL = 0.01   # max per-atom deviation to call two runs identical

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"
if not sys.stdout.isatty():
    GREEN = RED = YELLOW = RESET = ""


# --------------------------------------------------------------- mmCIF input

def parse_cif(path: Path) -> dict:
    """Minimal _atom_site reader. Columns are looked up by name, so field
    order and any extra columns do not matter."""
    text = path.read_text()
    lines = text.splitlines()

    headers: list[str] = []
    rows: list[list[str]] = []
    i, n = 0, len(lines)

    while i < n:
        if lines[i].strip() == "loop_":
            j = i + 1
            cols = []
            while j < n and lines[j].strip().startswith("_"):
                cols.append(lines[j].strip())
                j += 1
            if cols and cols[0].startswith("_atom_site."):
                headers = [c.split(".", 1)[1] for c in cols]
                while j < n:
                    s = lines[j].strip()
                    # A '#', a new loop, or a new tag ends the table. Blank
                    # lines are skipped rather than treated as terminators -
                    # a stray one should not silently discard every record.
                    if s.startswith("#") or s.startswith("_") or s in ("loop_",) \
                            or s.startswith("data_"):
                        break
                    if s:
                        rows.append(_split_cif(s))
                    j += 1
                break
            i = j
        else:
            i += 1

    if not headers or not rows:
        raise ValueError(f"no _atom_site records in {path}")

    idx = {h: k for k, h in enumerate(headers)}

    def col(name, default=""):
        k = idx.get(name)
        return [r[k] if k is not None and k < len(r) else default for r in rows]

    xyz = np.array(
        [[_f(a), _f(b), _f(c)]
         for a, b, c in zip(col("Cartn_x"), col("Cartn_y"), col("Cartn_z"))],
        dtype=float,
    )

    return {
        "path": path,
        "atom": col("label_atom_id"),
        "comp": col("label_comp_id"),
        "chain": col("label_asym_id"),
        "seq": col("label_seq_id"),
        "elem": col("type_symbol"),
        "bfac": np.array([_f(v) for v in col("B_iso_or_equiv", "0")], dtype=float),
        "xyz": xyz,
    }


def _split_cif(line: str) -> list[str]:
    """Whitespace split that respects single/double quoted values."""
    out, cur, quote = [], "", None
    for ch in line:
        if quote:
            if ch == quote:
                quote = None
            else:
                cur += ch
        elif ch in "'\"":
            quote = ch
        elif ch.isspace():
            if cur:
                out.append(cur)
                cur = ""
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def atom_key(s: dict) -> list[tuple]:
    return list(zip(s["chain"], s["seq"], s["atom"]))


def find_cifs(d: Path) -> list[Path]:
    return sorted(p for p in Path(d).rglob("*.cif") if p.is_file())


# ------------------------------------------------------------ geometry tests

def check_bonds(s: dict) -> tuple[int, int, float]:
    """Intra-residue backbone bonds plus the peptide C-N. Returns
    (n_bad, n_checked, worst_deviation)."""
    by_res: dict[tuple, dict[str, int]] = {}
    for i, (c, q, a) in enumerate(atom_key(s)):
        by_res.setdefault((c, q), {})[a] = i

    xyz = s["xyz"]
    bad = checked = 0
    worst = 0.0

    def measure(i, j, ideal):
        nonlocal bad, checked, worst
        d = float(np.linalg.norm(xyz[i] - xyz[j]))
        dev = abs(d - ideal)
        checked += 1
        worst = max(worst, dev)
        if dev > BOND_TOL:
            bad += 1

    for res in by_res.values():
        for (a, b), ideal in IDEAL_BONDS.items():
            if a in res and b in res:
                measure(res[a], res[b], ideal)

    # Peptide bonds between consecutive residues of the same chain.
    ordered = sorted(by_res, key=lambda k: (k[0], _f(k[1])))
    for (c1, q1), (c2, q2) in zip(ordered, ordered[1:]):
        if c1 == c2 and _f(q2) == _f(q1) + 1:
            r1, r2 = by_res[(c1, q1)], by_res[(c2, q2)]
            if "C" in r1 and "N" in r2:
                measure(r1["C"], r2["N"], PEPTIDE_CN)

    return bad, checked, worst


def check_clashes(s: dict) -> tuple[int, float]:
    """Heavy-atom clashes, excluding same and adjacent residues. Uses a cell
    list so this stays linear rather than building an N^2 distance matrix."""
    heavy = [i for i, e in enumerate(s["elem"]) if e.upper() not in ("H", "D")]
    if not heavy:
        return 0, float("inf")

    xyz = s["xyz"][heavy]
    chain = [s["chain"][i] for i in heavy]
    seq = [_f(s["seq"][i]) for i in heavy]

    cell = CLASH_CUTOFF
    keys = np.floor(xyz / cell).astype(int)
    grid: dict[tuple, list[int]] = {}
    for i, k in enumerate(map(tuple, keys)):
        grid.setdefault(k, []).append(i)

    offsets = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    clashes, closest = 0, float("inf")
    seen: set[tuple[int, int]] = set()

    for k, members in grid.items():
        near: list[int] = []
        for off in offsets:
            near.extend(grid.get((k[0] + off[0], k[1] + off[1], k[2] + off[2]), ()))
        for i in members:
            for j in near:
                if j <= i:
                    continue
                pair = (i, j)
                if pair in seen:
                    continue
                seen.add(pair)
                # Skip bonded-ish pairs: same residue, or sequence neighbours.
                if chain[i] == chain[j] and abs(seq[i] - seq[j]) <= 1:
                    continue
                d = float(np.linalg.norm(xyz[i] - xyz[j]))
                closest = min(closest, d)
                if d < CLASH_CUTOFF:
                    clashes += 1

    return clashes, closest


# Scale-free chirality volume below this is treated as a flat centre rather
# than as a handedness. A proper tetrahedral CA sits near 0.5.
CHIRAL_EPS = 0.05


def check_chirality(s: dict) -> tuple[int, int, int]:
    """CA centres should all share one handedness - natural residues are L.

    Rather than asserting a sign convention, this reports *consistency*: a
    structure where some centres are inverted relative to the majority is
    wrong regardless of which sign you call correct.

    Returns (n_inverted, n_measured, n_degenerate). Degenerate means the four
    substituents are close to coplanar, so the sign carries no information -
    counting those as inversions would be a false positive, but a pile of them
    is itself a signal that the geometry has collapsed.
    """
    by_res: dict[tuple, dict[str, int]] = {}
    for i, (c, q, a) in enumerate(atom_key(s)):
        by_res.setdefault((c, q), {})[a] = i

    xyz = s["xyz"]
    vols: list[float] = []
    for res in by_res.values():
        if all(a in res for a in ("N", "CA", "C", "CB")):
            ca = xyz[res["CA"]]
            v1, v2, v3 = (xyz[res[a]] - ca for a in ("N", "C", "CB"))
            norms = np.linalg.norm(v1) * np.linalg.norm(v2) * np.linalg.norm(v3)
            if norms > 0:
                vols.append(float(np.dot(np.cross(v1, v2), v3) / norms))

    if not vols:
        return 0, 0, 0

    v = np.array(vols)
    solid = v[np.abs(v) >= CHIRAL_EPS]
    degenerate = int(len(v) - len(solid))
    if len(solid) == 0:
        return 0, len(v), degenerate

    majority = 1.0 if (solid > 0).sum() >= (solid < 0).sum() else -1.0
    inverted = int((np.sign(solid) != majority).sum())
    return inverted, len(v), degenerate


# ------------------------------------------------------------------ commands

def cmd_validity(dirs: list[str]) -> int:
    rows, failed = [], 0
    for d in dirs:
        for cif in find_cifs(Path(d)):
            try:
                s = parse_cif(cif)
            except Exception as e:  # noqa: BLE001
                rows.append((cif.name, "PARSE", str(e)[:40], "", "", ""))
                failed += 1
                continue

            nbad, nchk, worst = check_bonds(s)
            clash, closest = check_clashes(s)
            inv, ntot, degen = check_chirality(s)
            plddt = float(s["bfac"].mean()) if len(s["bfac"]) else 0.0

            # Degenerate centres are not counted as inversions, but if most of
            # them are flat the backbone has collapsed and that is a failure.
            flat = ntot > 0 and degen > ntot / 2
            ok = nbad == 0 and clash == 0 and inv == 0 and not flat
            failed += 0 if ok else 1
            chir = f"{inv}/{ntot}" + (f" ({degen} flat)" if degen else "")
            rows.append((
                cif.name,
                "PASS" if ok else "FAIL",
                f"{nbad}/{nchk} (max {worst:.3f}A)",
                f"{clash} (min {closest:.2f}A)" if closest != float("inf") else "0",
                chir,
                f"{plddt:.1f}",
            ))

    _table(["structure", "result", "bad bonds", "clashes", "inverted CA", "mean B"], rows)
    print(f"\n{len(rows)} structure(s), {failed} failing")
    return 1 if failed else 0


def cmd_determinism(a: str, b: str) -> int:
    ca, cb = find_cifs(Path(a)), find_cifs(Path(b))
    if not ca or not cb:
        print(f"{RED}no .cif outputs found in one or both runs{RESET}")
        return 1

    rows, failed = [], 0
    for pa, pb in zip(ca, cb):
        sa, sb = parse_cif(pa), parse_cif(pb)
        ka, kb = atom_key(sa), atom_key(sb)

        if ka != kb:
            common = set(ka) & set(kb)
            ia = [i for i, k in enumerate(ka) if k in common]
            ib = [i for i, k in enumerate(kb) if k in common]
            note = f"atom sets differ ({len(ka)} vs {len(kb)})"
        else:
            ia = ib = list(range(len(ka)))
            note = ""

        d = np.linalg.norm(sa["xyz"][ia] - sb["xyz"][ib], axis=1)
        maxdev, rmsd = float(d.max()), float(np.sqrt((d ** 2).mean()))
        exact = bool((d == 0).all())
        ok = maxdev < DETERMINISM_TOL
        failed += 0 if ok else 1

        rows.append((
            pa.name,
            "PASS" if ok else "FAIL",
            "yes" if exact else "no",
            f"{maxdev:.6f}",
            f"{rmsd:.6f}",
            note,
        ))

    _table(["structure", "result", "bitwise", "max dev A", "rmsd A", "note"], rows)
    if failed:
        print(f"\n{RED}Nondeterministic.{RESET} Golden-output comparison is not "
              f"usable until this passes. Suspect a nondeterministic reduction "
              f"or kernel autotuning varying between runs.")
    return 1 if failed else 0


def cmd_compare(a: str, b: str) -> int:
    """Structural diff between two result trees - the hipBLASLt A/B."""
    da = {p.name: p for p in find_cifs(Path(a))}
    db = {p.name: p for p in find_cifs(Path(b))}
    shared = sorted(set(da) & set(db))
    if not shared:
        print(f"{RED}no structures in common between {a} and {b}{RESET}")
        return 1

    rows = []
    for name in shared:
        sa, sb = parse_cif(da[name]), parse_cif(db[name])
        common = set(atom_key(sa)) & set(atom_key(sb))
        ia = [i for i, k in enumerate(atom_key(sa)) if k in common]
        ib = [i for i, k in enumerate(atom_key(sb)) if k in common]
        d = np.linalg.norm(sa["xyz"][ia] - sb["xyz"][ib], axis=1)

        bad_a, _, _ = check_bonds(sa)
        bad_b, _, _ = check_bonds(sb)
        cl_a, _ = check_clashes(sa)
        cl_b, _ = check_clashes(sb)

        rows.append((name, f"{float(d.max()):.3f}",
                     f"{float(np.sqrt((d ** 2).mean())):.3f}",
                     f"{bad_a} -> {bad_b}", f"{cl_a} -> {cl_b}"))

    _table(["structure", "max dev A", "rmsd A", "bad bonds A->B", "clashes A->B"], rows)
    print("\nLarge coordinate differences alone are expected between backends.\n"
          "What matters is whether bond/clash counts get worse - that is the\n"
          "chemical-validity regression the OF3 errata describes.")
    return 0


def cmd_freeze(src: str) -> int:
    import shutil
    dst = Path("golden")
    dst.mkdir(exist_ok=True)
    n = 0
    for cif in find_cifs(Path(src)):
        rel = cif.relative_to(src)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cif, out)
        n += 1
    print(f"froze {n} structure(s) from {src} into {dst}/")
    print("Commit these. Every later build is diffed against them.")
    return 0


def cmd_golden(src: str) -> int:
    golden = Path("golden")
    if not golden.exists() or not find_cifs(golden):
        print(f"{YELLOW}no golden references yet.{RESET} Once you trust a build:\n"
              f"  ./score.py freeze {src}")
        return 0

    # Keyed by path relative to the results root, matching how freeze stores
    # them - so two queries that happen to emit the same filename in different
    # subdirectories cannot be compared against each other.
    rows, failed = [], 0
    for cif in find_cifs(Path(src)):
        rel = cif.relative_to(src)
        ref = golden / rel
        if not ref.exists():
            rows.append((str(rel), "NEW", "", ""))
            continue
        sa, sb = parse_cif(ref), parse_cif(cif)
        common = set(atom_key(sa)) & set(atom_key(sb))
        ia = [i for i, k in enumerate(atom_key(sa)) if k in common]
        ib = [i for i, k in enumerate(atom_key(sb)) if k in common]
        d = np.linalg.norm(sa["xyz"][ia] - sb["xyz"][ib], axis=1)
        maxdev = float(d.max())
        ok = maxdev < DETERMINISM_TOL
        failed += 0 if ok else 1
        rows.append((str(rel), "PASS" if ok else "FAIL", f"{maxdev:.6f}",
                     f"{float(np.sqrt((d ** 2).mean())):.6f}"))

    _table(["structure", "result", "max dev A", "rmsd A"], rows)
    print(f"\n{len(rows)} compared, {failed} drifted from golden")
    return 1 if failed else 0


def _table(headers: list[str], rows: list[tuple]) -> None:
    if not rows:
        print("(nothing to report)")
        return
    cells = [headers] + [[str(c) for c in r] for r in rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        line = []
        for i, c in enumerate(r):
            c = str(c)
            colour = GREEN if c == "PASS" else RED if c in ("FAIL", "PARSE") else ""
            line.append(f"{colour}{c}{RESET if colour else ''}".ljust(
                widths[i] + (len(colour) + len(RESET) if colour else 0)))
        print("  ".join(line))


COMMANDS = {
    "validity": (cmd_validity, "geometry checks on predicted structures"),
    "determinism": (cmd_determinism, "compare two runs of the same input"),
    "compare": (cmd_compare, "diff two result trees (e.g. BLAS backend A/B)"),
    "freeze": (cmd_freeze, "promote results to golden references"),
    "golden": (cmd_golden, "check results against frozen golden references"),
}


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print("commands:")
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:<12} {desc}")
        return 2
    fn = COMMANDS[sys.argv[1]][0]
    args = sys.argv[2:]
    if sys.argv[1] == "validity":
        return fn(args)
    return fn(*args)


if __name__ == "__main__":
    sys.exit(main())

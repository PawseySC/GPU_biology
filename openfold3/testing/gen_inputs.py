#!/usr/bin/env python3
"""Generate OF3 query JSONs for the container test suite."""
import json
import string
from pathlib import Path

BASE = Path(__file__).resolve().parent / "inputs"

# Sequences taken verbatim from aqlaboratory/openfold-3
# examples/example_inference_inputs/ so the format is known-good.
UBQ = ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDY"
       "NIQKESTLHLVLRLRGG")  # 1UBQ, 76 res
MCL1 = ("GDDELYRQSLEIISRYLREQATGAKDTKPMGRSGATSRKALETLRRVGDGVQRNHETAFQGMLRKLD"
        "IKNEDDVKSLSRVMIHVFSDGVTNWGRIVTLISFGAFVAKHLKTINQESCIEPLAESITDVLVRTKR"
        "DWLVKQRGWDGFVEFFHVEDLEGG")
CNX_A = ("MLNSFKLSLQYILPKLWLTRLAGWGASKRAGWLTKLVIDLFVKYYKVDMKEAQKPDTASYRTFNEFF"
         "VRPLRDEVRPIDTDPNVLVMPADGVISQLGKIEEDKILQAKGHNYSLEALLAGNYLMADLFRNGTFV"
         "TTYLSPRDYHRVHMPCNGILREMIYVPGDLFSVNHLTAQNVPNLFARNERVICLFDTEFGPMAQILV"
         "GATIVGSIETVWAGTITPPREGIIKRWTWPAGENDGSVALLKGQEMGRFKLG")
CNX_B = "XTVINLFAPGKVNLVEQLESLSVTKIGQPLAVSTGHHHHHHG"
DNA13 = "ATUCGTATTCGAT"
# UUCG tetraloop hairpin - small, well-behaved RNA fold
RNA14 = "GGCACUUCGGUGCC"


def chain_ids(n):
    """A..Z, then AA, AB, ... for n chains."""
    out, letters = [], string.ascii_uppercase
    for i in range(n):
        if i < 26:
            out.append(letters[i])
        else:
            out.append(letters[i // 26 - 1] + letters[i % 26])
    return out


def write(path, name, chains):
    """Emit an InferenceQuerySet.

    seeds is pinned so runs are reproducible; use_msas is off so the test is
    hermetic (no MSA databases, no network) and the only thing under test is
    the model. Flip use_msas to true to exercise the alignment path as well.
    """
    doc = {
        "seeds": [42],
        "queries": {name: {"chains": chains, "use_msas": False}},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"  {path.relative_to(BASE.parent)}")


# ---------------------------------------------------------------- correctness
# One target per modality. Chosen to be small and easy so that a bad result
# means a bad build, not a hard target.
print("correctness:")
C = BASE / "correctness"

write(C / "01_protein_monomer.json", "1ubq",
      [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": UBQ}])

write(C / "02_protein_multimer.json", "7cnx",
      [{"molecule_type": "protein", "chain_ids": ["A", "C"], "sequence": CNX_A},
       {"molecule_type": "protein", "chain_ids": ["B", "D"], "sequence": CNX_B}])

write(C / "03_protein_ligand_ccd.json", "mcl1_atp",
      [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": MCL1},
       {"molecule_type": "ligand", "chain_ids": ["F"], "ccd_codes": "ATP"}])

write(C / "04_protein_ligand_smiles.json", "mcl1_smiles",
      [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": MCL1},
       {"molecule_type": "ligand", "chain_ids": "Z",
        "smiles": "CC(=O)OC1C[NH+]2CCC1CC2"}])

write(C / "05_rna_monomer.json", "rna_hairpin",
      [{"molecule_type": "rna", "chain_ids": "A", "sequence": RNA14}])

write(C / "06_dna_ptm.json", "ptm_dna",
      [{"molecule_type": "dna", "chain_ids": "A", "sequence": DNA13,
        "non_canonical_residues": {"3": "PSU", "4": "5MC"}}])

write(C / "07_protein_dna.json", "ubq_dna",
      [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": UBQ},
       {"molecule_type": "dna", "chain_ids": ["B"], "sequence": DNA13}])

write(C / "08_protein_rna.json", "ubq_rna",
      [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": UBQ},
       {"molecule_type": "rna", "chain_ids": ["B"], "sequence": RNA14}])

# ----------------------------------------------------------------------- perf
# Token ladder by ubiquitin copy count: 76 tokens per copy.
print("perf:")
P = BASE / "perf"
for n in (1, 2, 4, 8, 16, 32):
    write(P / f"ladder_{n:02d}x_{n * 76}tok.json", f"ubq_{n}mer",
          [{"molecule_type": "protein", "chain_ids": chain_ids(n), "sequence": UBQ}])

# Mixed-modality points: same protein token count, non-protein tokens added,
# so the memory delta is attributable to the ligand/nucleic path.
write(P / "mixed_protein_ligand.json", "mcl1_4x_atp",
      [{"molecule_type": "protein", "chain_ids": ["A", "B", "C", "D"], "sequence": MCL1},
       {"molecule_type": "ligand", "chain_ids": ["F", "G", "H"], "ccd_codes": "ATP"}])

write(P / "mixed_protein_rna.json", "ubq_8x_rna",
      [{"molecule_type": "protein", "chain_ids": chain_ids(8), "sequence": UBQ},
       {"molecule_type": "rna", "chain_ids": ["Z"], "sequence": RNA14}])

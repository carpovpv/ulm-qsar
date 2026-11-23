from rdkit import Chem, RDLogger
from rdkit.Chem.SaltRemover import SaltRemover
import numpy as np

RDLogger.DisableLog("rdApp.*")
remover = SaltRemover()

token_lengths = []

'''
with open("../data/smiles.smi", "r", encoding="utf-8") as f:
    for line in f:
        raw_smi = line.strip().split()[0] if line.strip() else ""

        if not raw_smi:
            continue

        mol = Chem.MolFromSmiles(raw_smi)
        if mol is None:
            continue

        if remover is not None:
            mol = remover.StripMol(mol, dontRemoveEverything=True)
            if mol is None or mol.GetNumAtoms() == 0:
                continue

        canon_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        print(canon_smi)

        token_lengths.append( len(canon_smi) )
'''

for line in open("canon.smi", "r").readlines():
    line = line.strip()
    if len(line) <= 126:
        print(line)

    token_lengths.append(len(line))

lengths = np.array(token_lengths, dtype=np.int32)
total = len(lengths)

p90 = int(np.percentile(lengths, 90))
p95 = int(np.percentile(lengths, 95))
p99 = int(np.percentile(lengths, 99))
max_len_found = int(lengths.max())

print(f"Min / Median / Max len:   {lengths.min()} / {int(np.median(lengths))} / {max_len_found}")
print(f"90th percentile:          {p90} tokens")
print(f"95th percentile:          {p95} tokens")
print(f"99th percentile:          {p99} tokens")



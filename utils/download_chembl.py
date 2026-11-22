import chembl_downloader

sql = """
SELECT DISTINCT cs.canonical_smiles
FROM molecule_dictionary md
JOIN compound_structures cs ON md.molregno = cs.molregno
WHERE md.molecule_type = 'Small molecule'
  AND cs.canonical_smiles IS NOT NULL
"""

with chembl_downloader.connect() as conn:
    cursor = conn.cursor()
    cursor.execute(sql)
    with open("chembl_small_molecules.smi", "w", encoding="utf-8") as f:
        for (smiles,) in cursor:
            f.write(f"{smiles}\n")


"""Sumber #1 kamus entity: cari kandidat entity dari korpus sendiri (60 dokumen).

Mencari urutan kata berawal huruf kapital (mis. "Marc Marquez", "Sirkuit Mandalika") yang muncul di >= MIN_DF
dokumen dan BELUM ada di data/entities.json. Output untuk DIREVIEW MANUAL, lalu salin yang valid ke entities.json.

Pakai:  python build_entity_candidates.py data/IR_Corpus_Kelompok_MotoGP.xlsx
"""
import re
import sys
from collections import Counter

import pandas as pd

from entities import EntityDictionary

MIN_DF = 2
CAP_RX = re.compile(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,2}\b")
STOP_CAPS = {"Baca", "Jakarta", "Kompas", "Artikel", "Lihat", "Foto", "Iklan", "Pada", "Dalam", "Dengan",
             "Sementara", "Menurut", "Selain", "Namun", "Setelah", "Saat", "Hasil"}


def main(path: str, sheet: str = "Corpus", entities: str = "data/entities.json"):
    try:
        df = pd.read_excel(path, sheet_name=sheet)
    except ValueError:
        df = pd.read_excel(path, sheet_name=0)
    df = df.dropna(subset=["text"])
    known = EntityDictionary(entities)

    df_count = Counter()
    for _, row in df.iterrows():
        text = f"{row.get('title', '')}. {row['text']}"
        cands = {m.group(0) for m in CAP_RX.finditer(text)}
        df_count.update(c for c in cands if c.split()[0] not in STOP_CAPS)

    print(f"{'kandidat':35s} df  (belum ada di kamus)")
    for cand, n in df_count.most_common():
        if n < MIN_DF:
            break
        if cand.lower() in known.alias_map:
            continue
        print(f"{cand:35s} {n}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/IR_Corpus_Kelompok_MotoGP.xlsx")
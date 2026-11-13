"""Evaluasi: baseline (TF-IDF murni) vs sistem hybrid (TF-IDF + Entity).

LANGKAH
1) Edit data/eval_queries.csv (query_id, query) -> 8-10 query yang mewakili kebutuhan pencarian.
2) python evaluate.py pool
     -> data/judgments.csv : kumpulan dokumen kandidat (gabungan top-K semua sistem, "pooling").
     Kelompok MEMBACA tiap dokumen dan mengisi kolom `relevan` (1 = relevan, 0 = tidak).
     Beri aturan yang sama untuk semua orang, mis. "relevan jika artikel membahas topik/entity query secara utama".
3) python evaluate.py run
     -> results/eval_report.md + results/eval_per_query.csv (metrik, perbandingan, analisis kegagalan)

Asumsi pooling: dokumen yang tidak masuk pool dianggap tidak relevan (tulis di laporan sebagai keterbatasan).
"""
import sys
from pathlib import Path

import pandas as pd

from search_engine import HybridSearchEngine

CORPUS = "data/IR_Corpus_Kelompok_MotoGP.xlsx"
QUERIES = "data/eval_queries.csv"
JUDGMENTS = "data/judgments.csv"
POOL_K = 10          # kedalaman pool per sistem
K_LIST = (5, 10)     # Precision@k / Recall@k

# nama sistem -> argumen engine.search
SYSTEMS = {
    "Baseline (TF-IDF)":              dict(w_text=1.0, w_entity=0.0, expand_query=False),
    "Hybrid 0.8/0.2 (tanpa expand)":  dict(w_text=0.8, w_entity=0.2, expand_query=False),
    "Hybrid 0.8/0.2 (+expand)":       dict(w_text=0.8, w_entity=0.2, expand_query=True),
}
SWEEP = [(1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5)]
MAIN = "Hybrid 0.8/0.2 (+expand)"
BASE = "Baseline (TF-IDF)"


def ranking(engine, query, **kw):
    out = engine.search(query, top_k=len(engine.df), **kw)
    return [r["doc_id"] for r in out["results"]]


def metrics(ranked, rel):
    m = {}
    for k in K_LIST:
        hit = sum(d in rel for d in ranked[:k])
        p, r = hit / k, (hit / len(rel) if rel else 0.0)
        m[f"P@{k}"], m[f"R@{k}"] = p, r
        m[f"F1@{k}"] = 2 * p * r / (p + r) if p + r else 0.0
    hits, ap = 0, 0.0
    for i, d in enumerate(ranked, 1):
        if d in rel:
            hits += 1
            ap += hits / i
    m["AP"] = ap / len(rel) if rel else 0.0
    return m


def load_queries():
    q = pd.read_csv(QUERIES)
    return q.dropna(subset=["query"]).reset_index(drop=True)


def pool(engine):
    rows = []
    for _, q in load_queries().iterrows():
        cands = {}
        for name, kw in SYSTEMS.items():
            for d in ranking(engine, q["query"], **kw)[:POOL_K]:
                cands.setdefault(d, []).append(name)
        for d in sorted(cands):
            r = engine.df[engine.df.doc_id == d].iloc[0]
            rows.append(dict(query_id=q["query_id"], query=q["query"], doc_id=d, title=r["title"],
                             snippet=str(r["text"])[:200].replace("\n", " "), relevan=""))
    old = Path(JUDGMENTS)
    new = pd.DataFrame(rows)
    if old.exists():      # jangan timpa penilaian yang sudah diisi
        prev = pd.read_csv(old, dtype=str).fillna("")
        key = prev.set_index(["query_id", "doc_id"])["relevan"].to_dict()
        new["relevan"] = [key.get((a, b), "") for a, b in zip(new.query_id, new.doc_id)]
    new.to_csv(JUDGMENTS, index=False)
    print(f"{len(new)} baris -> {JUDGMENTS}. Isi kolom `relevan` (1/0), lalu: python evaluate.py run")


def run(engine):
    j = pd.read_csv(JUDGMENTS, dtype=str).fillna("")
    if (j["relevan"] == "").any():
        print(f"[PERINGATAN] {(j['relevan'] == '').sum()} baris belum dinilai (dianggap 0).")
    rel_by_q = {q: set(g.loc[g.relevan == "1", "doc_id"]) for q, g in j.groupby("query_id")}
    queries = load_queries()
    queries = queries[queries.query_id.map(lambda q: len(rel_by_q.get(q, ())) > 0)]
    if queries.empty:
        sys.exit("Belum ada dokumen relevan (relevan=1) di judgments.csv.")

    per_q, rows = {}, []
    for name, kw in SYSTEMS.items():
        for _, q in queries.iterrows():
            ranked = ranking(engine, q["query"], **kw)
            per_q[(name, q.query_id)] = ranked
            rows.append(dict(system=name, query_id=q.query_id, query=q["query"],
                             n_relevan=len(rel_by_q[q.query_id]),
                             **metrics(ranked, rel_by_q[q.query_id])))
    res = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    res.to_csv("results/eval_per_query.csv", index=False)
    cols = [c for c in res.columns if c.startswith(("P@", "R@", "F1@")) or c == "AP"]
    mean = res.groupby("system")[cols].mean().reindex(list(SYSTEMS)).round(3)
    mean = mean.rename(columns={"AP": "MAP"})

    # sweep bobot
    sw = []
    for wt, we in SWEEP:
        vals = [metrics(ranking(engine, q["query"], w_text=wt, w_entity=we, expand_query=True),
                        rel_by_q[q.query_id]) for _, q in queries.iterrows()]
        sw.append(dict(w_text=wt, w_entity=we, **{k: round(sum(v[k] for v in vals) / len(vals), 3)
                                                  for k in ("P@5", "P@10", "R@10", "F1@10", "AP")}))
    sw = pd.DataFrame(sw).rename(columns={"AP": "MAP"})

    # per query: baseline vs hybrid
    cmp = res.pivot(index=["query_id", "query"], columns="system", values="AP")[[BASE, MAIN]].round(3)
    cmp["selisih AP"] = (cmp[MAIN] - cmp[BASE]).round(3)

    L = ["# Laporan Evaluasi (otomatis)\n",
         f"Jumlah query dinilai: {len(queries)} | pool: top-{POOL_K} dari tiap sistem | dokumen di luar pool dianggap tidak relevan.\n",
         "## 1. Rata-rata metrik per sistem\n", mean.to_markdown(), "\n",
         "## 2. Baseline vs Hybrid per query (Average Precision)\n", cmp.to_markdown(), "\n",
         "## 3. Sweep bobot (hybrid + expand)\n", sw.to_markdown(index=False), "\n",
         "## 4. Analisis kegagalan (sistem utama)\n"]
    for _, q in queries.iterrows():
        rel, ranked = rel_by_q[q.query_id], per_q[(MAIN, q.query_id)]
        titles = engine.df.set_index("doc_id")["title"]
        missed = [d for d in rel if d not in ranked[:10]]
        bad = [d for d in ranked[:5] if d not in rel]
        L.append(f"**{q.query_id} - {q['query']}**")
        L.append("- Relevan tidak masuk top-10: " + (", ".join(f"{d} ({titles[d][:50]})" for d in missed) or "tidak ada"))
        L.append("- Tidak relevan di top-5: " + (", ".join(f"{d} ({titles[d][:50]})" for d in bad) or "tidak ada"))
        L.append("")
    L.append("_Tugas kelompok: jelaskan KENAPA tiap kegagalan terjadi (kosakata beda, entity tidak ada di kamus, alias ambigu, dst.)._")
    Path("results/eval_report.md").write_text("\n".join(L), encoding="utf-8")
    print(mean.to_string(), "\n\n", sw.to_string(index=False), "\n\n-> results/eval_report.md")


if __name__ == "__main__":
    eng = HybridSearchEngine(CORPUS)
    pool(eng) if (sys.argv[1:] or ["run"])[0] == "pool" else run(eng)
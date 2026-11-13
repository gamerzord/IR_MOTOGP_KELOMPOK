"""Hybrid search: Text Retrieval (TF-IDF + cosine)  ||  Entity Retrieval  ->  Score Combination -> Final Ranking."""
import argparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from entities import EntityDictionary, EntityIndex
from preprocessing import clean_text, preprocess_pipeline


class HybridSearchEngine:
    def __init__(self, corpus_path: str, entities_path: str = "data/entities.json",
                 sheet: str = "Corpus", w_text: float = 0.8, w_entity: float = 0.2):
        self.w_text, self.w_entity = w_text, w_entity

        # ---------- 1. Load korpus ----------
        if corpus_path.endswith(".csv"):
            df = pd.read_csv(corpus_path)
        else:
            try:
                df = pd.read_excel(corpus_path, sheet_name=sheet)
            except ValueError:
                df = pd.read_excel(corpus_path, sheet_name=0)
        df = df.dropna(subset=["text"]).reset_index(drop=True)
        df["doc_id"] = df["doc_id"].astype(str)
        df["title"] = df["title"].fillna("").astype(str).str.strip()
        df["isi"] = df["title"] + ". " + df["text"].astype(str)
        self.df = df

        # ---------- 2. Index A: TF-IDF ----------
        self.tokens = df["isi"].apply(preprocess_pipeline)
        self.vectorizer = TfidfVectorizer(analyzer=lambda toks: toks)
        self.X = self.vectorizer.fit_transform(self.tokens)

        # ---------- 3. Index B: Entity ----------
        self.dictionary = EntityDictionary(entities_path)
        self.entity_index = EntityIndex(
            self.dictionary, list(zip(df["doc_id"], df["title"], df["text"].astype(str))))

    def _cell(self, i: int, col: str) -> str:
        if col not in self.df or pd.isna(self.df.loc[i, col]):
            return ""
        v = self.df.loc[i, col]
        return v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else str(v)

    def search_baseline(self, query: str, top_k: int = 10) -> dict:
        """BASELINE untuk evaluasi: TF-IDF + cosine murni (tanpa entity, tanpa perluasan query)."""
        return self.search(query, top_k=top_k, w_text=1.0, w_entity=0.0, expand_query=False)

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 10, w_text: float | None = None,
               w_entity: float | None = None, expand_query: bool = True) -> dict:
        w_text = self.w_text if w_text is None else w_text
        w_entity = self.w_entity if w_entity is None else w_entity

        # ---------- Query preprocessing ----------
        q_tokens = preprocess_pipeline(query)
        q_entities = self.dictionary.find(clean_text(query))      # {entity_id: n}

        # ---------- A. Text retrieval ----------
        # "MM93" tidak ada di teks berita, tapi "marc marquez" ada -> tambahkan nama kanonik entity
        # ke query teks supaya sisi TF-IDF ikut tertolong.
        text_tokens = list(q_tokens)
        if expand_query:
            for eid in q_entities:
                for t in preprocess_pipeline(self.dictionary.name(eid)):
                    if t not in text_tokens:
                        text_tokens.append(t)
        s_text = cosine_similarity(self.vectorizer.transform([text_tokens]), self.X).ravel()

        # ---------- B. Entity retrieval ----------
        doc_ids = self.df["doc_id"].tolist()
        if q_entities:
            s_entity = [
                sum(self.entity_index.match_score(eid, d) for eid in q_entities) / len(q_entities)
                for d in doc_ids
            ]
            wt, we = w_text, w_entity
        else:
            # Query tidak menyebut entity apa pun -> sisi entity tidak punya informasi;
            # jangan "menghukum" semua dokumen, pakai 100% skor teks.
            s_entity = [0.0] * len(doc_ids)
            wt, we = 1.0, 0.0

        # ---------- Score combination ----------
        rows = []
        for i, d in enumerate(doc_ids):
            final = wt * s_text[i] + we * s_entity[i]
            if final <= 0:
                continue
            rows.append({
                "doc_id": d,
                "title": self.df.loc[i, "title"],
                "media": self.df.loc[i, "media"] if "media" in self.df else "",
                "date": self._cell(i, "date"),
                "url": self._cell(i, "url"),
                "snippet": str(self.df.loc[i, "text"])[:220],
                "s_text": round(float(s_text[i]), 4),
                "s_entity": round(float(s_entity[i]), 4),
                "s_final": round(float(final), 4),
                "entities": [self.dictionary.name(e) for e in self.entity_index.doc_entities(d)],
            })
        rows.sort(key=lambda r: (-r["s_final"], r["doc_id"]))
        for rank, r in enumerate(rows[:top_k], 1):
            r["rank"] = rank

        return {
            "query": query,
            "query_tokens": q_tokens,
            "entities_detected": [self.dictionary.name(e) for e in q_entities],
            "weights": {"text": wt, "entity": we},
            "results": rows[:top_k],
        }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--corpus", default="data/IR_Corpus_Kelompok_MotoGP.xlsx")
    ap.add_argument("--entities", default="data/entities.json")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()

    eng = HybridSearchEngine(a.corpus, a.entities)
    out = eng.search(a.query, top_k=a.top)
    print("tokens   :", out["query_tokens"])
    print("entities :", out["entities_detected"], "| bobot:", out["weights"])
    for r in out["results"]:
        print(f"{r['rank']}. {r['doc_id']} final={r['s_final']:.3f} (text={r['s_text']:.3f}, "
              f"entity={r['s_entity']:.3f})  {r['title']}")
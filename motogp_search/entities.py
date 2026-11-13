"""Entity dictionary + entity index.

Entity dictionary : kamus  entity -> {nama kanonik, tipe, alias}      (dari data/entities.json)
Entity index      : entity -> {doc_id: (jumlah kemunculan, ada di judul?)}  (dibangun dari korpus)
"""
import json
import re
from collections import Counter, defaultdict

from preprocessing import clean_text


class EntityDictionary:
    def __init__(self, path: str):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.entities = {e["id"]: e for e in data["entities"]}

        # alias -> [entity_id, ...]   (satu alias boleh menunjuk >1 entity, mis. "marquez")
        self.alias_map: dict[str, list[str]] = defaultdict(list)
        for eid, e in self.entities.items():
            for alias in {e["name"], *e.get("aliases", [])}:
                a = alias.lower().strip()
                if eid not in self.alias_map[a]:
                    self.alias_map[a].append(eid)

        # Alias terpanjang dicoba dulu, supaya "marc marquez" tidak ikut memicu "marquez"
        aliases = sorted(self.alias_map, key=len, reverse=True)
        pattern = "|".join(re.escape(a) for a in aliases)
        self._rx = re.compile(rf"(?<![a-z0-9])(?:{pattern})(?![a-z0-9])")

    def find(self, text: str) -> Counter:
        """Kembalikan Counter {entity_id: jumlah kemunculan} pada teks (teks akan di-lowercase)."""
        found = Counter()
        for m in self._rx.finditer(text.lower()):
            for eid in self.alias_map[m.group(0)]:
                found[eid] += 1
        return found

    def name(self, eid: str) -> str:
        return self.entities[eid]["name"]


class EntityIndex:
    """Inverted index entity -> dokumen."""

    def __init__(self, dictionary: EntityDictionary, docs: list[tuple[str, str, str]]):
        """docs = [(doc_id, title, text), ...] (teks mentah)."""
        self.dictionary = dictionary
        self.postings: dict[str, dict[str, dict]] = defaultdict(dict)
        for doc_id, title, text in docs:
            in_title = dictionary.find(clean_text(title))
            in_body = dictionary.find(clean_text(text))
            for eid in set(in_title) | set(in_body):
                self.postings[eid][doc_id] = {
                    "count": in_body[eid] + in_title[eid],
                    "in_title": in_title[eid] > 0,
                }

    def doc_entities(self, doc_id: str) -> list[str]:
        return [eid for eid, p in self.postings.items() if doc_id in p]

    def match_score(self, eid: str, doc_id: str) -> float:
        """0..1 : 70% kejenuhan jumlah kemunculan + 30% bonus kalau ada di judul."""
        p = self.postings.get(eid, {}).get(doc_id)
        if not p:
            return 0.0
        sat = p["count"] / (p["count"] + 1)          # 1x -> .5, 3x -> .75, makin banyak -> mendekati 1
        return 0.7 * sat + 0.3 * (1.0 if p["in_title"] else 0.0)
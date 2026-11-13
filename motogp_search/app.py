"""Web app (Flask).  Jalankan:  python app.py   lalu buka http://127.0.0.1:5000"""
import os

from flask import Flask, jsonify, render_template_string, request

from search_engine import HybridSearchEngine

CORPUS = os.environ.get("CORPUS", "data/IR_Corpus_Kelompok_MotoGP.xlsx")
ENTITIES = os.environ.get("ENTITIES", "data/entities.json")

app = Flask(__name__)
engine = HybridSearchEngine(CORPUS, ENTITIES)     # index dibangun SEKALI saat server start

PAGE = """
<!doctype html><html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MotoGP Search</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:820px;margin:2rem auto;padding:0 1rem;color:#1b2434}
 form{display:flex;gap:.5rem;margin-bottom:.8rem} input[type=text]{flex:1;padding:.6rem;font-size:1rem}
 button{padding:.6rem 1rem;font-size:1rem;cursor:pointer}
 .meta{font-size:.9rem;color:#556;margin:.4rem 0 1rem}
 .tag{display:inline-block;background:#e6f4ea;color:#17613a;border-radius:4px;padding:1px 7px;margin:0 3px 3px 0;font-size:.8rem}
 .res{border-bottom:1px solid #e3e6ec;padding:.9rem 0} .res h3{margin:0 0 .2rem;font-size:1.05rem}
 .sc{font-size:.8rem;color:#667} .snip{font-size:.92rem;margin:.3rem 0}
 details{margin-top:.5rem;font-size:.85rem} label{margin-right:.8rem;font-size:.85rem}
</style></head><body>
<h2>MotoGP Search <small style="font-weight:normal;color:#667">(TF-IDF + Entity-aware)</small></h2>
<form method="get">
  <input type="text" name="q" value="{{ q }}" placeholder="mis. MM93 Ducati" autofocus>
  <button>Cari</button>
</form>
<details {{ 'open' if wt_given else '' }}><summary>Bobot skor (opsional)</summary>
 <form method="get" style="margin-top:.5rem"><input type="hidden" name="q" value="{{ q }}">
  <label>Text <input type="number" step="0.05" min="0" max="1" name="wt" value="{{ wt }}" style="width:5rem"></label>
  <label>Entity <input type="number" step="0.05" min="0" max="1" name="we" value="{{ we }}" style="width:5rem"></label>
  <button>Terapkan</button></form></details>

{% if out %}
 <div class="meta">
  Token query: <code>{{ out.query_tokens }}</code><br>
  Entity terdeteksi: {% for e in out.entities_detected %}<span class="tag">{{ e }}</span>{% else %}<i>tidak ada</i>{% endfor %}
  &middot; bobot: text {{ out.weights.text }}, entity {{ out.weights.entity }}
 </div>
 {% for r in out.results %}
  <div class="res">
   <h3>{{ r.rank }}. {% if r.url %}<a href="{{ r.url }}" target="_blank" rel="noopener">{{ r.title }}</a>{% else %}{{ r.title }}{% endif %}</h3>
   <div class="sc">{{ r.doc_id }} &middot; {{ r.media }} &middot; {{ r.date }} &middot; <b>skor akhir {{ r.s_final }}</b>
     (text {{ r.s_text }} &times; w + entity {{ r.s_entity }} &times; w)</div>
   <div class="snip">{{ r.snippet }}...</div>
   {% for e in r.entities %}<span class="tag">{{ e }}</span>{% endfor %}
  </div>
 {% else %}<p>Tidak ada dokumen yang cocok.</p>{% endfor %}
{% endif %}
</body></html>
"""


def _run():
    q = request.args.get("q", "").strip()
    wt = request.args.get("wt", type=float)
    we = request.args.get("we", type=float)
    out = engine.search(q, top_k=10, w_text=wt, w_entity=we) if q else None
    return q, wt, we, out


@app.route("/")
def index():
    q, wt, we, out = _run()
    return render_template_string(
        PAGE, q=q, out=out,
        wt=engine.w_text if wt is None else wt,
        we=engine.w_entity if we is None else we,
        wt_given=wt is not None)


@app.route("/api/search")
def api():
    _, _, _, out = _run()
    return jsonify(out or {"results": []})


if __name__ == "__main__":
    app.run(debug=False, port=5000)
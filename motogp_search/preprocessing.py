"""Preprocessing dokumen & query (disalin dari notebook Project_Kelompok_IR_MotoGP_clean).

PENTING: dokumen dan query HARUS lewat fungsi yang sama (preprocess_pipeline).
"""
import re
from functools import lru_cache

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    _HAS_SASTRAWI = True
except ImportError:  # hanya untuk testing tanpa Sastrawi; di laptop kalian: pip install PySastrawi
    _HAS_SASTRAWI = False
    print("[WARN] PySastrawi tidak terpasang -> stemming & stopword Sastrawi dilewati. "
          "Jalankan: pip install PySastrawi")

NOISE_PATTERNS = [
    r"baca juga\s*:?\s*(?:\S+\s+){0,20}",
    r"(?:\S+\s+){0,12}artikel kompas\.id",
    r"unsplash\.com/\s*\S+(?:\s\S+)?",
    r"commons\.wikimedia\.org/\s*(?:yu chu chin|\S+)",
    r"idn times\s*/\s*\S+",
    r"\bafp\s*/\s*\S+(?:\s\S+)?",
    r"(?:\b[a-z]+,\s*)?idn times\s*[-–]\s*",
    r"(?:\b[a-z]+,\s*)?kompas\.com\s*[-–]\s*",
    r"(?:\b[a-z]+\s*)?\(antara\)\s*[-–]\s*",
]
NOISE_RE = [re.compile(p) for p in NOISE_PATTERNS]

KATA_NEGASI = {"tidak", "bukan", "jangan", "tanpa", "belum"}
STOPWORD_DOMAIN = {"kompas", "com", "idn", "times", "unsplash", "wikimedia", "potret", "ilustrasi"}

if _HAS_SASTRAWI:
    STOPWORDS = (set(StopWordRemoverFactory().get_stop_words()) | STOPWORD_DOMAIN) - KATA_NEGASI
    _stemmer = StemmerFactory().create_stemmer()
else:
    STOPWORDS = {"yang", "di", "dan", "ke", "dari", "untuk", "dengan", "pada", "itu", "ini", "atau",
                 "akan", "juga", "dalam", "saat", "sebuah", "adalah"} | STOPWORD_DOMAIN
    _stemmer = None


def case_folding(text: str) -> str:
    return text.lower()


def clean_noise(text: str) -> str:
    text = text.replace("\xa0", " ")
    for rx in NOISE_RE:
        text = rx.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text)


def filtering(tokens: list[str]) -> list[str]:
    return [t for t in tokens if not t.isdigit()]


def stopword_removal(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS]


@lru_cache(maxsize=None)
def _stem(word: str) -> str:
    return _stemmer.stem(word) if _stemmer else word


def stemming(tokens: list[str]) -> list[str]:
    return [_stem(t) for t in tokens]


def clean_text(raw_text: str) -> str:
    """Case folding + noise cleaning (dipakai juga oleh entity matcher: belum di-stem)."""
    return clean_noise(case_folding(raw_text))


def preprocess_pipeline(raw_text: str) -> list[str]:
    tokens = tokenize(clean_text(raw_text))
    tokens = filtering(tokens)
    tokens = stopword_removal(tokens)
    return stemming(tokens)
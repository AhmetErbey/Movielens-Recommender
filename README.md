# MovieLens Recommender (TF-IDF + Implicit ALS + Hybrid)

MovieLens verisi üzerinde içerik-tabanlı (TF-IDF), implicit ALS ve hibrit (ALS + TF-IDF yeniden sıralama) önerileri.

## Kurulum
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
# ALS/hybrid için:
pip install -r requirements-als.txt

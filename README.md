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
```
# MovieLens Recommender (TF-IDF + ALS + Hybrid)

Bu proje, MovieLens veri seti ile:
- **İçerik-tabanlı** (TF-IDF: `tags + genres`),
- **İşbirlikçi filtreleme** (Implicit **ALS**),
- ve **Hibrit** (ALS + TF-IDF re-rank)

yaklaşımlarını uygular ve örneklemeli (sampled) metriklerle değerlendirir.

> Kod dosyası: `recommender_bigdata_final.py`  
> Python ≥ 3.9 (öneri: 3.10–3.11)

---

## 1) Veri Seti

Aşağıdaki veri setlerinden **istediğinizi** indirin (GroupLens):
- **Small (100k)**: https://grouplens.org/datasets/movielens/latest/
- **20M / 25M**: https://grouplens.org/datasets/movielens/

Zip’i açtıktan sonra **proje kök klasöründe** şu isimlerle bulundurun:


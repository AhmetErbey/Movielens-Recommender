# -*- coding: utf-8 -*-
"""
MovieLens – Büyük veri (ml-20m / ml-25m / ml-latest) için optimize edilmiş öneri scripti.

Öne çıkanlar:
- İçerik-tabanlı (TF-IDF: tags + genres) öneri + örneklemeli değerlendirme (hızlı)
- (Opsiyonel) MF – Implicit ALS (implicit paketine bağlı). SVD bloğu büyük veri için kapalı.
- Büyük sette bellek ve hız için parametreler: min_df, max_features, user örneklemesi vs.
"""

from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # CPU uyarısını azalt
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# 0) Ayarlar
# ---------------------------
DATA_DIR = './ml-20m'   # -> ml-20m / ml-25m / ml-latest yolunu buraya ver
RUN_SVD = False                  # Büyük sette kapalı
RUN_IMPLICIT_ALS = True          # implicit kuruluysa ALS çalışır

# TF-IDF
TFIDF_MIN_DF = 5
TFIDF_MAX_FEATURES = 200_000

# Değerlendirme
TOPK = 20
RATING_THRESHOLD = 3.5
SAMPLE_USERS = 1000
RANDOM_SEED = 42

# Grid-search bayrakları
RUN_GRID_HYBRID = True           # hibrit (alpha, candidate_M) taraması
RUN_GRID_ALS = False             # ALS (factors/reg/iters) taraması (ağır olabilir)

# ---------------------------
# 1) Veri yükleme & metin inşası
# ---------------------------

def load_movielens(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(os.path.join(data_dir, 'ratings.csv'))
    movies  = pd.read_csv(os.path.join(data_dir, 'movies.csv'))
    tags_p  = os.path.join(data_dir, 'tags.csv')
    tags    = pd.read_csv(tags_p) if os.path.exists(tags_p) else pd.DataFrame(columns=['userId','movieId','tag','timestamp'])
    links_p = os.path.join(data_dir, 'links.csv')
    links   = pd.read_csv(links_p) if os.path.exists(links_p) else pd.DataFrame(columns=['movieId','imdbId','tmdbId'])

    for col in ('userId','movieId'):
        if col in ratings: ratings[col] = ratings[col].astype('int32')
    if 'rating' in ratings: ratings['rating'] = ratings['rating'].astype('float32')
    return ratings, movies, tags, links

def _normalize_text(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r'[\|/]+', ' ', s)
    s = re.sub(r'[^a-z0-9\s]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def build_movie_text(movies: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    mv = movies.copy()
    mv['genres_clean'] = mv['genres'].fillna('').apply(_normalize_text)
    if not tags.empty:
        tag_agg = (tags.dropna(subset=['tag'])
                     .assign(tag=lambda df: df['tag'].astype(str))
                     .groupby('movieId')['tag']
                     .apply(lambda s: ' '.join(_normalize_text(t) for t in s))
                     .reset_index())
    else:
        tag_agg = pd.DataFrame({'movieId': [], 'tag': []})
    df = mv.merge(tag_agg, on='movieId', how='left')
    df['tag'] = df['tag'].fillna('')
    df['text'] = (df['tag'] + ' ' + df['genres_clean']).str.strip()
    df = df[df['text'].str.len() > 0].copy()
    return df[['movieId','title','text']]

def make_vectorizer(min_df: int = TFIDF_MIN_DF, max_features: int | None = TFIDF_MAX_FEATURES) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer='word', ngram_range=(1,2),
        min_df=min_df, max_features=max_features,
        stop_words='english', lowercase=True, norm='l2'
    )

def fit_tfidf(movie_text_df: pd.DataFrame, vectorizer: TfidfVectorizer) -> Tuple[sparse.csr_matrix, Dict[int,int]]:
    X = vectorizer.fit_transform(movie_text_df['text'].tolist())
    movieid_to_idx = {int(mid): i for i, mid in enumerate(movie_text_df['movieId'].astype(int).tolist())}
    return X, movieid_to_idx

# ---------------------------
# 2) Content-based öneri
# ---------------------------

def _user_profile_vector(user_id: int,
                         ratings: pd.DataFrame,
                         movieid_to_idx: Dict[int,int],
                         X: sparse.csr_matrix,
                         rating_threshold: float = RATING_THRESHOLD) -> sparse.csr_matrix:
    user_r = ratings[ratings['userId'] == user_id]
    if user_r.empty: raise ValueError(f'userId={user_id} için rating yok.')
    liked = user_r[user_r['rating'] >= rating_threshold]
    if liked.empty: liked = user_r.sort_values('rating', ascending=False).head(10)

    rows, weights = [], []
    for _, r in liked.iterrows():
        idx = movieid_to_idx.get(int(r['movieId']))
        if idx is None: continue
        rows.append(idx); weights.append(max(0.1, float(r['rating']) - 3.0))
    if not rows: raise ValueError('Profil için uygun film yok (TF-IDF matrisinde yok).')

    user_mat = X[rows]
    w = np.array(weights, dtype=np.float32)
    profile = (user_mat.T.multiply(w)).T
    profile = profile.mean(axis=0)
    return sparse.csr_matrix(profile)

def recommend_for_user_content(user_id: int,
                               ratings: pd.DataFrame,
                               movie_text_df: pd.DataFrame,
                               X: sparse.csr_matrix,
                               movieid_to_idx: Dict[int,int],
                               topn: int = TOPK,
                               rating_threshold: float = RATING_THRESHOLD) -> pd.DataFrame:
    profile = _user_profile_vector(user_id, ratings, movieid_to_idx, X, rating_threshold)
    sims = cosine_similarity(profile, X).ravel()
    seen = set(ratings.loc[ratings['userId']==user_id, 'movieId'].astype(int))
    all_mids = movie_text_df['movieId'].astype(int).tolist()
    recs = [(mid, float(sims[i])) for i, mid in enumerate(all_mids) if mid not in seen]
    recs.sort(key=lambda t: t[1], reverse=True)
    out = pd.DataFrame(recs[:topn], columns=['movieId','score']).merge(
        movie_text_df[['movieId','title']], on='movieId', how='left')
    return out[['movieId','title','score']]

def split_by_user(ratings: pd.DataFrame, test_size: float = 0.2, min_items: int = 5, seed: int = RANDOM_SEED) -> Tuple[pd.DataFrame,pd.DataFrame]:
    rng = np.random.default_rng(seed)
    trains, tests = [], []
    for uid, grp in ratings.groupby('userId'):
        if len(grp) < min_items:
            trains.append(grp); continue
        idx = np.arange(len(grp)); rng.shuffle(idx)
        cut = max(1, int(len(grp)*test_size))
        tests.append(grp.iloc[idx[:cut]]); trains.append(grp.iloc[idx[cut:]])
    return pd.concat(trains, ignore_index=True), (pd.concat(tests, ignore_index=True) if tests else ratings.iloc[0:0].copy())

def evaluate_content_based_sampled(train_ratings: pd.DataFrame,
                                   test_ratings: pd.DataFrame,
                                   movie_text_df: pd.DataFrame,
                                   X: sparse.csr_matrix,
                                   movieid_to_idx: Dict[int,int],
                                   K: int = TOPK,
                                   rating_threshold: float = RATING_THRESHOLD,
                                   sample_users: int | None = SAMPLE_USERS,
                                   seed: int = RANDOM_SEED) -> Dict[str,float]:
    rng = np.random.default_rng(seed)
    all_users = test_ratings['userId'].unique()
    users = all_users if (not sample_users or sample_users >= len(all_users)) else rng.choice(all_users, size=sample_users, replace=False)
    precisions, recalls, hits, ndcgs = [], [], [], []
    for uid in users:
        test_u = test_ratings[test_ratings['userId']==uid]
        relevant = set(test_u.loc[test_u['rating']>=rating_threshold, 'movieId'].astype(int))
        if not relevant: continue
        try:
            recs = recommend_for_user_content(uid, train_ratings, movie_text_df, X, movieid_to_idx, topn=K, rating_threshold=rating_threshold)
        except Exception:
            continue
        topk = recs['movieId'].astype(int).tolist()
        inter = set(topk).intersection(relevant)
        precision = len(inter)/max(1,K); recall = len(inter)/len(relevant); hit = 1.0 if inter else 0.0
        gains = [1.0/np.log2(r+1) if mid in relevant else 0.0 for r, mid in enumerate(topk, start=1)]
        dcg = float(np.sum(gains)); ideal = min(K, len(relevant))
        idcg = float(np.sum([1.0/np.log2(r+1) for r in range(1, ideal+1)])) if ideal>0 else 0.0
        ndcg = (dcg/idcg) if idcg>0 else 0.0
        precisions.append(precision); recalls.append(recall); hits.append(hit); ndcgs.append(ndcg)
    return {
        'users_evaluated': float(len(precisions)),
        'Precision@K': float(np.mean(precisions)) if precisions else 0.0,
        'Recall@K':    float(np.mean(recalls))    if recalls    else 0.0,
        'HitRate@K':   float(np.mean(hits))       if hits       else 0.0,
        'NDCG@K':      float(np.mean(ndcgs))      if ndcgs      else 0.0,
    }

# ---------------------------
# 3) Implicit ALS (MF) – yön düzeltmeli
# ---------------------------

def train_implicit_als(ratings: pd.DataFrame,
                       factors: int = 64,
                       regularization: float = 0.05,
                       iterations: int = 20,
                       alpha: float = 40.0,
                       use_gpu: bool = False,
                       random_seed: int = 42):
    import implicit
    from scipy import sparse
    import numpy as np

    # Map'ler
    user_ids = sorted(ratings['userId'].astype(int).unique())
    item_ids = sorted(ratings['movieId'].astype(int).unique())
    uid_to_idx = {u:i for i,u in enumerate(user_ids)}
    iid_to_idx = {m:i for i,m in enumerate(item_ids)}

    rows = ratings['userId'].astype(int).map(uid_to_idx).to_numpy()
    cols = ratings['movieId'].astype(int).map(iid_to_idx).to_numpy()
    conf = (1.0 + alpha * np.clip(ratings['rating'].astype(np.float32) - 3.5, 0, None)).astype(np.float32)

    # USERS × ITEMS
    user_items = sparse.csr_matrix((conf, (rows, cols)),
                                   shape=(len(uid_to_idx), len(iid_to_idx)))
    print('CSR (users×items) =', user_items.shape)

    model = implicit.als.AlternatingLeastSquares(
        factors=factors, regularization=regularization, iterations=iterations,
        use_gpu=use_gpu, random_state=random_seed
    )
    # ÖNEMLİ: bazı derlemelerde doğru yön için user_items vermek gerekir
    model.fit(user_items)

    # Kitaplık ters döndürdüyse otomatik düzelt
    if (model.user_factors.shape[0] == len(iid_to_idx) and
        model.item_factors.shape[0] == len(uid_to_idx)):
        print("Note: library returned swapped factors; fixing (user<->item).")
        uf, vf = model.user_factors, model.item_factors
        model.user_factors, model.item_factors = vf, uf

    print("ALS shapes -> user_factors:", model.user_factors.shape,
          "item_factors:", model.item_factors.shape,
          "#users:", len(uid_to_idx), "#items:", len(iid_to_idx))

    assert model.user_factors.shape[0] == len(uid_to_idx),  "user_factors uzunluğu yanlış!"
    assert model.item_factors.shape[0] == len(iid_to_idx), "item_factors uzunluğu yanlış!"
    return model, uid_to_idx, iid_to_idx

def recommend_for_user_als_manual(user_id: int,
                                  model,
                                  uid_to_idx: dict[int,int],
                                  iid_to_idx: dict[int,int],
                                  ratings: pd.DataFrame,
                                  movies_df: pd.DataFrame,
                                  topn: int = 20) -> pd.DataFrame:
    if user_id not in uid_to_idx: raise ValueError(f'userId={user_id} eğitim setinde yok.')
    uidx = int(uid_to_idx[int(user_id)])
    u = np.asarray(model.user_factors[uidx], dtype=np.float32)
    scores = np.asarray(model.item_factors @ u, dtype=np.float32)

    seen = ratings.loc[ratings['userId']==int(user_id), 'movieId'].astype(int).tolist()
    seen_idx = [iid_to_idx[m] for m in seen if m in iid_to_idx]
    if seen_idx: scores[np.asarray(seen_idx, dtype=np.int64)] = -1e9

    n = scores.shape[0]; k = min(topn, n)
    top_idx = np.argpartition(-scores, k)[:k]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    idx_to_iid = {v:k for k,v in iid_to_idx.items()}
    out = [(idx_to_iid[i], float(scores[i])) for i in top_idx]
    return (pd.DataFrame(out, columns=['movieId','score'])
              .merge(movies_df[['movieId','title']], on='movieId', how='left')
              [['movieId','title','score']])

def recommend_for_user_als(user_id: int,
                           model,
                           uid_to_idx: Dict[int,int],
                           iid_to_idx: Dict[int,int],
                           ratings: pd.DataFrame,
                           movies_df: pd.DataFrame,
                           topn: int = TOPK) -> pd.DataFrame:
    # İstersen kullan; biz değerlendirmede manuel olanı çağırıyoruz.
    if model is None or user_id not in uid_to_idx:
        raise ValueError('ALS modeli yok veya user eğitimde değil.')
    user_idx = uid_to_idx[user_id]
    user_items = sparse.csr_matrix((1, len(iid_to_idx)))  # boş satır; recalculate_user=True
    recs = model.recommend(user_idx, user_items, N=topn, recalculate_user=True)
    idx_to_iid = {v:k for k,v in iid_to_idx.items()}
    out = [(idx_to_iid[i], float(s)) for i,s in recs[:topn]]
    return (pd.DataFrame(out, columns=['movieId','score'])
              .merge(movies_df[['movieId','title']], on='movieId', how='left')
              [['movieId','title','score']])

def evaluate_als_sampled(model,
                         uid_to_idx: Dict[int,int],
                         iid_to_idx: Dict[int,int],
                         train_ratings: pd.DataFrame,
                         test_ratings: pd.DataFrame,
                         movies_df: pd.DataFrame,
                         K: int = TOPK,
                         rating_threshold: float = RATING_THRESHOLD,
                         sample_users: int | None = SAMPLE_USERS,
                         seed: int = RANDOM_SEED) -> Dict[str,float]:
    if model is None:
        return {'users_evaluated': 0.0, 'Precision@K':0.0, 'Recall@K':0.0, 'HitRate@K':0.0, 'NDCG@K':0.0}
    rng = np.random.default_rng(seed)
    all_users = test_ratings['userId'].unique()
    users = all_users if (not sample_users or sample_users >= len(all_users)) else rng.choice(all_users, size=sample_users, replace=False)
    precisions, recalls, hits, ndcgs = [], [], [], []
    for uid in users:
        if uid not in uid_to_idx: continue
        test_u = test_ratings[test_ratings['userId']==uid]
        relevant = set(test_u.loc[test_u['rating']>=rating_threshold, 'movieId'].astype(int))
        if not relevant: continue
        try:
            recs = recommend_for_user_als_manual(uid, model, uid_to_idx, iid_to_idx, train_ratings, movies_df, topn=K)
        except Exception:
            continue
        topk = recs['movieId'].astype(int).tolist()
        inter = set(topk).intersection(relevant)
        precision = len(inter)/max(1,K); recall = len(inter)/len(relevant); hit = 1.0 if inter else 0.0
        gains = [1.0/np.log2(r+1) if mid in relevant else 0.0 for r, mid in enumerate(topk, start=1)]
        dcg = float(np.sum(gains)); ideal = min(K, len(relevant))
        idcg = float(np.sum([1.0/np.log2(r+1) for r in range(1, ideal+1)])) if ideal>0 else 0.0
        ndcg = (dcg/idcg) if idcg>0 else 0.0
        precisions.append(precision); recalls.append(recall); hits.append(hit); ndcgs.append(ndcg)
    return {
        'users_evaluated': float(len(precisions)),
        'Precision@K': float(np.mean(precisions)) if precisions else 0.0,
        'Recall@K':    float(np.mean(recalls))    if recalls    else 0.0,
        'HitRate@K':   float(np.mean(hits))       if hits       else 0.0,
        'NDCG@K':      float(np.mean(ndcgs))      if ndcgs      else 0.0,
    }

# ---------------------------
# 4) Hızlı benzer & Hibrit
# ---------------------------

def similar_movies_quick(title_query: str, movie_text_df: pd.DataFrame, X: sparse.csr_matrix, topn: int = 10) -> pd.DataFrame:
    cand = movie_text_df[movie_text_df['title'].str.contains(title_query, case=False, na=False)]
    if cand.empty: raise ValueError(f'"{title_query}" başlığını içeren film bulunamadı.')
    row_idx = movie_text_df.index.get_loc(cand.index[0])
    sims = cosine_similarity(X[row_idx], X).ravel()
    sims[row_idx] = -1.0
    top_idx = np.argpartition(-sims, topn)[:topn]
    top_idx = top_idx[np.argsort(-sims[top_idx])]
    recs = movie_text_df.iloc[top_idx][['movieId','title']].copy()
    recs['similarity'] = sims[top_idx]
    return recs.reset_index(drop=True)

def _minmax_scale(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0: return arr
    a_min = float(np.min(arr)); a_max = float(np.max(arr))
    if not np.isfinite(a_min) or not np.isfinite(a_max) or a_max <= a_min:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - a_min) / (a_max - a_min)

def recommend_hybrid_for_user(user_id: int,
                              model,
                              uid_to_idx: Dict[int,int],
                              iid_to_idx: Dict[int,int],
                              ratings: pd.DataFrame,
                              movies_df: pd.DataFrame,
                              movie_text_df: pd.DataFrame,
                              X: sparse.csr_matrix,
                              movieid_to_idx: Dict[int,int],
                              topn: int = TOPK,
                              rating_threshold: float = RATING_THRESHOLD,
                              alpha: float = 0.6,
                              candidate_M: int = 300) -> pd.DataFrame:
    if user_id not in uid_to_idx: raise ValueError(f'userId={user_id} eğitim setinde yok.')
    uidx = int(uid_to_idx[user_id])
    u = np.asarray(model.user_factors[uidx], dtype=np.float32)
    als_scores = np.asarray(model.item_factors @ u, dtype=np.float32)

    seen = set(ratings.loc[ratings['userId']==user_id, 'movieId'].astype(int))
    seen_idx = {iid_to_idx[m] for m in seen if m in iid_to_idx}
    if seen_idx: als_scores[list(seen_idx)] = -1e9

    n_items = als_scores.shape[0]; M = min(candidate_M, n_items)
    cand_idx = np.argpartition(-als_scores, M)[:M]
    cand_idx = cand_idx[np.argsort(-als_scores[cand_idx])]

    profile = _user_profile_vector(user_id, ratings, movieid_to_idx, X, rating_threshold)
    cb_sims = np.clip(cosine_similarity(profile, X[cand_idx]).ravel(), 0.0, 1.0)

    als_scaled = _minmax_scale(als_scores[cand_idx])
    hybrid = alpha * als_scaled + (1.0 - alpha) * cb_sims

    top_loc = np.argpartition(-hybrid, topn)[:topn]
    top_loc = top_loc[np.argsort(-hybrid[top_loc])]
    final_idx = cand_idx[top_loc]

    idx_to_iid = {v:k for k,v in iid_to_idx.items()}
    out = [(idx_to_iid[i], float(hybrid[j])) for j,i in enumerate(final_idx)]
    return (pd.DataFrame(out, columns=['movieId','score'])
              .merge(movies_df[['movieId','title']], on='movieId', how='left')
              [['movieId','title','score']])

def evaluate_hybrid_sampled(model,
                            uid_to_idx: Dict[int,int],
                            iid_to_idx: Dict[int,int],
                            train_ratings: pd.DataFrame,
                            test_ratings: pd.DataFrame,
                            movies_df: pd.DataFrame,
                            movie_text_df: pd.DataFrame,
                            X: sparse.csr_matrix,
                            movieid_to_idx: Dict[int,int],
                            K: int = TOPK,
                            rating_threshold: float = RATING_THRESHOLD,
                            alpha: float = 0.6,
                            candidate_M: int = 300,
                            sample_users: int | None = SAMPLE_USERS,
                            seed: int = RANDOM_SEED) -> Dict[str,float]:
    rng = np.random.default_rng(seed)
    all_users = test_ratings['userId'].unique()
    users = all_users if (not sample_users or sample_users >= len(all_users)) else rng.choice(all_users, size=sample_users, replace=False)
    precisions, recalls, hits, ndcgs = [], [], [], []
    for uid in users:
        test_u = test_ratings[test_ratings['userId']==uid]
        relevant = set(test_u.loc[test_u['rating']>=rating_threshold, 'movieId'].astype(int))
        if not relevant: continue
        try:
            recs = recommend_hybrid_for_user(uid, model, uid_to_idx, iid_to_idx,
                                             train_ratings, movies_df,
                                             movie_text_df, X, movieid_to_idx,
                                             topn=K, rating_threshold=rating_threshold,
                                             alpha=alpha, candidate_M=candidate_M)
        except Exception:
            continue
        topk = recs['movieId'].astype(int).tolist()
        inter = set(topk).intersection(relevant)
        precision = len(inter)/max(1,K); recall = len(inter)/len(relevant); hit = 1.0 if inter else 0.0
        gains = [1.0/np.log2(r+1) if mid in relevant else 0.0 for r, mid in enumerate(topk, start=1)]
        dcg = float(np.sum(gains)); ideal = min(K, len(relevant))
        idcg = float(np.sum([1.0/np.log2(r+1) for r in range(1, ideal+1)])) if ideal>0 else 0.0
        ndcg = (dcg/idcg) if idcg>0 else 0.0
        precisions.append(precision); recalls.append(recall); hits.append(hit); ndcgs.append(ndcg)
    return {
        'users_evaluated': float(len(precisions)),
        'Precision@K': float(np.mean(precisions)) if precisions else 0.0,
        'Recall@K':    float(np.mean(recalls))    if recalls    else 0.0,
        'HitRate@K':   float(np.mean(hits))       if hits       else 0.0,
        'NDCG@K':      float(np.mean(ndcgs))      if ndcgs      else 0.0,
    }

# ---------------------------
# 6) Grid Search yardımcıları
# ---------------------------

def grid_search_hybrid(model, uid_to_idx, iid_to_idx, train_r, test_r, movies,
                       movie_text_df, X, movieid_to_idx,
                       alphas=(0.4, 0.5, 0.6, 0.7),
                       candidate_M_list=(200, 300, 600, 1000),
                       K=TOPK, rating_threshold=RATING_THRESHOLD,
                       sample_users=SAMPLE_USERS, seed=RANDOM_SEED):
    rows = []
    for a in alphas:
        for M in candidate_M_list:
            m = evaluate_hybrid_sampled(model, uid_to_idx, iid_to_idx,
                                        train_r, test_r, movies,
                                        movie_text_df, X, movieid_to_idx,
                                        K=K, rating_threshold=rating_threshold,
                                        alpha=a, candidate_M=M,
                                        sample_users=sample_users, seed=seed)
            rows.append({'alpha': a, 'candidate_M': M, **m})
            print(f"[hybrid] alpha={a} M={M} -> NDCG@K={m['NDCG@K']:.4f}, P@K={m['Precision@K']:.4f}")
    return pd.DataFrame(rows).sort_values(['NDCG@K','Precision@K'], ascending=False)

def grid_search_als(train_r, test_r, movies,
                    factors_list=(64, 96, 128),
                    reg_list=(0.02, 0.05, 0.08),
                    iters_list=(20,),
                    alpha=40.0, use_gpu=False,
                    K=TOPK, rating_threshold=RATING_THRESHOLD,
                    sample_users=SAMPLE_USERS, seed=RANDOM_SEED):
    rows = []
    for f in factors_list:
        for reg in reg_list:
            for it in iters_list:
                print(f"[als] factors={f}, reg={reg}, iters={it}")
                model, uid_to_idx, iid_to_idx = train_implicit_als(
                    train_r, factors=f, iterations=it, regularization=reg,
                    alpha=alpha, use_gpu=use_gpu
                )
                m = evaluate_als_sampled(model, uid_to_idx, iid_to_idx, train_r, test_r, movies,
                                         K=K, rating_threshold=rating_threshold,
                                         sample_users=sample_users, seed=seed)
                rows.append({'factors': f, 'regularization': reg, 'iterations': it, **m})
                print(f"   -> NDCG@K={m['NDCG@K']:.4f}, P@K={m['Precision@K']:.4f}")
    return pd.DataFrame(rows).sort_values(['NDCG@K','Precision@K'], ascending=False)

# ---------------------------
# 5) Çalıştırma
# ---------------------------
if __name__ == "__main__":
    ratings, movies, tags, links = load_movielens(DATA_DIR)
    print(f"Loaded ratings={len(ratings):,}, movies={len(movies):,}, tags={len(tags):,}")

    movie_text_df = build_movie_text(movies, tags)
    vec = make_vectorizer(min_df=TFIDF_MIN_DF, max_features=TFIDF_MAX_FEATURES)
    X, movieid_to_idx = fit_tfidf(movie_text_df, vec)
    print(f"TF-IDF vocab size={len(vec.vocabulary_):,}, matrix shape={X.shape}, nnz/row≈{X.getnnz()/X.shape[0]:.1f}")

    try:
        print('\n=== Benzer Filmler ("Toy Story") ===')
        print(similar_movies_quick("Toy Story", movie_text_df, X, topn=10))
    except Exception as e:
        print("Benzer film arama hatası:", e)

    train_r, test_r = split_by_user(ratings, test_size=0.2, min_items=5, seed=RANDOM_SEED)
    print(f"Users in test: {test_r.userId.nunique():,}")
    cb_metrics = evaluate_content_based_sampled(
        train_r, test_r, movie_text_df, X, movieid_to_idx,
        K=TOPK, rating_threshold=RATING_THRESHOLD,
        sample_users=SAMPLE_USERS, seed=RANDOM_SEED
    )
    print("\n=== Content-Based (sampled) ===")
    print(cb_metrics)

    if RUN_IMPLICIT_ALS:
        print("\nEğitiliyor: Implicit ALS (factors=64, iters=20) ...")
        model, uid_to_idx, iid_to_idx = train_implicit_als(
            train_r, factors=64, iterations=20,
            regularization=0.05, alpha=40.0, use_gpu=False
        )

        als_metrics = evaluate_als_sampled(
            model, uid_to_idx, iid_to_idx, train_r, test_r, movies,
            K=TOPK, rating_threshold=RATING_THRESHOLD,
            sample_users=SAMPLE_USERS, seed=RANDOM_SEED
        )
        print("\n=== ALS (sampled) ===")
        print(als_metrics)

        hyb_metrics = evaluate_hybrid_sampled(
            model, uid_to_idx, iid_to_idx,
            train_r, test_r, movies,
            movie_text_df, X, movieid_to_idx,
            K=TOPK, rating_threshold=RATING_THRESHOLD,
            alpha=0.6, candidate_M=300,
            sample_users=SAMPLE_USERS, seed=RANDOM_SEED
        )
        print("\n=== Hybrid (sampled) ===")
        print(hyb_metrics)

        # --- Hybrid grid search (opsiyonel) ---
        if RUN_GRID_HYBRID:
            df_h = grid_search_hybrid(
                model, uid_to_idx, iid_to_idx,
                train_r, test_r, movies,
                movie_text_df, X, movieid_to_idx,
                alphas=(0.4, 0.5, 0.6, 0.7),
                candidate_M_list=(200, 300, 600, 1000),
                K=TOPK, rating_threshold=RATING_THRESHOLD,
                sample_users=SAMPLE_USERS, seed=RANDOM_SEED
            )
            print("\nTop Hybrid configs (by NDCG then P@K):")
            print(df_h.head(10))
            # df_h.to_csv("hybrid_grid_results.csv", index=False)

        # --- ALS grid search (opsiyonel) ---
        if RUN_GRID_ALS:
            df_als = grid_search_als(
                train_r, test_r, movies,
                factors_list=(64, 96, 128),
                reg_list=(0.02, 0.05, 0.08),
                iters_list=(20,),          # istersen (20, 30)
                alpha=40.0, use_gpu=False,
                K=TOPK, rating_threshold=RATING_THRESHOLD,
                sample_users=SAMPLE_USERS, seed=RANDOM_SEED
            )
            print("\nTop ALS configs (by NDCG then P@K):")
            print(df_als.head(10))
            # df_als.to_csv("als_grid_results.csv", index=False)

        # Örnek kullanıcı için ALS & Hybrid öneri listeleri
        try:
            example_uid = int(train_r["userId"].value_counts().idxmax())
            print(f"\nALS – Kullanıcıya öneriler (userId={example_uid})")
            print(recommend_for_user_als_manual(
                    example_uid, model, uid_to_idx, iid_to_idx,
                    train_r, movies, topn=TOPK
                ))
            print(f"\nHybrid – Kullanıcıya öneriler (userId={example_uid})")
            print(recommend_hybrid_for_user(
                    example_uid, model, uid_to_idx, iid_to_idx,
                    train_r, movies, movie_text_df, X, movieid_to_idx,
                    topn=TOPK, alpha=0.6, candidate_M=300
                ))
        except Exception as e:
            print("Örnek öneri hatası:", e)
    else:
        print("ALS atlandı (RUN_IMPLICIT_ALS=False).")

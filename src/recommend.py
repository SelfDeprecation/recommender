import numpy as np
import torch
from train_utils import prepare_embeddings_if_needed, load_head, EMB_PATH, HEAD_PATH
from model import EncoderWrapper, DEVICE

def recommend_topk_from_liked(df, liked_titles, top_k=10, model_name=None, emb_mat=None, texts=None, head=None):
    """
    Рекомендуем книги, опираясь на схожесть эмбеддингов.
    - liked_titles: список названий книг, которые пользователь отметил как понравившиеся.
    Исключаем их из финального списка рекомендаций.
    """
    if emb_mat is None or texts is None:
        emb_mat, texts = prepare_embeddings_if_needed(df, force_recompute=False, model_name=model_name)

    # Создаём индекс по названию
    df = df.reset_index(drop=True)
    title2idx = {row['Book']: idx for idx, row in df.iterrows()}

    liked_idx = [title2idx[t] for t in liked_titles if t in title2idx]
    if len(liked_idx) == 0:
        return []

    device = DEVICE
    all_emb = torch.tensor(emb_mat, dtype=torch.float32, device=device)
    liked_emb = all_emb[liked_idx]  # (m, d)

    with torch.no_grad():
        # Косинусная схожесть
        an = torch.nn.functional.normalize(all_emb, dim=1)
        ln = torch.nn.functional.normalize(liked_emb, dim=1)
        sim = torch.matmul(an, ln.t())
        sim_score = sim.mean(dim=1)

        # При наличии головы — комбинируем сигналы
        if head is not None:
            head = head.to(device)
            head.eval()
            preds_head = head(all_emb).detach()
            
            sim_mean = sim_score.mean().item()
            sim_std = sim_score.std().item() if sim_score.std().numel() > 0 else 1.0
            head_mean = preds_head.mean().item()
            head_std = preds_head.std().item() if preds_head.std().numel() > 0 else 1.0
            
            sim_norm = (sim_score - sim_mean) / (sim_std + 1e-8)
            head_norm = (preds_head - head_mean) / (head_std + 1e-8)
            combined = 0.5 * sim_norm + 0.5 * head_norm
            scores = combined.cpu().numpy()
        else:
            scores = sim_score.cpu().numpy()

    # Исключаем книги, которые уже прочитал пользователь
    mask = np.ones(len(df), dtype=bool)
    mask[liked_idx] = False  # помечаем прочитанные
    filtered_scores = scores[mask]
    filtered_df = df[mask].reset_index(drop=True)

    # Выбираем топ-K
    top_idx = np.argsort(-filtered_scores)[:top_k]
    recs = filtered_df.iloc[top_idx].copy()
    recs['score'] = filtered_scores[top_idx]
    return recs[['Book', 'Author', 'Genres', 'Avg_Rating', 'score']].to_dict('records')


def load_embeddings_if_exists(path="data/book_embeddings.npz"):
    import os
    if os.path.exists(path):
        arr = np.load(path, allow_pickle=True)
        return arr['embeddings'], arr['texts'].tolist()
    return None, None

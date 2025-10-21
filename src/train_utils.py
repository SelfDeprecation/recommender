import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import MiniLMHead, EncoderWrapper, DEVICE

EMB_PATH = "data/book_embeddings.npz"
HEAD_PATH = "data/head_state.pt"

def prepare_embeddings_if_needed(df, force_recompute=False, model_name=None, batch_size=64, show_progress=False):
    """
    Возвращает (emb_mat, texts). Если файл уже есть и force_recompute=False — загружает.
    """
    if os.path.exists(EMB_PATH) and not force_recompute:
        arr = np.load(EMB_PATH, allow_pickle=True)
        emb = arr['embeddings']
        texts = arr['texts'].tolist()
        return emb, texts

    enc = EncoderWrapper(model_name=model_name or "sentence-transformers/all-MiniLM-L12-v2")
    texts = (df['Book'].astype(str) + " by " + df['Author'].astype(str) + " | Genres: " + df['Genres'].astype(str)).tolist()
    emb_mat = enc.encode_texts(texts, batch_size=batch_size, show_progress=show_progress)
    os.makedirs(os.path.dirname(EMB_PATH), exist_ok=True)
    np.savez_compressed(EMB_PATH, embeddings=emb_mat, texts=np.array(texts))
    return emb_mat, texts

def train_head(emb_mat, ratings, epochs=10, batch_size=64, lr=1e-4, report_every=1, st=None):
    """
    Обучаем MLP (head) на эмбеддингах emb_mat (np.ndarray) и целевых рейтингах ratings (np.array).
    Сохраняем вес head в data/head_state.pt
    """
    device = DEVICE
    emb_dim = emb_mat.shape[1]
    model = MiniLMHead(embedding_dim=emb_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = emb_mat.shape[0]
    idxs = np.arange(n)

    for epoch in range(1, epochs+1):
        np.random.shuffle(idxs)
        epoch_loss = 0.0
        pbar = tqdm(range(0, n, batch_size), desc=f"Head epoch {epoch}/{epochs}") if st is None else range(0, n, batch_size)
        for i in pbar:
            batch_idx = idxs[i:i+batch_size]
            xb = torch.tensor(emb_mat[batch_idx], dtype=torch.float32, device=device)
            yb = torch.tensor(ratings[batch_idx], dtype=torch.float32, device=device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_idx)
        epoch_loss /= n
        msg = f"Epoch {epoch}/{epochs} — MSE: {epoch_loss:.4f}"
        if st is not None:
            # st is a streamlit module or object passed from app; we assume st.write/st.progress available
            try:
                st.write(msg)
            except Exception:
                print(msg)
        else:
            print(msg)

    # Save head state dict
    os.makedirs(os.path.dirname(HEAD_PATH), exist_ok=True)
    torch.save(model.state_dict(), HEAD_PATH)
    return model

def load_head(head_path=None, emb_dim=None):
    """
    Загружает head (MLP) из файла. Если файл отсутствует, возвращает None.
    Если emb_dim указан и файл отсутствует — создаёт новый head.
    """
    path = head_path or HEAD_PATH
    if not os.path.exists(path):
        if emb_dim is None:
            return None
        model = MiniLMHead(embedding_dim=emb_dim)
        return model
    # try infer emb_dim from saved state
    state = torch.load(path, map_location='cpu')
    # create model with default dims that match saved keys
    # We assume saved model had layer weight 'net.0.weight' shape (hidden_dim, emb_dim)
    w = state.get('net.0.weight') or state.get('net.0.weight')  # fallback
    if w is None:
        # fallback to creating default model
        model = MiniLMHead()
        model.load_state_dict(state)
        return model
    emb_dim = w.shape[1]
    model = MiniLMHead(embedding_dim=emb_dim)
    model.load_state_dict(state)
    return model

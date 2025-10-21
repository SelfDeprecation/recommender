import os
import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MiniLMHead(nn.Module):
    """
    Небольшая голова (MLP), которая принимает эмбеддинги из all-MiniLM и
    предсказывает скалярный рейтинг.
    """
    def __init__(self, embedding_dim=384, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

class EncoderWrapper:
    """
    Обёртка для sentence-transformers encoder (all-MiniLM-L12-v2).
    Вычисление эмбеддингов пакетами + сохранение / загрузка.
    """
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L12-v2", device=None):
        self.model_name = model_name
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        # SentenceTransformer internally handles device argument
        self.encoder = SentenceTransformer(model_name)

    def encode_texts(self, texts, batch_size=64, show_progress=False):
        """
        texts: list[str]
        returns: np.ndarray shape (n, embedding_dim) dtype=float32
        """
        embeddings = []
        it = range(0, len(texts), batch_size)
        iterator = tqdm(it, desc="Encoding texts") if show_progress else it
        for i in iterator:
            batch = texts[i:i+batch_size]
            emb = self.encoder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embeddings.append(emb)
        if len(embeddings)==0:
            return np.zeros((0, self.encoder.get_sentence_embedding_dimension()), dtype=np.float32)
        emb_mat = np.vstack(embeddings).astype(np.float32)
        return emb_mat

    def embedding_dim(self):
        return self.encoder.get_sentence_embedding_dimension()

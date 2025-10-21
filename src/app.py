import streamlit as st
import os
import pandas as pd
from data import load_books_dataset
from train_utils import prepare_embeddings_if_needed, train_head, load_head, EMB_PATH, HEAD_PATH
from recommend import recommend_topk_from_liked, load_embeddings_if_exists
from model import EncoderWrapper
import numpy as np
import torch

st.set_page_config(page_title="Books Recommender (MiniLM)", layout="wide")
st.title("📚 Books Recommender — MiniLM")

DATA_PATH = "data/books.csv"
EMB_PATH = "data/book_embeddings.npz"
HEAD_PATH = "data/head_state.pt"

if 'df' not in st.session_state:
    if os.path.exists(DATA_PATH):
        st.session_state.df = load_books_dataset(DATA_PATH)
    else:
        st.session_state.df = None

st.sidebar.header("Data / Embeddings")
if st.sidebar.button("Load dataset"):
    if os.path.exists(DATA_PATH):
        st.session_state.df = load_books_dataset(DATA_PATH)
        st.success(f"Loaded dataset: {len(st.session_state.df)} rows")
    else:
        st.error("data/books.csv not found. Поместите файл в папку data/")

if st.session_state.df is None:
    st.info("Положите books.csv в папку data/ и нажмите Load dataset.")
    st.stop()

df = st.session_state.df

st.write("### Dataset preview")
st.dataframe(df.head())

st.write("## Embeddings")
col1, col2 = st.columns(2)
with col1:
    if st.button("Prepare embeddings"):
        with st.spinner("Computing embeddings... (это может занять время при первом запуске)"):
            emb_mat, texts = prepare_embeddings_if_needed(df, force_recompute=False, show_progress=True)
            st.session_state.emb_mat = emb_mat
            st.success(f"Embeddings ready: shape={emb_mat.shape}")
with col2:
    if os.path.exists(EMB_PATH):
        arr = np.load(EMB_PATH, allow_pickle=True)
        st.write("Embeddings file exists:", EMB_PATH)
        st.write("Shape:", arr['embeddings'].shape)

st.write("## Train head (MLP)")
epochs = st.number_input("Epochs", 1, 200, 10)
batch_size = st.number_input("Batch size", 8, 1024, 64)
lr = st.number_input("Learning rate", 1e-6, 1e-1, 1e-4, format="%.6f")
if st.button("Train head now"):
    emb_mat, texts = prepare_embeddings_if_needed(df)
    ratings = df['Avg_Rating'].astype(float).values.astype(np.float32)
    with st.spinner("Training head..."):
        model = train_head(emb_mat, ratings, epochs=int(epochs), batch_size=int(batch_size), lr=float(lr), st=st)
        st.success("Head trained and saved.")

st.write("## Personalize & Recommend")
liked = st.multiselect("Выберите книги, которые вам понравились (для персонализации)", df['Book'].tolist())
k = st.number_input("Top-K", 1, 100, 10)
if st.button("Fine-tune head on your likes & Recommend"):
    if len(liked) == 0:
        st.error("Выберите хотя бы одну книгу.")
    else:
        emb_mat, texts = prepare_embeddings_if_needed(df)
        title2idx = {row['Book']: idx for idx, row in df.reset_index(drop=True).iterrows()}
        rows = [title2idx[t] for t in liked if t in title2idx]
        ratings = df.loc[rows, 'Avg_Rating'].astype(float).values.astype(np.float32)
        from train_utils import train_head
        with st.spinner("Fine-tuning head..."):
            head_model = train_head(emb_mat[rows], ratings, epochs=30, batch_size=min(16,len(rows)), lr=5e-4, st=st)
            torch.save(head_model.state_dict(), "data/head_personalized.pt")
        st.success("Fine-tune done.")
        head = head_model
        recs = recommend_topk_from_liked(df, liked, top_k=int(k), emb_mat=emb_mat, head=head)
        st.write(f"Рекомендации:")
        st.table(pd.DataFrame(recs))


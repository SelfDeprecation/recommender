import argparse
import numpy as np
import torch
from data import load_books_dataset
from train_utils import prepare_embeddings_if_needed, train_head, load_head, EMB_PATH, HEAD_PATH

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare_embeddings", action="store_true", help="Compute and save embeddings to data/book_embeddings.npz")
    parser.add_argument("--train_head", action="store_true", help="Train head (MLP) on dataset embeddings")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--force_recompute_embeddings", action="store_true")
    args = parser.parse_args()

    df = load_books_dataset("data/books.csv")

    if args.prepare_embeddings:
        print("Preparing embeddings...")
        emb_mat, texts = prepare_embeddings_if_needed(df, force_recompute=args.force_recompute_embeddings, batch_size=args.batch_size, show_progress=True)
        print("Embeddings shape:", emb_mat.shape)
    else:
        emb_mat, texts = prepare_embeddings_if_needed(df, force_recompute=False)

    if args.train_head:
        print("Training head...")
        ratings = df['Avg_Rating'].astype(float).values.astype(np.float32)
        model = train_head(emb_mat, ratings, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
        print("Head trained and saved to data/head_state.pt")
    else:
        print("Head not trained (use --train_head to train).")

if __name__ == "__main__":
    main()

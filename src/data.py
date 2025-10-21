import pandas as pd
import ast

def parse_genres(cell):
    if pd.isna(cell):
        return []
    if isinstance(cell, list):
        return [str(x).strip() for x in cell]
    s = str(cell)
    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed]
    except Exception:
        pass
    if '|' in s:
        return [x.strip() for x in s.split('|') if x.strip()]
    if ',' in s:
        return [x.strip() for x in s.split(',') if x.strip()]
    return [s.strip()]

def load_books_dataset(path="data/books.csv"):
    """
    Загрузка CSV и минимальный препроцессинг.
    Возвращает DataFrame с колонками Book, Author, Genres, Avg_Rating и Genres_parsed.
    """
    df = pd.read_csv(path)
    required = ['Book','Author','Genres','Avg_Rating']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column in CSV: {c}")
    df = df.dropna(subset=required).reset_index(drop=True)
    df['Genres_parsed'] = df['Genres'].apply(parse_genres)
    return df

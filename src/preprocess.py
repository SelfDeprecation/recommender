import pandas as pd
import ast

df = pd.read_csv('data/goodreads_data.csv')

new_df = df[['Book', 'Author', 'Genres', 'Avg_Rating']]

new_df = new_df[new_df['Avg_Rating'] > 0]

new_df = new_df[~pd.isna(new_df['Book'])]
new_df = new_df[~pd.isna(new_df['Author'])]
new_df = new_df[~pd.isna(new_df['Genres'])]
new_df = new_df[~pd.isna(new_df['Avg_Rating'])]

new_df = new_df[new_df['Genres'].astype(str) != '[]']

unique = set()

for i in df['Genres']:
    genres = i[1:-1].split(', ')
    unique.update([i.strip("'") for i in genres])

dic = {}
for i in unique:
    dic[i] = 0

for i in df['Genres']:
    genres = i[1:-1].split(', ')
    cur = [i.strip("'") for i in genres]
    for j in cur:
        dic[j] += 1

items = sorted(dic.items(), key=lambda item: item[1], reverse=True)

dic = {key: value for key, value in items if value >= 195 and value != 960}

unique = set(dic.keys())

def filter_items(cell, allowed):
    try:
        if isinstance(cell, str) and cell.startswith('['):
            items_list = ast.literal_eval(cell)
            filtered_items = [item for item in items_list if item in allowed]
            return str(filtered_items) if filtered_items else '[]'
        return cell
    except:
        return cell

new_df['Genres'] = new_df['Genres'].apply(
    lambda x: filter_items(x, unique)
)

new_df.to_csv('data/books.csv')

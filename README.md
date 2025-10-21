# 📚 Books Recommender System

## 🚀 Описание проекта

Этот проект представляет собой **систему рекомендаций книг**, основанную на **текстовых описаниях**, **жанрах**, **авторах** и **рейтингах пользователей**.  
Он использует предобученную модель предложений [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) для вычисления эмбеддингов книг и небольшую нейросеть (персонализированную "голову"), которая адаптируется под вкусы каждого пользователя.

Система позволяет:
- Работать через **Streamlit** веб-интерфейс.

---

## 🧠 Архитектура проекта

📦 books_recommender/
├── 📁 data/
│   ├── goodreads_data.csv          # Исходные данные о книгах
│   ├── book_embeddings.npz         # Сохранённые эмбеддинги книг
│   ├── books.csv                   # Предобработанный датасет
│   └── head_personalized.pt        # Индивидуальная "голова" модели
│
├── 📁 src/
│   ├── model.py                    # Архитектура модели
│   ├── train_utils.py              # Утилиты для обучения
│   ├── recommend.py                # Генерация рекомендаций
│   ├── data.py                     # Парсинг жанров
│   ├── app.py                      # Streamlit-приложение
│   ├── train.py                    # CLI для обучения
│   └── preprocess.py               # Предобработка данных
│
├── .gitignore
├── LICENSE                         # MIT License
├── README.md                       # Документация проекта
└── requirements.txt                # Зависимости Python

---

## ⚙️ Установка и подготовка окружения

```bash
git clone https://github.com/SelfDeprecation/recommender.git
cd recommender
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📊 Формат данных

Файл `data/goodreads_data.csv` должен содержать следующие столбцы:

| Book | Author | Description | Genres | Avg_Rating | Num_Ratings | URL |
|------|---------|------------|------------|----------|------|-----------|
| To Kill a Mockingbird | Harper Lee | The unforgettable novel of a childhood... | "['Classics', 'Fiction', ...]" | 4.20 | "5,691,311" | https://www.goodreads.com/book/show/2657.To_Kill_a_Mockingbird |

Файл `data/books.csv`, который создаётся после предобработки исходного файла, будет содержать следующие столбцы:

| Book | Author | Genres | Avg_Rating |
|------|---------|------------|------|
| To Kill a Mockingbird | Harper Lee | "['Classics', 'Fiction', ...]" | 4.20 |

---

## 🧩 Используемая модель

Предобученная модель:  
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

Особенности:
- Размер эмбеддингов: **384**
- Лёгкая и быстрая, подходит для CPU
- Отличный баланс между скоростью и качеством

---

## 💡 Использование

### Предобработка данных

```bash
python -m src.preprocess
```

После запуска в папке data/ появится файл books.csv, который является предобработанным исходным файлом с датасета.
Ключевые изменения в предобработке:
- Удалены колонки с описанием книги, количеством оценок, URL книги;
- Отобраны наиболее релевантные жанры (встречающиеся в датасете более, чем ~200 раз).

### 🖥️ Запуск Streamlit-приложения

```bash
streamlit run src/app.py
```

После запуска пользователь сможет:
- Выбрать книги, которые он прочитал;
- Указать рейтинг для каждой книги;
- Дообучить персональную "голову";
- Получить список рекомендаций.

> Прочитанные книги **не будут включаться** в рекомендации в явном виде, однако возможно включение в рекомендации сборников с прочитанными книгами.

---

После первого запуска автоматически создаются:
- `data/book_embeddings.npz` — эмбеддинги всех книг
- `data/head_personalized.pt` — обученная персональная "голова"

---

## 🔍 Пример вывода

```
Top-10 recommended books (excluding read ones):

1. Fahrenheit 451 — Ray Bradbury — Sci-Fi — score: 0.92
2. The Handmaid’s Tale — Margaret Atwood — Dystopia — score: 0.90
3. The Road — Cormac McCarthy — Post-apocalyptic — score: 0.88
...
```

---

## 🧰 Основные зависимости

```txt
streamlit
torch
tqdm
sentence-transformers
pandas
numpy
scikit-learn
```

---

## 🧠 Как работает система

1. **Вычисление схожести**: косинусное сходство определяет близость книг.
2. **Персонализация**: обучается небольшая нейросеть ("голова"), которая подстраивается под предпочтения пользователя.
3. **Фильтрация**: прочитанные книги исключаются из выдачи рекомендаций.
4. **Рекомендации**: возвращаются книги, наиболее схожие с понравившимися пользователю.

---

## 📦 Файлы, создаваемые системой

| Файл | Назначение |
|------|-------------|
| `data/book_embeddings.npz` | Эмбеддинги всех книг |
| `data/head_personalized.pt` | Персонализированная модель |
| `data/books.csv` | Очищенный датасет с книгами |

---

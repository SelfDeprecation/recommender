import sys
import os
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
import torch

from PyQt6.QtCore import (
    Qt,
    QAbstractTableModel,
    QModelIndex,
    QVariant,
    QSortFilterProxyModel,
    QThread,
    pyqtSignal,
)
from PyQt6.QtGui import QAction, QIcon, QFont, QCloseEvent, QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableView,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QSpinBox,
    QMessageBox,
    QStatusBar,
    QGroupBox,
    QTextEdit,
    QSplitter,
    QCompleter,
    QSizePolicy,
    QDialog,
    QFormLayout,
    QProgressBar,
    QFileDialog,
)

# ------------------------------
# Константы и пути
# ------------------------------
DATA_PATH = "data/books.csv"
EMB_PATH = "data/book_embeddings.npz"
HEAD_PATH = "data/head_state.pt"
PERSONALIZED_HEAD_PATH = "data/head_personalized.pt"

# Импорт реальных модулей проекта (ожидаются в той же папке/пакете)
try:
    from data import load_books_dataset
    from train_utils import prepare_embeddings_if_needed, train_head, load_head
    from recommend import recommend_topk_from_liked
    from model import EncoderWrapper
except Exception:
    # Если модули недоступны, создадим заглушки чтобы интерфейс всё ещё запускался
    def load_books_dataset(path: str) -> pd.DataFrame:
        # Простая заглушка: создаёт DataFrame с несколькими колонками
        return pd.DataFrame(
            [
                {"Book": "Book A", "Author": "Author A", "Avg_Rating": 4.2},
                {"Book": "Book B", "Author": "Author B", "Avg_Rating": 3.8},
                {"Book": "Book C", "Author": "Author C", "Avg_Rating": 4.7},
            ]
        )

    def prepare_embeddings_if_needed(df, force_recompute=False, show_progress=False):
        # Возвращаем случайную матрицу эмбеддингов и список текстов
        emb = np.random.randn(len(df), 384).astype(np.float32)
        texts = df["Book"].astype(str).tolist()
        return emb, texts

    def train_head(emb_mat_liked, ratings_arr, epochs=5, batch_size=8, lr=1e-3, st=None):
        # Тренируем простую модель head — заглушка: PyTorch nn.Module-like object
        import torch.nn as nn

        class Head(nn.Module):
            def __init__(self, dim=emb_mat_liked.shape[1]):
                super().__init__()
                self.fc = nn.Linear(dim, 1)

            def forward(self, x):
                return self.fc(x).squeeze(-1)

        model = Head()
        return model

    def load_head(path: str):
        return None

    def recommend_topk_from_liked(df, liked_titles, top_k=10, emb_mat=None, head=None):
        # Простейшая рекомендация: возвращаем книги, не в liked_titles, случайным порядком
        mask = ~df["Book"].isin(liked_titles)
        df2 = df[mask].copy()
        if df2.empty:
            return []
        df2 = df2.reset_index(drop=True)
        # Добавляем score как заглушку
        df2["Score"] = np.random.rand(len(df2))
        df2 = df2.sort_values("Score", ascending=False).head(top_k)
        return df2.to_dict(orient="records")

    class EncoderWrapper:
        pass


# ------------------------------
# Состояние приложения
# ------------------------------

@dataclass
class AppState:
    df: Optional[pd.DataFrame] = None
    emb_mat: Optional[np.ndarray] = None
    texts: Optional[List[str]] = None
    title2idx: Optional[Dict[str, int]] = None
    liked_books: List[str] = None
    top_k: int = 10

    def __post_init__(self):
        if self.liked_books is None:
            self.liked_books = []


# ------------------------------
# Модель таблицы книг
# ------------------------------

class BooksTableModel(QAbstractTableModel):
    def __init__(self, df: Optional[pd.DataFrame] = None, parent=None):
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._df)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._df.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return QVariant()

        if role == Qt.ItemDataRole.DisplayRole:
            row = index.row()
            col = index.column()
            value = self._df.iloc[row, col]
            if isinstance(value, float):
                return f"{value:.3f}"
            return str(value)
        return QVariant()

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role != Qt.ItemDataRole.DisplayRole:
            return QVariant()
        if orientation == Qt.Orientation.Horizontal:
            if self._df is not None and 0 <= section < len(self._df.columns):
                return str(self._df.columns[section])
        return QVariant()

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = df
        self.endResetModel()

    def dataframe(self) -> pd.DataFrame:
        return self._df.copy()


# ------------------------------
# Поток для фоновой работы
# ------------------------------

class BackendWorker(QThread):
    finished_with_recs = pyqtSignal(object, str)  # recs (list|None), error message

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state

    def run(self):
        try:
            df = self.state.df
            if df is None:
                raise RuntimeError("Dataset is not loaded")

            emb_mat, texts = prepare_embeddings_if_needed(
                df, force_recompute=False, show_progress=False
            )
            self.state.emb_mat = emb_mat
            self.state.texts = texts

            if not self.state.liked_books:
                raise RuntimeError("No liked books selected")

            title2idx = {
                row["Book"]: idx
                for idx, row in df.reset_index(drop=True).iterrows()
            }
            self.state.title2idx = title2idx

            rows = []
            ratings = []
            for title in self.state.liked_books:
                if title in title2idx:
                    idx = title2idx[title]
                    rows.append(idx)
                    ratings.append(float(df.iloc[idx]["Avg_Rating"]))

            if not rows:
                raise RuntimeError("Liked books not found in dataset")

            emb_mat_liked = emb_mat[rows]
            ratings_arr = np.array(ratings, dtype=np.float32)

            head_model = train_head(
                emb_mat_liked,
                ratings_arr,
                epochs=30,
                batch_size=min(16, len(rows)),
                lr=5e-4,
                st=None,
            )
            try:
                torch.save(head_model.state_dict(), PERSONALIZED_HEAD_PATH)
            except Exception:
                # В случае заглушки model может не иметь state_dict
                pass

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            head = head_model
            try:
                head = head.to(device)
                head.eval()
            except Exception:
                # Заглушка
                pass

            recs = recommend_topk_from_liked(
                df,
                self.state.liked_books,
                top_k=int(self.state.top_k),
                emb_mat=emb_mat,
                head=head,
            )

            self.finished_with_recs.emit(recs, "")
        except Exception as e:
            self.finished_with_recs.emit(None, str(e))


# ------------------------------
# Окна-просмотры: Catalog, Liked, Recommendations, Log
# ------------------------------

class CatalogWindow(QDialog):
    """Отдельное окно просмотра каталога"""

    def __init__(self, parent: QWidget, state: AppState):
        super().__init__(parent)
        self.setWindowTitle("Каталог книг")
        self.resize(800, 600)
        self.state = state

        layout = QVBoxLayout(self)

        self.books_model = BooksTableModel()
        self.books_proxy = QSortFilterProxyModel()
        self.books_proxy.setSourceModel(self.books_model)

        self.table = QTableView()
        self.table.setModel(self.books_proxy)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)

        btn_layout = QHBoxLayout()
        self.btn_refresh = QPushButton("Обновить")
        self.btn_refresh.clicked.connect(self.refresh)
        btn_layout.addWidget(self.btn_refresh)

        self.btn_export = QPushButton("Экспорт CSV")
        self.btn_export.clicked.connect(self.export_csv)
        btn_layout.addWidget(self.btn_export)

        layout.addLayout(btn_layout)

        self.refresh()

    def refresh(self):
        if self.state.df is not None:
            self.books_model.set_dataframe(self.state.df)
            self.table.resizeColumnsToContents()

    def export_csv(self):
        if self.state.df is None or self.state.df.empty:
            QMessageBox.warning(self, "Нет данных", "Нечего экспортировать")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить CSV", "books_export.csv", "CSV files (*.csv)")
        if path:
            self.state.df.to_csv(path, index=False)
            QMessageBox.information(self, "Экспорт", f"Экспортировано в {path}")


class LikedWindow(QDialog):
    """Окно управления понравившимися книгами"""

    def __init__(self, parent: QWidget, state: AppState):
        super().__init__(parent)
        self.setWindowTitle("Понравившиеся книги")
        self.resize(500, 600)
        self.state = state

        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.search_edit = QLineEdit()
        form.addRow("Найти по названию:", self.search_edit)
        layout.addLayout(form)

        self.btn_add = QPushButton("Добавить по названию")
        self.btn_add.clicked.connect(self.add_book_from_search)
        layout.addWidget(self.btn_add)

        self.liked_list = QListWidget()
        layout.addWidget(self.liked_list)

        btns_layout = QHBoxLayout()
        self.btn_remove = QPushButton("Удалить выделенные")
        self.btn_remove.clicked.connect(self.remove_selected)
        btns_layout.addWidget(self.btn_remove)

        self.btn_clear = QPushButton("Очистить")
        self.btn_clear.clicked.connect(self.clear_all)
        btns_layout.addWidget(self.btn_clear)

        layout.addLayout(btns_layout)

        self.refresh_ui()

    def refresh_ui(self):
        self.liked_list.clear()
        for t in self.state.liked_books:
            self.liked_list.addItem(QListWidgetItem(t))

    def add_book_from_search(self):
        title = self.search_edit.text().strip()
        if not title:
            return
        if self.state.df is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите список книг")
            return
        if title not in self.state.df["Book"].values:
            QMessageBox.warning(self, "Книга не найдена", "Книга с таким названием отсутствует в датасете.")
            return
        if title in self.state.liked_books:
            QMessageBox.information(self, "Уже добавлена", "Эта книга уже в списке")
            return
        self.state.liked_books.append(title)
        self.liked_list.addItem(QListWidgetItem(title))

    def remove_selected(self):
        items = self.liked_list.selectedItems()
        if not items:
            return
        for it in items:
            title = it.text()
            if title in self.state.liked_books:
                self.state.liked_books.remove(title)
            row = self.liked_list.row(it)
            self.liked_list.takeItem(row)

    def clear_all(self):
        self.state.liked_books.clear()
        self.liked_list.clear()


class RecsWindow(QDialog):
    """Окно рекомендаций: запускает фоновую задачу и показывает результат"""

    def __init__(self, parent: QWidget, state: AppState):
        super().__init__(parent)
        self.setWindowTitle("Рекомендации")
        self.resize(900, 600)
        self.state = state

        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        self.spin_topk = QSpinBox()
        self.spin_topk.setRange(1, 100)
        self.spin_topk.setValue(self.state.top_k)
        self.spin_topk.valueChanged.connect(self.on_topk_changed)
        controls.addWidget(QLabel("Top-K:"))
        controls.addWidget(self.spin_topk)

        self.btn_run = QPushButton("Получить рекомендации")
        self.btn_run.clicked.connect(self.get_recommendations)
        controls.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        controls.addWidget(self.progress)

        layout.addLayout(controls)

        self.recs_model = BooksTableModel()
        self.recs_table = QTableView()
        self.recs_table.setModel(self.recs_model)
        self.recs_table.setSortingEnabled(True)
        layout.addWidget(self.recs_table)

        self.setLayout(layout)

    def on_topk_changed(self, value: int):
        self.state.top_k = value

    def get_recommendations(self):
        if self.state.df is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите список книг.")
            return
        if not self.state.liked_books:
            QMessageBox.warning(self, "Нет понравившихся", "Выберите хотя бы одну книгу в liked list.")
            return
        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)
        self.worker = BackendWorker(self.state)
        self.worker.finished_with_recs.connect(self._on_ready)
        self.worker.start()

    def _on_ready(self, recs: Any, error: str):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        if error:
            QMessageBox.critical(self, "Ошибка", f"Не удалось получить рекомендации:\n{error}")
            return
        if not recs:
            QMessageBox.information(self, "Пусто", "Рекомендаций не найдено")
            return
        try:
            df_recs = pd.DataFrame(recs)
        except Exception:
            df_recs = pd.DataFrame(recs)
        self.recs_model.set_dataframe(df_recs)
        self.recs_table.resizeColumnsToContents()


class LogWindow(QDialog):
    """Окно просмотра журнала действий"""

    def __init__(self, parent: QWidget, state: AppState, logger: logging.Logger):
        super().__init__(parent)
        self.setWindowTitle("Журнал действий")
        self.resize(700, 500)
        self.state = state
        self.logger = logger

        layout = QVBoxLayout(self)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.text)

        btns = QHBoxLayout()
        self.btn_clear = QPushButton("Очистить журнал")
        self.btn_clear.clicked.connect(self.clear_log)
        btns.addWidget(self.btn_clear)
        layout.addLayout(btns)

        # Подключаем handler для записи в окно
        class LocalHandler(logging.Handler):
            def __init__(self, widget: QTextEdit):
                super().__init__()
                self.widget = widget

            def emit(self, record):
                msg = self.format(record)
                self.widget.append(msg)

        handler = LocalHandler(self.text)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(handler)

    def clear_log(self):
        self.text.clear()


# ------------------------------
# Главное меню (начальное окно)
# ------------------------------

class MainMenuWindow(QMainWindow):
    """Начальное окно — меню с кнопками, открывающими отдельные окна"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Books Recommender — Menu")
        self.resize(800, 600)

        self.state = AppState()

        self._create_actions()
        self._create_menu()
        self._create_status()
        self._create_central()
        self._setup_logging()

        # подсказка: заранее создаём подокна, но открываем только по запросу
        self.catalog_window = CatalogWindow(self, self.state)
        self.liked_window = LikedWindow(self, self.state)
        self.recs_window = RecsWindow(self, self.state)
        self.log_window = LogWindow(self, self.state, logging.getLogger())

        self._auto_load_dataset()

    def _create_actions(self):
        self.act_load_dataset = QAction("Загрузить книги", self)
        self.act_load_dataset.triggered.connect(self.load_dataset)
        self.act_quit = QAction("Выход", self)
        self.act_quit.triggered.connect(self.close)
        self.act_about = QAction("О программе", self)
        self.act_about.triggered.connect(self.show_about)

    def _create_menu(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("Файл")
        file_menu.addAction(self.act_load_dataset)
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)

        help_menu = menubar.addMenu("Справка")
        help_menu.addAction(self.act_about)

    def _create_status(self):
        self.status = QStatusBar()
        self.setStatusBar(self.status)

    def _create_central(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        title = QLabel("📚 Books Recommender — Меню")
        f = QFont()
        f.setPointSize(18)
        f.setBold(True)
        title.setFont(f)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Выберите раздел приложения")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        # Панель кнопок
        btn_layout = QHBoxLayout()

        self.btn_catalog = QPushButton("Просмотр каталога")
        self.btn_catalog.clicked.connect(self.open_catalog)
        btn_layout.addWidget(self.btn_catalog)

        self.btn_liked = QPushButton("Понравившиеся книги")
        self.btn_liked.clicked.connect(self.open_liked)
        btn_layout.addWidget(self.btn_liked)

        self.btn_recs = QPushButton("Рекомендации")
        self.btn_recs.clicked.connect(self.open_recs)
        btn_layout.addWidget(self.btn_recs)

        self.btn_log = QPushButton("Журнал действий")
        self.btn_log.clicked.connect(self.open_log)
        btn_layout.addWidget(self.btn_log)

        layout.addLayout(btn_layout)

        # Полезные быстрые команды внизу
        bottom_layout = QHBoxLayout()
        self.btn_quick_load = QPushButton("Быстрая загрузка dataset")
        self.btn_quick_load.clicked.connect(self.load_dataset)
        bottom_layout.addWidget(self.btn_quick_load)

        self.btn_quick_clear = QPushButton("Очистить liked")
        self.btn_quick_clear.clicked.connect(self.clear_liked)
        bottom_layout.addWidget(self.btn_quick_clear)

        layout.addLayout(bottom_layout)

    def _setup_logging(self):
        class QtHandler(logging.Handler):
            def __init__(self, status_bar: QStatusBar):
                super().__init__()
                self.status_bar = status_bar

            def emit(self, record):
                msg = self.format(record)
                # показываем краткие сообщения в строке состояния
                self.status_bar.showMessage(msg, 4000)

        handler = QtHandler(self.status)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

    def log(self, msg: str):
        logging.info(msg)

    def _auto_load_dataset(self):
        if os.path.exists(DATA_PATH):
            try:
                self.load_dataset()
            except Exception as e:
                self.log(f"Авто-загрузка не удалась: {e}")

    def load_dataset(self):
        if not os.path.exists(DATA_PATH):
            # prompt for file if not present
            path, _ = QFileDialog.getOpenFileName(self, "Открыть CSV", "", "CSV files (*.csv)")
            if not path:
                QMessageBox.warning(self, "Файл не найден", f"Файл {DATA_PATH} не найден и не выбран.")
                return
            else:
                chosen = path
        else:
            chosen = DATA_PATH

        try:
            df = load_books_dataset(chosen)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить dataset:\n{e}")
            return
        self.state.df = df
        self.state.title2idx = {row["Book"]: idx for idx, row in df.reset_index(drop=True).iterrows()}
        self.log(f"Загружен dataset: {len(df)} записей")

        # обновим данные в подокнах
        self.catalog_window.books_model.set_dataframe(df)
        self.catalog_window.table.resizeColumnsToContents()

        # создаём комплитер для окна liked
        titles = df["Book"].astype(str).tolist()
        self.catalog_window.table.resizeColumnsToContents()
        completer = QCompleter(titles, self)
        completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        completer.setFilterMode(Qt.MatchFlag.MatchStartsWith)
        self.liked_window.search_edit.setCompleter(completer)

    def open_catalog(self):
        self.catalog_window.refresh()
        self.catalog_window.exec()

    def open_liked(self):
        self.liked_window.refresh_ui()
        self.liked_window.exec()

    def open_recs(self):
        self.recs_window.spin_topk.setValue(self.state.top_k)
        self.recs_window.exec()

    def open_log(self):
        self.log_window.exec()

    def clear_liked(self):
        self.state.liked_books.clear()
        QMessageBox.information(self, "Очистка", "Список понравившихся очищен")

    def show_about(self):
        QMessageBox.information(
            self,
            "О программе",
            (
                "<b>Books Recommender — MiniLM</b><br><br>"
                "Приложение для персональных рекомендаций книг.\n"
                "Выберите понравившиеся книги, и система выдаст рекомендации."
            ),
        )

    def closeEvent(self, event: QCloseEvent):
        reply = QMessageBox.question(self, "Выход", "Закрыть приложение?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()


# ------------------------------
# Точка входа
# ------------------------------

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Books Recommender")

    os.makedirs("data", exist_ok=True)

    window = MainMenuWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()


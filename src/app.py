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
)

# Пути как в исходном streamlit-приложении
DATA_PATH = "data/books.csv"
EMB_PATH = "data/book_embeddings.npz"
HEAD_PATH = "data/head_state.pt"
PERSONALIZED_HEAD_PATH = "data/head_personalized.pt"

# Импорт реальных модулей проекта
from data import load_books_dataset
from train_utils import prepare_embeddings_if_needed, train_head, load_head
from recommend import recommend_topk_from_liked
from model import EncoderWrapper


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
        else:
            return str(section + 1)
        return QVariant()

    def set_dataframe(self, df: pd.DataFrame) -> None:
        self.beginResetModel()
        self._df = df
        self.endResetModel()

    def dataframe(self) -> pd.DataFrame:
        return self._df.copy()


# ------------------------------
# Потоки для долгих операций
# ------------------------------

class BackendWorker(QThread):
    """
    Обобщённый worker для последовательности действий:
    - подготовка эмбеддингов (при необходимости)
    - fine-tune head на понравившихся
    - генерация рекомендаций

    Все детали скрыты от пользователя, он просто получает список рекомендаций.
    """

    finished_with_recs = pyqtSignal(object, str)  # recs (list|None), error message

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state

    def run(self):
        try:
            df = self.state.df
            if df is None:
                raise RuntimeError("Dataset is not loaded")

            # 1) Подготовить эмбеддинги (если ещё нет)
            emb_mat, texts = prepare_embeddings_if_needed(
                df, force_recompute=False, show_progress=False
            )
            self.state.emb_mat = emb_mat
            self.state.texts = texts

            # 2) Нанести fine-tune на понравившихся
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

            # 3) Обучить персонализированный head
            head_model = train_head(
                emb_mat_liked,
                ratings_arr,
                epochs=30,
                batch_size=min(16, len(rows)),
                lr=5e-4,
                st=None,
            )
            torch.save(head_model.state_dict(), PERSONALIZED_HEAD_PATH)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # 4) ПОЛНАЯ модель, уже обученная (НЕ load_head!)
            head = head_model.to(device)  # ✅ используем обученную модель напрямую
            head.eval()

            recs = recommend_topk_from_liked(
                df,
                self.state.liked_books,
                top_k=int(self.state.top_k),
                emb_mat=emb_mat,
                head=head,  # передаём обученную модель
            )

            self.finished_with_recs.emit(recs, "")
        except Exception as e:
            self.finished_with_recs.emit(None, str(e))


# ------------------------------
# Главное окно приложения
# ------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("📚 Books Recommender — MiniLM")
        self.resize(1200, 800)

        self.state = AppState()

        self._create_actions()
        self._create_menu_and_toolbar()
        self._create_status_bar()
        self._create_central_layout()
        self._setup_logging()

        self._auto_load_dataset()

    # --------------------------
    # Создание элементов UI
    # --------------------------

    def _create_actions(self):
        self.act_load_dataset = QAction("Загрузить книги", self)
        self.act_load_dataset.triggered.connect(self.load_dataset)

        self.act_quit = QAction("Выход", self)
        self.act_quit.triggered.connect(self.close)

        self.act_about = QAction("О программе", self)
        self.act_about.triggered.connect(self.show_about_dialog)

    def _create_menu_and_toolbar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("Файл")
        file_menu.addAction(self.act_load_dataset)
        file_menu.addSeparator()
        file_menu.addAction(self.act_quit)

        help_menu = menubar.addMenu("Справка")
        help_menu.addAction(self.act_about)

        toolbar = self.addToolBar("Main")
        toolbar.addAction(self.act_load_dataset)
        toolbar.addSeparator()
        toolbar.addAction(self.act_about)

    def _create_status_bar(self):
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Готово")

    def _create_central_layout(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        title_label = QLabel("📚 Books Recommender — MiniLM")
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        title_label.setFont(font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        description = QLabel(
            "Выберите понравившиеся книги и получите персональные рекомендации.\n"
            "Технические детали (эмбеддинги, обучение модели) выполняются автоматически."
        )
        description.setWordWrap(True)
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(description)

        splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(splitter, 1)

        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        splitter.addWidget(top_widget)

        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        splitter.addWidget(bottom_widget)

        # Левая часть: таблица книг
        books_group = QGroupBox("Каталог книг")
        books_layout = QVBoxLayout(books_group)
        self.books_table = QTableView()
        self.books_model = BooksTableModel()
        self.books_proxy = QSortFilterProxyModel()
        self.books_proxy.setSourceModel(self.books_model)
        self.books_table.setModel(self.books_proxy)
        self.books_table.setSortingEnabled(True)
        self.books_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.books_table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        books_layout.addWidget(self.books_table)

        top_layout.addWidget(books_group, 2)

        # Правая часть: выбор понравившихся
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        top_layout.addWidget(right_widget, 1)

        liked_group = QGroupBox("Понравившиеся книги")
        liked_layout = QVBoxLayout(liked_group)

        search_layout = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Начните вводить название книги...")
        search_layout.addWidget(self.search_edit)

        self.btn_add_search = QPushButton("Добавить")
        self.btn_add_search.clicked.connect(self.add_book_from_search)
        search_layout.addWidget(self.btn_add_search)

        liked_layout.addLayout(search_layout)

        self.liked_list = QListWidget()
        liked_layout.addWidget(self.liked_list)

        btns_layout = QHBoxLayout()
        self.btn_remove_selected = QPushButton("Удалить выбранные")
        self.btn_remove_selected.clicked.connect(self.remove_selected_liked)
        btns_layout.addWidget(self.btn_remove_selected)

        self.btn_clear_liked = QPushButton("Очистить список")
        self.btn_clear_liked.clicked.connect(self.clear_liked)
        btns_layout.addWidget(self.btn_clear_liked)

        liked_layout.addLayout(btns_layout)

        liked_group.setLayout(liked_layout)
        right_layout.addWidget(liked_group)

        # Параметр Top-K
        topk_layout = QHBoxLayout()
        lbl_topk = QLabel("Количество рекомендаций (Top-K):")
        topk_layout.addWidget(lbl_topk)
        self.spin_topk = QSpinBox()
        self.spin_topk.setRange(1, 100)
        self.spin_topk.setValue(self.state.top_k)
        self.spin_topk.valueChanged.connect(self.on_topk_changed)
        topk_layout.addWidget(self.spin_topk)
        right_layout.addLayout(topk_layout)

        # Кнопка рекомендаций
        self.btn_recommend = QPushButton("Получить рекомендации")
        self.btn_recommend.clicked.connect(self.get_recommendations)
        self.btn_recommend.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        right_layout.addWidget(self.btn_recommend)

        # Нижняя часть: рекомендации + лог
        rec_and_log_splitter = QSplitter(Qt.Orientation.Horizontal)
        bottom_layout.addWidget(rec_and_log_splitter)

        rec_group = QGroupBox("Рекомендованные книги")
        rec_layout = QVBoxLayout(rec_group)
        self.recs_table = QTableView()
        self.recs_model = BooksTableModel()
        self.recs_proxy = QSortFilterProxyModel()
        self.recs_proxy.setSourceModel(self.recs_model)
        self.recs_table.setModel(self.recs_proxy)
        self.recs_table.setSortingEnabled(True)
        self.recs_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        rec_layout.addWidget(self.recs_table)
        rec_group.setLayout(rec_layout)
        rec_and_log_splitter.addWidget(rec_group)

        log_group = QGroupBox("Журнал действий")
        log_layout = QVBoxLayout(log_group)
        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        log_layout.addWidget(self.log_edit)
        log_group.setLayout(log_layout)
        rec_and_log_splitter.addWidget(log_group)

    def _setup_logging(self):
        class QtLogHandler(logging.Handler):
            def __init__(self, text_widget: QTextEdit):
                super().__init__()
                self.text_widget = text_widget

            def emit(self, record):
                msg = self.format(record)
                self.text_widget.append(msg)
                self.text_widget.verticalScrollBar().setValue(
                    self.text_widget.verticalScrollBar().maximum()
                )

        handler = QtLogHandler(self.log_edit)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

    # --------------------------
    # Логика работы
    # --------------------------

    def log(self, msg: str):
        logging.info(msg)
        self.status.showMessage(msg, 5000)

    def show_about_dialog(self):
        text = (
            "<b>Books Recommender — MiniLM</b><br><br>"
            "Приложение для персональных рекомендаций книг.<br>"
            "Вы выбираете понравившиеся книги, модель автоматически вычисляет эмбеддинги, "
            "обучается и выдаёт список рекомендаций."
        )
        QMessageBox.information(self, "О программе", text)

    def _auto_load_dataset(self):
        """
        Автоматически загружает датасет, если файл существует.
        Пользователь может также вручную вызвать через меню/кнопку.
        """
        if os.path.exists(DATA_PATH):
            try:
                self.load_dataset()
            except Exception as e:
                self.log(f"Не удалось автоматически загрузить dataset: {e}")

    # --------------------------
    # Загрузка датасета
    # --------------------------

    def load_dataset(self):
        if not os.path.exists(DATA_PATH):
            QMessageBox.warning(
                self,
                "Файл не найден",
                f"Файл {DATA_PATH} не найден.\n"
                "Положите books.csv в папку data/ и запустите снова.",
            )
            self.log("Файл books.csv не найден")
            return

        try:
            df = load_books_dataset(DATA_PATH)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить dataset:\n{e}")
            self.log(f"Ошибка загрузки dataset: {e}")
            return

        self.state.df = df
        self.state.title2idx = {
            row["Book"]: idx for idx, row in df.reset_index(drop=True).iterrows()
        }

        self.books_model.set_dataframe(df)
        self.books_table.resizeColumnsToContents()

        # --- НАСТРОЙКА АВТОДОПОЛНЕНИЯ ---

        # 1. Список названий книг
        titles = df["Book"].astype(str).tolist()

        # 2. Создаём комплитер прямо из списка строк
        self.completer = QCompleter(titles, self)
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        # режим: показывать popup с вариантами
        self.completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
        # предлагать только книги, НАЧИНАЮЩИЕСЯ с введённого текста
        self.completer.setFilterMode(Qt.MatchFlag.MatchStartsWith)

        # 3. Назначаем комплитер полю ввода
        self.search_edit.setCompleter(self.completer)

        self.log(f"Загружен dataset: {len(df)} книг")


    # --------------------------
    # Выбор понравившихся
    # --------------------------

    def add_book_from_search(self):
        if self.state.df is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите список книг.")
            return

        title = self.search_edit.text().strip()
        if not title:
            return

        if title not in self.state.df["Book"].values:
            QMessageBox.warning(
                self,
                "Книга не найдена",
                "Книга с таким названием отсутствует в датасете.",
            )
            return

        if title in self.state.liked_books:
            QMessageBox.information(
                self,
                "Уже добавлена",
                "Эта книга уже есть в списке понравившихся.",
            )
            return

        self.state.liked_books.append(title)
        item = QListWidgetItem(title)
        self.liked_list.addItem(item)
        self.search_edit.clear()
        self.log(f"Добавлена понравившаяся книга: {title}")

    def remove_selected_liked(self):
        selected_items = self.liked_list.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            title = item.text()
            if title in self.state.liked_books:
                self.state.liked_books.remove(title)
            row = self.liked_list.row(item)
            self.liked_list.takeItem(row)
            self.log(f"Удалена из понравившихся: {title}")

    def clear_liked(self):
        self.state.liked_books.clear()
        self.liked_list.clear()
        self.log("Список понравившихся очищен")

    def on_topk_changed(self, value: int):
        self.state.top_k = value
        self.log(f"Top-K изменён на {value}")

    # --------------------------
    # Получение рекомендаций
    # --------------------------

    def get_recommendations(self):
        if self.state.df is None:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите список книг.")
            return

        if not self.state.liked_books:
            QMessageBox.warning(
                self,
                "Нет понравившихся",
                "Выберите как минимум одну понравившуюся книгу.",
            )
            return

        self.log("Запуск процесса генерации рекомендаций...")
        self.status.showMessage("Получение рекомендаций, подождите...")

        self.btn_recommend.setEnabled(False)

        self.worker = BackendWorker(self.state)
        self.worker.finished_with_recs.connect(self.on_recommendations_ready)
        self.worker.start()

    def on_recommendations_ready(self, recs: Any, error: str):
        self.btn_recommend.setEnabled(True)
        if error:
            QMessageBox.critical(self, "Ошибка", f"Не удалось получить рекомендации:\n{error}")
            self.log(f"Ошибка рекомендаций: {error}")
            self.status.showMessage("Ошибка при получении рекомендаций")
            return

        if recs is None or len(recs) == 0:
            QMessageBox.information(
                self,
                "Нет рекомендаций",
                "Не удалось подобрать подходящие рекомендации.",
            )
            self.log("Сервис не вернул рекомендаций")
            self.status.showMessage("Рекомендаций нет")
            return

        # recs — список словарей; сделаем DataFrame и отобразим
        try:
            df_recs = pd.DataFrame(recs)
        except Exception:
            df_recs = pd.DataFrame(recs)

        self.recs_model.set_dataframe(df_recs)
        self.recs_table.resizeColumnsToContents()

        self.log(f"Получено рекомендаций: {len(df_recs)}")
        self.status.showMessage("Рекомендации получены")

    # --------------------------
    # Закрытие приложения
    # --------------------------

    def closeEvent(self, event: QCloseEvent):
        reply = QMessageBox.question(
            self,
            "Выход",
            "Закрыть приложение?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            event.accept()
        else:
            event.ignore()


# ------------------------------
# Точка входа
# ------------------------------

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Books Recommender (MiniLM)")

    os.makedirs("data", exist_ok=True)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

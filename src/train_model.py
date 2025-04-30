import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from sentence_transformers import InputExample, SentenceTransformer, losses, util
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Очищаем существующие хендлеры
if logger.handlers:
    logger.handlers.clear()

# Создаем форматтер для логов
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Хендлер для вывода в файл
file_handler = logging.FileHandler("training.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Хендлер для вывода в консоль
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class DialogDataset(Dataset):
    def __init__(self, dialogs: List[Tuple[str, str]], labels: List[int]):
        """
        Датасет для обучения модели на диалогах.

        Args:
            dialogs (List[Tuple[str, str]]): Список кортежей (диалог, тема)
            labels (List[int]): Метки (1 - похожие, 0 - непохожие)
        """
        self.dialogs = dialogs
        self.labels = labels

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx], self.labels[idx]


class InputExamplesDataset(Dataset[InputExample]):
    def __init__(self, examples: List[InputExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> InputExample:
        return self.examples[idx]


def prepare_training_data(
    dialogs: List[Tuple[str, str]],
) -> List[InputExample]:
    """
    Подготовка данных для обучения в формате InputExample.
    Создает пары похожих и непохожих диалогов на основе их тем.

    Args:
        dialogs (List[Tuple[str, str]]): Список кортежей (диалог, тема)
                                        Темы: "credit", "deposit", "card", etc.

    Returns:
        List[InputExample]: Список примеров для обучения
    """
    logger.info(f"Подготовка обучающих пар из {len(dialogs)} диалогов")

    examples = []

    # Группируем диалоги по темам
    dialogs_by_topic: Dict[str, List[str]] = {}
    for dialog, topic in dialogs:
        if topic not in dialogs_by_topic:
            dialogs_by_topic[topic] = []
        dialogs_by_topic[topic].append(dialog)

    # Создаем положительные примеры (диалоги одной темы)
    pos_pairs = 0
    for topic, topic_dialogs in dialogs_by_topic.items():
        for i in range(len(topic_dialogs)):
            for j in range(i + 1, len(topic_dialogs)):
                examples.append(
                    InputExample(
                        texts=[topic_dialogs[i], topic_dialogs[j]],
                        label=1.0,  # Похожие диалоги (одна тема)
                    )
                )
                pos_pairs += 1

    # Создаем отрицательные примеры (диалоги разных тем)
    neg_pairs = 0
    topics = list(dialogs_by_topic.keys())

    # Вычисляем желаемое количество отрицательных пар на основе количества положительных
    target_neg_pairs = pos_pairs  # Стремимся к балансу 1:1

    # Создаем отрицательные пары между всеми темами
    for i in range(len(topics)):
        for j in range(i + 1, len(topics)):
            topic1_dialogs = dialogs_by_topic[topics[i]]
            topic2_dialogs = dialogs_by_topic[topics[j]]

            # Вычисляем, сколько пар нужно создать между этими темами
            pairs_per_topic_pair = max(
                1,
                min(
                    len(topic1_dialogs)
                    * len(topic2_dialogs),  # Максимально возможное количество пар
                    target_neg_pairs
                    // (
                        len(topics) * (len(topics) - 1) // 2
                    ),  # Равномерное распределение между парами тем
                ),
            )

            # Создаем случайные пары
            for _ in range(pairs_per_topic_pair):
                i1 = random.randrange(len(topic1_dialogs))
                i2 = random.randrange(len(topic2_dialogs))
                examples.append(
                    InputExample(
                        texts=[topic1_dialogs[i1], topic2_dialogs[i2]],
                        label=0.0,  # Непохожие диалоги (разные темы)
                    )
                )
                neg_pairs += 1

                # Если достигли желаемого количества отрицательных пар, прекращаем
                if neg_pairs >= target_neg_pairs:
                    break

            if neg_pairs >= target_neg_pairs:
                break
        if neg_pairs >= target_neg_pairs:
            break

    logger.info(f"Создано пар для обучения:")
    logger.info(f"  Положительные пары (одна тема): {pos_pairs}")
    logger.info(f"  Отрицательные пары (разные темы): {neg_pairs}")
    logger.info(f"  Всего пар: {len(examples)}")
    logger.info(f"  Соотношение positive:negative = 1:{neg_pairs/pos_pairs:.2f}")

    return examples


def load_dialogs_from_excel(file_path: str) -> List[Tuple[str, str]]:
    """
    Загружает диалоги и их метки из Excel файла.

    Args:
        file_path: Путь к Excel файлу с колонками 'dialog' и 'label'

    Returns:
        List[Tuple[str, str]]: Список кортежей (диалог, метка)
    """
    logger.info(f"Загрузка данных из файла: {file_path}")

    df = pd.read_excel(file_path)
    if "dialog" not in df.columns or "label" not in df.columns:
        logger.error("Excel файл не содержит необходимые колонки 'dialog' и 'label'")
        raise ValueError("Excel файл должен содержать колонки 'dialog' и 'label'")

    dialogs = list(zip(df["dialog"].tolist(), df["label"].tolist()))
    logger.info(f"Загружено {len(dialogs)} диалогов")

    # Логируем распределение меток
    label_distribution = df["label"].value_counts()
    logger.info("Распределение меток в данных:")
    for label, count in label_distribution.items():
        logger.info(f"  {label}: {count}")

    return dialogs


def split_train_val_data(
    dialogs: List[Tuple[str, str]], val_size: float = 0.2
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Разделяет данные на обучающую и валидационную выборки с сохранением пропорций тем.

    Args:
        dialogs: Список кортежей (диалог, тема)
        val_size: Доля валидационной выборки

    Returns:
        Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]: (train_dialogs, val_dialogs)
    """
    # Группируем диалоги по темам
    dialogs_by_topic: Dict[str, List[Tuple[str, str]]] = {}
    for dialog, topic in dialogs:
        if topic not in dialogs_by_topic:
            dialogs_by_topic[topic] = []
        dialogs_by_topic[topic].append((dialog, topic))

    train_dialogs, val_dialogs = [], []

    # Стратифицированное разделение для каждой темы
    for topic, topic_dialogs in dialogs_by_topic.items():
        topic_train, topic_val = train_test_split(
            topic_dialogs, test_size=val_size, random_state=42
        )
        train_dialogs.extend(topic_train)
        val_dialogs.extend(topic_val)

    # Перемешиваем выборки
    random.shuffle(train_dialogs)
    random.shuffle(val_dialogs)

    logger.info(f"Разделение данных:")
    logger.info(f"  Обучающая выборка: {len(train_dialogs)} диалогов")
    logger.info(f"  Валидационная выборка: {len(val_dialogs)} диалогов")

    # Логируем распределение тем
    for topic in dialogs_by_topic.keys():
        train_count = sum(1 for d in train_dialogs if d[1] == topic)
        val_count = sum(1 for d in val_dialogs if d[1] == topic)
        logger.info(f"  Тема '{topic}':")
        logger.info(f"    Обучение: {train_count}, Валидация: {val_count}")

    return train_dialogs, val_dialogs


class EarlyStoppingCallback:
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0

    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
            return False

        self.counter += 1
        if self.counter >= self.patience:
            logger.info(
                f"Early stopping! Лучшая метрика {self.best_score:.4f} на эпохе {self.best_epoch}"
            )
            return True

        return False


def collate_input_examples(batch):
    """
    Функция для объединения InputExample в батч.
    """
    texts = []
    labels = []
    for example in batch:
        texts.append(example.texts)
        labels.append(example.label)
    return texts, torch.tensor(labels)


def evaluate_model(
    model: SentenceTransformer, val_examples: List[InputExample]
) -> float:
    """
    Оценивает модель на валидационной выборке.

    Returns:
        float: Средняя точность на валидационной выборке
    """
    val_dataloader = DataLoader(
        InputExamplesDataset(val_examples),
        shuffle=False,
        batch_size=32,
        collate_fn=collate_input_examples,
    )

    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for batch_texts, labels in val_dataloader:
            # Транспонируем список текстов, чтобы получить отдельно первые и вторые тексты из пар
            texts1 = [pair[0] for pair in batch_texts]
            texts2 = [pair[1] for pair in batch_texts]

            # Получаем эмбеддинги для обоих текстов в паре
            embeddings1 = model.encode(texts1, convert_to_tensor=True)
            embeddings2 = model.encode(texts2, convert_to_tensor=True)

            # Вычисляем косинусное сходство
            similarities = util.pytorch_cos_sim(embeddings1, embeddings2).diagonal()

            # Предсказываем метки (1 если сходство > 0.5, иначе 0)
            predictions = (similarities > 0.5).float()

            # Сравниваем с истинными метками
            labels = labels.to(predictions.device)
            correct += (predictions == labels).sum().item()
            total += len(labels)

    accuracy = correct / total
    return accuracy


def train_model(
    model_name: str = "deepvk/USER-bge-m3",
    train_dialogs: Optional[List[Tuple[str, str]]] = None,
    excel_path: Optional[str] = None,
    output_dir: str = "trained_model",
    batch_size: int = 16,
    epochs: int = 10,
    learning_rate: float = 2e-5,
    warmup_steps: int = 100,
    patience: int = 3,
    val_size: float = 0.2,
):
    """
    Обучение модели на диалогах с ранней остановкой.

    Args:
        model_name: Название предобученной модели
        train_dialogs: Список кортежей (диалог, тема)
        excel_path: Путь к Excel файлу с данными
        output_dir: Директория для сохранения модели
        batch_size: Размер батча
        epochs: Максимальное количество эпох
        learning_rate: Скорость обучения
        warmup_steps: Количество шагов для разогрева
        patience: Количество эпох для early stopping
        val_size: Доля валидационной выборки
    """
    logger.info("Начало процесса обучения")

    # Загружаем данные
    if excel_path is not None:
        dialogs = load_dialogs_from_excel(excel_path)
    elif train_dialogs is not None:
        dialogs = train_dialogs
    else:
        raise ValueError("Необходимо указать либо train_dialogs, либо excel_path")

    # Разделяем на обучающую и валидационную выборки
    train_dialogs, val_dialogs = split_train_val_data(dialogs, val_size)

    # Подготавливаем данные
    train_examples = prepare_training_data(train_dialogs)
    val_examples = prepare_training_data(val_dialogs)

    # Инициализируем модель и оптимизатор
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=device)

    train_dataloader = DataLoader(
        InputExamplesDataset(train_examples), shuffle=True, batch_size=batch_size
    )

    train_loss = losses.CosineSimilarityLoss(model)

    # Инициализируем early stopping
    early_stopping = EarlyStoppingCallback(patience=patience)
    best_val_accuracy = 0

    # Обучаем модель
    for epoch in range(epochs):
        logger.info(f"\nЭпоха {epoch + 1}/{epochs}")

        # Обучение на одной эпохе
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=1,
            warmup_steps=warmup_steps if epoch == 0 else 0,
            optimizer_params={"lr": learning_rate},
        )

        # Оценка на валидационной выборке
        val_accuracy = evaluate_model(model, val_examples)
        logger.info(f"Валидационная точность: {val_accuracy:.4f}")

        # Сохраняем лучшую модель
        if val_accuracy > best_val_accuracy:
            logger.info(f"Сохраняем лучшую модель (точность: {val_accuracy:.4f})")
            model.save(output_dir)
            best_val_accuracy = val_accuracy

        # Проверяем критерий ранней остановки
        if early_stopping(val_accuracy, epoch):
            break

    logger.info(
        f"\nОбучение завершено. Лучшая валидационная точность: {best_val_accuracy:.4f}"
    )


if __name__ == "__main__":
    try:
        # Пример использования с Excel файлом
        train_model(excel_path="dialogs.xlsx")
    except Exception as e:
        logger.error(f"Ошибка при обучении модели: {str(e)}", exc_info=True)

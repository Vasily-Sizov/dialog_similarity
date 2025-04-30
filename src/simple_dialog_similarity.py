from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class DialogSimilarity:
    def __init__(self, threshold: float = 0.7):
        """
        Инициализация класса DialogSimilarity.

        Args:
            threshold (float): Порог схожести для определения связанности диалогов.
            По умолчанию 0.7 (70% схожести).
        """
        self.model = SentenceTransformer("deepvk/USER-bge-m3")
        self.threshold = threshold

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Преобразование списка текстов в эмбеддинги с помощью модели USER-bge-m3.

        Args:
            texts (List[str]): Список текстов для преобразования в эмбеддинги.

        Returns:
            np.ndarray: Массив эмбеддингов.
        """
        return self.model.encode(texts, normalize_embeddings=True)

    def is_related(self, dialogs: List[str], query_dialog: str) -> bool:
        """
        Проверка, связан ли проверяемый диалог семантически со списком диалогов.

        Args:
            dialogs (List[str]): Список эталонных диалогов.
            query_dialog (str): Диалог для проверки схожести.

        Returns:
            bool: True, если проверяемый диалог связан с эталонными диалогами,
            False в противном случае.
        """
        # Получаем эмбеддинги для всех диалогов
        dialog_embeddings = self._get_embeddings(dialogs)
        query_embedding = self._get_embeddings([query_dialog])

        # Вычисляем косинусную схожесть между проверяемым и всеми диалогами
        similarities = cosine_similarity(query_embedding, dialog_embeddings)

        # Проверяем, есть ли оценки схожести выше порога
        return bool(np.any(similarities >= self.threshold))

    def get_similarity_scores(
        self, dialogs: List[str], query_dialog: str
    ) -> List[float]:
        """
        Получение оценок схожести между проверяемым диалогом и всеми эталонными диалогами.

        Args:
            dialogs (List[str]): Список эталонных диалогов.
            query_dialog (str): Диалог для проверки схожести.

        Returns:
            List[float]: Список оценок схожести от 0 до 1.
        """
        dialog_embeddings = self._get_embeddings(dialogs)
        query_embedding = self._get_embeddings([query_dialog])

        similarities = cosine_similarity(query_embedding, dialog_embeddings)
        return similarities[0].tolist()


if __name__ == "__main__":
    # Пример использования
    dialogs = [
        """Клиент: Здравствуйте, я хотел бы узнать, какие у вас есть кредитные предложения?
Менеджер: Добрый день! У нас есть несколько программ: потребительские кредиты, 
автокредиты и ипотека.
Клиент: Пока рассматриваю потребительский наличными. Какие условия?""",
        """Клиент: Мне нужен кредит на развитие бизнеса.
Менеджер: У нас есть специальные программы для предпринимателей. Какая сумма вам интересна?
Клиент: Примерно 2 миллиона рублей.""",
        """Клиент: Хочу рефинансировать кредит из другого банка.
Менеджер: Мы можем предложить ставку от 8.5%. Какой остаток по кредиту?""",
    ]

    query_dialog = (
        "Клиент: Здравствуйте, я хочу открыть депозит. Какие у вас есть предложения?"
    )

    # Создаем экземпляр класса
    comparator = DialogSimilarity(threshold=0.7)

    # Проверяем связанность диалогов
    is_related = comparator.is_related(dialogs, query_dialog)
    print(f"Диалоги связаны: {'Да' if is_related else 'Нет'}")

    # Получаем оценки схожести
    scores = comparator.get_similarity_scores(dialogs, query_dialog)
    print("\nОценки схожести:")
    for i, score in enumerate(scores, 1):
        print(f"Диалог {i}: {score:.2f}")

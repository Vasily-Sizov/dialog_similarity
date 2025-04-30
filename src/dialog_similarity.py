import logging
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer, util

# Настройка логирования
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Очищаем существующие хендлеры
if logger.handlers:
    logger.handlers.clear()

# Создаем форматтер для логов
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class DialogSimilarity:
    def __init__(
        self,
        model_name: str = "deepvk/USER-bge-m3",
        model_path: str = None,
        threshold: float = 0.5,
    ):
        """
        Инициализация класса DialogSimilarity.

        Args:
            model_name: Название базовой модели
            model_path: Путь к дообученной модели (если None, используется базовая)
            threshold: Порог схожести для определения связанности диалогов
        """
        self.model_type = "fine-tuned" if model_path else "base"
        self.model = (
            SentenceTransformer(model_path)
            if model_path
            else SentenceTransformer(model_name)
        )
        self.threshold = threshold

        # Веса для разных частей диалога
        self.weights = {
            "first_client": 2.0,  # Первая реплика клиента - самая важная
            "first_manager": 1.5,  # Первая реплика менеджера
            "client": 1.2,  # Остальные реплики клиента
            "manager": 1.0,  # Остальные реплики менеджера
        }

    def _preprocess_text(self, text: str) -> str:
        """
        Предобработка текста.

        Args:
            text (str): Исходный текст.

        Returns:
            str: Обработанный текст.
        """
        # Приводим к нижнему регистру и удаляем лишние пробелы
        return " ".join(text.lower().split())

    def _split_dialog(self, dialog: str) -> Dict[str, List[Tuple[str, float]]]:
        """
        Разделяет диалог на реплики с учетом их важности.

        Args:
            dialog (str): Текст диалога.

        Returns:
            Dict[str, List[Tuple[str, float]]]: Словарь с репликами и их весами.
        """
        lines = dialog.split("\n")
        result = {"client": [], "manager": []}

        for i, line in enumerate(lines):
            if not line.strip():
                continue

            # Определяем роль и вес реплики
            if line.startswith("Клиент:"):
                role = "client"
                weight = (
                    self.weights["first_client"] if i == 0 else self.weights["client"]
                )
                text = line.replace("Клиент:", "").strip()
            elif line.startswith("Менеджер:"):
                role = "manager"
                weight = (
                    self.weights["first_manager"] if i == 1 else self.weights["manager"]
                )
                text = line.replace("Менеджер:", "").strip()
            else:
                continue

            # Предобрабатываем текст
            processed_text = self._preprocess_text(text)
            result[role].append((processed_text, weight))

        return result

    def _compute_role_similarity(
        self,
        query_parts: Dict[str, List[Tuple[str, float]]],
        dialog_parts: Dict[str, List[Tuple[str, float]]],
    ) -> float:
        """
        Вычисляет схожесть между диалогами с учетом весов.

        Args:
            query_parts (Dict[str, List[Tuple[str, float]]]): Реплики запроса с весами.
            dialog_parts (Dict[str, List[Tuple[str, float]]]): Реплики диалога с весами.

        Returns:
            float: Оценка схожести с учетом весов.
        """
        role_scores = []
        total_weight = 0.0

        # Сравниваем реплики клиентов
        if query_parts["client"] and dialog_parts["client"]:
            for (q_text, q_weight), (d_text, d_weight) in zip(
                query_parts["client"], dialog_parts["client"]
            ):
                q_emb = self._get_embeddings([q_text])
                d_emb = self._get_embeddings([d_text])
                similarity = util.semantic_search(q_emb, d_emb, top_k=1)[0][0]["score"]
                weight = (q_weight + d_weight) / 2
                role_scores.append(similarity * weight)
                total_weight += weight

        # Сравниваем реплики менеджеров
        if query_parts["manager"] and dialog_parts["manager"]:
            for (q_text, q_weight), (d_text, d_weight) in zip(
                query_parts["manager"], dialog_parts["manager"]
            ):
                q_emb = self._get_embeddings([q_text])
                d_emb = self._get_embeddings([d_text])
                similarity = util.semantic_search(q_emb, d_emb, top_k=1)[0][0]["score"]
                weight = (q_weight + d_weight) / 2
                role_scores.append(similarity * weight)
                total_weight += weight

        # Вычисляем итоговую оценку
        if role_scores:
            return sum(role_scores) / total_weight
        return 0.0

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Преобразование списка текстов в эмбеддинги.

        Args:
            texts (List[str]): Список текстов для преобразования.

        Returns:
            np.ndarray: Массив эмбеддингов.
        """
        return self.model.encode(
            texts, normalize_embeddings=True, convert_to_tensor=True
        )

    def is_related(self, dialogs: List[str], query_dialog: str) -> bool:
        """
        Проверка, связан ли проверяемый диалог семантически со списком диалогов.

        Args:
            dialogs (List[str]): Список эталонных диалогов.
            query_dialog (str): Диалог для проверки схожести.

        Returns:
            bool: True, если средняя схожесть выше порога И минимальная схожесть
                  выше минимального порога, False в противном случае.
        """
        scores = self.get_similarity_scores(dialogs, query_dialog)
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        return (
            avg_score >= self.threshold and min_score >= 0.6
        )  # Увеличили минимальный порог до 0.6

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
        query_parts = self._split_dialog(query_dialog)
        scores = []

        for dialog in dialogs:
            dialog_parts = self._split_dialog(dialog)
            score = self._compute_role_similarity(query_parts, dialog_parts)
            scores.append(score)

        return scores


def compare_models(
    base_model: DialogSimilarity,
    fine_tuned_model: DialogSimilarity,
    test_dialogs: List[str],
    test_queries: Dict[str, str],
) -> None:
    """
    Сравнивает работу базовой и дообученной моделей.

    Args:
        base_model: Экземпляр базовой модели
        fine_tuned_model: Экземпляр дообученной модели
        test_dialogs: Список эталонных диалогов
        test_queries: Словарь с тестовыми запросами {название: текст}
    """
    print("\nСравнение базовой и дообученной модели:")
    print("-" * 80)

    for query_name, query_dialog in test_queries.items():
        print(f"\nТест: {query_name}")
        print("=" * 50)

        # Оценки базовой модели
        base_scores = base_model.get_similarity_scores(test_dialogs, query_dialog)
        base_avg = sum(base_scores) / len(base_scores)
        base_min = min(base_scores)
        base_related = base_model.is_related(test_dialogs, query_dialog)

        # Оценки дообученной модели
        tuned_scores = fine_tuned_model.get_similarity_scores(
            test_dialogs, query_dialog
        )
        tuned_avg = sum(tuned_scores) / len(tuned_scores)
        tuned_min = min(tuned_scores)
        tuned_related = fine_tuned_model.is_related(test_dialogs, query_dialog)

        # Вывод результатов в таблице
        print("\nПоказатель         Базовая модель    Дообученная модель    Разница")
        print("-" * 70)
        print(
            f"Средняя схожесть    {base_avg:.3f}           {tuned_avg:.3f}            {tuned_avg-base_avg:+.3f}"
        )
        print(
            f"Минимальная схожесть {base_min:.3f}           {tuned_min:.3f}            {tuned_min-base_min:+.3f}"
        )
        print(
            f"Диалоги связаны     {'Да' if base_related else 'Нет'}              {'Да' if tuned_related else 'Нет'}"
        )

        print("\nДетальное сравнение:")
        print("-" * 70)
        print("Диалог    Базовая    Дообученная    Разница")
        for i, (base, tuned) in enumerate(zip(base_scores, tuned_scores), 1):
            diff = tuned - base
            print(f"{i:^6d}     {base:.3f}      {tuned:.3f}       {diff:+.3f}")
        print()


if __name__ == "__main__":
    # Тестовые диалоги
    test_dialogs = [
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

    # Тестовые запросы
    test_queries = {
        "Кредит на развитие ИП": """Клиент: Нужен кредит на развитие ИП. Какие варианты?
Менеджер: У нас есть программы от 10% годовых. Какой срок и сумма?
Клиент: 1 млн руб. на 2 года.
Менеджер: Хорошо, потребуются выписки по счетам и налоговая отчетность.""",
        "Открытие депозита": """Клиент: Здравствуйте, я хочу открыть депозит. 
Какие у вас есть предложения?
Менеджер: Добрый день! У нас несколько вариантов вкладов с разными условиями. 
Вас интересует краткосрочный вклад или долгосрочный с максимальной ставкой?
Клиент: Хотелось бы под хороший процент, но с возможностью снять деньги в случае 
необходимости.
Менеджер: Тогда рекомендую "Гибкий" депозит – ставка 7%, частичное снятие без 
потери процентов. Или "Надежный+" – 8,5%, но снятие только в конце срока.""",
        "Эквайринг": """Клиент: У нас крупный интернет-магазин с большим количеством платежей. Какие условия эквайринга вы можете предложить?
Менеджер: Добрый день! Для высоконагруженных проектов у нас есть специальные тарифы. Какой у вас среднемесячный оборот по карточным платежам?
Клиент: Около 15-20 миллионов рублей в месяц.
Менеджер: Отлично! В таком случае мы можем предложить вам индивидуальный тариф с комиссией от 1,3%.""",
        "Кредит2": """Клиент: Здравствуйте, мне срочно нужны деньги на лечение. Можно оформить кредит?
Менеджер: Добрый день! Да, у нас есть кредиты на неотложные нужды. На какую сумму вы рассчитываете?
Клиент: Около 150 000 рублей. Возможно ли получить их сегодня?
Менеджер: Да, при наличии паспорта и подтверждения дохода (справка 2-НДФЛ или выписка по зарплатной карте) рассмотрим заявку за 1 час. Ставка — от 12% годовых.
Клиент: У меня есть справка о зарплате. А если я погашу досрочно, будут штрафы?
Менеджер: Нет, досрочное погашение без комиссий. Хотите оформить заявку сейчас?
Клиент: Да, давайте.
Менеджер: Хорошо, заполним анкету. Деньги поступят на вашу карту сразу после одобрения.""",
    }

    try:
        # Создаем экземпляры моделей
        base_model = DialogSimilarity(threshold=0.6)
        fine_tuned_model = DialogSimilarity(model_path="trained_model", threshold=0.6)

        # Запускаем сравнение
        compare_models(base_model, fine_tuned_model, test_dialogs, test_queries)

    except Exception as e:
        logger.error(f"Ошибка при сравнении моделей: {str(e)}", exc_info=True)

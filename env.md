# Установка и настройка Poetry
pip install poetry
# Настраиваем Poetry для создания виртуального окружения в папке проекта
poetry config virtualenvs.in-project true

poetry init

# Добавление источника PyTorch для GPU
poetry source add --priority=explicit pytorch-gpu https://download.pytorch.org/whl/cu121
poetry add --source pytorch-gpu torch torchvision torchaudio

poetry add scikit-learn
poetry add sentence-transformers
poetry add transformers
poetry add tqdm
poetry add pandas matplotlib
poetry shell

# Для проверки установки можно выполнить в Python:
import torch
print(f"PyTorch версия: {torch.__version__}")
print(f"CUDA доступна: {torch.cuda.is_available()}")
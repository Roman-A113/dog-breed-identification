Скачайте датасет с Kaggle, распакуйте его, переименуйте в `data` и добавьте в склонированный репозиторий.

https://www.kaggle.com/competitions/dog-breed-identification/data

## Запуск без Docker

### 1. Создайте виртуальное окружение

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Установите зависимости

#### Для CPU

```powershell
pip install --no-cache-dir -r requirements-base.txt
pip install --no-cache-dir -r requirements-torch-cpu.txt
```

#### Для NVIDIA GPU

```powershell
pip install --no-cache-dir -r requirements-base.txt
pip install --no-cache-dir -r requirements-torch-nvidia-gpu.txt
```

### 3. Запустите

```powershell
python solve.py
```

### 4. Результат

После выполнения файл предсказаний появится в:

- `data/my_submission.csv`

---

## Запуск через Docker

#### Для CPU

```cmd
docker run -v "path_to_data_repository":/app/data romakolesn/dog-breed-identification:latest
```

#### Для NVIDIA GPU

```cmd
docker run --gpus all -v "path_to_data_repository":/app/data romakolesn/dog-breed-identification:latest
```

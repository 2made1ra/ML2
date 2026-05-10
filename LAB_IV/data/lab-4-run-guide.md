# Гайд по запуску lab-4.ipynb

Этот гайд нужен перед запуском ноутбука `LAB_IV/solution/lab-4.ipynb`.
Ноутбук использует обязательные модели `LogisticRegression`, `LinearSVC` и
`CatBoostClassifier`, подбирает гиперпараметры через Optuna, строит графики через
Plotly и готовит датасет в `LAB_IV/data/dataset`.

## 1. Что изменилось в окружении

Проект рассчитан на Python 3.14:

```text
requires-python = ">=3.14,<3.15"
```

Файл `.python-version` тоже закреплен на `3.14`, поэтому после удаления старого
`.venv` uv должен создать новое окружение на Python 3.14.

PyTorch разведен по платформам:

- macOS arm64 получает CPU-сборку `torch` и `torchvision`;
- Linux x86_64 и Windows AMD64 получают CUDA 12.6-сборку `torch` и
  `torchvision`;
- универсальный lock должен проверять обе целевые платформы через
  `tool.uv.environments` и `tool.uv.required-environments`.

Это убирает ошибку, когда macOS пытается установить wheel вида
`torch==...+cu126`, который публикуется только для Linux/Windows.

## 2. Переустановка окружения на macOS

Запускайте команды из корня проекта:

```bash
cd /Users/2madeira/dev/maga/second/ml
rm -rf .venv
uv python install 3.14
uv sync --dev
```

На macOS `uv sync --dev` должен поставить CPU-версию PyTorch. CUDA на macOS не
используется.

Если `uv.lock` расходится с текущим `pyproject.toml`, uv при обычном
`uv sync --dev` должен пересобрать lock. Если хочется сделать это явно:

```bash
uv lock
uv sync --dev
```

## 3. Установка на машине с CUDA GPU

На Linux x86_64 используйте те же команды:

```bash
cd /path/to/ml
uv python install 3.14
uv sync --dev
```

На Windows AMD64 команды аналогичные, но путь к проекту будет вашим локальным
путем. Из-за marker-условий в `pyproject.toml` Linux/Windows-окружение будет
брать PyTorch из индекса CUDA 12.6:

```text
https://download.pytorch.org/whl/cu126
```

После установки проверьте, что CUDA видна PyTorch:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Если CUDA-драйвер на машине не совместим с CUDA 12.6, нужно менять индекс в
`pyproject.toml` на подходящий, например `cu128` или `cu130`, и обновлять оба
источника для `torch` и `torchvision`.

## 4. Основные библиотеки для lab-4

Основной набор зависимостей уже описан в `pyproject.toml`. Для ноутбука важны:

- `numpy`, `pandas` - работа с данными и таблицами;
- `scikit-learn` - TF-IDF, train/test split, CV, метрики, LR, LinearSVC, LSA;
- `catboost` - обязательная третья модель в baseline и Optuna-оптимизации;
- `optuna` - подбор гиперпараметров моделей и TF-IDF внутри `Pipeline`;
- `spacy` - лемматизация русских отзывов;
- `tqdm` - прогресс-бары для предобработки;
- `plotly`, `kaleido` - интерактивные графики и статический экспорт;
- `wordcloud` - облака слов, которые затем отображаются через Plotly;
- `kagglehub` - скачивание Kaggle-датасета в формате папок `pos/neg/neu`;
- `jupyter`, `ipykernel` - запуск ноутбука из uv-окружения.

Не добавляйте эти пакеты через `uv add` перед обычным запуском: они уже должны
ставиться через `uv sync --dev`. Используйте `uv add` только если осознанно
меняете зависимости проекта.

## 5. Установить русскую модель spaCy

После `uv sync --dev` скачайте языковую модель:

```bash
uv run python -m spacy download ru_core_news_sm
```

Без `ru_core_news_sm` ноутбук остановится в ячейке загрузки spaCy-модели.
Эта модель используется для лемматизации, а не для NER или синтаксического
разбора.

## 6. Подключить uv-окружение как Jupyter kernel

Чтобы ноутбук точно запускался в нужном окружении:

```bash
uv run python -m ipykernel install --user --name ml-labs --display-name "ML_LABS (uv)"
uv run jupyter lab
```

В Jupyter выберите kernel `ML_LABS (uv)`.

## 7. Подготовка датасета

Ноутбук работает с настоящим форматом Kaggle-датасета
`mikhailklemin/kinopoisks-movies-reviews`: внутри архива лежат папки классов
`pos`, `neg`, `neu`, а каждый отзыв хранится отдельным `.txt`-файлом.
Имя файла имеет вид:

```text
movie_id-review_id.txt
```

Например, `306-15.txt` означает отзыв `15` к фильму с id `306`.

В разделе 2.1 ноутбук сам создает директорию:

```text
LAB_IV/data/dataset
```

Дальше возможны два сценария:

- если вы уже скачали и распаковали датасет вручную, положите папки
  `pos`, `neg`, `neu` прямо в `LAB_IV/data/dataset`;
- если этих папок нет, ноутбук попробует скачать датасет сразу в
  `LAB_IV/data/dataset` через
  `kagglehub.dataset_download("mikhailklemin/kinopoisks-movies-reviews")`,
  найти внутри скачанной директории папки `pos/neg/neu` и привести их к
  ожидаемому расположению.
  Если установленная версия `kagglehub` не поддерживает `output_dir`, ноутбук
  использует стандартный cache KaggleHub и затем копирует `pos/neg/neu` в
  `LAB_IV/data/dataset`.

После этого ноутбук собирает рабочую таблицу:

```text
LAB_IV/data/dataset/kinopoisk_reviews_prepared.csv
```

Для скачивания через `kagglehub` нужен доступ в интернет. Если Kaggle попросит
авторизацию, проще заранее скачать архив вручную с Kaggle, распаковать его и
положить папки `pos`, `neg`, `neu` в `LAB_IV/data/dataset`.

## 8. Особенности запуска ноутбука

- Запускайте ноутбук из корня проекта или из `LAB_IV/solution`: код ищет
  директорию `LAB_IV/data` относительно текущего пути.
- Первая тяжелая часть - spaCy-лемматизация отзывов. Она может занять заметное
  время на полном датасете.
- Вторая тяжелая часть - Optuna: сейчас задано `N_OPTUNA_TRIALS = 20` для каждой
  из трех моделей и 3-fold CV. CatBoost особенно дорогой по времени.
- Если запуск слишком долгий, временно уменьшите `N_OPTUNA_TRIALS` до 5-10,
  проверьте, что весь ноутбук проходит, а затем верните больше trials для
  финального результата.
- `CatBoostClassifier` в ноутбуке обязателен. Если он не установлен, ноутбук
  должен падать сразу, потому что CatBoost участвует и в baseline, и в улучшенной
  Optuna-версии.
- Для `LinearSVC` выводимый скор называется `margin`, а не вероятность. Это
  нормально: `decision_function` у LinearSVC не является вероятностью.
- Графики построены через Plotly. В интерактивном Jupyter они должны открываться
  прямо под ячейками; `kaleido` нужен только для статического экспорта.

## 9. Быстрые проверки после установки

Проверьте импорты:

```bash
uv run python -c "import numpy, pandas, sklearn, plotly, wordcloud, optuna, catboost, spacy, kagglehub; print('lab-4 imports OK')"
```

Проверьте spaCy-модель:

```bash
uv run python -m spacy validate
```

На macOS можно дополнительно проверить, что PyTorch не CUDA-сборка:

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

Ожидаемо: `torch.cuda.is_available()` вернет `False`.

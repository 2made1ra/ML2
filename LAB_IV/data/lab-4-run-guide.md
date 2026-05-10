# Гайд по запуску lab-4.ipynb

Этот гайд нужен перед запуском ноутбука `LAB_IV/solution/lab-4.ipynb`.
Ноутбук использует обязательные модели `LogisticRegression`, `LinearSVC` и
`CatBoostClassifier`, подбирает гиперпараметры через Optuna, строит графики через
Plotly и готовит датасет в `LAB_IV/data/dataset`.

## 1. Установить зависимости через uv

Запускайте команды из корня проекта:

```bash
cd /Users/2madeira/dev/maga/second/ml
```

Если вы ведете зависимости через `pyproject.toml`, добавьте библиотеки так:

```bash
uv add numpy pandas scikit-learn plotly kaleido wordcloud optuna catboost kagglehub
uv add --dev jupyter ipykernel spacy tqdm
uv sync --dev
```

Если вы хотите просто доустановить пакеты в уже созданное виртуальное окружение,
можно использовать `uv pip`:

```bash
uv pip install numpy pandas scikit-learn plotly kaleido wordcloud optuna catboost kagglehub
uv pip install jupyter ipykernel spacy tqdm
```

Зачем нужны основные библиотеки:

- `numpy`, `pandas` - работа с данными и таблицами;
- `scikit-learn` - TF-IDF, train/test split, CV, метрики, LR, LinearSVC, LSA;
- `catboost` - обязательная третья модель в baseline и Optuna-оптимизации;
- `optuna` - подбор гиперпараметров моделей и TF-IDF внутри `Pipeline`;
- `spacy` - лемматизация русских отзывов;
- `tqdm` - прогресс-бары для предобработки;
- `plotly`, `kaleido` - интерактивные графики и возможность статического экспорта;
- `wordcloud` - облака слов, которые затем отображаются через Plotly;
- `kagglehub` - скачивание CSV, если локальный файл не найден;
- `jupyter`, `ipykernel` - запуск ноутбука из uv-окружения.

## 2. Установить русскую модель spaCy

После установки `spacy` обязательно скачайте языковую модель:

```bash
uv run python -m spacy download ru_core_news_sm
```

Без `ru_core_news_sm` ноутбук остановится в ячейке загрузки spaCy-модели.
Эта модель используется для лемматизации, а не для NER или синтаксического
разбора.

## 3. Подключить uv-окружение как Jupyter kernel

Чтобы ноутбук точно запускался в нужном окружении:

```bash
uv run python -m ipykernel install --user --name ml-labs --display-name "ML_LABS (uv)"
uv run jupyter lab
```

В Jupyter выберите kernel `ML_LABS (uv)`.

## 4. Подготовка датасета

Ноутбук ожидает итоговый файл здесь:

```text
LAB_IV/data/dataset/kinopoisk_reviews.csv
```

В разделе 2.1 ноутбук сам создает директорию `LAB_IV/data/dataset`.
Дальше возможны два сценария:

- если CSV уже лежит в `LAB_IV/data` или `LAB_IV/data/dataset`, он будет скопирован
  в `LAB_IV/data/dataset/kinopoisk_reviews.csv`;
- если локального CSV нет, ноутбук попробует скачать датасет
  `mikhailklemin/kinopoisks-movies-reviews` через `kagglehub`.

Для скачивания через `kagglehub` нужен доступ в интернет. Если Kaggle попросит
авторизацию, проще заранее скачать CSV вручную и положить его в
`LAB_IV/data/dataset/kinopoisk_reviews.csv`.

## 5. Особенности запуска ноутбука

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

## 6. Если uv не может собрать окружение

В `pyproject.toml` сейчас указан `requires-python = ">=3.14"`. Для некоторых ML
пакетов бинарные колеса под Python 3.14 могут появляться позже, чем под Python
3.12 или 3.13. Если `uv sync` падает именно на совместимости Python-версии,
проблема не в ноутбуке: нужно запускать проект на версии Python, под которую есть
колеса `catboost`, `spacy`, `scikit-learn` и `numpy`.

Перед финальным запуском полезно проверить:

```bash
uv run python -c "import numpy, pandas, sklearn, plotly, wordcloud, optuna, catboost, spacy, kagglehub; print('OK')"
uv run python -m spacy validate
```

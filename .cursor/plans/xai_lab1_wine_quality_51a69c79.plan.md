---
name: XAI_lab1_wine_quality
overview: "Обновлённый план ноутбука: мультиклассовая классификация качества красного вина (RF+CatBoost) с валидацией и полным циклом XAI (Permutation, PDP top-5, SHAP summary/heatmap/waterfall, LIME) при строгих учебных ограничениях (только hints)."
todos:
  - id: todo-1773838176630-n1g7h6btg
    content: "Создать notebooks/lab_1.ipynb и Главу I (Загрузка данных): Инициализировать ноутбук. Написать ячейку импортов (добавить optuna, Plotly, Pathlib, sklearn, shap, lime, catboost). Реализовать загрузку/кэширование winequality-red.csv через pathlib.Path, вывести .head() и .info()."
    status: completed
  - id: todo-1773838178514-tsvl40u5f
    content: "Написать Главу II (EDA на исходных данных): Сгенерировать ячейки визуализации строго на Plotly: дисбаланс (px.histogram), catplot-аналоги (px.box/strip), сводная таблица, объединенный px.violin (через melt) и матрица корреляций (px.imshow). Перед каждым графиком добавить markdown-hints."
    status: completed
  - id: todo-1773838185431-w59soos87
    content: "Написать Главу III (Feature Engineering и Препроцессинг): Создать 3 обязательных признака. Выполнить кодирование таргета (LabelEncoder), стратифицированное разбиение на Train/Val/Test и применить StandardScaler."
    status: completed
  - id: todo-1773838193341-859v29nkg
    content: "Написать Главы IV и V (Базовые модели и Оптимизация через Optuna): Обучить baseline-версии RandomForest и CatBoost, добавить метрики (classification_report, ROC AUC) внутрь их разделов. Написать оптимизацию через optuna (строго n_trials=10). Для RF использовать cross_val_score, для CatBoost — eval_set и early_stopping. Обучить финальные модели и добавить markdown-подсказки о выборе гиперпараметров."
    status: completed
  - id: todo-1773838199980-1vv6az0pj
    content: "Написать Главу VI (Итоговое сравнение и Анализ ошибок): Сделать сводную таблицу метрик и график Plotly. Написать логику пересечения ошибок (wrong_both = wrong_rf & wrong_cb), посчитать доли и вывести 2-5 примеров строк с ошибками обеих моделей, добавив подсказки для студента."
    status: completed
  - id: todo-1773838206315-78nrgu75p
    content: "Написать Главу VII, Часть 1 (Глобальный XAI): Написать DRY-функции. Реализовать Permutation Importance (px.bar) и PDP (нативный sklearn, строго для Топ-5) для ОБЕИХ моделей. Построить SHAP Summary для обеих моделей (обязательно параметр plot_cmap='viridis')."
    status: completed
  - id: todo-1773838211730-cav9ebh0y
    content: "Написать Главу VII, Часть 2 (Локальный XAI) и Главу VIII (Выводы): Реализовать алгоритм отбора ровно 10 верных и 10 ошибочных объектов (общий список). Построить SHAP heatmap для 20 объектов, 4 точечных SHAP waterfall (1+1 на каждую модель) и применить LIME (1-2 объекта). Завершить ноутбук Markdown-шаблоном итоговых выводов."
    status: completed
isProject: false
---

## 0) Подтверждение ключевых ограничений (для самого ноутбука)

- В markdown-ячейках **не пишем финальные выводы** по EDA/метрикам/XAI.
- Пишем только **краткие hints (1–2 предложения)** и оставляем “незаконченные” формулировки, чтобы студент дописал.
- Перед каждым графиком — markdown-hint, после — (при необходимости) ещё одна markdown-заглушка.

## 1) Структура заголовков ноутбука (как будет выглядеть оглавление)

Перестраиваем иерархию **в стиле эталона** `part_2_CAT_x_XGB.ipynb`: сначала “Оглавление/Импорты”, затем главы с римской нумерацией и подпунктами. Важно: **оценка качества находится внутри раздела каждой модели**, а не отдельной главой. Дополнительно соблюдаем классический пайплайн: **EDA делаем на исходных данных ДО feature engineering и препроцессинга**.

- `# Лабораторная 1. Explainable AI (XAI): Wine Quality (multiclass)`
- `## Оглавление`
- `## Импорты`
- `# I. Загрузка и подготовка данных`
  - `## 1. Загрузка датасета и первичный осмотр`
  - `## 2. Минимальные проверки качества данных`
- `# II. EDA (предварительный анализ исходных данных)`
  - `## 1. Дисбаланс классов (Plotly)`
  - `## 2. Catplot-аналоги (Plotly): alcohol / volatile acidity / sulphates`
  - `## 3. Группировки по качеству: mean/median (таблица)`
  - `## 4. Компактный violinplot через melt (Plotly)`
  - `## 5. Корреляционная матрица (Plotly)`
- `# III. Feature Engineering и Препроцессинг`
  - `## 1. Feature Engineering (создание новых признаков)`
  - `## 2. Подготовка целевой переменной (LabelEncoder)`
  - `## 3. Разбиение Train/Val/Test + StandardScaler`
- `# IV. Обучение базовых моделей`
  - `## 1. Метрики для сравнения моделей (что считаем и почему)`
  - `## 2. RandomForestClassifier (baseline)`
    - `### 2.1 Обучение`
    - `### 2.2 Оценка качества (classification_report + ROC AUC per-class)`
  - `## 3. CatBoostClassifier (baseline)`
    - `### 3.1 Обучение`
    - `### 3.2 Оценка качества (classification_report + ROC AUC per-class)`
  - `## 4. Сравнение базовых моделей (Plotly)`
- `# V. Оптимизация выбранных моделей`
  - `## 1. Оптимизация RandomForest`
    - `### 1.1 Оптимизация через Optuna (CV-3)`
    - `### 1.2 Оценка и сравнение с baseline`
  - `## 2. Оптимизация CatBoost`
    - `### 2.1 Early stopping на eval_set`
    - `### 2.2 Оптимизация гиперпараметров через Optuna`
    - `### 2.3 Оценка и сравнение с baseline`
- `# VI. Итоговое сравнение и анализ ошибок`
  - `## 1. Итоговое сравнение метрик (Plotly)`
  - `## 2. Пересечение ошибок моделей (set-операции + примеры строк)`
- `# VII. Интерпретируемость (XAI)`
  - `## 1. Глобальная интерпретация (классические методы)`
    - `### 1.1 Permutation Importance (Plotly) — для RF и CatBoost`
    - `### 1.2 PDP (scikit-learn) — только top-5, для RF и CatBoost`
  - `## 2. Глобальная интерпретация (SHAP Summary)`
    - `### 2.1 SHAP для RandomForest (plot_cmap='viridis')`
    - `### 2.2 SHAP для CatBoost (plot_cmap='viridis')`
  - `## 3. Локальная интерпретация (SHAP + LIME)`
    - `### 3.1 Выбор 10 correct + 10 wrong (общий список)`
    - `### 3.2 SHAP heatmap для 20 объектов (оптом)`
    - `### 3.3 SHAP waterfall: 1 correct + 1 wrong (точечно)`
    - `### 3.4 LIME: 1–2 объекта (альтернативно)`
- `# VIII. Выводы (шаблон для студента)`

## 2) Правила разбиения на ячейки (строго по вашему паттерну)

- Отдельная ячейка: **imports**.
- Отдельная ячейка: **загрузка + первичный осмотр** (`head`, `info`).
- Каждая модель — **отдельная ячейка обучения**.
- Каждый метод XAI (Permutation, PDP, SHAP summary, SHAP heatmap/waterfall, LIME) — **отдельные ячейки**, чтобы запускались независимо.

## 3) Шаги реализации (что именно будет сделано в коде)

### Шаг 1. Imports

- `pandas, numpy, matplotlib`
- `plotly.express as px`, `plotly.graph_objects as go`
- `from pathlib import Path`
- `sklearn`: split, scaler, metrics, permutation importance, PDP, `cross_val_score`
- `optuna`
- `catboost`
- `shap`, `lime` (+ `LimeTabularExplainer`)
- Общие настройки: стиль графиков, размеры, `random_state=42`.

### Шаг 2. Загрузка данных (без Kaggle API)

- Качаем CSV **напрямую из UCI** (тот же датасет Red Wine Quality):
  - `https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv`
- Кэшируем локально **строго через `pathlib.Path`**:
  - `data_dir = Path(\"LAB_I/data\")`
  - `file_path = data_dir / \"winequality-red.csv\"`
- Показать `df.head()` и `df.info()`.

### Шаг 3. EDA (глава II): анализ исходных данных до FE/препроцессинга

Обязательное и компактное:

- **Дисбаланс классов** (Plotly): `px.histogram(df, x='quality', color='quality')` (или без `color`, если мешает легенда).
- **Обязательные catplot-аналоги на Plotly** для `alcohol`, `volatile acidity`, `sulphates`:
  - компактно через `px.box(..., x='quality', y=feature, color='quality')` или `px.strip(...)` (выберем один единый стиль, чтобы не перегружать ноутбук).
- **Сводная таблица**: `df.groupby('quality')[top5_features].agg(['mean','median'])`.
  - Топ‑5 “логически важных” зафиксируем заранее (например: `alcohol`, `volatile acidity`, `sulphates`, `density`, `total_acidity`) и используем те же в сводной.
- **Один объединённый violinplot** (Plotly):
  - `melt` 3–4 ключевых фичи → `px.violin(melted, x='feature', y='value', color='quality', box=True, points='outliers')`.
- **Корреляционная матрица** (Plotly):
  - `corr = df[feature_cols + ['quality']].corr()` → `px.imshow(corr, color_continuous_scale='Viridis')`.
- Везде markdown-hints вместо выводов.

### Шаг 4. Feature Engineering и Препроцессинг (глава III)

#### 4.1 Feature Engineering (3 обязательных признака)

- `total_acidity = fixed acidity + volatile acidity + citric acid`
- `sugar_to_acid_ratio = residual sugar / (total_acidity + 1e-5)`
- `alcohol_density_product = alcohol * density`
- Сформировать `feature_cols = все числовые признаки кроме quality + 3 новые`.

#### 4.2 Подготовка целевой переменной (LabelEncoder)

- `y = df['quality']` → `LabelEncoder` → `y_enc` (классы 0..K-1).

#### 4.3 Разбиение Train/Val/Test + StandardScaler

- `X = df[feature_cols]`
- Разбиение:
  - сначала `train_val` vs `test` со `stratify=y_enc`.
  - затем `train` vs `val` тоже стратифицированно.
- `StandardScaler` на `X_train`, затем transform для `val/test`.

### Шаг 5. Обучение базовых моделей (глава IV)

#### 5.1 RandomForestClassifier (baseline) — обучение и оценка внутри раздела модели

- Балансировка: `class_weight='balanced'`.
- Обучить baseline на `X_train_scaled`.
- Получить предсказания/вероятности на `train` и `test` (для hint-диагностики переобучения).
- Посчитать `classification_report` и ROC AUC per-class (через `label_binarize`) на `test`.
- Сохранить метрики и артефакты в общую структуру результатов (для сравнения).

#### 5.2 CatBoostClassifier (baseline) — обучение и оценка внутри раздела модели

- Балансировка: `auto_class_weights='Balanced'` (или явные `class_weights` — выберем один и используем последовательно).
- `eval_set=(X_val_scaled, y_val)` + `early_stopping_rounds`.
- Обучить baseline (с ранней остановкой), затем оценить на `test`:\n  - `classification_report` и ROC AUC per-class.\n- Сохранить метрики и артефакты в общую структуру результатов.

#### 5.3 Сравнение базовых моделей (Plotly)

- Собрать baseline-результаты в компактную таблицу и 1 график сравнения (Plotly) + markdown-hint вместо выводов.

### Шаг 6. Оптимизация выбранных моделей (глава V)

#### 6.1 Оптимизация RandomForest

- Используем **Optuna**:\n  - `study = optuna.create_study(direction='maximize')`\n  - `objective(trial)`: семплируем 2–3 параметра (`n_estimators`, `max_depth`, `min_samples_leaf`/`max_features`), считаем качество через `cross_val_score(..., cv=3, scoring=...)`.\n  - Метрика в `objective`: `f1_macro` (рекомендуемо для дисбаланса) или `roc_auc_ovo` (если стабильно поддерживается для мультикласса в вашем sklearn); в плане фиксируем одну и используем везде последовательно.\n  - Ограничение по скорости: `**n_trials=10` или `15`** (по умолчанию 10).\n- После поиска обучить финальный `RandomForestClassifier(**best_params, class_weight='balanced', random_state=42)` на `X_train_scaled`, оценить на `test` теми же метриками, обновить результаты.\n- Markdown: вывести лучшие параметры + hint “почему Optuna могла выбрать такие значения …” и “почему ограничиваем trials …”.

#### 6.2 Оптимизация CatBoost

- Сохраняем `eval_set` + `early_stopping_rounds` и добавляем **Optuna** для подбора 2–3 параметров:\n  - `study = optuna.create_study(direction='maximize')`\n  - `objective(trial)`: семплируем, например, `depth`, `learning_rate`, `l2_leaf_reg` (и фиксируем остальное), обучаем CatBoost с `eval_set=(X_val_scaled, y_val)` и `early_stopping_rounds`.\n  - Качество возвращаем по выбранной метрике (например, `f1_macro` на `val` или агрегированное `roc_auc_ovr/ovo` на `val`; в плане фиксируем подход заранее).\n  - Ограничение по скорости: `**n_trials=10`**.\n- После поиска обучить финальную модель на лучших параметрах, оценить на `test`, обновить результаты.\n- Markdown: параметры + hint “почему early stopping важен …” и “почему trials ограничены …”.

### Шаг 7. Итоговое сравнение и анализ ошибок (глава VI)

#### 7.1 Итоговое сравнение метрик (Plotly)

- Сводная таблица + 1 компактный график сравнения (Plotly).

#### 7.2 Пересечение ошибок моделей

- Для test:
  - `wrong_rf = set(df_test_indices[y_pred_rf != y_test])`
  - `wrong_cb = set(df_test_indices[y_pred_cb != y_test])`
  - `wrong_both = wrong_rf & wrong_cb`
- Посчитать долю общих ошибок:
  - `len(wrong_both)/len(test)` и `len(wrong_both)/len(wrong_rf ∪ wrong_cb)` (оба показателя полезны, выведем оба).
- Показать 2–5 строк `df.loc[list(wrong_both)[:k], ...]` (несколько ключевых признаков + таргет).
- Markdown-hint: на какие “аномалии/крайние значения/редкие классы” стоит смотреть.

### Шаг 8. Интерпретируемость (XAI) (глава VII)

#### 8.1 Глобальная интерпретация (классические методы): Permutation + PDP — строго для обеих моделей

- Напишем DRY-функции:
  - `compute_permutation_importance(model, X_test_scaled, y_test, feature_names)`
  - `plot_pdp_topk(model, X_train_scaled, feature_names, top_features, k=5)`
- Для **RF и CatBoost**:
  - `permutation_importance` на test (все признаки) + bar chart **на Plotly** (например, `px.bar` по mean importance).
  - Выбрать **top-5** признаков по permutation importance.
  - Построить **PDP только для top-5** (нативный `scikit-learn`, без замены на Plotly).

#### 8.2 Глобальная интерпретация (SHAP Summary) — для обеих моделей

- DRY-функции:
  - `make_shap_explainer(model, X_background)`
  - `plot_shap_summary(explainer, X_sample, feature_names, plot_cmap='viridis')`
- Использовать подвыборку (`X_val_scaled` или `X_test_scaled` subset) для скорости.
- Для мультикласса: аккуратно обработать формат `shap_values` (list/array) и обеспечить корректные summary (например, по каждому классу или агрегировано; выберем вариант, который стабильно работает в установленной версии shap).
- **Во всех SHAP графиках**: `plot_cmap='viridis'`.

#### 8.3 Локальная интерпретация (SHAP + LIME) (обновлено)

- Отбор общих объектов:
  - Выбрать **ровно 10 корректных** и **ровно 10 ошибочных** объектов по предсказаниям (на test). Чтобы список был “общим для обеих моделей”, используем правило:
    - correct: `оба_верны` (RF и CatBoost оба угадали) — если хватает, берём первые 10; если не хватает, используем приоритет “оба верны, иначе хотя бы одна верна” (и явно фиксируем правило в markdown-hint).
    - wrong: `оба_ошиблись` — если хватает, берём первые 10; если не хватает, добираем “ошибся хотя бы один”.
  - В итоге получаем 20 индексов и показываем небольшую таблицу с `y_true`, `pred_rf`, `pred_cb`.
- SHAP “оптом”:
  - `shap.plots.heatmap(...)` для этих 20 объектов (по выбранному классу объяснения; зафиксируем правило: объясняем вклад в **предсказанный класс** конкретной модели).
  - Отдельно для RF и для CatBoost (две ячейки), чтобы не перегружать.
- SHAP “точечно”:
  - 1 waterfall для корректного и 1 waterfall для ошибочного (по одному объекту) — отдельно для каждой модели или для одной (решение: сделать 2 waterfall на модель = 4 графика, но компактно; либо 2 графика на “лучшую” модель. Чтобы соответствовать “за модель”, предпочтительнее по 1+1 на каждую).
- LIME:
  - 1–2 объекта (верный/ошибочный) как альтернативный метод.
  - Обёртки `predict_proba` для RF и CatBoost; `LimeTabularExplainer` обучаем на `X_train_scaled`.

### Шаг 9. Выводы (шаблон для студента) (глава VIII)

- Markdown с пустыми пунктами, которые студент заполняет:
  - сравнение моделей,
  - совпадение/расхождение важных фич по разным XAI,
  - гипотезы о причинах общих ошибок.

## 4) Контроль соответствия требованиям

- EDA (Plotly): groupby-таблица + violin(melt) + box/strip аналоги catplot + histogram (countplot-аналог) + коррматрица через `px.imshow`.
- Валидация/оптимизация (в стиле эталона): оценка качества “внутри” раздела каждой модели; RF и CatBoost оптимизируем через Optuna (ограниченные `n_trials`), CatBoost обязательно с `eval_set` + early stopping.
- Ошибки: множества ошибок + пересечение + доля + показ строк.
- Глобальный XAI: permutation + PDP(top-5) для **обеих** моделей.
- Локальный XAI: 10 correct + 10 wrong; SHAP heatmap; по 1 waterfall (верный/ошибочный); LIME 1–2 объекта.
- SHAP colormap: везде `plot_cmap='viridis'`.


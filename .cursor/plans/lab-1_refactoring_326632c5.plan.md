---
name: Lab-1 Refactoring
overview: Полный рефакторинг `LAB_I/notebooks/lab_1.ipynb` — переписать структуру, код и Markdown по образцу part_1/part_2, адаптировать под мультиклассовую классификацию (RF + CatBoost), исправить все сломанные блоки, добавить обязательный EDA и XAI.
todos:
  - id: header-imports
    content: Переписать заголовок, оглавление и блок импортов по шаблону part_2
    status: completed
  - id: eda-section
    content: "Раздел I: добавить groupby-таблицу, px.box с color=quality, markdown hints"
    status: completed
  - id: preprocessing
    content: "Раздел II: расширить add_wine_features до 5 признаков, добавить вывод маппинга LabelEncoder, убрать val-сплит"
    status: completed
  - id: metrics-functions
    content: "Раздел III: написать calculate_multiclass_metrics, adapt add_result и plot_model_report_separate под классификацию, вставить таблицу-пояснение метрик в Markdown"
    status: completed
  - id: baseline-models
    content: "Раздел III: обучение baseline RF и CatBoost, вывод метрик, сравнение"
    status: completed
  - id: optuna-optimization
    content: "Раздел IV: исправить Optuna (30 trials, корректный objective), CatBoost с eval_set"
    status: completed
  - id: error-analysis
    content: "Раздел V: исправить IndexError (.flatten()), добавить Confusion Matrix и ROC-кривые по классам"
    status: completed
  - id: xai-global
    content: "Раздел VI.1: исправить PDP (параметр target=), SHAP Summary с check_additivity=False и plot_cmap='viridis'"
    status: completed
  - id: xai-local
    content: "Раздел VI.2: корректный отбор 20 общих объектов, SHAP heatmap, waterfall, LIME без xgb_for_xai"
    status: completed
  - id: conclusions
    content: "Раздел VII: заменить пустые выводы на плейсхолдеры '> Вывод: [Место для вашего вывода...]'"
    status: completed
isProject: false
---

# Полный рефакторинг lab_1.ipynb

## Контекст и диагностика текущего состояния

Текущий [lab_1.ipynb](LAB_I/notebooks/lab_1.ipynb) содержит:

- Сломанные блоки: PDP (`ValueError: target must be specified`), SHAP (`ExplainerError: Additivity check failed`), LIME (`NameError: xgb_for_xai`)
- Отсутствует: confusion matrix, визуальные ROC-кривые, `balanced_accuracy_score`, `weighted` F1
- Feature engineering только 3 признака (вместо 5 в эталоне)
- `y_pred_cb` из CatBoost возвращает 2D массив — `IndexError` в анализе ошибок

Эталоны:

- [part_1_sklearn.ipynb](LAB_I/docs/raw/part_1_sklearn.ipynb) / [part_2_CAT_x_XGB.ipynb](LAB_I/docs/raw/part_2_CAT_x_XGB.ipynb) — шаблон структуры, функций, Plotly-визуализации, Markdown-стиля
- [part_3_bonus.ipynb](LAB_I/docs/raw/part_3_bonus.ipynb) — паттерн мультиклассовых метрик и guard для пустых классов

---

## Целевая структура ноутбука

```
# Заголовок (автор, цель, датасет, модели)
## Оглавление
## Импорты

# I.   Загрузка и подготовка данных (EDA)
# II.  Предварительная обработка данных
# III. Обучение базовых моделей
  ## 1. Метрики для сравнения
  ## 2. Выбор моделей
  ## 3. Random Forest (базовая)
  ## 4. CatBoost (базовая)
  ## 5. Сравнение базовых моделей
# IV.  Оптимизация
  ## 1. Оптимизация Random Forest (Optuna)
  ## 2. Оптимизация CatBoost (Optuna)
  ## 3. Сравнение оптимизированных моделей
# V.   Итоговое сравнение и анализ ошибок
  ## 1. Сводная таблица results_df
  ## 2. Confusion Matrix
  ## 3. ROC-кривые по классам
  ## 4. Анализ пересечения ошибок
# VI.  Интерпретируемость (XAI)
  ## 1. Глобальная: Permutation Importance, PDP, SHAP Summary
  ## 2. Локальная: SHAP Heatmap, Waterfall, LIME
# VII. Выводы (плейсхолдеры для студента)
```

---

## Ключевые изменения по разделам

### Раздел I — EDA (добавить)

- `df.groupby('quality')[features].agg(['mean', 'median'])` — таблица средних/медиан по классам
- `px.histogram` для `quality` (уже есть, оставить)
- `px.box(df, x='quality', y=feature, color='quality')` — catplot-аналог (топ-3 признака)
- `px.violin` для Топ-4 признаков через `df.melt` (уже есть, доработать)
- Markdown-hints: `> **Подсказка:** ...` обращающие внимание на различия между элитными и плохими винами

### Раздел II — Препроцессинг

- Расширить `add_wine_features` до 5 признаков (как в part_1/part_2): `total_acidity`, `vol_fixed_ratio`, `free_so2_ratio`, `alc_density_prod`, `sugar_alc_ratio`
- `LabelEncoder` — добавить явный вывод маппинга классов (паттерн из part_3_bonus)
- Стратифицированный train/test сплит (убрать val-сплит, он не используется в дальнейшем)
- `StandardScaler` — оба датасета (base + enhanced) масштабируются

### Раздел III — Функции и метрики

Новая функция `calculate_multiclass_metrics(y_true, y_pred, y_proba)`:

```python
def calculate_multiclass_metrics(y_true, y_pred, y_proba, model_name):
    return {
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'F1 Macro':          f1_score(y_true, y_pred, average='macro', zero_division=0),
        'F1 Weighted':       f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'ROC AUC (OvR)':     roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro'),
    }
```

Таблица-пояснение метрик вставляется в Markdown прямо перед секцией III.1 (как указано в требованиях).

Функция `add_result` адаптируется под классификацию: хранит Balanced Acc, F1 Macro, F1 Weighted, ROC AUC для train и test.

`plot_model_report_separate` — grouped bar chart по 4 метрикам (Train vs Test), горизонтальный bar для feature importance. Паттерн из part_2.

### Раздел IV — Оптимизация (Optuna)

- RF: Optuna, 30 trials, `objective = f1_macro` на CV-3 — исправить текущий код (10 trials → 30)
- CatBoost: Optuna с `eval_set` + `early_stopping_rounds=50`, `verbose=0`
- Каждая модель: вывод лучших параметров + `calculate_multiclass_metrics` на test

### Раздел V — Анализ ошибок (починить)

Исправление `IndexError` для CatBoost:

```python
y_pred_cb = cb_best.predict(X_test).flatten()  # .flatten() для 2D → 1D
```

Добавить Confusion Matrix:

```python
fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                title='Confusion Matrix — CatBoost')
```

ROC-кривые по классам (matplotlib, паттерн из part_3_bonus с guard `if y_true_bin[:, i].sum() > 0`).

### Раздел VI — XAI (исправить всё сломанное)

**Глобальная интерпретация:**

- Permutation Importance: уже работает, оставить
- PDP — исправление `ValueError`:

```python
  PartialDependenceDisplay.from_estimator(
      rf_for_xai, X_test_scaled, top5_features,
      target=TARGET_CLASS,  # обязательный параметр для мультикласса
      kind='average'
  )
  

```

- SHAP Summary для RF:

```python
  explainer_rf = shap.TreeExplainer(rf_for_xai)
  sv_rf = explainer_rf.shap_values(X_test_scaled, check_additivity=False)
  shap.summary_plot(sv_rf[TARGET_CLASS], ..., plot_cmap='viridis')
  

```

- SHAP Summary для CatBoost:

```python
  explainer_cb = shap.TreeExplainer(cb_for_xai)
  sv_cb = explainer_cb.shap_values(Pool(X_test_scaled, ...))
  shap.summary_plot(sv_cb, ..., plot_cmap='viridis')
  

```

**Локальная интерпретация:**

```python
# Общие 10 верных + 10 ошибочных
correct_rf  = set(np.where(y_pred_rf  == y_test)[0])
correct_cb  = set(np.where(y_pred_cb  == y_test)[0])
wrong_rf    = set(np.where(y_pred_rf  != y_test)[0])
wrong_cb    = set(np.where(y_pred_cb  != y_test)[0])
shared_correct = list(correct_rf & correct_cb)[:10]
shared_wrong   = list(wrong_rf   & wrong_cb)[:10]
local_idx = shared_correct + shared_wrong  # 20 объектов
```

- SHAP Heatmap на 20 объектах: `shap.plots.heatmap(...)`, `plot_cmap='viridis'`
- SHAP Waterfall (1 верный + 1 ошибочный): `shap.plots.waterfall(...)`
- LIME (1 верный + 1 ошибочный): убрать `xgb_for_xai`, заменить на `rf_for_xai` и `cb_for_xai`

### Markdown-стиль (везде)

- Плейсхолдеры выводов: `> **Вывод:** [Место для вашего вывода...]`
- Подтверждения ячеек: `print("✅ Функция calculate_multiclass_metrics() создана")`
- Hints: `> **Подсказка:** ...`
- Нет финальных заключений за студента

---

## Что убрать

- Все упоминания `xgb_for_xai` и `XGBoostClassifier`
- Validation set (`X_val`) — не используется нигде содержательно, упростить до train/test
- Пустые bullet-пункты в выводах (заменить на плейсхолдеры)


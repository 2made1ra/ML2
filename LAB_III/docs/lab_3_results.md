# Лабораторная работа III — отчёт о текущем состоянии

Документ составлен по ноутбуку [`solution/lab_3.ipynb`](../solution/lab_3.ipynb). **Таблица гиперпараметров** отражает актуальную ячейку `Config` в коде. **Числа метрик, логи эпох и время** взяты из **сохранённого вывода** ячеек того же файла (зафиксированный прогон). Дата ссылки: *25.04.2026*; в примере устройства: `cuda`, NVIDIA GeForce RTX 4090, окружение `d:\DEV\ML2\.venv`.

---

## Цель

Мультиклассовая **классификация жанра** картины по изображению на датасете [huggan/wikiart](https://huggingface.co/datasets/huggan/wikiart), фреймворк **PyTorch**.

---

## Окружение (сохранённый вывод ноутбука)

| Параметр | Значение                 |
| -------- | ------------------------ |
| Device   | `cuda`                   |
| GPU      | NVIDIA GeForce RTX 4090  |

---

## Параметры эксперимента (`Config` — актуальный код)

| Параметр                 | Значение              |
| ------------------------ | --------------------- |
| `dataset_name`           | `huggan/wikiart`     |
| `target_column`          | `genre`              |
| `max_samples`            | 5000                 |
| `cache_image_size`       | 320                  |
| `custom_image_size`      | 192                  |
| `pretrained_image_size`  | 224                  |
| `batch_size`             | 192                  |
| `num_workers`            | 0                    |
| `custom_epochs`          | 30                   |
| `transfer_head_epochs`   | 30                   |
| `fine_tune_epochs`       | 20                   |
| `lr_custom`              | 6e-4                 |
| `lr_head`                | 2e-3                 |
| `lr_finetune`            | 3e-5                 |
| `weight_decay`           | 2e-4                 |
| `label_smoothing`        | 0.05                 |
| `use_class_weights`      | `True`               |

*Примечание:* в **сохранённых stream-логах** обучения EfficientNet в ноутбуке зафиксированы **8 эпох** фазы `efficientnet_head` и **12 эпох** `efficientnet_finetune` (а своя CNN — 30 эпох). Это соответствует прогону с меньшим числом эпох transfer, чем в текущей ячейке `Config` выше. После полного «Run All» с актуальным `Config` метрики и длительность этапов обновятся.

---

## Данные

- **Целевой признак:** `genre`.
- **Загрузка:** streaming с Hugging Face, локальный кэш в `data_lab3/wikiart_genre_cache/`, `metadata.csv`.
- **Классы:** `ClassLabel` из стрима или `FALLBACK_GENRE_NAMES` (11 жанров, включая `Unknown Genre`):

  `abstract_painting`, `cityscape`, `genre_painting`, `illustration`, `landscape`, `nude_painting`, `portrait`, `religious_painting`, `sketch_and_study`, `still_life`, `Unknown Genre`.

- **Разбиение:** стратифицированный `train_test_split` — 30% во временный пул, затем 50/50 на валидацию и тест. Доли **70% / 15% / 15%**. Для `max_samples = 5000` в сохранённом выводе: **3500 / 750 / 750** (train / val / test).
- **Пайплайны:** своя CNN — **192×192**; **EfficientNet-B0** — **224** и нормализация ImageNet. Аугментации train для своей сети: `RandomResizedCrop`, `RandomHorizontalFlip`, `ColorJitter` (через `RandomApply`), `RandomGrayscale`, `RandomErasing`; для предобученной модели — умеренный `RandomResizedCrop` / `ColorJitter` / `RandomHorizontalFlip`. Оценка: resize+center crop под размер входа. В обучении на CUDA: AMP, `channels_last` для тензоров и модели, при `use_class_weights` — веса классов в лоссе.

---

## Модели

1. **WikiArtSmallCNN** (в сводной таблице: «Custom separable residual CNN») — собственная CNN с **inverted residual**-блоками в духе MobileNetV2/EfficientNet, **SE (squeeze–excitation)**, `BatchNorm`, `SiLU`, стохастическая глубина, `AdaptiveAvgPool2d` и линейный классификатор. Обучение: AdamW, **label smoothing**, cosine schedule, опционально **class weights**, `fit_model` на `custom_*` лоадерах.
2. **EfficientNet-B0 (ImageNet)** — сначала **голова** (`lr_head`, замороженный backbone), затем **fine-tune** последних блоков `features` с `lr_finetune`.

**Оценка:** `run_epoch` (AMP), `fit_model`, `predict_model`, графики, `confusion_matrix`, `classification_report`, `summarize_metrics` (accuracy, macro precision/recall/F1), выбор `best_model` по test accuracy.

---

## Метрики на тесте (n = 750)

Сводка из `summarize_metrics` (одинаковая тестовая выборка для обеих моделей).

| Модель | `test_loss` | accuracy | macro precision | macro recall | macro F1 |
| ------ | ----------: | -------: | --------------: | -----------: | -------: |
| Custom separable residual CNN (WikiArtSmallCNN) | 1.8928 | 0.3427 | 0.3179 | 0.3905 | 0.3018 |
| EfficientNet-B0 (transfer + fine-tune) | 1.4731 | 0.5107 | 0.4319 | 0.5241 | 0.4529 |

**Лучшая модель по test accuracy:** EfficientNet-B0 (0.5107).

По `classification_report` (итоги из сохранённого вывода): своя CNN — **accuracy 0,34**, weighted F1 **0,32**; EfficientNet-B0 — **accuracy 0,51**, weighted F1 **0,50**.

---

## Валидация и время обучения (сохранённые логи)

| Этап | Лучшая `val_acc` | Время (лог `fit_model`) |
| ---- | ---------------: | ----------------------: |
| `custom_cnn`, 30 эпох | 0.3560 | 16.3 мин |
| `efficientnet_head` (8 эпох в логе) | 0.5507 | 2.5 мин |
| `efficientnet_finetune` (12 эпох в логе) | 0.5613 | 3.3 мин |

*Веса в ноутбуке — по лучшей валидационной эпохе внутри каждой фазы `fit_model`.*

---

## Анализ и инференс (по структуре ноутбука)

- **Анализ ошибок:** топ-10 пар путаниц по `confusion_matrix`, сетка примеров, эвристика `error_comment`. В сохранённом прогоне: **367** неверно классифицированных примеров **из 750** на лучшей модели.
- **Внешние изображения:** `data_lab3/external_images` и при необходимости загрузка с Wikimedia Commons; предсказание через `best_model` и `best_transform`.

---

## Выводы (из раздела «Выводы» ноутбука + факты прогона)

По **accuracy** и **macro F1** на тесте выигрывает **EfficientNet-B0** (таблица выше): перенос с ImageNet даёт выигрыш по сравнению с обучением усиленной custom CNN с нуля на подвыборке 5000. Ошибки в жанровой задаче по-прежнему отчасти **структурные**: путаница близких жанров и редких классов, неоднозначная разметка.

**Идеи улучшения (из ноутбука):** увеличить `max_samples` и число эпох, balanced sampler, более сильные backbone (`convnext_tiny`, `efficientnet_v2_s`).

---

*Метрики воспроизводимы запуском `lab_3.ipynb` с тем же `Config`, кэшем и seed. При смене кода `Config`, кэша или весов EfficientNet цифры отличатся. Если сохранённый вывод и текущий `Config` различаются по числу эпох transfer, перезапустите соответствующие ячейки или весь ноутбук и при необходимости обновите этот документ.*

# Итерация V6: более глубокий ViT fine-tuning и усиленный лёгкий CNN

Документ фиксирует плановые изменения V6 в `LAB_III/solution/lab_3.ipynb`. Метрики V6 нужно заполнить после полного `Run All`.

---

## База V5

Текущие сохранённые результаты V5:

| Модель | accuracy | macro F1 | Время |
| --- | ---: | ---: | ---: |
| `Custom WikiArtTinyCNN` | `0.4187` | `0.38` | n/a |
| `ViT-B/16 ImageNet transfer/fine-tune` | `0.6340` | `0.61` | `18.1 min` |

V3 baseline для сравнения:

- `EfficientNet-B0`: `accuracy = 0.5107`, `macro F1 = 0.4529`.

---

## Цель V6

- Улучшить ViT validation/test accuracy за счёт разморозки большего числа encoder layers.
- Приблизиться к целевому порядку `~50M` активных параметров без полной разморозки ViT.
- Немного улучшить custom baseline без возврата к тяжёлым residual/SE-блокам и без большой потери скорости.

---

## Изменения `Config`

| Параметр | V5 | V6 |
| --- | ---: | ---: |
| `vit_batch_size` | `192` | `128` |
| `vit_head_epochs` | `3` | `3` |
| `vit_finetune_epochs` | `10` | `16` |
| `vit_unfreeze_layers` | `4` | `7` |
| `early_stopping_patience` | `4` | `5` |
| `vit_lr_head` | `8e-4` | `6e-4` |
| `vit_lr_finetune` | `1.5e-5` | `8e-6` |
| `custom_epochs` | `5` | `8` |
| `lr_custom` | `1e-3` | `8e-4` |

Без изменений:

- `pretrained_model_name = "google/vit-base-patch16-224"`;
- `target_column = "genre"`;
- `max_samples = 40_000`;
- `custom_image_size = 160`;
- `vit_image_size = 224`;
- `batch_size = 512`;
- `num_workers = 0`;
- `weight_decay = 1e-4`;
- `vit_weight_decay = 0.05`;
- `balanced_train_sampler = True`;
- `use_class_weights = False`.

---

## Архитектура и обучение

Предобученная модель не меняется: используется тот же `google/vit-base-patch16-224`.

ViT fine-tuning:

- head training остаётся коротким: `3` эпохи;
- размораживаются последние `7` encoder layers;
- fine-tuning увеличен до `16` эпох;
- `vit_lr_finetune` снижен до `8e-6`, потому что trainable-параметров стало больше;
- checkpoint по-прежнему выбирается по лучшему `val_acc`;
- early stopping: `patience = 5`.

Custom baseline остаётся лёгким, но получает один дополнительный conv-блок:

- `ConvNormAct(3, 32, stride=2)`;
- `ConvNormAct(32, 64, stride=2)`;
- `ConvNormAct(64, 128, stride=2)`;
- `ConvNormAct(128, 160, stride=2)`;
- `ConvNormAct(160, 192, stride=1)`;
- `AdaptiveAvgPool2d(1)`;
- `Dropout(0.25)`;
- `Linear(192, NUM_CLASSES)`.

Не возвращаются:

- `RandomErasing`;
- `RandomGrayscale`;
- squeeze-excitation;
- stochastic depth;
- residual blocks.

---

## Fallback

- Если будет OOM: снизить `vit_batch_size` с `128` до `96`.
- Если память всё ещё нестабильна: снизить `vit_unfreeze_layers` с `7` до `6`.
- Если DataLoader снова станет bottleneck, отдельной итерацией сравнить `num_workers=0/2`, но в V6 оставить стабильное значение `0`.

---

## Метрики после запуска

Заполнено по текущему сохранённому output из `lab_3.ipynb` (последний прогон V6).

| Модель | `test_loss` | accuracy | macro precision | macro recall | macro F1 | Время |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Custom WikiArtTinyCNN` | n/a в output | 0.3560 | 0.33 | 0.43 | 0.33 | 8.0 min |
| `ViT-B/16 ImageNet transfer/fine-tune` | n/a в output | 0.6675 | 0.62 | 0.72 | 0.65 | 31.4 min (head 5.0 + finetune 26.4) |

Критерий успеха относительно V5:

- ViT accuracy выше `0.6340`;
- ViT macro F1 выше `0.61`;
- custom CNN желательно выше `0.4187` accuracy без существенного роста времени.

Факт по текущему прогону:

- ViT улучшил метрики относительно V5: `accuracy 0.6675` и `macro F1 0.65` (критерии выполнены).
- Custom CNN ухудшился относительно V5: `accuracy 0.3560`, `macro F1 0.33` (критерий не выполнен).

# Code Agent Guard — пайплайны M2: обучение и оценка

Классификатор bash-команд AI-агента на классы `allow` / `ask` / `deny` (v0). Два независимых пайплайна:

| Пайплайн | Вход | Выход |
|---|---|---|
| `train.py` — обучение | train-данные, модель из `configs/models.yaml` | папка чекпойнта + run в MLflow |
| `evaluate.py` — оценка | test-данные + один источник предсказаний | метрики, таблицы, графики + run в MLflow |

Связь между ними — папка чекпойнта: `train.py` её пишет, `evaluate.py --checkpoint` читает.
Оценка не зависит от обучения: тем же пайплайном оцениваются голые модели без дообучения и LLM-судьи через API.

## Установка

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128   # под свою CUDA
pip install -r requirements.txt
```

Для 8B-моделей нужна GPU с поддержкой bitsandbytes (NF4); энкодеры обучаются на любой CUDA-GPU, оцениваются и на CPU.

## Данные

CSV с 9 колонками по гайдлайну: `command, description, util, severity, category, risk, source, seed, class`.
`--data` — CSV-файл или папка: берутся `commands_<split>.csv` в ней и в подпапках
(для test, если нет `commands_test.csv`, — `commands.csv`, как в ветках `feat/data_CAT-*`).
`--cats` фильтрует по категории (`CAT-1`) или подкатегории (`CAT-1.2`).
Классы `modify` / `defer` в схеме `v0_drop` выкидываются, в `v0_merge` сливаются в `deny` / `ask`.

## Обучение

```bash
python train.py --model granite_small --data ../benchmark --cats CAT-4 --seeds 42 1337 2024
python train.py --model seed_coder_8b --data ../benchmark --cats CAT-4
```

1. train-сплит нужных категорий;
2. val — `--val-data` или стратифицированно 15% от train (`--val-frac`, `--val-seed`);
3. обучение с early stopping по val loss: энкодеры — полностью, 8B — QLoRA (NF4 + LoRA r=16);
4. пороги под FPR 0.5 / 1 / 2 / 5% на val;
5. чекпойнт `checkpoints/<model>-s<seed>-<категории>/`;
6. MLflow, эксперимент `m2-train`: гиперпараметры, кривые обучения по эпохам, метрики на val, `meta.json`.

Test в обучении не участвует.

Модели (`configs/models.yaml`): энкодеры `granite_small` (47M), `ettin150m`, `modernbert`, `unixcoder`, `codebert`;
8B — `seed_coder_8b`, `granite_8b`. Гиперпараметры там же.

### Чекпойнт

| Файл | Содержимое |
|---|---|
| `weights.pt` | обучаемые параметры: у энкодеров все (~180 МБ у granite_small), у 8B — LoRA + голова (~160–190 МБ) |
| `meta.json` | spec модели, `max_len`, пороги с val, данные обучения и их хэши, seed, id run в MLflow |
| `train_commands.json` | команды train + val — для поиска почти-дубликатов в test |

## Оценка

```bash
python evaluate.py --checkpoint checkpoints/granite_small-s42-CAT-4 --data ../benchmark
python evaluate.py --zeroshot seed_coder_8b --data ../benchmark --cats CAT-1 CAT-2
python evaluate.py --judge d1_ultra --data ../benchmark
python evaluate.py --judge gpt_oss_120b --data ../benchmark --limit 40 --no-mlflow   # пробный прогон
```

Источник предсказаний — ровно один:

| Флаг | Что оценивается |
|---|---|
| `--checkpoint DIR` | дообученная модель из `train.py` |
| `--zeroshot KEY` | голая модель без дообучения: энкодеры — по косинусному сходству с описаниями классов, 8B — по вероятностям токенов `allow`/`ask`/`deny` |
| `--judge KEY` | LLM-судья через OpenRouter: `gpt_oss_safeguard_20b`, `gpt_oss_120b`, `jev_1_14`, `d1_ultra` |

Судьям нужен `OPENROUTER_API_KEY`. Ответы кэшируются в `cache/judges/<judge>.jsonl`: повторный прогон запрашивает только
новые команды, при полном кэше ключ не нужен. Запросы идут только к провайдерам без сбора данных.

### Метрики

unsafe = `ask` ∪ `deny`. FPR — доля безопасных команд, остановленных зря (цель ≤ 1%);
FNR — доля опасных, пропущенных как `allow` (цель < 5%).

| Метрика | Что значит |
|---|---|
| `fnr_at_fpr_{0p5,1,2,5}pct_oracle` | FNR при пороге, подобранном на самом test под заданный FPR — потолок модели, для сравнения моделей |
| `valthr_{…}_fpr / _fnr` | FPR и FNR при пороге, подобранном на val при обучении — честная оценка, как в проде (только у чекпойнтов) |
| `verdict_fpr_argmax / verdict_fnr_argmax` | FPR и FNR по собственному ответу модели без порога — так работают судьи |
| `deny_to_allow_rate_argmax` | доля `deny`, пропущенных как `allow` — самая дорогая ошибка |
| `allow_vs_ask/…`, `allow_vs_deny/…`, `allow_vs_all/…` | те же FNR@FPR и detection rate (= 1 − FNR) отдельно для ask, для deny и для всех опасных; score и безопасные одни и те же, поэтому порог общий — видно, какой класс guard пропускает. Есть варианты oracle, val-порог (`valthr_…_detection_rate`) и по ответу модели (`verdict_detection_rate`) |
| `unsafe_roc_auc`, `unsafe_pr_auc` | качество ранжирования safe / unsafe без порога |
| `macro_f1_argmax`, `ovr_auc_*`, `ovo_auc_*` | многоклассовое качество allow / ask / deny |
| `test_clean/…` | те же метрики без почти-дубликатов train в test (cos ≥ 0.9 по char-n-gram TF-IDF; только у чекпойнтов) |
| `cat/<CAT>/…` | AUROC, FNR@FPR 1% / 5% (свой порог в каждой категории), verdict FPR / FNR, FPR / FNR при val-пороге |

### Результат

`outputs/<run>/` и тот же набор в MLflow (эксперименты `m2-eval-finetuned`, `m2-eval-zeroshot`, `m2-eval-judge`):

| Файл | Содержимое |
|---|---|
| `metrics.json` | все метрики плоским словарём |
| `tables/by_cat_code.csv`, `by_risk_code.csv`, `by_severity.csv` | метрики по категориям, подкатегориям, severity |
| `tables/pairs.csv`, `pairs_by_cat_code.csv` | allow vs ask / deny / all: FNR и detection rate при FPR 1% и 5%, в целом и по категориям |
| `tables/confusion_*.csv`, `ovo_auc.csv`, `recall_risk_x_class.csv` | матрицы ошибок, one-vs-one AUC, recall подкатегория × класс |
| `predictions.csv`, `errors.csv` | вероятности по каждой команде; ошибки, отсортированные по score |
| `plots/*.png` | ROC / FNR-vs-FPR / PR, матрицы ошибок, тепловые карты по категориям и подкатегориям |
| `prompt.txt` | промпт (zero-shot и судьи) |

## MLflow

По умолчанию — локальный `mlflow.db` в этой папке: `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`.
Общий сервер — через переменную `MLFLOW_TRACKING_URI`. Без MLflow — флаг `--no-mlflow` (результаты только в `outputs/`).

## Структура

```
train.py                пайплайн обучения
evaluate.py             пайплайн оценки
configs/models.yaml     модели и гиперпараметры
guard/
  data.py               загрузка CSV, схемы классов, train/val, почти-дубликаты
  models.py             классификатор: бэкбон + пулинг + голова (энкодеры, 8B QLoRA)
  trainer.py            цикл обучения, инференс, пороги с val
  checkpoint.py         сохранение / загрузка чекпойнта
  metrics.py            все метрики
  report.py             метрики + таблицы + графики в папку
  plots.py              графики
  tracking.py           MLflow
  prompts.py            политика классов для zero-shot и судей
  predictors/           источники предсказаний для оценки: checkpoint, zeroshot, judge
```

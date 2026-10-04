# Команды проекта. Список с описаниями: make help
#
# Модель выбирается именем конфига из configs/ без расширения:
#   make full MODEL=scm_0_5b
# Доступные конфиги: make models
#
# Рецепты состоят из одной команды каждая, поэтому работают и в sh, и в cmd.exe.

MODEL ?= qwen3guard_baseline
PROFILE ?= smoke2
CONFIG := configs/$(MODEL).yaml
NOTEBOOKS := notebooks/02_streaming_benchmark.ipynb notebooks/04_scm_benchmark.ipynb notebooks/06_model_comparison.ipynb

.DEFAULT_GOAL := help
.PHONY: help models install install-models data test lint check run smoke full rerun reports notebooks figures

help: ## Показать этот список команд
	@uv run python -c "import re; [print(f'  make {m[1]:<16}{m[2]}') for m in re.finditer(r'^([a-z-]+):.*?## (.*)$$', open('Makefile', encoding='utf-8').read(), re.M)]"

models: ## Показать конфиги моделей, которые можно передать в MODEL=
	@uv run python -c "from pathlib import Path; [print(' ', p.stem) for p in sorted(Path('configs').glob('*.yaml'))]"

# --- Окружение ---------------------------------------------------------------

install: ## Окружение для чтения ноутбуков и тестов (без весов моделей)
	uv sync --extra analysis --extra dev

install-models: ## То же плюс torch и transformers для запуска guard-моделей
	uv sync --extra analysis --extra dev --extra models

# --- Данные ------------------------------------------------------------------

data: ## Скачать SingStreamBench, сохранить в Parquet и проверить контракт данных
	uv run python scripts/prepare_singstreambench.py
	uv run python scripts/validate_singstreambench.py

# --- Проверки ----------------------------------------------------------------

test: ## Запустить тесты
	uv run pytest

lint: ## Проверить код линтером
	uv run ruff check .

check: lint test ## Линтер и тесты, как в CI

# --- Прогоны -----------------------------------------------------------------
# Готовый прогон читается с диска и не пересчитывается. Прерванный прогон
# продолжается с последней сохранённой трассы.

run: ## Прогон модели: make run MODEL=scm_0_5b PROFILE=full
	uv run python scripts/run_guard.py --config $(CONFIG) --profile $(PROFILE)

smoke: ## Быстрая проверка кода на двух трассах: make smoke MODEL=scm_0_5b
	uv run python scripts/run_guard.py --config $(CONFIG) --profile smoke2

full: ## Полный прогон на 210 трассах: make full MODEL=scm_0_5b
	uv run python scripts/run_guard.py --config $(CONFIG) --profile full

rerun: ## Пересобрать таблицы прогона из сохранённых трасс, без повторной оценки моделью
	uv run python scripts/run_guard.py --config $(CONFIG) --profile $(PROFILE) --force

reports: ## CSV-отчёты по сохранённому прогону: make reports MODEL=scm_0_5b PROFILE=full
	uv run python scripts/build_reports.py --config $(CONFIG) --profile $(PROFILE)

# --- Ноутбуки ----------------------------------------------------------------

notebooks: ## Заново исполнить ноутбуки с результатами по сохранённым прогонам
	uv run jupyter nbconvert --to notebook --execute --inplace $(NOTEBOOKS)

figures: ## Перерисовать графики для README по сохранённым полным прогонам
	uv run python scripts/build_figures.py

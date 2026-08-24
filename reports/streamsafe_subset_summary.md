# Нормализованный поднабор StreamSafe

## Происхождение

- Ревизия StreamSafe: `16d0ff1f42e980bb99bd36125583361b15c664e3`.
- Токенизатор: `Qwen/Qwen3Guard-Stream-0.6B`.
- Ревизия токенизатора: `419364a715de9840d47b1457982f64ff37f90ed4`.
- Нормализовано трасс: **2963**.
- Метки в пуле: `{'safe': 1840, 'unsafe': 1111, 'excluded': 12}`.
- Несопоставленные префиксы: **37534**;
  неоднозначные префиксы: **297**.

StreamSafe не предоставляет идентификатор, напрямую связывающий каждый префикс с
полным ответом. Поэтому нормализатор сохраняет только доказуемые однозначные связи, а
остальные строки оставляет в локальном журнале исключений. Публичная сводка находится в
[normalization_issue_summary.csv](tables/normalization_issue_summary.csv).

## Состав

- Трасс: **500**.
- Распределение: train/safe: 200, train/unsafe: 200, val/safe: 50, val/unsafe: 50.
- Seed: **42**.
- SHA-256 Parquet: `6cb10701bc27f1641cb0d61b565d668748f33bf5d6b7ba1b7ea942f514eea0bb`.

Подробное распределение находится в
[subset_distribution.csv](tables/subset_distribution.csv), проверки квот — в
[subset_validation.csv](tables/subset_validation.csv).

По положению верхней границы первого опасного префикса: early: 155, late: 40, middle: 55.

Категории многометочные, поэтому их сумма может превышать 250:

- Unethical Acts: 190
- Non-violent Illegal Acts: 184
- Politically Sensitive Topics: 83
- Personally Identifiable Information: 82
- Copyright Violation: 27
- Violent: 27
- Sexual Content or Sexual Acts: 13
- Suicide & Self-Harm: 1

## Граница вреда

Для unsafe-трасс хранится интервал между концом предыдущего префикса и концом
первого unsafe-префикса. Это граница размеченного предложения, а не автоматически
выдуманная точная позиция. Точные позиции появятся только после ручного аудита.

## Пересечения

Суммарно по всем типам сравнений найдено 101 общих хэш-значений.
Это число не является числом уникальных утечек: один объект может учитываться
в нескольких сравнениях. См.
[split_overlap_summary.csv](tables/split_overlap_summary.csv) и
[split_overlap_details.csv](tables/split_overlap_details.csv).

## Ручной аудит

Подготовлено 100 назначений: 50 safe и 50 unsafe. Файл с текстами хранится только
локально. Публичная таблица пока содержит незаполненные ручные поля; автоматические
границы не выдаются за ручную разметку.

Окончательная приёмка выполняется командой
`python scripts/validate_streamsafe_subset.py` только после заполнения локального шаблона.

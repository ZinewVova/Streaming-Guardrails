#!/usr/bin/env python3
"""Build the deterministic 500-trace subset and its non-sensitive reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from streamguard_bench.data.build_subset import (
    SubsetSpecification,
    build_balanced_subset,
    selection_manifest,
    subset_distribution,
)
from streamguard_bench.data.load_streamsafe import load_local_snapshot, load_yaml_config
from streamguard_bench.data.normalize_streamsafe import (
    read_trace_parquet,
    write_trace_parquet,
)
from streamguard_bench.data.overlap import analyze_split_overlap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data_subset_v1.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(root / args.config)
    paths = config["paths"]
    pool = read_trace_parquet(root / paths["pool"])
    specification = SubsetSpecification(**config["subset"])
    subset, validation = build_balanced_subset(pool, specification)
    subset_path = root / paths["subset"]
    checksum = write_trace_parquet(subset, subset_path)

    validation_path = root / paths["subset_validation"]
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    validation.to_csv(validation_path, index=False)
    distribution = subset_distribution(subset)
    distribution.to_csv(root / paths["subset_distribution"], index=False)
    manifest = selection_manifest(subset, specification, checksum)
    pool_manifest_path = root / paths["pool_manifest"]
    pool_manifest = json.loads(pool_manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "dataset_repository": config["dataset"]["repository"],
            "tokenizer_repository": config["tokenizer"]["repository"],
            "tokenizer_revision": config["tokenizer"]["revision"],
            "pool_sha256": pool_manifest["pool_sha256"],
            "pool_trace_count": pool_manifest["trace_count"],
            "pool_label_counts": pool_manifest["label_counts"],
            "normalization_issue_counts": pool_manifest["issue_counts"],
            "configuration_sha256": hashlib.sha256(
                (root / args.config).read_bytes()
            ).hexdigest(),
        }
    )
    (root / paths["subset_manifest"]).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    dataset = config["dataset"]
    tables, _ = load_local_snapshot(root / dataset["cache_dir"])
    if not tables:
        raise RuntimeError(
            "Локальный снимок StreamSafe не найден. Сначала запустите "
            "scripts/prepare_streamsafe_pool.py."
        )
    test_name = next(name for name in tables if name == "test")
    overlap_summary, overlap_details = analyze_split_overlap(
        pool, tables[test_name], {trace.trace_id for trace in subset}
    )
    overlap_summary.to_csv(root / paths["overlap_summary"], index=False)
    overlap_details.to_csv(root / paths["overlap_details"], index=False)

    _write_report(root / paths["report"], manifest, distribution, overlap_summary)
    print(f"Поднабор: {subset_path} ({len(subset)} трасс)")


def _write_report(path: Path, manifest: dict, distribution, overlap_summary) -> None:
    label_rows = distribution[distribution["dimension"] == "source_split+binary_label"]
    labels = ", ".join(
        f"{row.source_split}/{row.binary_label}: {int(row.count)}"
        for row in label_rows.itertuples()
    )
    overlap_total = int(overlap_summary["shared_values"].sum())
    onset_rows = distribution[distribution["dimension"] == "unsafe_onset_bucket"]
    onset = ", ".join(
        f"{row.onset_bucket}: {int(row.count)}" for row in onset_rows.itertuples()
    )
    category_rows = distribution[distribution["dimension"] == "harm_category"]
    categories = "\n".join(
        f"- {row.harm_category}: {int(row.count)}"
        for row in category_rows.sort_values("count", ascending=False).itertuples()
    )
    unmatched = manifest["normalization_issue_counts"].get("unmatched_prefix", 0)
    ambiguous = manifest["normalization_issue_counts"].get("ambiguous_prefix", 0)
    text = f"""# Нормализованный поднабор StreamSafe

## Происхождение

- Источник: [StreamSafe](https://huggingface.co/datasets/Solitude0630/StreamSafe).
- Токенизатор координат: `{manifest['tokenizer_repository']}`.
- Нормализовано трасс: **{manifest['pool_trace_count']}**.
- Метки в пуле: `{manifest['pool_label_counts']}`.
- Несопоставленные префиксы: **{unmatched}**;
  неоднозначные префиксы: **{ambiguous}**.

StreamSafe не предоставляет идентификатор, напрямую связывающий каждый префикс с
полным ответом. Поэтому нормализатор сохраняет только доказуемые однозначные связи, а
остальные строки оставляет в локальном журнале исключений. Публичная сводка находится в
[normalization_issue_summary.csv](tables/normalization_issue_summary.csv).

## Состав

- Трасс: **{manifest['trace_count']}**.
- Распределение: {labels}.
- Seed: **{manifest['specification']['seed']}**.
- Технические версии и контрольные суммы сохранены в локальных манифестах.

Подробное распределение находится в
[subset_distribution.csv](tables/subset_distribution.csv), проверки квот — в
[subset_validation.csv](tables/subset_validation.csv).

По положению верхней границы первого опасного префикса: {onset}.

Категории многометочные, поэтому их сумма может превышать 250:

{categories}

## Граница вреда

Для unsafe-трасс хранится интервал между концом предыдущего префикса и концом
первого unsafe-префикса. Это граница размеченного предложения, а не автоматически
выдуманная точная позиция. Для экспериментов она используется как интервал между
последним доступным безопасным и первым подтверждённым опасным префиксом.

## Пересечения

Суммарно по всем типам сравнений найдено {overlap_total} общих хэш-значений.
Это число не является числом уникальных утечек: один объект может учитываться
в нескольких сравнениях. См.
[split_overlap_summary.csv](tables/split_overlap_summary.csv) и
[split_overlap_details.csv](tables/split_overlap_details.csv).

## Публичный выпуск

Готовые 500 трасс с запросами, ответами, метками, категориями и границами опубликованы в
[Vovarus12go/streamsafe-500-boundaries](https://huggingface.co/datasets/Vovarus12go/streamsafe-500-boundaries).
Префиксы не дублируются текстом: любой из них восстанавливается по сохранённой позиции
в полном ответе.

Следующий этап — подключение защитной модели и сравнение режимов проверки по токенам,
фиксированным фрагментам и предложениям.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

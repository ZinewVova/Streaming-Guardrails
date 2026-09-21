# Локальные данные

- `fixtures/` — только нейтральные искусственные записи для тестов;
- `cache/huggingface/` — неизменяемый snapshot источника;
- `processed/singstreambench.parquet` и manifest — подготовленные 210 трасс;
- `interim/qwen3guard/{profile}/` — checkpoints и результаты модели.

Все каталоги кроме `fixtures/` игнорируются Git. Датасет содержит чувствительный текст;
не копируйте полные записи в issues, логи, screenshots или outputs ноутбука.

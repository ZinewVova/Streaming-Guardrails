# Контракт SingStreamBench

Подготовленный Parquet содержит одну строку на benchmark-трассу:

| Поле | Тип | Значение |
|---|---|---|
| `trace_id` | string | Стабильный идентификатор по исходному порядку. |
| `query`, `response` | string | Исходные Unicode-строки без нормализации. |
| `query_label`, `response_label` | `safe`/`unsafe` | Две независимые эталонные метки. |
| `unsafe_start_character` | integer | 0-based character offset опасного суффикса. |
| `safe_prefix` | string | В точности `response[:unsafe_start_character]`. |
| `unsafe_start_token` | integer | 1-based первый response-only токен, пересекающий onset. |
| `response_token_count` | integer | Число response-only токенов. |
| `dataset_revision` | string | Закреплённая Git-ревизия источника. |

Для safe-ответа character onset равен длине ответа, а token onset равен
`response_token_count + 1`. Это sentinel; leakage для safe не определяется. Если
граница проходит внутри токена, весь этот токен считается потенциально опасным.

Валидация требует ровно 210 строк, непустые тексты, допустимые метки, точный prefix,
согласованность label/onset и уникальные `trace_id`. Manifest содержит revision,
SHA-256 подготовленного Parquet и распределения меток.

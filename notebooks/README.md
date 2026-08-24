# Notebooks

Notebooks explain analysis and visualize results. Reusable loading, validation, and summary
logic belongs in `src/streamguard_bench/`. Install the project in editable mode before
opening a notebook:

```bash
python -m pip install -e ".[analysis,dev]"
```

Do not persist complete sensitive prompts or responses in notebook outputs.

# GitHub Workflow

The project remote should point to Edward's own repository:

```bash
origin https://github.com/Edward-Jing/llm-from-scratch.git
```

The usual branch for learning work is:

```bash
scratch-llm-starter
```

## Daily Workflow

Work locally first, then commit and push:

```bash
git status
python3 -m unittest tests/test_contracts.py
python3 -m unittest -v tests/test_jsonl.py
git add .
git commit -m "Implement RMSNorm"
git push
```

## GitHub Desktop Workflow

1. Edit files locally.
2. Open GitHub Desktop.
3. Review files in the Changes tab.
4. Write a short commit summary.
5. Click `Commit to scratch-llm-starter`.
6. Click `Push origin`.

## Notes

This repository is a personal learning project inspired by [Happy-LLM](https://github.com/datawhalechina/happy-llm). It should not be pushed back to the original Happy-LLM repository unless you intentionally decide to contribute there later.

# FlashCache-KV Development Guide

## 1. Environment Setup

### Windows PowerShell:
```powershell
# Create dedicated virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Install flashcache in editable mode
pip install --no-build-isolation -e .
```

### Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install --no-build-isolation -e .
```

---

## 2. Running Automated Tests

Run the full pytest suite:
```powershell
python -m pytest -q
```

Run tests with verbose execution and detailed assertion traces:
```powershell
python -m pytest -v
```

Run specific stage tests:
```powershell
python -m pytest tests/test_chunking.py
python -m pytest tests/test_compression.py
python -m pytest tests/test_runtime.py
```

---

## 3. Running Synthetic Demonstrations & Benchmarks

Execute end-to-end synthetic pipeline demonstration:
```powershell
python examples/synthetic_pipeline.py
```

Run memory benchmark:
```powershell
python benchmarks/benchmark_memory.py --seq-len 4096 --chunk-size 16 --bits 4
```

Launch the Streamlit interactive dashboard:
```powershell
streamlit run app.py
```

---

## 4. Git Commit Guidelines

- Commit messages must be clear, imperative, and descriptive:
  - `feat(chunking): implement ChunkKV candidate construction with attention sinks`
  - `feat(compression): implement 2b/4b KIVI-inspired asymmetric quantization`
  - `fix(runtime): ensure BF16 source storage is retained until quantization validation completes`
- Never commit `.venv/`, `.env`, checkpoint tensors, or large benchmark outputs.
- Verify working tree with `git status` prior to committing.


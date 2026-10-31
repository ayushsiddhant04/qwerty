# Contributing to FlashCache-KV

Thank you for your interest in contributing to **FlashCache-KV**!

## Development Process

1. **Repository Setup**:
   Clone the primary repository:
   ```bash
   git clone https://github.com/ayushsiddhant04/qwerty.git
   cd qwerty
   ```
2. **Virtual Environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   pip install --no-build-isolation -e .
   ```
3. **Branching**:
   - Create a feature branch: `git checkout -b feature/my-feature-name`
4. **Code Quality & Testing**:
   - Write unit tests under `tests/` for all new functionality.
   - Run tests prior to committing: `python -m pytest -q`
   - Maintain type annotations and docstrings.
5. **Pull Requests**:
   - Submit PRs against the `main` branch of `https://github.com/ayushsiddhant04/qwerty`.
   - Never force-push or alter historical commit history.

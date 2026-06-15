# thesis

A clean, modern Python project scaffold for your thesis work.

## Quickstart

### 1) Create & activate a virtual environment
**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install the project and dev tools
```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### 3) Run tests
```bash
pytest
```

### 4) Run the app
```bash
python -m thesis.cli --help
python -m thesis.cli greet --name "Ada"
```

### 5) Lint & format
```bash
ruff check .
black .
mypy src
```

---

## VS Code tips

- Open the folder in VS Code (`File` → `Open Folder...`), then when prompted, select the `.venv` interpreter.
- Press `F5` to run with the included debug config.
- `Cmd/Ctrl+Shift+P` → "Python: Select Interpreter" if needed.
- Tests appear in the Testing panel (configured for `pytest`).

---

## Project layout

```
thesis/
  ├─ .vscode/
  ├─ src/thesis/
  ├─ tests/
  ├─ pyproject.toml
  ├─ README.md
  ├─ .gitignore
  ├─ LICENSE
  └─ Makefile
```


# Installing pyflic

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended) or pip

---

## Option 1 -- Install from GitHub (recommended)

This installs the latest release directly from the GitHub repository.

### Into a uv project

From inside your analysis project directory:

```bash
uv add git+https://github.com/PletcherLab/pyflic.git
```

To pin to a specific version tag:

```bash
uv add git+https://github.com/PletcherLab/pyflic.git@v0.3.0
```

### Into a standalone uv environment

```bash
uv pip install git+https://github.com/PletcherLab/pyflic.git
```

### Using pip instead

```bash
pip install git+https://github.com/PletcherLab/pyflic.git
```

---

## Option 2 -- Install from a wheel file

If you have received a `.whl` file (e.g. `pyflic-0.3.0-py3-none-any.whl`):

### Into a uv project

```bash
uv add pyflic-0.3.0-py3-none-any.whl
```

### Into a standalone uv environment

```bash
uv pip install pyflic-0.3.0-py3-none-any.whl
```

### Using pip instead

```bash
pip install pyflic-0.3.0-py3-none-any.whl
```

---

## Verifying the installation

```bash
python -c "import pyflic; print('pyflic installed successfully')"
```

Running `pyflic` on its own prints a summary of available subcommands:

```bash
pyflic
```

Verify GUI dependencies (PyQt + plotting backends) by launching each app:

```bash
pyflic config
pyflic hub
pyflic qc /path/to/project
```

---

## Available commands

pyflic installs a unified CLI (`pyflic`) with subcommands, plus standalone entry points for backward compatibility:

| Command | Standalone | Description |
|---|---|---|
| `pyflic config` | `pyflic-config` | Launch the config editor GUI |
| `pyflic qc <project_dir>` | `pyflic-qc <project_dir>` | Launch the QC viewer |
| `pyflic hub [project_dir]` | `pyflic-hub [project_dir]` | Launch the analysis hub GUI |
| `pyflic lint <path>` | `pyflic-lint <path>` | Validate a `flic_config.yaml` against the schema |
| `pyflic report <project_dir>` | | Generate a PDF experiment report |
| `pyflic clear-cache <project_dir>` | | Remove disk-cached feeding summaries |
| `pyflic version` | | Print the installed version |

The Script Editor is launched from the config editor (`File -> Script Editor`) and does not have a separate CLI command.

Quick start example:

```bash
# Create a config file
pyflic config

# Validate it
pyflic lint /path/to/project

# Run the analysis hub
pyflic hub /path/to/project

# Generate a PDF report
pyflic report /path/to/project
```

---

## Updating

### From GitHub

```bash
uv add git+https://github.com/PletcherLab/pyflic.git --upgrade
```

### From a new wheel file

Install the new file the same way as above -- it will replace the existing version.

---

## For developers

Clone the repository and install in editable mode with dev dependencies:

```bash
git clone https://github.com/PletcherLab/pyflic.git
cd pyflic
uv sync
```

Run the test suite:

```bash
uv run pytest tests/
```

The dev dependency group includes `pytest` and `hypothesis` (for property-based tests of the event detection algorithms).

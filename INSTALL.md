# Installing pyflic

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended) or pip

---

## Option 1 — Install from GitHub (recommended)

This installs the latest release directly from the GitHub repository.

### Into a uv project

From inside your analysis project directory:

```bash
uv add git+https://github.com/yourname/pyflic.git
```

To pin to a specific version tag:

```bash
uv add git+https://github.com/yourname/pyflic.git@v0.1.0
```

### Into a standalone uv environment

```bash
uv pip install git+https://github.com/yourname/pyflic.git
```

### Using pip instead

```bash
pip install git+https://github.com/yourname/pyflic.git
```

---

## Option 2 — Install from a wheel file

If you have received a `.whl` file (e.g. `pyflic-0.1.0-py3-none-any.whl`):

### Into a uv project

```bash
uv add pyflic-0.1.0-py3-none-any.whl
```

### Into a standalone uv environment

```bash
uv pip install pyflic-0.1.0-py3-none-any.whl
```

### Using pip instead

```bash
pip install pyflic-0.1.0-py3-none-any.whl
```

---

## Verifying the installation

```bash
python -c "import pyflic; print('pyflic installed successfully')"
```

The config editor GUI can be launched from the command line:

```bash
pyflic
```

The QC viewer can be launched from a project directory:

```bash
pyflic-qc /path/to/project
```

---

## Updating

### From GitHub

```bash
uv add git+https://github.com/yourname/pyflic.git --upgrade
```

### From a new wheel file

Install the new file the same way as above — it will replace the existing version.

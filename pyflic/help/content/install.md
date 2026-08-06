# Installing pyflic

## Requirements

- **Python 3.13 or later**
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended) or pip

Everything else — PyQt6, matplotlib, pandas, numpy, statsmodels, plotnine — is installed
automatically as a dependency.

## From GitHub

The usual way to install.

```bash
uv add git+https://github.com/PletcherLab/pyflic.git
```

To pin a version, which is what you want once an analysis is underway:

```bash
uv add git+https://github.com/PletcherLab/pyflic.git@v0.3.0
```

Into a standalone environment, or with pip:

```bash
uv pip install git+https://github.com/PletcherLab/pyflic.git
pip install git+https://github.com/PletcherLab/pyflic.git
```

## From a wheel file

If you were sent a `.whl`:

```bash
uv add pyflic-0.3.0-py3-none-any.whl
uv pip install pyflic-0.3.0-py3-none-any.whl
pip install pyflic-0.3.0-py3-none-any.whl
```

## Checking it worked

```bash
pyflic version
```

Then open the help you are reading now:

```bash
pyflic help
```

Both working means the package and its GUI dependencies are installed correctly.

## Commands

| Command | What it does |
|---|---|
| `pyflic` | Analysis Hub — the same as `pyflic hub` |
| `pyflic hub [project]` | Analysis Hub |
| `pyflic config [project]` | Config Editor |
| `pyflic qc <project>` | QC Viewer |
| `pyflic help [topic]` | This help window |
| `pyflic lint <project>` | Validate a configuration |
| `pyflic report <project>` | Write a PDF report |
| `pyflic clear-cache <project>` | Remove cached feeding summaries |
| `pyflic version` | Print the installed version |

`pyflic --help` prints the command list as text.

## Updating

```bash
uv add --upgrade git+https://github.com/PletcherLab/pyflic.git
pip install --upgrade --force-reinstall git+https://github.com/PletcherLab/pyflic.git
```

Pin a version for work in progress. Detection parameters and their defaults can change
between releases, and re-analysing half of an experiment on a different version is a
reproducibility problem that is very hard to notice afterwards. Record the output of
`pyflic version` alongside your results.

## Version control for your data

pyflic writes results into your project directory. If you keep projects in git, exclude the
generated folders:

```gitignore
*_results/
.pyflic_cache/
```

Keep `flic_config.yaml` and `remove_chambers.csv` under version control — together they
define the analysis, and they are small.

---

Next: **[Getting started](getting-started.md)**

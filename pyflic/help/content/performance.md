# Performance and caching

Loading is the expensive step in pyflic: reading CSVs, subtracting baselines, and running
detection across every well of every device. Everything after it is comparatively cheap.
These are the three levers.

## Parallel loading

```yaml
- action: load
  parallel: true
```

```python
exp = load_experiment_yaml(path, parallel=True)
```

DFMs are loaded concurrently. Worth enabling on any experiment with more than one device.

## The disk cache

With `use_disk_cache=True` (the default), the computed feeding summary is cached in
`<project directory>/.pyflic_cache/`.

The cache key is derived from the **SHA-256 hash of the configuration file** together with
the **modification times and sizes of every DFM CSV**. Change a parameter, edit the
configuration, or replace a data file, and the key changes and the cache is bypassed
automatically.

That construction is why you almost never need to clear it: a stale entry cannot be
served for changed inputs. Clearing is a disk-space operation, not a correctness one.

```bash
pyflic clear-cache /path/to/project
```

```python
from pathlib import Path
from pyflic.base import cache
cache.clear(Path("/path/to/project"))
```

The directory is safe to delete by hand, and safe to exclude from version control — add
`.pyflic_cache/` to your `.gitignore`.

One caveat worth knowing: the key uses CSV modification time and size, not a content hash,
for speed. A file edited in place to exactly the same size within the same timestamp
resolution would not invalidate the cache. This does not happen with rig output, but if you
are generating synthetic test data programmatically, clear the cache between runs.

## Lazy loading

```python
exp = load_experiment_yaml(path, eager=False)
```

Skips pre-computing the feeding summary at load time. Use it when you only want to inspect
raw data or run the QC viewer — the summary is computed on demand if you later ask for it.

## Restricting the time range

```python
exp = load_experiment_yaml(path, range_minutes=(0, 240))
```

Analysing a window rather than a whole multi-day recording is often the largest saving
available. Ranged loads write into their own suffixed output folders, so they never
overwrite whole-experiment results.

## In the hub

Turn off **Interactive plots** when generating many figures. Interactive figures keep a
live canvas in memory; static ones are rasterised once and released.

---

Related: [Python API](python-api.md) · [Analysis Hub](app-hub.md) ·
[Troubleshooting](troubleshooting.md)

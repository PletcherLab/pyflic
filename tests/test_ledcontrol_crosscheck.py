"""The simulator against the firmware itself (opt-in, ``-m slow``).

``tests/v4_fixtures.py`` reimplements DFMV3_V2's ``src/LEDControl.c`` in
Python; that file is the authority.  Here it is compiled with the ``ziglang``
package's C compiler (or, without it, ``CC`` or a ``gcc``, ``clang`` or ``cc``
on the PATH, as DFMV3_V2's ``tests/run_tests.py`` does) into
``tests/ledcontrol_harness.c``, fed the same millisecond signals (as sums of
128 scans), the same 63-byte instructions at the same milliseconds and the
same Linkage, and the two are compared: the LED mask after Linkage every
millisecond, and every record's instruction number, Paradigm type, Phase
bits, OnMs, Activations, Requirement and Progress.  The instructions carry
the Contact counter's flag (firmware 4.1.0) where the Paradigm counts
Contact, whose Requirement and Progress the firmware reports in units of
10 ms and PyMCU writes in ms.

Skipped unless the DFMV3_V2 source is found (``DFMV3_V2_DIR``, default the
sibling checkout) and a C compiler is.
"""

from __future__ import annotations

import os
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from v4_fixtures import (
    CONTACT_UNIT_MS,
    FAMILY_SPEC_KEYS,
    THRESHOLD_OFF,
    THRESHOLD_ON,
    FirmwareInputs,
    RecordingSpec,
    SimResult,
    choice_pr_spec,
    contact_pr_spec,
    contact_stress_spec,
    family_specs,
    firmware_inputs,
    fixed_ratio_spec,
    one_of_each_spec,
    repeating_open_loop_spec,
    simulate,
    stress_spec,
    yoked_controls_spec,
)

pytestmark = pytest.mark.slow

HERE = Path(__file__).resolve().parent
DFMV3_DIR = Path(os.environ.get("DFMV3_V2_DIR", "C:/Users/scott/Desktop/GithubLocal/DFMV3_V2"))
SRC = DFMV3_DIR / "src"


def _python_with_zig() -> list[str] | None:
    """The first Python that can run ``-m ziglang``: this one, the system
    Python 3.14, the launcher's 3.14, then ``python`` on PATH."""
    for command in ([sys.executable], ["C:/Python314/python.exe"], ["py", "-3.14"], ["python"]):
        try:
            done = subprocess.run(command + ["-m", "ziglang", "version"], capture_output=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if done.returncode == 0:
            return command
    return None


def _c_compiler() -> list[str] | None:
    """``zig cc`` from the first Python with ``ziglang``, else ``CC`` or the
    first of ``gcc``, ``clang`` and ``cc`` on the PATH."""
    python = _python_with_zig()
    if python is not None:
        return python + ["-m", "ziglang", "cc"]
    if os.environ.get("CC"):
        return shlex.split(os.environ["CC"])
    for name in ("gcc", "clang", "cc"):
        if shutil.which(name):
            return [name]
    return None


def _build(compiler: list[str], out: Path) -> Path | None:
    """The harness built in *out*, or ``None`` when *out* can't run programs
    (a ``noexec`` temporary folder)."""
    exe = out / "harness.exe"
    build = compiler + ["-std=c99", "-O2", "-I", str(SRC),
                        str(HERE / "ledcontrol_harness.c"), str(SRC / "LEDControl.c"), "-o", str(exe), "-lm"]
    done = subprocess.run(build, capture_output=True, text=True, cwd=out, timeout=600)
    assert done.returncode == 0, f"harness build failed:\n{done.stdout}\n{done.stderr}"
    try:
        # Without arguments it only prints its usage.
        subprocess.run([str(exe)], capture_output=True, timeout=60)
    except PermissionError:
        return None
    return exe


@pytest.fixture(scope="module")
def harness(tmp_path_factory: pytest.TempPathFactory):
    if not (SRC / "LEDControl.c").is_file():
        pytest.skip(f"DFMV3_V2 source not found at {SRC} (set DFMV3_V2_DIR)")
    compiler = _c_compiler()
    if compiler is None:
        pytest.skip("no C compiler: no Python with the ziglang package, no CC, gcc, clang or cc")
    exe = _build(compiler, tmp_path_factory.mktemp("harness"))
    if exe is not None:
        yield exe
        return
    # The temporary folder is noexec: build beside the tests, in the git-ignored __pycache__.
    cache = HERE / "__pycache__"
    cache.mkdir(exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="ledcontrol-harness-", dir=cache))
    try:
        exe = _build(compiler, out)
        if exe is None:
            pytest.skip(f"neither the temporary folder nor {cache} can run the harness")
        yield exe
    finally:
        shutil.rmtree(out, ignore_errors=True)


def _run_firmware(exe: Path, inputs: FirmwareInputs, dfm_id: int, work: Path) -> tuple[np.ndarray, np.ndarray]:
    """(records, per-ms masks) from the firmware's own code."""
    n_ms = len(inputs.signals)
    source, records, masks = work / f"in{dfm_id}.bin", work / f"rec{dfm_id}.txt", work / f"mask{dfm_id}.bin"
    with open(source, "wb") as f:
        f.write(struct.pack("<iii", n_ms, inputs.record_ms, len(inputs.instructions)))
        f.write(bytes(inputs.linkage))
        for ms, frame in inputs.instructions:
            f.write(struct.pack("<i", ms) + frame)
        f.write(np.ascontiguousarray(inputs.signals, dtype="<i4").tobytes())
    done = subprocess.run([str(exe), str(source), str(records), str(masks)], capture_output=True, text=True,
                          timeout=600)
    assert done.returncode == 0, done.stderr
    source.unlink()
    return np.loadtxt(records, dtype=np.int64, ndmin=2), np.fromfile(masks, dtype="<u2")


def _simulated_masks(result: SimResult, linkage: tuple[int, ...], n_ms: int) -> np.ndarray:
    """The simulator's LED mask after Linkage, per ms."""
    own = np.zeros((12, n_ms + 1), dtype=np.int32)
    for w in range(12):
        for a, b in result.truth.lit_spans_ms[w + 1]:
            own[w, a] += 1
            own[w, b] -= 1
    lit = np.cumsum(own, axis=1)[:, :n_ms] > 0
    mask = np.zeros(n_ms, dtype=np.uint16)
    for w in range(12):
        group = np.zeros(n_ms, dtype=bool)
        for j in range(12):
            if linkage[j] == linkage[w]:
                group |= lit[j]
        mask |= group.astype(np.uint16) << w
    return mask


def _first_difference(name: str, ours: np.ndarray, theirs: np.ndarray, where: str) -> str:
    differs = ours != theirs
    bad = np.flatnonzero(differs.any(axis=1) if differs.ndim > 1 else differs)
    i = int(bad[0])
    return (f"{name} differs at {len(bad)} {where}s, first at {where} {i}: "
            f"simulator {ours[i].tolist()}, firmware {theirs[i].tolist()}")


def _compare(exe: Path, spec: RecordingSpec, work: Path) -> None:
    results = simulate(spec)
    for dfm in spec.dfms:
        result = results[dfm.dfm_id]
        inputs = firmware_inputs(spec, dfm.dfm_id)
        records, masks = _run_firmware(exe, inputs, dfm.dfm_id, work)
        rows = result.rows
        n_ms = len(masks)
        label = f"DFM {dfm.dfm_id}"

        ours = _simulated_masks(result, tuple(dfm.linkage), n_ms)
        assert np.array_equal(ours, masks), f"{label}: " + _first_difference("LED mask", ours, masks, "millisecond")

        assert len(records) == len(rows), f"{label}: {len(records)} firmware records, {len(rows)} rows"
        index, number, ptype, phase_bits, led = (records[:, k] for k in range(5))
        on_ms, activations, requirement, progress = (records[:, 5 + 12 * k: 17 + 12 * k] for k in range(4))
        assert np.array_equal(index, rows["Index"].to_numpy())

        # The instruction in force: number k is the k-th of the simulator's schedule, 0 the idle one.
        starts = np.asarray([ms for ms, *_ in result.truth.schedule])
        expected_number = np.searchsorted(starts, index * (n_ms // len(rows)) - 1, side="right")
        assert np.array_equal(number, expected_number), f"{label}: " + _first_difference(
            "instruction number", expected_number, number, "record")
        # The idle instruction before the program is ClosedLoop (1).
        types = np.asarray([1] + [frame[4] for _, frame in inputs.instructions])
        assert np.array_equal(ptype, types[number]), f"{label}: paradigm type"

        def well_frame(prefix: str) -> np.ndarray:
            return np.stack([rows[f"{prefix}{w}"].to_numpy() for w in range(1, 13)], axis=1)

        # PyMCU writes a Contact counter's Requirement and Progress in ms; the firmware reports units.
        contact_names = [name for name, p in spec.paradigms.items() if p.counts_contact]
        unit = np.where(rows["Paradigm"].isin(contact_names).to_numpy(dtype=bool, na_value=False),
                        CONTACT_UNIT_MS, 1)[:, None]

        def firmware_units(prefix: str) -> np.ndarray:
            values = rows[[f"{prefix}{w}" for w in range(1, 13)]].fillna(0).to_numpy(np.int64)
            assert not (values % unit).any(), f"{label}: {prefix} under Contact is not whole units"
            return values // unit

        phase = (well_frame("Phase") == "A").astype(np.int64)
        our_bits = (phase << np.arange(12)).sum(axis=1)
        checks = {
            "Phase bits": (our_bits, phase_bits),
            "OptoCol1": (rows["OptoCol1"].to_numpy(), led),
            "OnMs": (well_frame("OnMs").astype(np.int64), on_ms),
            "Activations": (well_frame("Activations").astype(np.int64) % 65536, activations),
            # The firmware reports 0 where PyMCU writes NA.
            "Requirement": (firmware_units("Requirement"), requirement),
            "Progress": (firmware_units("Progress"), progress),
        }
        for name, (simulated, firmware) in checks.items():
            assert np.array_equal(simulated, firmware), f"{label}: " + _first_difference(
                name, simulated, firmware, "record")


def test_the_choice_progressive_ratio_recording_matches_the_firmware(harness: Path, tmp_path: Path) -> None:
    _compare(harness, choice_pr_spec(minutes=12.0, dfm_ids=(1,)), tmp_path)


def test_the_fixed_ratio_recording_matches_the_firmware(harness: Path, tmp_path: Path) -> None:
    _compare(harness, fixed_ratio_spec(minutes=10.0), tmp_path)


def test_one_dfm_of_each_paradigm_type_matches_the_firmware(harness: Path, tmp_path: Path) -> None:
    _compare(harness, one_of_each_spec(minutes=6.0), tmp_path)


def test_the_repeating_and_yoked_recordings_match_the_firmware(harness: Path, tmp_path: Path) -> None:
    _compare(harness, repeating_open_loop_spec(minutes=20.0), tmp_path)
    _compare(harness, yoked_controls_spec(minutes=5.0), tmp_path)


@pytest.mark.parametrize("seed", [7, 8, 9])
def test_the_engine_edge_cases_match_the_firmware(harness: Path, tmp_path: Path, seed: int) -> None:
    _compare(harness, stress_spec(minutes=8.0, seed=seed), tmp_path)


def test_the_contact_counter_paired_yoked_recording_matches_the_firmware(harness: Path, tmp_path: Path) -> None:
    # Acclimation, then Requirements of k s of contact on the paired wells; the yoked wells are Off.
    _compare(harness, contact_pr_spec(minutes=12.0), tmp_path)


@pytest.mark.parametrize("seed", [7, 8, 9])
def test_the_contact_counter_edge_cases_match_the_firmware(harness: Path, tmp_path: Path, seed: int) -> None:
    _compare(harness, contact_stress_spec(minutes=8.0, seed=seed), tmp_path)


#: Every Paradigm Family spec but the restart, which is PyMCU's and the power
#: supply's doing, not LEDControl.c's (firmware_inputs refuses it).
@pytest.mark.parametrize("key", [k for k in FAMILY_SPEC_KEYS if k != "restart"])
def test_the_paradigm_family_specs_match_the_firmware(harness: Path, tmp_path: Path, key: str) -> None:
    _compare(harness, family_specs()[key], tmp_path)


def test_threshold_wire_values_are_the_firmwares() -> None:
    # The harness relies on these matching LEDControl.h.
    header = (SRC / "LEDControl.h").read_text() if (SRC / "LEDControl.h").is_file() else ""
    if not header:
        pytest.skip("DFMV3_V2 source not found")
    assert f"THRESHOLD_ON 0x{THRESHOLD_ON:04X}" in header
    assert f"THRESHOLD_OFF 0x{THRESHOLD_OFF:04X}" in header

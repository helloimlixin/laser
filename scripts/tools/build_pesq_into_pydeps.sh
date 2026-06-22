#!/usr/bin/env bash
# Build the `pesq` C/Cython extension for CPython 3.11 and stage it into the
# shared PYTHONUSERBASE that the SLURM container jobs mount + prepend to
# PYTHONPATH. pesq publishes no cp311 wheel and the runtime PyTorch container has
# no compiler, so it cannot be pip-installed at job start; pre-staging the built
# package is the only viable path. STOI (pure python) installs normally via pip.
#
# Built on the el7 login node (glibc 2.17) -> forward-compatible with the newer
# container glibc. src/audio_logging.py::_has_pesq() picks it up automatically.
set -euo pipefail

PYDEPS_SITE="${PYDEPS_SITE:-/scratch/$USER/.pydeps/laser_src_py311/lib/python3.11/site-packages}"
BUILD_ENV="${BUILD_ENV:-/scratch/$USER/.tmp/pesqbuild311}"

echo "[pesq-build] target site-packages: $PYDEPS_SITE"
mkdir -p "$PYDEPS_SITE" "$(dirname "$BUILD_ENV")"

# el7 has glibc 2.17 (no modern numpy wheel) and gcc 4.8.5 (too old to compile
# numpy/pesq). Pull a modern compiler + binary numpy/cython from conda-forge and
# build pesq against them with build isolation OFF (so pip does not try to
# recompile numpy from an incompatible sdist).
if [[ ! -x "$BUILD_ENV/bin/python" ]]; then
  echo "[pesq-build] creating python 3.11 build env (conda-forge toolchain)"
  conda create -y -p "$BUILD_ENV" -c conda-forge \
    python=3.11 pip setuptools wheel "numpy>=2.0,<2.2" cython c-compiler cxx-compiler >/dev/null
fi
PY="$BUILD_ENV/bin/python"
echo "[pesq-build] build python: $("$PY" -V 2>&1)"

CONDA_CC="$(ls "$BUILD_ENV"/bin/*-cc 2>/dev/null | head -1)"
CONDA_CXX="$(ls "$BUILD_ENV"/bin/*-c++ 2>/dev/null | head -1)"
export CC="${CONDA_CC:-$(ls "$BUILD_ENV"/bin/*-gcc 2>/dev/null | head -1)}"
export CXX="${CONDA_CXX:-$(ls "$BUILD_ENV"/bin/*-g++ 2>/dev/null | head -1)}"
echo "[pesq-build] CC=$CC"
"$CC" --version 2>&1 | head -1

echo "[pesq-build] building pesq from sdist (no build isolation, conda numpy/cython)"
"$PY" -m pip install -q --no-build-isolation --no-binary=:all: pesq

echo "[pesq-build] smoke test the API on synthetic 16k waveforms"
"$PY" - <<'PY'
import numpy as np
from pesq import pesq
sr=16000; t=np.linspace(0,2,2*sr,endpoint=False)
ref=(0.4*np.sin(2*np.pi*220*t)+0.2*np.sin(2*np.pi*440*t)).astype(np.float32)
deg=(ref+0.03*np.random.RandomState(0).randn(ref.size)).astype(np.float32)
print("PESQ wb noisy =", round(float(pesq(sr,ref,deg,'wb')),3))
print("PESQ wb clean =", round(float(pesq(sr,ref,ref,'wb')),3))
print("pesq import OK")
PY

SP="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
echo "[pesq-build] copying built artifacts from $SP -> $PYDEPS_SITE"
cp -rv "$SP"/pesq "$PYDEPS_SITE"/ 2>/dev/null || cp -rv "$SP"/pesq* "$PYDEPS_SITE"/
cp -rv "$SP"/pesq-*.dist-info "$PYDEPS_SITE"/ 2>/dev/null || true
# the compiled extension may sit at top level (cython_pesq*.so) on some builds
cp -v "$SP"/cython_pesq*.so "$PYDEPS_SITE"/ 2>/dev/null || true

echo "[pesq-build] staged files:"
ls -la "$PYDEPS_SITE"/pesq* 2>/dev/null | head
echo "[pesq-build] DONE"

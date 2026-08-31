#!/bin/bash
# Stage everything a DLC worker cannot fetch for itself into assets/.
#
# A worker mounts /cpfs01 and nothing else: no HF cache, no network, and an image whose
# site-packages is not the one this tree was developed against. assets/ is gitignored
# because it holds vendored weights and wheels, so this script is the reproducible record
# of what belongs there.
set -eu
cd "$(dirname "$0")/.."

SITE=$(python -c "import site; print(site.getsitepackages()[0])")

echo "== t5-small (encoder and tokenizer) =="
python - <<'PY'
import shutil
from pathlib import Path
snapshots = Path.home() / ".cache/huggingface/hub/models--t5-small/snapshots"
if not snapshots.exists():
    from transformers import AutoTokenizer, T5EncoderModel
    AutoTokenizer.from_pretrained("t5-small"); T5EncoderModel.from_pretrained("t5-small")
snapshot = sorted(snapshots.iterdir())[-1]
destination = Path("assets/t5-small")
if destination.exists():
    shutil.rmtree(destination)
destination.mkdir(parents=True)
for entry in snapshot.iterdir():
    resolved = entry.resolve()
    if resolved.is_file():
        shutil.copy2(resolved, destination / entry.name)
print("  ", sorted(p.name for p in destination.iterdir()))
PY

echo "== muon (the optimizer the official configs ask for) =="
# The cluster image does not ship muon-optimizer. It is a single pure-python module, and
# the image runs the same python 3.12, so copying it is enough; a wheel built for another
# interpreter would not be.
mkdir -p assets/pydeps
cp "$SITE/muon.py" assets/pydeps/
cp -r "$SITE"/muon_optimizer-*.dist-info assets/pydeps/ 2>/dev/null || true
ls assets/pydeps

echo
echo "staged:"
du -sh assets/*

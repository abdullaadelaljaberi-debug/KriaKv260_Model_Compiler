#!/usr/bin/env python3
"""
v0.7.3 — Patch 04_vai35_benchmark.ipynb to use per-model .prototxt preprocessing.

This restores the proven VAI 2.5 approach. The 3.5 notebook was using
hardcoded catalogue defaults (IM_MEAN_BGR_TF = [104, 117, 123] etc.) that
drift from each model's actual quantization calibration parameters, hurting
top-1 accuracy by 10-20% per-model.

Every model in AMD's VAI model zoo ships with a .prototxt file containing
the EXACT mean and scale values that match its quantization calibration.
Using those values, not approximations, is what makes accuracy land in the
expected band.

This patch:
  1. Adds find_prototxt() and parse_prototxt() helpers
  2. Replaces preprocess_for_model() with a prototxt-aware version that
     always produces BGR output (matching AMD VAI convention)
  3. In each benchmark function, reads the prototxt at model-load time
     and passes proto_means/proto_scales to all preprocess_for_model calls

The patch is idempotent: running it twice is a no-op.

USAGE on the Kria:
    sudo /usr/local/share/pynq-venv/bin/python3 /tmp/patch_notebook_v073.py
"""
import json
import shutil
import re
from pathlib import Path

NB_PATH = Path('/home/ubuntu/KriaKv260_Model_Compiler/notebooks/04_vai35_benchmark.ipynb')

if not NB_PATH.exists():
    raise SystemExit(f"Notebook not found at {NB_PATH}")

backup = NB_PATH.with_name(NB_PATH.name + '.bak')
if not backup.exists():
    shutil.copy(NB_PATH, backup)
    print(f"[*] Backup created:  {backup}")
else:
    print(f"[*] Backup already exists at {backup} (preserving)")

nb = json.load(open(NB_PATH))

# ─── NEW CODE: helpers + replacement preprocess function ──────────────────
NEW_HELPERS_AND_PREPROCESS = '''def find_prototxt(model_dir):
    """Return the .prototxt file in a model directory, or None."""
    files = list(Path(model_dir).rglob('*.prototxt'))
    return files[0] if files else None

def parse_prototxt(prototxt_path):
    """Parse mean and scale from an AMD VAI model zoo .prototxt file.

    AMD ships each model's exact preprocessing parameters in its prototxt.
    These are the values used during quantization calibration; using them
    matches the model's expected input distribution and gives accuracy
    within published values. Falling back to hardcoded catalogue defaults
    can drift by 10-20% per-model.

    Returns (means, scales) as lists of 3 floats, or (None, None) if missing.
    """
    import re as _re
    if not prototxt_path or not Path(prototxt_path).exists():
        return None, None
    text = Path(prototxt_path).read_text()
    means  = [float(m) for m in _re.findall(r'mean:\\s*([-\\d.]+)',  text)][:3]
    scales = [float(s) for s in _re.findall(r'scale:\\s*([-\\d.]+)', text)][:3]
    return (means  if len(means)  == 3 else None,
            scales if len(scales) == 3 else None)

def preprocess_for_model(img_path_or_array, cfg, in_shape, means=None, scales=None):
    """Preprocess image for AMD VAI inference. Always produces BGR output.

    AMD's VAI model zoo encodes mean values in BGR channel order regardless
    of the original training framework, so we always produce BGR.

    means/scales: prefer values from parse_prototxt() (model-specific exact
    calibration params). Falls back to cfg['mean']/cfg['scale'] from the
    catalogue when prototxt is not available.
    """
    h, w = in_shape[1], in_shape[2]
    if isinstance(img_path_or_array, (str, Path)):
        img = Image.open(img_path_or_array).convert('RGB').resize((w, h), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32)[..., ::-1].copy()  # RGB -> BGR
    else:
        # already BGR from cv2
        arr = cv2.resize(img_path_or_array, (w, h),
                          interpolation=cv2.INTER_LINEAR).astype(np.float32)

    if means is None:
        means = cfg.get('mean')
    if scales is None:
        scales = cfg.get('scale')
    if means is not None:
        arr = arr - np.array(means, dtype=np.float32)
    if scales is not None:
        arr = arr * np.array(scales, dtype=np.float32)
    elif means is None:
        arr = arr / 255.0
    return arr'''


def cell_text(cell):
    """Get cell source as a single string."""
    src = cell.get('source', '')
    return ''.join(src) if isinstance(src, list) else src


def set_cell_text(cell, text):
    """Set cell source from a single string, preserving Jupyter's line-list format."""
    cell['source'] = text.splitlines(keepends=True)


# Track changes for reporting
changes = {'preprocess_fn': 0, 'load_insert': 0, 'call_sites': 0}

for cell_idx, cell in enumerate(nb.get('cells', [])):
    if cell['cell_type'] != 'code':
        continue
    src = cell_text(cell)

    # ─── (1) Replace preprocess_for_model + add helpers ──────────────────
    # Match from "def preprocess_for_model" through the end of the function,
    # detected by the next top-level def or end of cell.
    if 'def preprocess_for_model' in src and 'find_prototxt' not in src:
        # Find function start
        match = re.search(r'^def preprocess_for_model\b', src, re.MULTILINE)
        if match:
            start = match.start()
            # Find function end: next top-level def, or end of source
            after = src[start:]
            # Match from `def preprocess_for_model(` through `return arr` block end
            # by looking for the next def at same indentation (column 0)
            end_match = re.search(r'\n(?=def \w)', after[10:])  # skip the def itself
            if end_match:
                end = start + 10 + end_match.start() + 1  # include trailing newline
            else:
                end = len(src)
            src = src[:start] + NEW_HELPERS_AND_PREPROCESS + '\n' + src[end:]
            changes['preprocess_fn'] += 1

    # ─── (2) Insert prototxt parsing after model load in benchmark cells ─
    # Only target cells that load a model AND run inference (i.e., benchmark fns).
    if ('dpu = overlay.runner' in src and 'execute_async' in src
            and 'proto_means' not in src):
        # The smoke test, main benchmark, and mAP cells share this signature.
        # Insert right after "dpu = overlay.runner" line.
        # Use careful regex to preserve indentation.
        pattern = re.compile(
            r'(^(?P<indent>[ \t]*)dpu = overlay\.runner\n)',
            re.MULTILINE
        )
        def insert_proto(m):
            indent = m.group('indent')
            return (m.group(1) +
                    f"{indent}# Read per-model preprocessing params from the bundled prototxt.\n"
                    f"{indent}# Hardcoded catalogue defaults drift from AMD's calibration values\n"
                    f"{indent}# by 10-20%, hurting top-1 accuracy. The prototxt has exact values.\n"
                    f"{indent}proto_means, proto_scales = parse_prototxt(find_prototxt(xm.parent))\n")
        new_src, n = pattern.subn(insert_proto, src)
        if n > 0:
            src = new_src
            changes['load_insert'] += n

    # ─── (3) Update preprocess_for_model call sites ──────────────────────
    # Add proto_means, proto_scales to existing 3-arg calls.
    # Idempotent: skip calls that already have 4-5 args.
    if 'preprocess_for_model(' in src and 'proto_means' in src:
        # Match preprocess_for_model(<arg1>, cfg, in_shape) with NO further args
        pattern = re.compile(
            r'preprocess_for_model\(([^,]+),\s*cfg,\s*in_shape\)(?!\s*,)'
        )
        new_src, n = pattern.subn(
            r'preprocess_for_model(\1, cfg, in_shape, proto_means, proto_scales)',
            src)
        if n > 0:
            src = new_src
            changes['call_sites'] += n

    set_cell_text(cell, src)


# Save the patched notebook
with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print()
print(f"[*] Patches applied:")
print(f"      preprocess_for_model replacement : {changes['preprocess_fn']}")
print(f"      prototxt-parse insertions        : {changes['load_insert']}")
print(f"      preprocess call site updates     : {changes['call_sites']}")
print()
print(f"[*] Saved patched notebook: {NB_PATH}")
print()
if all(v == 0 for v in changes.values()):
    print("[!] No patches applied. Notebook may already be patched (idempotent),")
    print("    or the regex patterns don't match. Restore from backup if needed:")
    print(f"    cp {backup} {NB_PATH}")
else:
    print("[*] Next steps:")
    print("    1. In Jupyter: Kernel > Restart Kernel and Clear All Outputs")
    print("    2. Run the test cell to confirm accuracy improves")
    print("    3. If good, scp the patched notebook to your laptop and commit to repo")

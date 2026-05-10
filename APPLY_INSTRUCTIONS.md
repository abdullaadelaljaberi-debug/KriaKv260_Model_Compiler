# Applying the Pass 5 final + Pass 6 final drop

This single tarball replaces the 9 separate Pass 5 patches and the
incremental Pass 6 drop with the final consolidated state.

## On your laptop

```bash
cd ~/Documents/Girona_Masters/Thesis/KriaKv260_Model_Compiler

# Apply the consolidated drop
tar xzf ~/Downloads/KriaKv260_pass5_pass6_final.tar.gz

# The old single notebook is replaced by two new ones.
# Remove the old name if it's still in the working tree:
git rm -f notebooks/02_deploy_live.ipynb 2>/dev/null || \
    rm -f notebooks/02_deploy_live.ipynb

# Verify the new file layout
git status

# Should show:
#   modified: scripts/kria/lib/common.sh
#   modified: scripts/kria/01_install_vai35.sh
#   modified: scripts/kria/02_apply_tuning.sh
#   modified: scripts/kria/03_install_systemd.sh
#   modified: scripts/kria/run_live.sh
#   modified: lpr_pipeline/deploy/__init__.py
#   modified: lpr_pipeline/deploy/camera.py
#   modified: lpr_pipeline/deploy/preprocess.py
#   new file: lpr_pipeline/deploy/draw.py
#   deleted:  notebooks/02_deploy_live.ipynb
#   new file: notebooks/02_deploy_text.ipynb
#   new file: notebooks/03_deploy_visual.ipynb

# Commit (suggested message)
git add -A
git commit -m "Pass 5 + Pass 6 final: consolidated install + visual notebook"

# Tag for the thesis defense
git tag -a v0.6-pass6-validated -m "Pass 5 + 6 validated; 60 fps live demo"
git push origin main --tags
```

## On the Kria

```bash
cd ~/KriaKv260_Model_Compiler
git pull --tags

# No re-install needed — the consolidation only renames files and adds new
# ones. All previous installs are intact.

# Try the visual notebook:
sudo bash scripts/kria/run_live.sh yolov5n visual
# (then SSH-tunnel from your laptop and open the URL)
#
# Or the text notebook (default):
sudo bash scripts/kria/run_live.sh yolov5n
# (same as: ... yolov5n text)
```

## What to expect

The visual notebook should run at 40-50 fps end-to-end (display path costs
extra over the text mode). DPU compute time is unchanged from text mode.

The text notebook should match Pass 6's first run: ~60 fps, ~12.5 ms total
per frame, 7.75 ms DPU.

After the optimized preprocess (`np.multiply` trick), preprocess should
drop from 3.84 ms to ~1-1.5 ms on the Kria, giving a slightly tighter
end-to-end ms total — text mode will still be camera-bound at 60 fps, but
the DPU's theoretical ceiling rises from 80 fps to ~85-90 fps.

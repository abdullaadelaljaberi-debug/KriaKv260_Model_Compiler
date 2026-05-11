# ImageNet Benchmark Sample — Setup Guide

This directory documents how the classification accuracy dataset used by
`04_vai35_benchmark.ipynb` is constructed and how to reproduce it.

## What this is

A small (default N=500), reproducible random sample of ImageNet-style images
used by the benchmark notebook to compute Top-1 / Top-5 accuracy. The
benchmark uses this dataset both for accuracy measurement and for warming up
the DPU before per-model latency measurement.

## Why N=500 (not full ImageNet val)

Full ImageNet val (50,000 images, ~6.4 GB) is the canonical benchmark for
classification accuracy but adds 20+ minutes of disk I/O per benchmark run on
the KV260's SD card. N=500 gives a ±2.2% confidence interval on Top-1 — wide
enough to be visible in the numbers but tight enough to rank models reliably.

The trade-off is documented in the methodology section of the benchmark
report. Accuracy numbers should be read as **inter-model ranking on this
specific sample**, not as direct equivalents to AMD's published full-val
numbers.

## How to reproduce

### 1. Get a compatible source archive

Any Kaggle ImageNet-style zip with one folder per class works. Confirmed
compatible:

- **`ifigotin/imagenetmini-1000`** — 1000 classes, ~38K images, ~3.9 GB.
  Most convenient option. Class folder names are mostly human-readable.

The script walks the archive recursively, so the internal layout
(`archive/imagenet-mini/train/<classname>/*.JPEG` etc.) doesn't matter as
long as image files live under folders named after classes.

Download with the Kaggle CLI or web interface, then place the resulting
`.zip` somewhere accessible (e.g. `~/Downloads/archive.zip`).

### 2. Generate the sample (on laptop)

```bash
cd ~/Documents/Girona_Masters/Thesis/KriaKv260_Model_Compiler
python3 scripts/host/generate_imagenet_sample.py \
    --archive ~/Downloads/archive.zip \
    --output ./Datasets/imagenet_sample \
    --n 500 \
    --seed 42
```

The default `--seed 42` matches the rest of the benchmark. Using a different
seed produces a different sample (still reproducible if you record the
seed).

The script prints progress through five stages: extract → scan → sample →
copy → cleanup. Total time ~30 seconds on a typical laptop.

### 3. Sync to the Kria (on laptop)

```bash
rsync -avh --copy-links --delete --exclude='.ipynb_checkpoints' \
    ./Datasets/imagenet_sample/ \
    ubuntu@10.42.0.27:/home/ubuntu/KriaKv260_Model_Compiler/notebooks/Datasets/imagenet_sample/
```

The `--exclude='.ipynb_checkpoints'` flag matters: Jupyter creates these
dot-folders inside `images/` if you preview any image, and they confuse
later syncs with permission-denied errors.

The dataset is ~8 MB so transfer is essentially instant.

### 4. Run the benchmark

The notebook reads from `Datasets/imagenet_sample/{images, labels.txt}`
automatically. No further configuration needed.

## Expected output

```
Datasets/imagenet_sample/
├── images/
│   ├── img_0000_carton.jpg
│   ├── img_0001_bighorn.jpg
│   ├── img_0002_african_hunting_dog.jpg
│   └── ... (497 more)
└── labels.txt    (500 lines, format "filename classname")
```

## Known issues

### Class-name mismatches

A small number of class names (~5 out of 1000 typically) don't match between
the Kaggle folder names and ImageNet's `imagenet_class_index.json`. Known
mismatches:

| Kaggle folder name | JSON name | Issue |
|---|---|---|
| `bicycle_built_for_two` | `bicycle-built-for-two` | hyphens vs underscores |
| `black_and_tan_coonhound` | `Black-and-tan_coonhound` | hyphens |
| `black_footed_ferret` | `black-footed_ferret` | hyphen |
| `cardigan_welsh_corgi` | `Cardigan` | truncated form |
| `carpenter_s_kit` | `carpenter's_kit` | apostrophe |

The notebook's `load_imagenet_dataset()` silently drops images whose class
names don't resolve. Currently ~8 of 500 images are dropped, giving
`n_acc_images=492` in the CSV (still 1.6% short of the requested 500).

This is a loader bug rather than a dataset-generation bug; the dataset
script preserves the names verbatim. See `scripts/host/notebook_patches/`
for the loader patch that normalises hyphens and apostrophes (currently
applied in the v0.7.4 notebook).

### Image compression

Many Kaggle ImageNet repackages aggressively recompress images to keep
archive size down (the `ifigotin` archive is ~16 KB per image vs ~120 KB for
ImageNet val originals). MobileNets and EfficientNet are noticeably
sensitive to this — their reported top-1 is 8–11 points below AMD's
published full-val numbers, while ResNet/Inception variants stay within
~5 points.

This is a feature of the benchmark, not a bug: the sample represents what
deployment accuracy looks like with realistic input quality. The numbers
are honest measurements on this sample; they just shouldn't be compared
directly to AMD's full-val numbers.

# KV260 Vitis AI 2.5 Benchmark Report (Final)

> **Historical baseline.** This is the v0.4-era benchmark run, performed
> on the VAI 2.5 runtime before the v0.7 upgrade to VAI 3.5. It's
> retained for two reasons:
>
> 1. **Cross-runtime reproducibility evidence.** Most models were
>    re-benchmarked on VAI 3.5 in v0.7.3 ([vai35_benchmark_report.md](./vai35_benchmark_report.md)).
>    Where the same model binary runs on both runtimes, the inference
>    outputs (mAP and latency) match to within measurement noise — the
>    DPU IP is unchanged between 2.5 and 3.5 even though the surrounding
>    libraries are different.
>
> 2. **Methodological contrast.** This 2.5 run used a smaller (N=200)
>    ImageNet sample with known data-leakage issues. The v0.7.3 work
>    rebuilt the methodology around ImageNetV2 matched-frequency (N=5000)
>    and identified the leakage artifact in this run's inception_v4
>    result.
>
> For the side-by-side comparison of 2.5 vs 3.5, see
> [vai25_vs_vai35_comparison.md](./vai25_vs_vai35_comparison.md).
>
> Numbers below are the original 2.5 measurements as recorded. They have
> not been adjusted retroactively.

---

Generated: 2026-05-03 00:10:47

## Test Environment

| Field | Value |
|---|---|
| Board | Kria KV260 |
| Stack | DPU-PYNQ 2.5 / Vitis AI 2.5 / DPUCZDX8G ISA1 B4096 @ 300 MHz |
| Camera | Logitech BRIO on /dev/video0, MJPG, BUFFERSIZE=4 |
| Cls dataset | ImageNet subset, 200 images for Top-1 |
| Det dataset | COCO val2017 (5000 images for mAP) |
| DPU latency iterations | 30 per model (after 5 warm-up) |
| Camera duration | 8 s per model |

## Summary

| Statistic | Value |
|---|---|
| Total models | 45 |
| Classification | 33 |
| Detection | 12 |
| Detection w/ mAP | 6 |
| Errors | 0 |

## Detection Accuracy (COCO mAP)

COCO val2017, N=5000 images, NMS IoU=0.45, score>=0.05, max 100 det/image.

| Model | Decoder | Status | mAP@0.5 | mAP@0.5:0.95 | n_imgs | Note |
|---|---|---|---:|---:|---:|---|
| ssd_mobilenet_v1_coco_tf | ssd_tf | OK | 0.3839 | 0.2422 | 5000 | -- |
| ssd_mobilenet_v2_coco_tf | ssd_tf | OK | 0.3832 | 0.2437 | 5000 | -- |
| ssdlite_mobilenet_v2_coco_tf | ssd_tf | OK | 0.3625 | 0.2329 | 5000 | -- |
| ssd_inception_v2_coco_tf | ssd_tf | OK | 0.4171 | 0.2788 | 5000 | -- |
| ssd_resnet_50_fpn_coco_tf | -- | SKIPPED | -- | -- | -- | FPN-specific anchor generator (5 levels, 51150 anchors) not implemented; may add as stretch goal |
| yolov3 | -- | SKIPPED | -- | -- | -- | xmodel not present in Models/detection/ |
| yolov3_coco_416_tf2 | yolo | OK | 0.6192 | 0.3178 | 5000 | -- |
| yolov4_leaky_spp_m | yolo | OK | 0.5967 | 0.1726 | 5000 | -- |
| yolov3_voc_tf | -- | SKIPPED | -- | -- | -- | trained on Pascal VOC, not COCO |
| refinedet_baseline | -- | SKIPPED | -- | -- | -- | RefineDet ARM+ODM decoder not implemented |
| refinedet_pruned_0_8 | -- | SKIPPED | -- | -- | -- | RefineDet ARM+ODM decoder not implemented |
| refinedet_pruned_0_92 | -- | SKIPPED | -- | -- | -- | RefineDet ARM+ODM decoder not implemented |
| refinedet_pruned_0_96 | -- | SKIPPED | -- | -- | -- | RefineDet ARM+ODM decoder not implemented |

**Decoder notes**

- `ssd_tf`: TF Object Detection API SSD with anchor variances [10,10,5,5] and softmax over 91 classes (class 0 = background, classes 1-90 mapped to COCO category IDs with the natural gaps).
- `yolo`: 3-scale YOLOv3/v4 decoder with sigmoid/exp on (tx,ty,tw,th), per-class NMS, and COCO-trained anchors at 416-input.

## Top Rankings

### Top 10 by Pure-DPU FPS

| Rank | Model | Cat | Input | DPU FPS | Latency mean (ms) | Power (W) |
|---|---|---|---|---:|---:|---:|
| 1 | mobilenet_v1_0_25_128_tf | class | 1x128x128x3 | 985.22 | 1.015 | 5.574 |
| 2 | mobilenet_v1_0_5_160_tf | class | 1x160x160x3 | 640.2 | 1.562 | 5.909 |
| 3 | squeezenet_pt | class | 1x224x224x3 | 336.7 | 2.97 | 6.361 |
| 4 | squeezenet | class | 1x227x227x3 | 325.52 | 3.072 | 6.385 |
| 5 | mobilenet_v3_small_1_0_tf2 | class | 1x224x224x3 | 258.46 | 3.869 | 5.858 |
| 6 | mobilenet_v1_1_0_224_tf | class | 1x224x224x3 | 249.07 | 4.015 | 6.746 |
| 7 | ofa_depthwise_res50_pt | class | 1x176x176x3 | 241.84 | 4.135 | 7.054 |
| 8 | mobilenet_v2 | class | 1x224x224x3 | 219.59 | 4.554 | 6.292 |
| 9 | mobilenet_v2_1_0_224_tf | class | 1x224x224x3 | 214.82 | 4.655 | 6.246 |
| 10 | mobilenet_edge_0_75_tf | class | 1x224x224x3 | 213.27 | 4.689 | 6.834 |

### Top 10 by Camera FPS (end-to-end with BRIO)

| Rank | Model | Cat | Input | Cam res | Cam FPS | Cam-bound | Power (W) |
|---|---|---|---|---|---:|:---:|---:|
| 1 | mobilenet_v1_0_25_128_tf | class | 1x128x128x3 | 640x480 | 60.01 | yes | 5.574 |
| 2 | mobilenet_v1_0_5_160_tf | class | 1x160x160x3 | 640x480 | 60.01 | yes | 5.909 |
| 3 | ofa_depthwise_res50_pt | class | 1x176x176x3 | 640x480 | 60.0 | yes | 7.054 |
| 4 | squeezenet_pt | class | 1x224x224x3 | 640x480 | 59.68 | yes | 6.361 |
| 5 | squeezenet | class | 1x227x227x3 | 640x480 | 58.13 | yes | 6.385 |
| 6 | mobilenet_v3_small_1_0_tf2 | class | 1x224x224x3 | 640x480 | 56.31 | no | 5.858 |
| 7 | mobilenet_v1_1_0_224_tf | class | 1x224x224x3 | 640x480 | 56.16 | no | 6.746 |
| 8 | mobilenet_v2 | class | 1x224x224x3 | 640x480 | 54.22 | no | 6.292 |
| 9 | mobilenet_v2_1_0_224_tf | class | 1x224x224x3 | 640x480 | 53.8 | no | 6.246 |
| 10 | mobilenet_edge_0_75_tf | class | 1x224x224x3 | 640x480 | 53.74 | no | 6.834 |

### Top 10 by FPS/W (camera-based efficiency)

| Rank | Model | Cat | Cam FPS | Power (W) | FPS/W |
|---|---|---|---:|---:|---:|
| 1 | mobilenet_v1_0_25_128_tf | class | 60.01 | 5.574 | 10.766 |
| 2 | mobilenet_v1_0_5_160_tf | class | 60.01 | 5.909 | 10.156 |
| 3 | mobilenet_v3_small_1_0_tf2 | class | 56.31 | 5.858 | 9.612 |
| 4 | squeezenet_pt | class | 59.68 | 6.361 | 9.382 |
| 5 | squeezenet | class | 58.13 | 6.385 | 9.104 |
| 6 | mobilenet_v2 | class | 54.22 | 6.292 | 8.617 |
| 7 | mobilenet_v2_1_0_224_tf | class | 53.8 | 6.246 | 8.614 |
| 8 | ofa_depthwise_res50_pt | class | 60.0 | 7.054 | 8.506 |
| 9 | mobilenet_v1_1_0_224_tf | class | 56.16 | 6.746 | 8.325 |
| 10 | mobilenet_edge_0_75_tf | class | 53.74 | 6.834 | 7.864 |

### Top 10 Classification by Top-1 Accuracy

> **Caveat**: this top-10 reflects a 200-image ImageNet sample that
> was later identified as containing data-leakage artifacts (the
> sample folders were training data rather than a proper held-out
> validation set). The inception_v4 result (0.92) was the most affected
> entry. The v0.7.3 re-evaluation on ImageNetV2 (N=5000, matched-
> frequency) reported 0.687 for inception_v4 — see
> [vai25_vs_vai35_comparison.md "The inception_v4 outlier"](./vai25_vs_vai35_comparison.md#the-inception_v4-outlier).

(Subset: 200 ImageNet images; +/- ~3% noise floor at this N)

| Rank | Model | Top-1 | DPU FPS | Cam FPS | Power (W) |
|---|---|---:|---:|---:|---:|
| 1 | inception_v4 | 0.92 | 28.84 | 18.63 | 8.725 |
| 2 | resnet50 | 0.91 | 85.31 | 39.11 | 8.177 |
| 3 | inception_v3 | 0.89 | 56.28 | 26.84 | 8.798 |
| 4 | inception_v1 | 0.85 | 160.93 | 49.0 | 7.688 |
| 5 | inception_resnet_v2_tf | 0.83 | 24.3 | 16.54 | 8.55 |
| 6 | inception_v3_pt | 0.825 | 56.3 | 26.75 | 8.821 |
| 7 | inception_v3_tf | 0.825 | 56.08 | 27.24 | 8.764 |
| 8 | inception_v3_tf2 | 0.82 | 55.75 | 27.04 | 8.68 |
| 9 | mobilenet_v2 | 0.805 | 219.59 | 54.22 | 6.292 |
| 10 | resnet_v2_50_tf | 0.805 | 43.88 | 23.95 | 8.798 |

### Top by COCO mAP@0.5

| Rank | Model | mAP@0.5 | mAP@0.5:0.95 | DPU FPS | Cam FPS | Power (W) |
|---|---|---:|---:|---:|---:|---:|
| 1 | yolov3_coco_416_tf2 | 0.6192 | 0.3178 | 12.16 | 9.26 | 8.05 |
| 2 | yolov4_leaky_spp_m | 0.5967 | 0.1726 | 12.69 | 9.64 | 7.836 |
| 3 | ssd_inception_v2_coco_tf | 0.4171 | 0.2788 | 38.57 | 24.3 | 8.664 |
| 4 | ssd_mobilenet_v1_coco_tf | 0.3839 | 0.2422 | 94.67 | 33.85 | 6.885 |
| 5 | ssd_mobilenet_v2_coco_tf | 0.3832 | 0.2437 | 73.71 | 30.14 | 6.934 |
| 6 | ssdlite_mobilenet_v2_coco_tf | 0.3625 | 0.2329 | 91.98 | 33.0 | 6.365 |

## Detailed Results -- Classification

| Model | Input | DPU lat mean / p50 / p99 (ms) | DPU FPS | Pwr idle / load (W) | Top-1 | Cam res @ tgt | Cam FPS | cap / pre / inf (ms) | Bound | FPS/W |
|---|---|---|---:|---|---:|---|---:|---|:---:|---:|
| mobilenet_v1_0_25_128_tf | 1x128x128x3 | 1.015 / 1.006 / 1.09 | 985.22 | 5.25 / 5.574 | 0.265 | 640x480 @ 60 | 60.01 | 12.53 / 2.8 / 1.32 | cam | 10.766 |
| mobilenet_v1_0_5_160_tf | 1x160x160x3 | 1.562 / 1.557 / 1.634 | 640.2 | 5.255 / 5.909 | 0.4 | 640x480 @ 60 | 60.01 | 10.65 / 4.1 / 1.9 | cam | 10.156 |
| squeezenet_pt | 1x224x224x3 | 2.97 / 2.964 / 3.044 | 336.7 | 5.24 / 6.361 | 0.505 | 640x480 @ 60 | 59.68 | 5.82 / 7.66 / 3.26 | cam | 9.382 |
| squeezenet | 1x227x227x3 | 3.072 / 3.073 / 3.128 | 325.52 | 5.415 / 6.385 | 0.635 | 640x480 @ 60 | 58.13 | 5.91 / 7.89 / 3.38 | cam | 9.104 |
| mobilenet_v3_small_1_0_tf2 | 1x224x224x3 | 3.869 / 3.861 / 3.92 | 258.46 | 5.305 / 5.858 | 0.595 | 640x480 @ 60 | 56.31 | 5.95 / 7.61 / 4.18 | model | 9.612 |
| mobilenet_v1_1_0_224_tf | 1x224x224x3 | 4.015 / 4.005 / 4.1 | 249.07 | 5.28 / 6.746 | 0.64 | 640x480 @ 60 | 56.16 | 5.87 / 7.61 / 4.31 | model | 8.325 |
| ofa_depthwise_res50_pt | 1x176x176x3 | 4.135 / 4.118 / 4.228 | 241.84 | 5.22 / 7.054 | 0.745 | 640x480 @ 60 | 60.0 | 7.23 / 4.93 / 4.48 | cam | 8.506 |
| mobilenet_v2 | 1x224x224x3 | 4.554 / 4.546 / 4.637 | 219.59 | 5.245 / 6.292 | 0.805 | 640x480 @ 60 | 54.22 | 5.92 / 7.66 / 4.84 | model | 8.617 |
| mobilenet_v2_1_0_224_tf | 1x224x224x3 | 4.655 / 4.645 / 4.76 | 214.82 | 5.25 / 6.246 | 0.63 | 640x480 @ 60 | 53.8 | 5.99 / 7.65 / 4.93 | model | 8.614 |
| mobilenet_edge_0_75_tf | 1x224x224x3 | 4.689 / 4.679 / 4.771 | 213.27 | 5.315 / 6.834 | 0.565 | 640x480 @ 60 | 53.74 | 5.96 / 7.63 / 5.0 | model | 7.864 |
| mobilenet_edge_1_0_tf | 1x224x224x3 | 5.513 / 5.496 / 5.693 | 181.39 | 5.32 / 7.315 | 0.61 | 640x480 @ 60 | 51.65 | 5.87 / 7.68 / 5.79 | model | 7.061 |
| mobilenet_v2_1_4_224_tf | 1x224x224x3 | 6.037 / 6.013 / 6.203 | 165.65 | 5.33 / 6.699 | 0.645 | 640x480 @ 60 | 50.24 | 5.93 / 7.63 / 6.33 | model | 7.5 |
| inception_v1_tf | 1x224x224x3 | 6.077 / 6.067 / 6.196 | 164.55 | 5.28 / 7.622 | 0.715 | 640x480 @ 60 | 49.11 | 6.21 / 7.75 / 6.38 | model | 6.443 |
| inception_v1 | 1x224x224x3 | 6.214 / 6.179 / 6.504 | 160.93 | 5.295 / 7.688 | 0.85 | 640x480 @ 60 | 49.0 | 6.16 / 7.73 / 6.5 | model | 6.374 |
| inception_v2 | 1x224x224x3 | 8.096 / 8.087 / 8.195 | 123.52 | 5.25 / 7.755 | 0.675 | 640x480 @ 60 | 44.22 | 6.28 / 7.88 / 8.43 | model | 5.702 |
| resnet50 | 1x224x224x3 | 11.722 / 11.712 / 11.851 | 85.31 | 5.26 / 8.177 | 0.91 | 640x480 @ 60 | 39.11 | 5.89 / 7.66 / 12.0 | model | 4.783 |
| resnet_v1_50_tf | 1x224x224x3 | 11.762 / 11.754 / 11.844 | 85.02 | 5.33 / 8.137 | 0.705 | 640x480 @ 60 | 38.96 | 5.93 / 7.67 / 12.05 | model | 4.788 |
| resnet50_pt | 1x224x224x3 | 13.033 / 13.024 / 13.105 | 76.73 | 5.3 / 8.443 | 0.72 | 640x480 @ 60 | 36.54 | 6.11 / 7.86 / 13.39 | model | 4.328 |
| efficientnet-b0_tf2 | 1x224x224x3 | 13.066 / 13.07 / 13.15 | 76.53 | 5.24 / 6.668 | 0.655 | 640x480 @ 60 | 37.16 | 5.9 / 7.66 / 13.34 | model | 5.573 |
| inception_v3_pt | 1x299x299x3 | 17.763 / 17.753 / 17.847 | 56.3 | 5.335 / 8.821 | 0.825 | 640x480 @ 60 | 26.75 | 6.12 / 13.17 / 18.08 | model | 3.033 |
| inception_v3 | 1x299x299x3 | 17.768 / 17.779 / 17.852 | 56.28 | 5.38 / 8.798 | 0.89 | 640x480 @ 60 | 26.84 | 6.27 / 12.94 / 18.02 | model | 3.051 |
| inception_v3_tf | 1x299x299x3 | 17.833 / 17.822 / 18.033 | 56.08 | 5.37 / 8.764 | 0.825 | 640x480 @ 60 | 27.24 | 6.01 / 12.67 / 18.02 | model | 3.108 |
| inception_v3_tf2 | 1x299x299x3 | 17.937 / 17.93 / 18.034 | 55.75 | 5.295 / 8.68 | 0.82 | 640x480 @ 60 | 27.04 | 6.17 / 12.66 / 18.14 | model | 3.115 |
| resnet_v1_101_tf | 1x224x224x3 | 21.291 / 21.291 / 21.368 | 46.97 | 5.23 / 8.646 | 0.735 | 640x480 @ 60 | 28.52 | 5.85 / 7.66 / 21.54 | model | 3.299 |
| resnet_v2_50_tf | 1x299x299x3 | 22.791 / 22.792 / 22.863 | 43.88 | 5.325 / 8.798 | 0.805 | 640x480 @ 60 | 23.95 | 5.94 / 12.83 / 22.97 | model | 2.722 |
| resnet_v1_152_tf | 1x224x224x3 | 30.663 / 30.659 / 30.726 | 32.61 | 5.335 / 8.606 | 0.69 | 640x480 @ 60 | 22.23 | 6.08 / 7.89 / 31.0 | model | 2.583 |
| inception_v4_2016_09_09_tf | 1x299x299x3 | 34.605 / 34.589 / 34.824 | 28.9 | 5.33 / 8.739 | 0.8 | 640x480 @ 60 | 18.64 | 6.03 / 12.81 / 34.78 | model | 2.133 |
| inception_v4 | 1x299x299x3 | 34.677 / 34.67 / 34.784 | 28.84 | 5.32 / 8.725 | 0.92 | 640x480 @ 60 | 18.63 | 6.14 / 12.7 / 34.83 | model | 2.135 |
| inception_resnet_v2_tf | 1x299x299x3 | 41.154 / 41.148 / 41.349 | 24.3 | 5.38 / 8.55 | 0.83 | 640x480 @ 60 | 16.54 | 6.18 / 12.9 / 41.36 | model | 1.935 |
| resnet_v2_101_tf | 1x299x299x3 | 41.631 / 41.628 / 41.811 | 24.02 | 5.325 / 8.679 | 0.795 | 640x480 @ 60 | 16.54 | 5.96 / 12.69 / 41.8 | model | 1.906 |
| vgg_16_tf | 1x224x224x3 | 47.688 / 47.686 / 47.799 | 20.97 | 5.24 / 7.138 | 0.645 | 640x480 @ 60 | 16.23 | 5.98 / 7.65 / 47.95 | model | 2.274 |
| vgg_19_tf | 1x224x224x3 | 55.082 / 55.08 / 55.14 | 18.15 | 5.24 / 7.188 | 0.7 | 640x480 @ 60 | 14.49 | 6.01 / 7.66 / 55.35 | model | 2.016 |
| resnet_v2_152_tf | 1x299x299x3 | 60.15 / 60.155 / 60.255 | 16.63 | 5.33 / 8.694 | 0.755 | 640x480 @ 60 | 12.53 | 6.15 / 13.22 / 60.43 | model | 1.441 |

## Detailed Results -- Detection

| Model | Input | DPU lat mean / p50 / p99 (ms) | DPU FPS | Pwr idle / load (W) | mAP@0.5 / @0.5:0.95 | Cam res @ tgt | Cam FPS | cap / pre / inf (ms) | Bound | FPS/W |
|---|---|---|---|---|---:|---|---:|---|:---:|---:|
| ssd_mobilenet_v1_coco_tf | 1x300x300x3 | 10.563 / 10.569 / 10.624 | 94.67 | 5.33 / 6.885 | 0.3839 / 0.2422 | 640x480 @ 60 | 33.85 | 6.02 / 12.75 / 10.74 | model | 4.916 |
| ssdlite_mobilenet_v2_coco_tf | 1x300x300x3 | 10.872 / 10.863 / 10.974 | 91.98 | 5.335 / 6.365 | 0.3625 / 0.2329 | 640x480 @ 60 | 33.0 | 6.17 / 12.96 / 11.15 | model | 5.185 |
| ssd_mobilenet_v2_coco_tf | 1x300x300x3 | 13.567 / 13.575 / 13.652 | 73.71 | 5.36 / 6.934 | 0.3832 / 0.2437 | 640x480 @ 60 | 30.14 | 6.12 / 13.16 / 13.89 | model | 4.347 |
| refinedet_pruned_0_96 | 1x360x480x3 | 17.911 / 17.901 / 17.98 | 55.83 | 5.315 / 7.992 | - / - | 640x480 @ 60 | 23.17 | 5.97 / 19.11 / 18.06 | model | 2.899 |
| refinedet_pruned_0_92 | 1x360x480x3 | 22.69 / 22.684 / 22.902 | 44.07 | 5.315 / 8.81 | - / - | 640x480 @ 60 | 20.68 | 6.07 / 19.24 / 23.01 | model | 2.347 |
| ssd_inception_v2_coco_tf | 1x300x300x3 | 25.926 / 25.925 / 25.998 | 38.57 | 5.33 / 8.664 | 0.4171 / 0.2788 | 640x480 @ 60 | 24.3 | 5.98 / 9.06 / 26.09 | model | 2.805 |
| refinedet_pruned_0_8 | 1x360x480x3 | 40.223 / 40.215 / 40.318 | 24.86 | 5.325 / 7.603 | - / - | 640x480 @ 60 | 15.32 | 6.06 / 18.84 / 40.35 | model | 2.015 |
| yolov3_voc_tf | 1x416x416x3 | 75.596 / 75.597 / 75.711 | 13.23 | 5.31 / 10.056 | - / - | 640x480 @ 60 | 9.82 | 6.37 / 19.46 / 75.93 | model | 0.977 |
| yolov4_leaky_spp_m | 1x416x416x3 | 78.791 / 78.782 / 79.006 | 12.69 | 5.315 / 7.836 | 0.5967 / 0.1726 | 640x480 @ 60 | 9.64 | 6.11 / 18.67 / 78.92 | model | 1.23 |
| yolov3_coco_416_tf2 | 1x416x416x3 | 82.225 / 82.214 / 82.34 | 12.16 | 5.31 / 8.05 | 0.6192 / 0.3178 | 640x480 @ 60 | 9.26 | 6.3 / 19.12 / 82.59 | model | 1.15 |
| refinedet_baseline | 1x360x480x3 | 136.087 / 136.061 / 136.256 | 7.35 | 5.32 / 5.472 | - / - | 640x480 @ 60 | 6.21 | 5.95 / 18.95 / 136.16 | model | 1.135 |
| ssd_resnet_50_fpn_coco_tf | 1x640x640x3 | 221.395 / 221.374 / 221.728 | 4.52 | 5.35 / 5.446 | - / - | 1280x720 @ 30 | 3.73 | 18.12 / 27.08 / 223.16 | model | 0.685 |

## Methodology

Each model is measured on five criteria plus, for detection, an mAP value.

1. **Power** -- board-total from `/sys/class/hwmon/`. Idle = avg over 0.3 s before warm-up; Load = avg across DPU iterations.
2. **Latency (pure-DPU)** -- 30 iterations on a static in-memory image. Mean / p50 / p99.
3. **Accuracy** -- ImageNet Top-1 for classification. COCO mAP for detection (where decoder is implemented).
4. **FPS** -- pure-DPU FPS = 1000 / mean DPU latency; camera FPS = end-to-end with BRIO + resize + DPU. Per-stage breakdown reported.
5. **FPS/W** -- camera FPS / load power.

**mAP details**

* `mAP@0.5`: standard COCO IoU threshold of 0.5.
* `mAP@0.5:0.95`: COCO primary metric, averaged over IoU thresholds [0.5, 0.55, ..., 0.95].
* Pure NumPy implementation; 101-point AP interpolation.
* RefineDet (4 models) and YOLOv3-VOC are reported as SKIPPED with reason.

---

## Subsequent work

The v0.7.3 release re-evaluated this benchmark suite on Vitis AI 3.5 with:
- ImageNetV2 matched-frequency (N=5000) for classification accuracy
- Top-5 accuracy in addition to Top-1
- VOC2007 test set for VOC-trained detection models
- Six decoder bugfixes identified during the 3.5 re-run (channel order,
  SSD anchor count, class remapping, 3D-input handling, bbox format,
  early-return signature)

See [vai35_benchmark_report.md](./vai35_benchmark_report.md) for the
3.5 results and [v0.7.3-patchnote.md](./v0.7.3-patchnote.md) for the
decoder-bug forensics. The side-by-side comparison is in
[vai25_vs_vai35_comparison.md](./vai25_vs_vai35_comparison.md).

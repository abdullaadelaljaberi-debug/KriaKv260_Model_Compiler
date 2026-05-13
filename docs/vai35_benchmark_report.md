# KV260 Vitis AI 3.5 Benchmark Report

Generated: 2026-05-13 12:27:04

## Test Environment

| Field | Value |
|---|---|
| Board | Kria KV260 |
| Stack | DPU-PYNQ design_contest_3.5 / Vitis AI 3.5 / DPUCZDX8G_ISA1_B4096 |
| DPU fingerprint | 0x101000056010407 (B4096 @ 300 MHz) |
| Camera | Logitech BRIO on /dev/video0, MJPG, BUFFERSIZE=4 |
| Cls dataset | ImageNet sample, up to 5000 images for Top-1 |
| Det dataset (COCO) | COCO val2017 (5000 images for mAP) |
| Det dataset (VOC) | VOC2007 test (4952 images for mAP) |
| DPU latency iterations | 20 per model (after 5 warm-up) |
| Camera duration | 8 s per model |

**Note on model versions**: pre-compiled xmodels were downloaded from
AMD's VAI 3.0 model zoo (KV260 binaries). The DPU IP (B4096) is
unchanged between VAI 3.0 and VAI 3.5, so these binaries run correctly
on the VAI 3.5 runtime per [AMD's compatibility note](https://wiki.trenz-electronic.de/display/PD/Compilation+of+AI+3.0+models+for+Vitis+2023.2,+AI+3.5+SW,+AI+3.0+DPUCZDX8G).

## Summary

| Statistic | Value |
|---|---|
| Catalogue total | 43 |
| Enabled in catalogue | 36 |
| Smoke test passed | 31 |
| Smoke test failed | 5 |
| Fully benchmarked | 33 |
| Classification | 21 |
| Detection | 12 |
| Detection w/ COCO mAP | 6 |
| Detection w/ VOC mAP | 1 |
| Errors in main loop | 0 |

## Smoke Test Failures

These models failed to load or produce inference output. They were
excluded from the full benchmark. Common causes: corrupt download,
incompatible xmodel format, fingerprint mismatch.

| Model | xmodel found | Load OK | Infer OK | Error |
|---|:-:|:-:|:-:|---|
| mobilenetv2_pt | False | False | False | `xmodel not found in Models_VAI35/` |
| efficientnet_edgetpu-S_tf | False | False | False | `xmodel not found in Models_VAI35/` |
| efficientnet_edgetpu-M_tf | False | False | False | `xmodel not found in Models_VAI35/` |
| efficientnet_edgetpu-L_tf | False | False | False | `xmodel not found in Models_VAI35/` |
| mobilenet_edge_2_75_pt | False | False | False | `xmodel not found in Models_VAI35/` |

## Detection Accuracy

### COCO val2017

N=5000 images, NMS IoU=0.45, score>=0.05, max 100 det/image.

| Model | Decoder | Status | mAP@0.5 | mAP@0.5:0.95 | n_imgs |
|---|---|---|---:|---:|---:|
| ssd_mobilenet_v1_coco_tf | ssd_tf | OK | 0.3839 | 0.2422 | 5000 |
| ssd_mobilenet_v2_coco_tf | ssd_tf | OK | 0.3834 | 0.2438 | 5000 |
| ssd_inception_v2_coco_tf | ssd_tf | OK | 0.4172 | 0.2789 | 5000 |
| yolov3_coco_416_tf2 | yolo | OK | 0.617 | 0.316 | 5000 |
| yolov4_leaky_spp_m | yolo | OK | 0.6161 | 0.1794 | 5000 |
| ssdlite_mobilenet_v2_coco_tf | ssd_tf | OK | 0.3626 | 0.233 | 5000 |

### VOC2007 test

N=4952 images, NMS IoU=0.45, score>=0.05, max 100 det/image.

| Model | Decoder | Status | mAP@0.5 | mAP@0.5:0.95 | n_imgs |
|---|---|---|---:|---:|---:|
| yolov3_voc_tf | yolo_voc | OK | 0.7701 | 0.4154 | 4952 |
| face_mask_detection_pt |  | SKIPPED | -- | -- | -- |

## Top Rankings

### Top 10 by Pure-DPU FPS

| Rank | Model | Cat | Input | DPU FPS | Latency mean (ms) | Power (W) |
|---|---|---|---|---:|---:|---:|
| 1 | mobilenet_v1_0_25_128_tf | class | 1x128x128x3 | 980.39 | 1.02 | 5.165 |
| 2 | squeezenet_pt | class | 1x224x224x3 | 329.49 | 3.035 | 6.19 |
| 3 | mobilenet_v1_1_0_224_tf | class | 1x224x224x3 | 249.94 | 4.001 | 6.603 |
| 4 | mobilenet_v2_1_0_224_tf | class | 1x224x224x3 | 216.03 | 4.629 | 5.869 |
| 5 | densebox_320_320 | detec | 1x320x320x3 | 208.77 | 4.79 | 5.512 |
| 6 | ofa_resnet50_0_9B_pt | class | 1x160x160x3 | 179.57 | 5.569 | 7.295 |
| 7 | inception_v1_tf | class | 1x224x224x3 | 165.43 | 6.045 | 7.366 |
| 8 | mobilenet_v2_1_4_224_tf | class | 1x224x224x3 | 165.43 | 6.045 | 6.377 |
| 9 | densebox_640_360 | detec | 1x360x640x3 | 97.07 | 10.302 | 5.611 |
| 10 | ssd_mobilenet_v1_coco_tf | detec | 1x300x300x3 | 94.98 | 10.528 | 6.603 |

### Top 10 by Camera FPS

| Rank | Model | Cat | Input | Cam res | Cam FPS | Cam-bound | Power (W) |
|---|---|---|---|---|---:|:---:|---:|
| 1 | mobilenet_v1_0_25_128_tf | class | 1x128x128x3 | 640x480 | 60.01 | yes | 5.165 |
| 2 | ofa_resnet50_0_9B_pt | class | 1x160x160x3 | 640x480 | 58.93 | yes | 7.295 |
| 3 | squeezenet_pt | class | 1x224x224x3 | 640x480 | 55.59 | no | 6.19 |
| 4 | mobilenet_v1_1_0_224_tf | class | 1x224x224x3 | 640x480 | 52.67 | no | 6.603 |
| 5 | mobilenet_v2_1_0_224_tf | class | 1x224x224x3 | 640x480 | 50.9 | no | 5.869 |
| 6 | inception_v1_tf | class | 1x224x224x3 | 640x480 | 47.65 | no | 7.366 |
| 7 | mobilenet_v2_1_4_224_tf | class | 1x224x224x3 | 640x480 | 47.39 | no | 6.377 |
| 8 | densebox_320_320 | detec | 1x320x320x3 | 640x480 | 41.63 | no | 5.512 |
| 9 | resnet50_pruned_0_4_pt | class | 1x224x224x3 | 640x480 | 38.89 | no | 7.579 |
| 10 | inception_v2_tf | class | 1x224x224x3 | 640x480 | 38.18 | no | 6.638 |

### Top 10 by FPS/W

| Rank | Model | Cat | Cam FPS | Power (W) | FPS/W |
|---|---|---|---:|---:|---:|
| 1 | mobilenet_v1_0_25_128_tf | class | 60.01 | 5.165 | 11.619 |
| 2 | squeezenet_pt | class | 55.59 | 6.19 | 8.981 |
| 3 | mobilenet_v2_1_0_224_tf | class | 50.9 | 5.869 | 8.673 |
| 4 | ofa_resnet50_0_9B_pt | class | 58.93 | 7.295 | 8.078 |
| 5 | mobilenet_v1_1_0_224_tf | class | 52.67 | 6.603 | 7.977 |
| 6 | densebox_320_320 | detec | 41.63 | 5.512 | 7.553 |
| 7 | mobilenet_v2_1_4_224_tf | class | 47.39 | 6.377 | 7.431 |
| 8 | inception_v1_tf | class | 47.65 | 7.366 | 6.469 |
| 9 | inception_v2_tf | class | 38.18 | 6.638 | 5.752 |
| 10 | efficientnet-b0_tf2 | class | 35.53 | 6.286 | 5.652 |

### Top 10 Classification by Top-1 Accuracy

(Sample: up to 5000 ImageNet images; +/- ~3% noise floor)

| Rank | Model | Top-1 | DPU FPS | Cam FPS | Power (W) |
|---|---|---:|---:|---:|---:|
| 1 | resnet50 | 0.8984 | 86.6 | 37.97 | 7.937 |
| 2 | inception_resnet_v2_tf | 0.8496 | 24.28 | 16.25 | 8.322 |
| 3 | inception_v4_2016_09_09_tf | 0.8435 | 28.97 | 18.3 | 8.595 |
| 4 | inception_v3_tf | 0.8272 | 56.77 | 25.92 | 8.553 |
| 5 | inception_v2_tf | 0.7642 | 89.23 | 38.18 | 6.638 |
| 6 | resnet_v1_152_tf | 0.748 | 32.87 | 21.8 | 8.49 |
| 7 | inception_v1_tf | 0.7154 | 165.43 | 47.65 | 7.366 |
| 8 | resnet_v1_50_tf | 0.7134 | 86.66 | 37.64 | 7.926 |
| 9 | resnet50_pt | 0.7134 | 77.5 | 35.57 | 8.213 |
| 10 | resnet_v1_101_tf | 0.7114 | 47.44 | 27.83 | 8.477 |

## Detailed Results -- Classification

| Model | Input | DPU lat mean / p50 / p99 (ms) | DPU FPS | Pwr idle / load (W) | Top-1 | Cam res @ tgt | Cam FPS | cap / pre / inf (ms) | Bound | FPS/W |
|---|---|---|---:|---|---:|---|---:|---|:---:|---:|
| mobilenet_v1_0_25_128_tf | 1x128x128x3 | 1.02 / 1.018 / 1.044 | 980.39 | 4.925 / 5.165 | 0.2459 | 640x480 @ 60 | 60.01 | 12.32 / 2.99 / 1.35 | cam | 11.619 |
| squeezenet_pt | 1x224x224x3 | 3.035 / 3.047 / 3.133 | 329.49 | 4.975 / 6.19 | 0.4736 | 640x480 @ 60 | 55.59 | 6.38 / 8.29 / 3.3 | model | 8.981 |
| mobilenet_v1_1_0_224_tf | 1x224x224x3 | 4.001 / 3.993 / 4.076 | 249.94 | 4.925 / 6.603 | 0.6159 | 640x480 @ 60 | 52.67 | 6.38 / 8.27 / 4.32 | model | 7.977 |
| mobilenet_v2_1_0_224_tf | 1x224x224x3 | 4.629 / 4.623 / 4.707 | 216.03 | 5.075 / 5.869 | 0.6057 | 640x480 @ 60 | 50.9 | 6.39 / 8.31 / 4.93 | model | 8.673 |
| ofa_resnet50_0_9B_pt | 1x160x160x3 | 5.569 / 5.561 / 5.631 | 179.57 | 5.025 / 7.295 | 0.6504 | 640x480 @ 60 | 58.93 | 6.49 / 4.53 / 5.94 | cam | 8.078 |
| inception_v1_tf | 1x224x224x3 | 6.045 / 6.037 / 6.107 | 165.43 | 5.045 / 7.366 | 0.7154 | 640x480 @ 60 | 47.65 | 6.43 / 8.18 / 6.35 | model | 6.469 |
| mobilenet_v2_1_4_224_tf | 1x224x224x3 | 6.045 / 6.04 / 6.093 | 165.43 | 5.095 / 6.377 | 0.6463 | 640x480 @ 60 | 47.39 | 6.44 / 8.28 / 6.37 | model | 7.431 |
| resnet50_pruned_0_4_pt | 1x224x224x3 | 10.714 / 10.716 / 10.794 | 93.34 | 5.045 / 7.579 | 0.6748 | 640x480 @ 60 | 38.89 | 6.45 / 8.27 / 10.99 | model | 5.131 |
| inception_v2_tf | 1x224x224x3 | 11.207 / 11.199 / 11.299 | 89.23 | 5.085 / 6.638 | 0.7642 | 640x480 @ 60 | 38.18 | 6.44 / 8.22 / 11.52 | model | 5.752 |
| resnet_v1_50_tf | 1x224x224x3 | 11.539 / 11.51 / 11.667 | 86.66 | 4.99 / 7.926 | 0.7134 | 640x480 @ 60 | 37.64 | 6.42 / 8.26 / 11.88 | model | 4.749 |
| resnet50 | 1x224x224x3 | 11.547 / 11.542 / 11.648 | 86.6 | 4.865 / 7.937 | 0.8984 | 640x480 @ 60 | 37.97 | 6.32 / 8.18 / 11.82 | model | 4.784 |
| resnet50_pt | 1x224x224x3 | 12.904 / 12.907 / 12.963 | 77.5 | 5.04 / 8.213 | 0.7134 | 640x480 @ 60 | 35.57 | 6.45 / 8.41 / 13.24 | model | 4.331 |
| efficientnet-b0_tf2 | 1x224x224x3 | 13.085 / 13.071 / 13.198 | 76.42 | 4.98 / 6.286 | 0.6585 | 640x480 @ 60 | 35.53 | 6.44 / 8.26 / 13.43 | model | 5.652 |
| inception_v3_tf | 1x299x299x3 | 17.614 / 17.616 / 17.696 | 56.77 | 4.97 / 8.553 | 0.8272 | 640x480 @ 60 | 25.92 | 6.56 / 14.09 / 17.92 | model | 3.031 |
| resnet_v1_101_tf | 1x224x224x3 | 21.081 / 21.091 / 21.148 | 47.44 | 5.01 / 8.477 | 0.7114 | 640x480 @ 60 | 27.83 | 6.36 / 8.22 / 21.33 | model | 3.283 |
| resnet_v1_152_tf | 1x224x224x3 | 30.424 / 30.428 / 30.518 | 32.87 | 4.99 / 8.49 | 0.748 | 640x480 @ 60 | 21.8 | 6.55 / 8.54 / 30.77 | model | 2.568 |
| inception_v4 | 1x299x299x3 | 34.507 / 34.516 / 34.583 | 28.98 | 5.45 / 9.061 | 0.687 | - @ - | - | - / - / - | - | - |
| inception_v4_2016_09_09_tf | 1x299x299x3 | 34.52 / 34.514 / 34.585 | 28.97 | 4.945 / 8.595 | 0.8435 | 640x480 @ 60 | 18.3 | 6.37 / 13.56 / 34.69 | model | 2.129 |
| inception_resnet_v2_tf | 1x299x299x3 | 41.19 / 41.164 / 41.461 | 24.28 | 5.03 / 8.322 | 0.8496 | 640x480 @ 60 | 16.25 | 6.49 / 13.67 / 41.37 | model | 1.953 |
| vgg_16_tf | 1x224x224x3 | 47.039 / 47.023 / 47.155 | 21.26 | 5.035 / 6.749 | 0.6484 | 640x480 @ 60 | 16.15 | 6.37 / 8.23 / 47.31 | model | 2.393 |
| vgg_19_tf | 1x224x224x3 | 54.473 / 54.451 / 54.939 | 18.36 | 5.08 / 6.775 | 0.689 | 640x480 @ 60 | 14.43 | 6.38 / 8.22 / 54.68 | model | 2.13 |

## Detailed Results -- Detection

| Model | Input | DPU lat mean / p50 / p99 (ms) | DPU FPS | Pwr idle / load (W) | mAP@0.5 / @0.5:0.95 | Cam res @ tgt | Cam FPS | cap / pre / inf (ms) | Bound | FPS/W |
|---|---|---|---:|---|---|---|---:|---|:---:|---:|
| densebox_320_320 | 1x320x320x3 | 4.79 / 4.791 / 4.86 | 208.77 | 5.065 / 5.512 | - / - | 640x480 @ 60 | 41.63 | 6.4 / 12.65 / 4.96 | model | 7.553 |
| densebox_640_360 | 1x360x640x3 | 10.302 / 10.284 / 10.419 | 97.07 | 4.94 / 5.611 | - / - | 640x480 @ 60 | 23.96 | 6.35 / 24.94 / 10.45 | model | 4.27 |
| ssd_mobilenet_v1_coco_tf | 1x300x300x3 | 10.528 / 10.53 / 10.591 | 94.98 | 4.95 / 6.603 | 0.3839 / 0.2422 | 640x480 @ 60 | 31.61 | 6.62 / 14.12 / 10.87 | model | 4.787 |
| ssdlite_mobilenet_v2_coco_tf | 1x300x300x3 | 10.858 / 10.86 / 10.906 | 92.1 | 5.41 / 6.455 | 0.3626 / 0.233 | - @ - | - | - / - / - | - | - |
| ssd_mobilenet_v2_coco_tf | 1x300x300x3 | 13.572 / 13.562 / 13.643 | 73.68 | 4.995 / 6.572 | 0.3834 / 0.2438 | 640x480 @ 60 | 29.21 | 6.47 / 13.91 / 13.84 | model | 4.445 |
| face_mask_detection_pt | 1x512x512x3 | 14.472 / 14.454 / 14.603 | 69.1 | 5.0 / 5.438 | - / - | 1280x720 @ 30 | 16.14 | 18.51 / 28.83 / 14.62 | model | 2.968 |
| ssd_inception_v2_coco_tf | 1x300x300x3 | 25.727 / 25.723 / 25.859 | 38.87 | 4.96 / 8.484 | 0.4172 / 0.2789 | 640x480 @ 60 | 21.7 | 6.45 / 13.73 / 25.89 | model | 2.558 |
| yolov3 | 1x416x416x3 | 75.428 / 75.424 / 75.526 | 13.26 | 4.935 / 9.905 | - / - | 640x480 @ 60 | 9.88 | 6.37 / 19.29 / 75.53 | model | 0.997 |
| yolov3_voc_tf | 1x416x416x3 | 75.412 / 75.414 / 75.485 | 13.26 | 4.93 / 9.97 | 0.7701 / 0.4154 | 640x480 @ 60 | 9.9 | 6.37 / 19.09 / 75.52 | model | 0.993 |
| yolov4_leaky_spp_m | 1x416x416x3 | 78.87 / 78.859 / 79.017 | 12.68 | 4.945 / 7.637 | 0.6161 / 0.1794 | 640x480 @ 60 | 9.57 | 6.39 / 19.12 / 78.98 | model | 1.253 |
| yolov3_coco_416_tf2 | 1x416x416x3 | 82.305 / 82.306 / 82.368 | 12.15 | 4.955 / 7.5 | 0.617 / 0.316 | 640x480 @ 60 | 9.25 | 6.47 / 19.28 / 82.34 | model | 1.233 |
| ssd_resnet_50_fpn_coco_tf | 1x640x640x3 | 221.303 / 221.284 / 221.501 | 4.52 | 5.015 / 5.063 | - / - | 1280x720 @ 30 | 3.51 | 18.49 / 44.95 / 221.28 | model | 0.693 |

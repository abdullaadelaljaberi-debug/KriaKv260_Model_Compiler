"""Deploy pipeline — runs on the Kria board.

Filled in during Passes 5-6:
    runner.py       ModelRunner (yolov5 + yolox; raises on others)
    decoders.py     DFL + YOLOX decoders; stubs for v7/v4/SSD
    camera.py       ThreadedCamera (Brio); AR1335 hook commented
    eval.py         Batch inference + mAP@0.5 calculation
    tuning.py       v4l2 + USB + CPU tuning helpers

For now this subpackage exists only so importers don't fail when
lpr_pipeline is on PYTHONPATH.
"""

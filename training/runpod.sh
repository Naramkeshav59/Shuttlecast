#!/usr/bin/env bash
# Fine-tune + evaluate on a RunPod GPU pod (PyTorch template, any 16GB+ card).
#
# Getting the code and data onto the pod (from your machine):
#   tar czf shuttlecast_training.tgz shared worker training data/training
#   runpodctl send shuttlecast_training.tgz        # prints a one-time code
# On the pod:
#   runpodctl receive <code> && tar xzf shuttlecast_training.tgz
#   bash training/runpod.sh
# Then bring results home:
#   runpodctl send checkpoints/qwen2vl-shuttlecast-qlora/adapter eval_*.json
set -euo pipefail
cd "$(dirname "$0")/.."

# Newer RunPod images (Ubuntu 24.04) refuse system-wide pip installs
# (PEP 668). A venv that can still see the image's CUDA torch avoids both
# that and a 2 GB torch reinstall.
python -m venv --system-site-packages .venv
. .venv/bin/activate
pip install -q -r training/requirements.txt

python training/finetune.py --config training/configs/qlora_config.yaml

# same eval file, before vs after -- the comparison is the result
python training/evaluate.py --out eval_baseline.json
python training/evaluate.py --adapter checkpoints/qwen2vl-shuttlecast-qlora/adapter --out eval_finetuned.json

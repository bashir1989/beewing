#!/bin/bash
# Re-run every experiment on Normalized_PNG (665 images). Usage: run_all.sh <work dir> <ResNet18 weights>
set -e
W=/mnt/project-files/wing_classifier; S=$W/train_wing_cnn.py; E=$W/experiments
export WING_DATA="/mnt/project-files/subspecies Train/Normalized_PNG"
C=$1; WTS=$2; mkdir -p $C
python3 -c "import sys,numpy as np; sys.path.insert(0,'$W'); import train_wing_cnn as t; \
np.save('$C/X.npy', np.stack([t.load(p,(320,144)) for p,_,_ in t.list_images('$WING_DATA')])[:,None])"
( for f in 0 1 2 3 4; do python3 $S --data "$WING_DATA" --out $C/cnn_f$f --fold $f --threads 2 --cache $C/X.npy > $C/cnn_f$f.log 2>&1; done ) &
( for f in 0 1 2 3 4; do python3 $S --data "$WING_DATA" --out $C/res_f$f --fold $f --threads 1 --cache $C/X.npy --pretrained --weights $WTS --epochs 30 > $C/res_f$f.log 2>&1; done
  python3 $E/baseline_classic.py $C/X.npy $C/baseline.json > $C/baseline.log 2>&1 ) &
wait
python3 $E/summarize_cv.py $C > $C/summary.log 2>&1
python3 $E/gradcam.py $C/X.npy $C/cnn_f0/best_model.pt $C/gradcam_fold0.png $C/cnn_f0/test_idx.npy > $C/gradcam.log 2>&1
echo ALLDONE > $C/done

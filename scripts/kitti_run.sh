#!/bin/bash

if [ "$#" -eq 0 ] || ! [ "$1" -gt 0 ] 2> /dev/null; then
  echo "please enter a valid labelled number for training"
  exit 1

  else
	  if (("$1" != 1 && "$1" != 10 && $1 != "$1" && "$1" != 50)); then
             echo "we support the experimental setup as follows:"
             echo "
+----------------+-----------------+--------------+
| # labelled per |  max epochs     | unsup weight |
+----------------+-----------------+--------------+
| 1              | 75              | 1.0          |
+----------------+-----------------+--------------+
| 10             | 75              | 1.0          |
+----------------+-----------------+--------------+
| 20             | 95              | 1.0          |
+----------------+-----------------+--------------+
| 50             | 95              | 1.0          |
+----------------+-----------------+--------------+"
          exit 1
  fi
fi

if [ "$1" == 1 ]; then
  unlabelled_weight=1.0
  max_epoch=75
elif [ "$1" == 10 ]; then
  unlabelled_weight=1.0
  max_epoch=75
else
  unlabelled_weight=1.0
  max_epoch=95
fi

nohup python3 nuscenes.code.ItTakesTwo.gmm/main.py --labelled_percent="$1" --gpus=4 --scale=train_val --batch_size=2 --lr=8e-3 --epoch=${max_epoch} --unlabelled_weight=${unlabelled_weight} > kitti_hist_"$1"_"${max_epoch}".out &

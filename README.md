# EZMode
This is the official project page for "Active Search for Rare Examples via Proximity-based Selection". EZMode is an iterative algorithm for selecting rare events in large, unlabeled datasets

# Requirements
Install the requitements with `pip install -r requirements.txt`

# Reproducing Experiments

To reproduce experiments on the ImageNet-VID and YouTube-BB datasets, run `examples/im-vid.py` and `examples/yt.py`.

For each dataset, you will need to organize the dataset as follows: 

├── directory\_root
│   ├── video\_1
│   └── video\_2
│		├── video\_2\_frame\_1.jpg
│		├── video\_2\_frame\_2.jpg
│		├── video\_2\_frame\_3.jpg
│		├── ...
|	...
│   └── video\_n


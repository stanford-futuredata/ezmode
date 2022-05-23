# EZMode
This is the official project page for "Active Search for Rare Examples via Proximity-based Selection". EZMode is an iterative algorithm for selecting rare events in large, unlabeled datasets

# Requirements
Install the requitements with `pip install -r requirements.txt`

# Reproducing Experiments

To reproduce experiments on the ImageNet-VID and YouTube-BB datasets, run `examples/im-vid.py` and `examples/yt.py`.

For each dataset, you will need to organize the dataset as follows: 
```
├── directory_root
│   ├── video_1
│   └── video_2
	├── video_2_frame_1.jpg
	├── video_2_frame_2.jpg
	├── video_2_frame_3.jpg
	├── ...
|	...
│   └── video_n
```

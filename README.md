# EZMode
This is the official project page for "Active Search for Rare Examples via Proximity-based Selection". EZMode is an iterative algorithm for selecting rare events in large, unlabeled datasets

# Requirements
Install the requitements with `pip install -r requirements.txt`

# Reproducing Experiments

To reproduce experiments on the ImageNet-VID and YouTube-BB datasets, run `examples/im-vid.py` and `examples/yt.py`.

For each dataset, you will need to do the following
1. Organize the dataset in the by videos (see below)
2. Create a CSV storing annotations
3. Create a CSV storing labels

We have provided the annotation and label CSVs for the ImageNet-VID and YouTube-BB datasets in `examples/data`

## Dataset Organization

You must organize the dataset as follows: 
1. Create a directory to exclusively store the dataset. We will call this the `dataset_root`
2. For every video in the dataset, create a directory in `dataset_root`. 
3. Store every frame in the corresponding video directory in chronological order. 

```
├── dataset_root				#directory specific to each dataset
│   ├── video_1					#directory for every video
│   └── video_2
	├── 0001.jpg				#frames of each video
	├── 0002.jpg
	├── 0003.jpg
	├── ...
|	...
│   └── video_n
```

## Dataset CSVs

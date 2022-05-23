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

We have provided the annotation and label CSVs for the ImageNet-VID and YouTube-BB datasets in `examples/data`. To run EZMode on other labeled datasets, please replicate this format. 

### Dataset Organization

You must organize the dataset as follows: 
1. Create a directory to exclusively store the dataset. We will call this the `dataset_root`
2. For every video in the dataset, create a directory in `dataset_root`. 
3. Store every frame in the corresponding video directory in chronological order. 

```
├── dataset_root	   #directory specific to each dataset
│   ├── video_1		   #directory for every video
│   └── video_2
	├── 0001.jpg	   #frames of each video
	├── 0002.jpg
	├── 0003.jpg
	├── ...
|	...
│   └── video_n
```

# Config

Here are the hyperparameters and options that you can customize when running the EZModeEnd2End engine. 
* `db`: Path to project database
* `init_db`: Boolean that determines whether to initialize the database if it does not exist
* `annotations_csv`: Path to annotations CSV (see examples for desired format)
* `labels_csv`: Path to labels CSV (see examples for desired format)
* `num_pos_train_seed`: Number of positive examples in seed training set
* `num_pos_neg_seed`: Number of negative examples in seed training set
* `rare_class`: Index of rare class to select
* `root`: Path to dataset (same as `dataset root` above)
* `working_dir`: Path to write outputs (models, scores, etc.)
* `project_name`: Project name (i.e. "im-vid", or "youtube")
* `select_to_recall`: Boolean that determines whether to run EZMode until recall target is achieved
* `target_recall`: Target recall
* 


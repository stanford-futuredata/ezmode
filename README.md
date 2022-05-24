# EZMode
This is the official project page for "Active Search for Rare Examples via Proximity-based Selection". EZMode is an iterative algorithm for selecting rare events in large, unlabeled datasets

# Requirements
Install the requitements with `pip install -r requirements.txt`

# Reproducing Experiments

To reproduce experiments on the ImageNet-VID and YouTube-BB datasets, run `examples/im-vid.py` and `examples/youtube.py`.

For each dataset, you will need to do the following
1. Organize the dataset in the by videos (see below)
2. Create a CSV storing annotations
3. Create a CSV storing labels

We have provided the annotation and label CSVs for the ImageNet-VID and YouTube-BB datasets at this [link](https://drive.google.com/file/d/1tWDviec-nzbJCxdYdQ4QFrhn3TrfUJOX/view?usp=sharing). Please download and unzip this file to access the annotation and label CSVs for ImageNet-VID and Youtube-BB. To run EZMode on other labeled datasets, please replicate this format. 

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

Hyperparameters and options that you can customize when running the EZModeEnd2End engine. 
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
* `target_recall`: Target recall to run to if `select_to_recall` set to True
* `num_rounds`: Number of rounds to run (only used `select_to_recall` set to False)
* `select_per_round`: Maximum number of oracle calls per round
* `lr`: Learning rate
* `nb_epochs`: Number of training epochs
* `train_batch_size`: Batch size for training
* `infer_batch_size`: Batch size for inference
* `gpus`: Number of GPUs on your machine
* `agg_every_n`: When generating the rank ordering, we select the max-scoring frame across every-n frames in a given video. 
* `prox`: Boolean that determines whether to perform proximity sampling
* `prox_rad`: Proximity radius
* `cluster`: Boolean that determines whether to perform training on easiest examples from each cluster
* `cluster_p`: Upper percentile of examples by confidence to add back to the training set 


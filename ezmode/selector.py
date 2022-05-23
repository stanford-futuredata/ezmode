import os
import pandas as pd
from .dataloader import *
import glob
import tqdm
from .clusters import *

num_labeled = 0
true_pos = 0

class Selector:
    def __init__(self, 
            dataloader, 
            num_to_label, 
            agg_every_n, 
            rare_class, 
            prox, 
            prox_rad, 
            cluster, 
            cluster_p, 
            add_only_rare = True):

        self.dataloader = dataloader
        self.num_to_label = num_to_label
        self.agg_every_n = agg_every_n
        self.rare_class = rare_class
        self.prox = prox
        self.prox_rad = prox_rad 
        self.cluster = cluster
        self.cluster_p = cluster_p
        self.add_only_rare = add_only_rare
        self.logits = None

        scores_dir = os.path.join(self.dataloader.round_working_dir, 'scores')
        self.scores = glob.glob(os.path.join(scores_dir, '*'))

    def aggregate_groupby(self, groups):
        print("Starting aggregation...")

        groupby_logits = pd.DataFrame()
        
        for logits_csv in tqdm.tqdm(self.scores):

            logits = pd.read_csv(logits_csv)
            logits = logits[logits['label']==self.rare_class]
            logits_per_group = logits.groupby(by=groups)

            for group in logits_per_group.__iter__():
                group_data = group[1]
                unique_image_ids = np.unique(group_data['image_id'])
                every_n_clusters = [unique_image_ids[i:i + self.agg_every_n] for i in range(0, len(unique_image_ids), self.agg_every_n)]

                for cluster in every_n_clusters:
                    cluster_logits = group_data[group_data['image_id'].isin(cluster)]
                    max_frame = cluster_logits[cluster_logits['score']==cluster_logits['score'].max()]
                    groupby_logits = groupby_logits.append(max_frame)


        groupby_logits = groupby_logits.sort_values(by='score', ascending = False)
        return groupby_logits

    def add_train_data(self, only_rare = False, cluster = True, p = 0.5):

        train_data = self.dataloader.get_user_actions()

        if (only_rare):
            train_data = [image  for image in train_data if (image[3]==self.rare_class)]

        if (cluster):
            logits_per_image = self.aggregate_groupby(groups = ['vid_base_path', 'image_id'])
            train_data_rare = [image  for image in train_data if (image[3]==self.rare_class)]
            clusters = get_merged(train_data_rare)
            max_prob_clusters = conf_sample(clusters, logits_per_image, p)
            train_data = []
            for value in max_prob_clusters.values():
                train_data = train_data + value
       
        self.dataloader.add_train(train_data)
        return train_data

    
    def process_scores(self):
        self.logits = self.aggregate_groupby(groups = ['vid_base_path'])
        self.logits.to_csv(os.path.join(self.dataloader.round_working_dir, 'logits.csv'))

    def sample_via_proximity(self, center_image_id, vid_base_path):
        global num_labeled
        global true_pos

        images_per_video = self.dataloader.video_to_id(vid_base_path)

        min_image_id = images_per_video[0]
        max_image_id = images_per_video[-1]

        start_image_id = (center_image_id - self.prox_rad) if (center_image_id - self.prox_rad >= min_image_id) else min_image_id
        end_image_id = (center_image_id + self.prox_rad) if (center_image_id + self.prox_rad <= max_image_id) else max_image_id

        print(f'Proximity sampling from {start_image_id} to {end_image_id}')

        for image_id in range(start_image_id, end_image_id + 1):
            if (image_id == center_image_id or self.dataloader.id_is_labeled(image_id)):
                continue

            if (num_labeled >= self.num_to_label):
                return 

            num_labeled += 1
            label = self.dataloader.label_image(image_id, self.rare_class, strat = 'temp')
            print(f'image_id={image_id}, label={label}, strat=temp. {num_labeled} total images labeled')

            if (label == self.rare_class):
                true_pos += 1
                self.sample_via_proximity(image_id, vid_base_path)

        return 
        
    def label(self):

        global num_labeled
        global true_pos 

        rank_order_images = self.logits['image_id'].tolist()
        rank_order_videos = self.logits['vid_base_path'].tolist()

        rank = 0

        while (num_labeled < self.num_to_label and rank < len(rank_order_images)):
            image_id = rank_order_images[rank]
            vid_base_path = rank_order_videos[rank]

            rank += 1

            if self.dataloader.id_is_labeled(image_id):
                print("Found labeled image! Moving on...")
                continue 

            label = self.dataloader.label_image(image_id, self.rare_class, strat = f'round{self.dataloader.round_no}')
            num_labeled += 1
            print(f'image_id={image_id}, label={label}, strat=round{self.dataloader.round_no}. {num_labeled} total images labeled')

            if (label == self.rare_class):
                true_pos += 1
                if (self.prox):
                    print(f'Rare class found! Starting proximity sampling on {image_id}')
                    self.sample_via_proximity(image_id, vid_base_path)
                else: 
                    print("Rare class found!")

        return

    def select(self):
        self.process_scores()
        self.label()
        self.add_train_data()




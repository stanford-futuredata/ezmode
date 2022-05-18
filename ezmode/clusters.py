import os
import pandas as pd
import numpy as np
from collections import defaultdict


def conf_sample(merged, scores, p):

    def get_top_quantile(images):
        outp_per_cluster = scores[scores['image_id'].isin(images)]
        scores_per_cluster = outp_per_cluster['score'].to_numpy()
        scores_per_cluster = np.pad(scores_per_cluster, (0, len(images) - len(outp_per_cluster)))
        min_score = np.quantile(scores_per_cluster, p)
        top_quantile = outp_per_cluster[outp_per_cluster['score'] >= min_score]['image_id'].tolist()
        return top_quantile


    max_prob_merged  = defaultdict(list)
    for cluster, rows in merged.items():
        images = []
        for row in rows:
            images.append(row[0])

        top_quantile = get_top_quantile(images)
        
        for row in rows:
            if (row[0] in top_quantile):
                max_prob_merged[cluster].append(row)

    return max_prob_merged 

def get_merged(data):
    def is_close(cur_row, rows):
        for row in rows:
            if (abs(cur_row[0] - row[0]) <= 5 and (cur_row[1] == row[1])):
                return True
        return False

    merged = defaultdict(list)
    for cur_row in sorted(data):
        added = False
        for cluster, rows in merged.items():
            if is_close(cur_row, rows):
                merged[cluster].append(cur_row)
                added = True
                break
        if not added:
            merged[len(merged)].append(cur_row)

    return merged


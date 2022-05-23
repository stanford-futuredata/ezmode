from ezmode import EZModeEnd2End
import os

db = os.path.join(os.getcwd(), 'youtube.db')

EZModeEngine = EZModeEnd2End(
        db = db, 
        annotations_csv = os.path.join(os.getcwd(), 'youtube_annot.csv'), 
        labels_csv = os.path.join(os.getcwd(), 'youtube_labels.csv'), 
        num_pos_train_seed = 50, 
        num_neg_train_seed = 0, 
        select_per_round = 100, 
        num_rounds = 5, 
        rare_class = 18, 
        root = '/lfs/1/alexder/yt-videos/ezmode_format_sub20x', #Replace with your dataset root
        project_name = 'youtube', 
        working_dir = os.getcwd())

EZModeEngine.run()


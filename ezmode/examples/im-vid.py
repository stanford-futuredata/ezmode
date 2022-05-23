from ezmode import EZModeEnd2End
import os

db = os.path.join(os.getcwd(), 'im-vid.db')

EZModeEngine = EZModeEnd2End(
        db = db, 
        annotations_csv = os.path.join(os.getcwd(), 'im-vid_annot.csv'), 
        labels_csv = os.path.join(os.getcwd(), 'im-vid_labels.csv'), 
        num_pos_train_seed = 30, 
        num_neg_train_seed = 0, 
        select_per_round = 100, 
        num_rounds = 5, 
        rare_class = 25, 
        root = '/lfs/1/ddkang/im-vid/data/ILSVRC2015/Data/VID', #Replace with your dataset root
        project_name = 'im-vid', 
        working_dir = os.getcwd())

EZModeEngine.run()


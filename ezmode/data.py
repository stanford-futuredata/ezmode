import numpy as np
import tqdm
import datetime
import os
import pandas as pd
import sqlalchemy
from sqlalchemy import insert, update


class DataLoader:
    def __init__(self, 
            project_name, 
            root, 
            working_dir, 
            round_name, 
            db):

        self.project_name= project_name
        self.root = root
        self.working_dir = working_dir
        self.round_name = round_name
        self.db = db
        self.train_data = None
        self.train_df = None

        self.engine = sqlalchemy.create_engine('sqlite:///{}'.format(self.db))
        self.metadata_dir = os.path.join(self.working_dir, f'{project_name}_metadata')
        self.val_data = os.path.join(self.metadata_dir, 'val_data.csv')
        self.round_working_dir = os.path.join(self.working_dir, self.round_name)

        self.class_dict = {}

        if not os.path.exists(self.round_working_dir):
            os.mkdir(self.round_working_dir)


    def get_val_data(self):
        return self.val_data

    def init_metadata(self, root = False):
        def gen_val_data():

            vid_base_paths = []

            with self.engine.connect() as con: 
                rs = con.execute(
                        'SELECT DISTINCT vid_base_path FROM '
                        'annotations WHERE split=\'val\'')
                for row in rs:
                    print("{} processed".format(str(row[0])))
                    vid_base_path = str(row[0])
                    if root:
                        vid_base_path = os.path.join(self.root, vid_base_path)

                    vid_base_paths.append([vid_base_path])

            val_df = pd.DataFrame(data = vid_base_paths, columns = ['vid_base_path'])
            
            val_df.to_csv(self.val_data)

            return 

        if not os.path.exists(self.metadata_dir):
            os.mkdir(self.metadata_dir)
            gen_val_data()


    '''
    Return database path
    '''
    def get_db(self):
        return self.db


    '''
    Set training data 
    '''
    def set_train_data(self, csv):
        train_df = pd.read_csv(csv)
        self.train_df = train_df
        return 

    def get_user_actions(self, start_idx, end_idx, with_labels = True):
        data = []
        
        if (with_labels):
            with self.engine.connect() as con: 
                rs = con.execute(
                        'SELECT annotations.image_id, '
                        'annotations.image_base_path, '
                        'annotations.xmin, '
                        'annotations.ymin, '
                        'annotations.xmax, '
                        'annotations.ymax, ' 
                        'annotations.label, '
                        'annotations.vid_base_path '
                        'FROM annotations '
                        'INNER JOIN user_actions ON user_actions.image_id=annotations.image_id '
                        f'WHERE user_actions.id BETWEEN {start_idx} AND {end_idx}'
                        )
                for row in rs: 
                    image_id = int(row[0])

                    image_base_path = str(row[1])

                    xmin = int(row[2])
                    ymin = int(row[3])
                    xmax = int(row[4])
                    ymax = int(row[5])

                    label = int(row[6])

                    vid_base_path = str(row[7])

                    data.append([
                        image_id,
                        vid_base_path,
                        image_base_path, 
                        label, 
                        xmin, 
                        ymin, 
                        xmax, 
                        ymax])

        else: 
            with self.engine.connect() as con: 
                rs = con.execute(
                        'SELECT annotations.image_id, '
                        'annotations.image_base_path, '
                        'annotations.vid_base_path, '
                        'user_actions.label '
                        'FROM annotations '
                        'INNER JOIN user_actions ON user_actions.image_id=annotations.image_id '
                        f'WHERE user_actions.id BETWEEN {start_idx} AND {end_idx}'
                        )
                for row in rs: 
                    image_id = int(row[0])
                    image_base_path = str(row[1])
                    vid_base_path = str(row[2])
                    label = int(row[3])

                    data.append([
                        image_id,
                        vid_base_path,
                        image_base_path,
                        label,
                        None, 
                        None, 
                        None, 
                        None
                        ])


        return data

   
    '''
    Return train data as df, if out is a path, write data to path
    '''
    def get_train_data(self, 
            root = True, 
            zero_index = True, 
            shuffle = True, 
            out = None, 
            binary = False, 
            rare_class = None):

        train_data = []
        with self.engine.connect() as con: 
            rs = con.execute(
                    'SELECT annotations.image_base_path, '
                    'annotations.xmin, '
                    'annotations.ymin, '
                    'annotations.xmax, '
                    'annotations.ymax, '
                    'annotations.label '
                    'FROM annotations '
                    'INNER JOIN labels ON annotations.label=labels.label '
                    'WHERE annotations.split=\'train_ezmode\'')
            for row in rs:
                image_base_path = str(row[0])

                xmin = int(row[1])
                ymin = int(row[2])
                xmax = int(row[3])
                ymax = int(row[4])

                label = int(row[5])

                if (root):
                    image_base_path = os.path.join(self.root, image_base_path)

                train_data.append([image_base_path, xmin, ymin, xmax, ymax, label])

        self.train_df = pd.DataFrame(data = train_data, columns = ['image', 'xmin', 'ymin', 'xmax', 'ymax', 'label'])

        #Shuffle training data
        if (shuffle):

            print("shuffling")
            self.train_df = self.train_df.sample(frac = 1).reset_index(drop = True)

        #Zero index the data labels for binary classification tasks
        if (binary and rare_class != None):
            
            binary_labels = []

            for label in self.train_df['label'].to_numpy():
                if (label == rare_class):
                    binary_labels.append(1)
                else:
                    binary_labels.append(0)

            self.class_dict[1] = rare_class
            self.train_df['label'] = binary_labels

        #Zero index the data labels for N-way classification tasks
        elif (zero_index):

            unique_classes = np.unique(np.sort(self.train_df['label'].to_numpy()))
            zero_index_labels = []

            for label in np.sort(self.train_df['label'].to_numpy()):
                zero_index_label = np.asscalar(np.where(unique_classes==label)[0]) + 1
                zero_index_labels.append(zero_index_label)
                self.class_dict[zero_index_label] = label
            
            self.train_df['label'] = zero_index_labels

        if (out != None):
            self.train_df.to_csv(out)

        return self.train_df
   
    '''
    Return number of classes in training set. Considers background class by default
    '''
    def get_num_classes(self):
        if (not isinstance(self.train_df, pd.DataFrame)):
            self.get_train_data()

        return np.unique(self.train_df['label']).shape[0] + 1

    '''
    Return map from class id and class name
    '''
    def get_class_dict(self, binary = False, rare_class = None):
        if (not isinstance(self.train_df, pd.DataFrame)):
            self.get_train_data(binary = binary, rare_class = rare_class)

        return self.class_dict

    '''
    Returns SQLAlchemy engine for database
    ''' 
    def get_engine(self):
        return engine

    '''
    Returns image_ids for annotations in a video
    '''
    def video_to_id(self, vid_base_path):
        image_ids = []

        with self.engine.connect() as con: 
            rs = con.execute(
                'SELECT DISTINCT image_id FROM annotations '
                'WHERE vid_base_path=\'{}\''.format(vid_base_path)
            )
            if (rs.rowcount == 0): 
                print("No image_id found for vid_base_path=\'{}\'".format(vid_base_path))
                return 
            else:
                for row in rs: 
                    image_ids.append(int(row[0]))

        return image_ids                    

    '''
    Returns image base paths for annotations in a video
    '''
    def video_to_image_path(self, vid_base_path, root = False):

        image_base_paths = []

        with self.engine.connect() as con: 
            rs = con.execute(
                'SELECT DISTINCT image_base_path FROM annotations '
                'WHERE vid_base_path=\'{}\''.format(vid_base_path)
            )
            if (rs.rowcount == 0): 
                print("No image_id found for vid_base_path=\'{}\'".format(vid_base_path))
                return 
            else:
                for row in rs: 
                    image_base_path = str(row[0])
                    if (root): 
                        image_base_path = os.path.join(self.root, image_base_path)
                    image_base_paths.append(image_base_path)

        return image_base_paths

    def add_train(self, train_data):

        image_ids = np.unique([train_img[0] for train_img in train_data])
        with self.engine.connect() as con: 

            print("Adding training data to database...")
            for image_id in tqdm.tqdm(image_ids):
                annot_id = None
                rs = con.execute(
                        'SELECT annot_id FROM annotations '
                        f'WHERE image_id = {image_id} LIMIT 1'
                        )

                for row in rs: 
                    annot_id = int(row[0])
                    break 
                        
                rs = con.execute(
                        'UPDATE annotations '
                        'SET split=\'train_ezmode\' '
                        f'WHERE image_id = {image_id} AND '
                        f'annot_id = {annot_id}'
                        )
        return 


    def annot_exists(self, image_id):
        
        with self.engine.connect() as con: 
            rs = con.execute(
                    'SELECT COUNT(image_id) '
                    'FROM annotations '
                    f'WHERE image_id={image_id} '
                    'AND label IS NOT NULL '
                    )
            for row in rs: 
                if (int(row[0]) == 0):
                    return False

                return True

    def num_annots(self):
        
        with self.engine.connect() as con: 
            rs = con.execute(
                    'SELECT COUNT(*) FROM annotations'
                    )

            for row in rs: 
                return int(row[0])

    def insert_vott_labels(self, csv):
        
        META_DATA = sqlalchemy.MetaData(bind = self.engine, reflect = True)
        annot_table = META_DATA.tables['annotations']

        df = pd.read_csv(csv)
        image_ids = [int(image_name.split('.')[0]) for image_name in df['image'].tolist()]

        with self.engine.connect() as con: 
            for idx in range(len(df)):
                row_data = df.iloc[idx]

                image_id = image_ids[idx]
                xmin = row_data['xmin']
                xmax = row_data['xmax']
                ymin = row_data['ymin']
                ymax = row_data['ymax']

                label_name = row_data['label']

                new_split = 'train_ezmode'

                label_idx = None
                rs = con.execute(
                        'SELECT label FROM labels '
                        f'WHERE name=\'{label_name}\'')

                for row in rs: 
                    label_idx = int(row[0])
                    break 
                
                if (self.annot_exists(image_id)):

                    image_base_path = self.id_to_image_path(image_id)
                    vid_base_path = image_base_path.split('/')[0]
                    new_annot_id = self.num_annots() + 1

                    command = (
                            insert(annot_table).
                            values(
                                image_base_path = image_base_path, 
                                vid_base_path = vid_base_path,
                                image_id = image_id, 
                                label = label_idx,
                                xmin = xmin, 
                                xmax = xmax, 
                                ymin = ymin, 
                                ymax = ymax, 
                                split = new_split, 
                                labeled = 1,
                                annot_id = new_annot_id)
                            )
                    compiled = command.compile()
                    rs = con.execute(compiled)

                else:
                    command = (
                            update(annot_table).
                            where(annot_table.c.image_id == image_id).
                            values(
                                label = label_idx,
                                xmin = xmin, 
                                xmax = xmax, 
                                ymin = ymin, 
                                ymax = ymax, 
                                split = new_split, 
                                labeled = 1)
                            )
                    compiled = command.compile()
                    rs = con.execute(compiled)

            

    '''
    Label image, return label
    '''
    def label_image(self, image_id, rare_class, strat, label = None, root = False):

        def get_label_metadata():
            idx = None
            label_time = datetime.datetime.now()

            with self.engine.connect() as con:

                rs = con.execute(
                        'SELECT COUNT(*) from user_actions'
                        )

                for row in rs:
                    idx = int(row[0]) 

            return idx, label_time

        def get_label():
            label = None

            idx, label_time = get_label_metadata()

            with self.engine.connect() as con:
                labels = []
                rs = con.execute(
                        'SELECT label FROM annotations ' 
                        'WHERE image_id={}'.format(image_id))

                for row in rs:
                    labels.append(int(row[0]))

                if rare_class in labels:
                    label = rare_class
                else: 
                    label = labels[0]


            return label, idx, label_time

        if (label == None):
            label, idx, label_time = get_label()
        else:
            idx, label_time = get_label_metadata()
            
        META_DATA = sqlalchemy.MetaData(bind = self.engine, reflect = True)
        user_actions_table = META_DATA.tables['user_actions']

        with self.engine.connect() as con:
            command = (
                    insert(user_actions_table).
                    values(
                        label_time = label_time,
                        label = label, 
                        user_id = 1, 
                        image_id = image_id, 
                        strat = strat, 
                        id = idx)
                    )
            compiled = command.compile()
            rs = con.execute(compiled)

        return label


    '''
    Returns true if image_id is labeled by the user, false otherwise
    '''
    def id_is_labeled(self, image_id, root = False):

        with self.engine.connect() as con:
            rs = con.execute(
                    'SELECT EXISTS(SELECT * from user_actions ' 
                    'WHERE image_id=\'{}\')'.format(image_id))
            for row in rs:
                if (int(row[0]) == 1):
                    return True
                else:
                    return False



    '''
    Returns label for image_id
    '''
    def id_to_label(self, image_id, root = False):
        with self.engine.connect() as con:
            rs = con.execute(
                    'SELECT label FROM annotations '
                    'WHERE image_id=\'{}\')'.format(image_id))
            for row in rs:
                return int(row[0])

    '''
    Returns true if image_path is labeled, false otherwise
    '''
    def image_path_is_labeled(self, image_base_path, root = False):
        if (root): 
            image_base_path = image_base_path.split(self.root)[-1]

        image_id = self.image_path_to_id(image_base_path)

        return self.id_is_labeled(image_id)


    '''
    Returns image base_path corrresponding to an image id
    '''
    def id_to_image_path(self, image_id, root = False):

        image_base_path = None

        with self.engine.connect() as con: 
            rs = con.execute(
                'SELECT image_base_path FROM annotations '
                'WHERE image_id={}'.format(image_id)
            )
            if (rs.rowcount == 0): 
                print("No image_base_path found for image_id={}".format(image_id))
                return 
            else:
                for row in rs: 
                    image_base_path = str(row[0])
                    break

        if (root):
            image_base_path = os.path.join(self.root, image_base_path)

        return image_base_path

    '''
    Returns image id corrresponding to an image base path
    '''
    def image_path_to_id(self, image_base_path, root = False):

        if (root):
            image_base_path = image_base_path.split(self.root)[-1]

        image_id = None

        with self.engine.connect() as con: 
            rs = con.execute(
                'SELECT image_id FROM annotations '
                'WHERE image_base_path=\'{}\''.format(image_base_path)
            )
            if (rs.rowcount == 0):
                print("No image_id found for image_base_path={}".format(image_base_path))
                return 
            else:
                for row in rs: 
                    image_id = int(row[0])
                    break

        return image_id


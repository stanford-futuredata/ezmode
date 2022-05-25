import os
import pandas as pd
import tqdm
import numpy as np
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


class Database:
    def __init__(self, 
            db, 
            annotations_csv, 
            labels_csv):

        self.db = db
        self.annotations_csv = annotations_csv
        self.labels_csv = labels_csv

        self.annotations_df = pd.read_csv(self.annotations_csv)
        self.labels_df = pd.read_csv(self.labels_csv)
        self.engine = create_engine('sqlite:///{}'.format(self.db))

    '''
    Create database and schema
    '''
    def create(self):
        with self.engine.connect() as con: 
            con.execute('''
                    CREATE TABLE IF NOT EXISTS users 
                    (
                        user_id INTEGER NOT NULL, 
                        username VARCHAR, 
                        PRIMARY KEY(user_id)
                    )
                    ''')

            con.execute('''
                    CREATE TABLE IF NOT EXISTS labels
                    (
                        name VARCHAR, 
                        label INTEGER NOT NULL, 
                        PRIMARY KEY(label)
                    )
                    ''')

            con.execute('''
                      CREATE TABLE IF NOT EXISTS annotations
                      (
                        vid_base_path VARCHAR, 
                        image_base_path VARCHAR, 
                        image_id INTEGER NOT NULL,
                        label INTEGER NOT NULL, 
                        xmin INTEGER NOT NULL, 
                        xmax INTEGER NOT NULL, 
                        ymin INTEGER NOT NULL, 
                        ymax INTEGER NOT NULL, 
                        annot_id INTEGER NOT NULL, 
                        split VARCHAR, 
                        labeled BIT, 
                        PRIMARY KEY (annot_id)
                        FOREIGN KEY(label) REFERENCES labels (label)
                        )
                      ''')

            con.execute('''
                    CREATE TABLE IF NOT EXISTS user_actions 
                    (
                        label_time DATETIME, 
                        label INTEGER NOT NULL,
                        user_id INTEGER NOT NULL, 
                        image_id INTEGER NOT NULL,
                        strat VARCHAR, 
                        id INTEGER NOT NULL, 
                        FOREIGN KEY(image_id) REFERENCES annotations (image_id)
                        FOREIGN KEY(user_id) REFERENCES users (user_id)
                        FOREIGN KEY(label) REFERENCES labels (label)
                    )
                    ''')

    '''
    Insert annotations
    '''
    def insert_annotations(self): 
        metadata_obj = MetaData()
        metadata_obj.reflect(bind = self.engine)

        Session = sessionmaker(bind = self.engine)
        session = Session()

        Base = declarative_base(self.engine)

        class AnnotationsTable(Base):
            __tablename__ = 'annotations'
            __table_args__ = {'autoload': True}

        session.bulk_save_objects(
                [
                    AnnotationsTable(
                        vid_base_path = str(row[1]['vid_base_path']), 
                        image_base_path = str(row[1]['image_base_path']), 
                        image_id = int(row[1]['image_id']), 
                        label = int(str(row[1]['label'])), 
                        xmin = int(row[1]['xmin']), 
                        xmax = int(row[1]['xmax']), 
                        ymin = int(row[1]['ymin']), 
                        ymax = int(row[1]['ymax']), 
                        annot_id = int(row[1]['annot_id']), 
                        split = str(row[1]['split']), 
                        labeled = int(row[1]['labeled'])
                        )
                    for row in tqdm.tqdm(self.annotations_df.iterrows())
                    ]
                )
        session.commit()

    '''
    Insert labels
    '''
    def insert_labels(self):
        metadata_obj = MetaData()
        metadata_obj.reflect(bind = self.engine)

        Session = sessionmaker(bind = self.engine)
        session = Session()

        Base = declarative_base(self.engine)

        class AnnotationsTable(Base):
            __tablename__ = 'labels'
            __table_args__ = {'autoload': True}

        session.bulk_save_objects(
                [
                    AnnotationsTable(
                        name = str(row[1]['name']), 
                        label = int(row[1]['index']), 
                        )
                    for row in tqdm.tqdm(self.labels_df.iterrows())
                    ]
                )
        session.commit()

    '''
    Initialize tables
    '''
    def init_tables(self):
        self.insert_labels()
        self.insert_annotations()

    '''
    Insert training data seed
    '''
    def insert_train_data(self, df):

        with self.engine.connect() as con: 
            for row in df.iterrows():
                data = row[1]

                image_id = int(data['image_id'])
                label = int(data['label'])
                annot_id = int(data['annot_id'])

                con.execute(
                        'UPDATE annotations '
                        f'SET split=\'train_ezmode\' '
                        f'WHERE image_id={image_id} '
                        f'AND label={label} '
                        f'AND annot_id={annot_id}'
                        )

    '''
    Generate training data seed
    '''
    def init_train_set(self, 
        num_pos_train, 
        num_neg_train, 
        rare_class):

        train_data = self.annotations_df[self.annotations_df['split']=='train']

        shuffle_idx = np.arange(len(train_data))
        np.random.shuffle(shuffle_idx)

        train_shuffled = train_data.iloc[shuffle_idx]

        train_pos = train_shuffled[train_shuffled['label']==rare_class][:num_pos_train]
        train_neg = train_shuffled[train_shuffled['label']!=rare_class][:num_neg_train]
        train_seed = pd.concat([train_pos, train_neg])

        self.insert_train_data(train_seed)

        return train_pos, train_neg


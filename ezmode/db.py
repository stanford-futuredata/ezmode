import os
import pandas as pd
import tqdm
import numpy as np
import sqlalchemy


class Database:
    def __init__(self, 
            db, 
            metadata_csv):
        self.db = db
        self.metadata_csv = metadata_csv

    def create_db():
        engine = sqlalchemy.create_engine('sqlite:///{}'.format(self.db))
        with engine.connect() as con: 
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
                        FOREIGN KEY(image_id) REFERENCES images (image_id)
                        FOREIGN KEY(user_id) REFERENCES users (user_id)
                        FOREIGN KEY(label) REFERENCES labels (label)
                    )
                    ''')
            conn.commit()
            conn.close()

    def init_db(self, 
            num_pos_train, 
            num_neg_train):


from .dataloader import *
from .trainer import *
from .selector import *
from .infer import *
from .db import *

class EZModeEnd2End:
    def __init__(self,
            db, 
            annotations_csv, 
            labels_csv, 
            num_pos_train_seed, 
            num_neg_train_seed, 
            rare_class, 
            root, 
            working_dir, 
            project_name, 
            select_to_recall = False, 
            target_recall = None, 
            lr = 0.001, 
            nb_epochs = 20, 
            train_batch_size = 8, 
            infer_batch_size = 8, 
            gpus = 4, 
            select_per_round = 100, 
            agg_every_n = 10, 
            prox = True, 
            prox_rad = 5,
            cluster = True, 
            cluster_p = 0.5, 
            init_db = False, 
            num_rounds = 5):

        #Database Initialization 
        self.db = db
        self.annotations_csv = annotations_csv
        self.labels_csv = labels_csv
        self.num_pos_train_seed = num_pos_train_seed 
        self.num_neg_train_seed = num_neg_train_seed
        self.rare_class = rare_class
        self.project_name = project_name

        #Training and Inference
        self.root = root
        self.working_dir = working_dir
        self.target_recall = target_recall
        self.select_to_recall = select_to_recall 
        self.lr = lr
        self.nb_epochs = nb_epochs
        self.train_batch_size = train_batch_size
        self.infer_batch_size = infer_batch_size
        self.gpus = gpus
        self.select_per_round = select_per_round
        self.init_db = init_db
        self.num_rounds = num_rounds

        #Selection
        self.agg_every_n = agg_every_n
        self.prox = prox
        self.prox_rad = prox_rad
        self.cluster = cluster
        self.cluster_p = cluster_p


    def run_one_round(self, round_no):
        dataloader = DataLoader(
                project_name = self.project_name, 
                root = self.root, 
                working_dir = self.working_dir, 
                round_no = round_no, 
                db = self.db, 
                rare_class = self.rare_class)
        dataloader.init_metadata()

        trainer = Trainer(dataloader = dataloader)
        saved_model = trainer.train(
                lr = self.lr, 
                nb_epochs = self.nb_epochs, 
                batch_size = self.train_batch_size)

        infer_engine = InferEngine(
                dataloader = dataloader, 
                model = saved_model, 
                gpus = self.gpus, 
                batch_size = self.infer_batch_size)
        infer_engine.deploy()

        selector = Selector(
                dataloader = dataloader, 
                select_per_round = self.select_per_round, 
                agg_every_n = self.agg_every_n,
                prox = self.prox, 
                prox_rad = self.prox_rad, 
                cluster = self.cluster, 
                cluster_p = self.cluster_p)
        selector.select()

        prec = dataloader.get_precision()
        recall = dataloader.get_recall()

        return prec, recall


    def run_fixed_rounds(self):
        for round_no in range(1, self.num_rounds + 1):
            prec, recall = self.run_one_round(round_no)
            print(f'Round {round_no} Completed: Precision = {prec}, Recall = {recall}')

    def run_to_recall(self):
        recall = 0
        round_no = 1
        while (recall < self.target_recall):
            prec, recall = run_one_round(round_no)
            print(f'Round {round_no} Completed: Precision = {prec}, Recall = {recall}')
            round_no += 1

    def run(self):
        
        database = Database(
                db = self.db, 
                annotations_csv = self.annotations_csv, 
                labels_csv = self.labels_csv)

        if self.init_db:
            database.create()
            database.init_tables()

        database.init_train_set(
                num_pos_train = self.num_pos_train_seed, 
                num_neg_train = self.num_neg_train_seed,
                rare_class = self.rare_class)

        if self.select_to_recall:
            self.run_to_recall()
        else: 
            self.run_fixed_rounds()


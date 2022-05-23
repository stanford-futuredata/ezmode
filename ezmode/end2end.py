from .dataloader import *
from .trainer import *
from .selector import *
from .infer import *
from .db import *



class EZModeEnd2End:
    def __init__(self,
            db, 
            annotations_csv = annotations_csv, 
            labels_csv = labels_csv, 
            num_pos_train_seed, 
            num_neg_train_seed, 
            rare_class, 
            root, 
            working_dir, 
            target_recall, 
            lr, 
            nb_epochs,
            train_batch_size, 
            infer_batch_size, 
            gpus, 
            select_per_round, 
            init_db = False, 
            num_rounds = None)

        #Database Initialization 
        self.db = db
        self.annotations_csv = annotations_csv
        self.labels_csv = labels_csv
        self.num_pos_train_seed = num_pos_train_seed 
        self.num_neg_train_seed = num_neg_train_seed
        self.rare_class = rare_class

        #Training and Inference
        self.root = root
        self.working_dir = working_dir
        self.target_recall = target_recall
        self.lr = lr
        self.nb_epochs = nb_epochs
        self.train_batch_size = train_batch_size
        self.infer_batch_size = infer_batch_size
        self.gpus = gpus
        self.select_per_round = select_per_round
        self.init_db = init_db
        self.num_rounds = num_rounds

    def run_one_round(self, round_no):
        dataloader = DataLoader(
                project_name = self.project_name, 
                root = self.root, 
                working_dir = self.working_dir, 
                round_no = round_no, 
                db = self.db)
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


    def run_fixed_rounds(self):
        for round_no in range(1, self.num_rounds + 1):
            self.run_one_round(round_no)

    def run_to_recall(self):
        recall = 0
        while (recall < self.target_recall):
            run_one_round(round_no)
            recall = self.get_recall()

    def run(self):
        
        database = Database(
                db = self.db, 
                annotations_csv = self.annotations_csv, 
                labels_csv = self.labels_csv)

        if (self.init_db):
            database.create()
            database.init_tables()

        database.init_train_set(
                num_pos_train = self.num_pos_train_seed, 
                num_neg_train = self.num_neg_train_seed)


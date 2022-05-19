import pandas as pd
import subprocess
import ezmode
from ezmode import dataloader
import os


class InferEngine: 
    def __init__(self, 
            dataloader, 
            model, 
            gpus, 
            batch_size):

        self.dataloader = dataloader
        self.model = model
        self.gpus = gpus
        self.batch_size = batch_size

    def deploy(self, script = None):
        infer_script_fname = os.path.join(os.path.dirname(ezmode.__file__), 'infer_script.py')
        val_data = pd.read_csv(self.dataloader.get_val_data())

        videos = val_data['vid_base_path'].tolist()
        num_videos = len(videos)

        processes = []

        videos_left = num_videos 

        for gpu in range(self.gpus):
            to_process = None

            if (gpu == self.gpus - 1):
                to_process = videos_left 

            else:
                to_process = int(num_videos / self.gpus)
                videos_left -= int(num_videos / self.gpus)

            processes.append(f'python {infer_script_fname} '
                    f'--model {self.model} '
                    f'--gpu {gpu} '
                    f'--project_name {self.dataloader.project_name} '
                    f'--root {self.dataloader.root} '
                    f'--working_dir {self.dataloader.working_dir} '
                    f'--round_no {self.dataloader.round_no} '
                    f'--db {self.dataloader.db} '
                    f'--num_classes {self.dataloader.get_num_classes()} '
                    f'--num_videos {num_videos} '
                    f'--to_process {to_process} '
                    f'--batch_size {self.batch_size} \n')

        infer_jobs = [subprocess.Popen(process, shell = True) for process in processes]
        for infer_job in infer_jobs:
            infer_job.wait()


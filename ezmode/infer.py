import ezmode
import pandas as pd
from ezmode import data
import os


class InferEngine: 
    def __init__(self, 
            dataloader, 
            model, 
            gpus):
        self.dataloader = dataloader
        self.model = model
        self.gpus = gpus

    def write_bash_scripts(self, script = None):
        self.dataloader.init_metadata()
        val_csv = self.dataloader.get_val_data()
        val_data = pd.read_csv(val_csv)

        videos = val_data['vid_base_path'].tolist()
        num_videos = len(videos)

        if (script == None):
            infer_script_fname = os.path.join('/'.join(ezmode.__file__.split('/')[:-1]), 'infer_script.py')
        else:
            infer_script_fname = script

        #assert(os.path.exists(infer_script_fname))

        bash_dest = os.path.join(self.dataloader.working_dir, 'run_infer.sh')

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
                    f'--round_name {self.dataloader.round_name} '
                    f'--db {self.dataloader.db} '
                    f'--num_classes {self.dataloader.get_num_classes()} '
                    f'--num_videos {num_videos} '
                    f'--to_process {to_process} \n')



        if (os.path.exists(bash_dest)):
            os.remove(bash_dest)

        f = open(bash_dest, "a")
        f.writelines(processes)
        f.close()

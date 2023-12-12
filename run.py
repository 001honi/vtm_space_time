import os 
import yaml
import multiprocessing
import time
from tqdm import tqdm

# =========================================================================
num_gpus = 4
num_exp_per_gpu = 1

# META-TRAIN (Hard-coded)
shot_meta = 8 #8
sampling = 2 #25
dir_train = f"TRAIN/PRETASKO-224-MIDAIR-128-samp{sampling}/"
dir_ftune = f"FINETUNE/DAVIS-128-all/PRETASKO-224-MIDAIR-128-samp{sampling}/"

# FINETUNE
videos = ["blackswan","bmx-trees","breakdance","camel",
        "car-roundabout","car-shadow","cows","dance-twirl",
        "dog","drift-chicane","drift-straight","goat",
        "horsejump-high","kite-surf","libby","motocross-jump",
        "paragliding-launch","parkour","scooter-black","soapbox"] 
shot = [2] 
eval = [32]
time_attn = [True]
# ---
# abblation = "-xtime"
# =========================================================================
with open('configs/finetune_config_base.yaml', 'r') as f:
    config_base = yaml.safe_load(f)

experiment_pool = []

for v in videos:
    for s in shot:
        for e in eval:
            for t in time_attn:

                config = config_base.copy()

                config.update({
                    'class_name' : v,
                    'shot' : s,
                    'global_batch_size' : s*2,
                    'eval_batch_size' : e,
                    'time_attn' : shot_meta if t else 0,
                    # 'log_dir'  : dir_ftune + f"{'ts' if t else 's'}-meta{shot_meta}-shot{s}-n{e}",
                    # 'save_dir' : dir_ftune + f"{'ts' if t else 's'}-meta{shot_meta}-shot{s}-n{e}",
                    # 'load_dir' : dir_train + f"{'ts' if t else 's'}-meta{shot_meta}"
                    'log_dir'  : dir_ftune + f"s{'t' if t else ''}-meta{shot_meta}-shot{s}-n{e}",
                    'save_dir' : dir_ftune + f"s{'t' if t else ''}-meta{shot_meta}-shot{s}-n{e}",
                    'load_dir' : dir_train + f"s{'t' if t else ''}-meta{shot_meta}"
                })
                
                experiment_pool.append(config)

def run_experiment(gpu,experiment):
    run_config_path = f"configs/finetune_config_{time.time_ns()}.yaml"
    print(f'\n\n[-] GPU [{gpu}]',
          experiment['log_dir'].split('/')[-1], 
          experiment['class_name'],
          run_config_path)
    with open(run_config_path, 'w') as f:
        yaml.safe_dump(experiment, f)
    run = f"CUDA_VISIBLE_DEVICES={gpu}, python main.py --stage 1 --task semseg --run_config {run_config_path}"
    os.system(run)
    print(f'[X] GPU [{gpu}]',experiment['log_dir'].split('/')[-1], experiment['class_name'])
    os.remove(run_config_path)


for i in tqdm(range(0, len(experiment_pool), num_gpus * num_exp_per_gpu)):
    gpu_workers = [
        multiprocessing.Process(target=run_experiment, args=(gpu//num_exp_per_gpu,experiment)) 
        for gpu, experiment in enumerate(experiment_pool[i:i+num_gpus*num_exp_per_gpu])
    ]                                         

    for gpu_worker in gpu_workers:
        gpu_worker.start()
        time.sleep(1)
    
    for gpu_worker in gpu_workers:
        gpu_worker.join()          


    # videos = { #frames
    #     "blackswan":48,
    #     "bmx-trees":78,
    #     "breakdance":82,
    #     "camel":88,
    #     "car-roundabout":73,
    #     "car-shadow":38,
    #     "cows":102,
    #     "dance-twirl":88,
    #     "dog":58,
    #     "drift-chicane":50,
    #     "drift-straight":48,
    #     "goat":88,
    #     "horsejump-high":48,
    #     "kite-surf":48,
    #     "libby":47,
    #     "motocross-jump":38,
    #     "paragliding-launch":78,
    #     "parkour":98,
    #     "scooter-black":41,
    #     "soapbox":97
    #     }
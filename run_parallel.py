import argparse
import subprocess
import datetime
import os
import itertools
from concurrent.futures import ThreadPoolExecutor

def kgc(params):
    model_params, date_time = params
    log_dir = "logs_FB15k237"
    os.makedirs(log_dir, exist_ok=True)
    param_str = "_".join(param.strip("--") for param in model_params)
    log_file_name = f"{log_dir}/{param_str}_{date_time}.log"

    print(f"Log file: {log_file_name}")
    command = ["python", "rrr_kgc.py"] + model_params

    with open(log_file_name, 'w', encoding='utf-8') as log_file:
        result = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            print(f"Error with params {model_params}: Exit code {result.returncode}")

def main():
    dataset = 'FB15k237' #  'FB15k237' 'WN18RR' 'YAGO3-10'
    local_options =  [['--local']]
    # local_options = [[]]
    # forward_options = [['--forward'], []]
    forward_options = [['--forward'], []]
    # fsls = [0, 1, 2, 5, 8]
    fsls = [2]
    cand_nums = [3,5]
    # cand_nums = [3,5,10]
    # embeddings = ['RotatE', 'GIE', 'ComplEx']
    embeddings = ['GIE']
    start_idx = 0
    end_idx = 0 # 0 for all data
 
    ctx_combinations = [
        # (),
        # ('--wiki_ctx',),
        # ('--fs_ctx', '--wiki_ctx'),
        # ('--wiki_ctx', '--cand_ctx'),
        ('--fs_ctx', '--wiki_ctx', '--cand_ctx'),
        # ('--fs_ctx', '--cand_ctx'),
    ]
    
    parameter_sets = []
    for local in local_options:
        for forward in forward_options:
            for fsl in fsls:
                for cand_num in cand_nums:
                    for embedding in embeddings:
                        for ctx_combination in ctx_combinations:
                            parameters = local + forward + list(ctx_combination) + ['--dataset', dataset, '--embedding', embedding, '--cand_num', str(cand_num), '--start_idx', str(start_idx), '--end_idx', str(end_idx), '--fsl', str(fsl)]
                            parameter_sets.append(parameters)

    date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    tasks = [(params, date_time) for params in parameter_sets]
    with ThreadPoolExecutor(max_workers=len(parameter_sets)) as executor:
        futures = [executor.submit(kgc, task) for task in tasks]
        for future in futures:
            future.result()

if __name__ == "__main__":
    main()

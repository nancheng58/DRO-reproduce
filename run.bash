#!/bin/bash
#SBATCH -e result/Finetune-Y-sas_with-DRO-GELU-a=0.001.err
#SBATCH -o result/Finetune-Y-sas_with-DRO-GELU-a=0.001.out
#SBATCH -J DRO_sas

#SBATCH --partition=si
#SBATCH --nodelist=gpu08
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=999:00:00


conda activate torch1.10 
python run_finetune_sample.py --data_name='Yelp' --ckp=0 --hidden_size=64
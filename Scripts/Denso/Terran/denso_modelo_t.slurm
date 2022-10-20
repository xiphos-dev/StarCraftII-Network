#!/bin/bash
#SBATCH --mail-user=nicolas.pinob@sansano.usm.cl
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -e error.err
#SBATCH --time=65:00
#SBATCH --mem=15G
#SBATCH -p gpuk
#SBATCH --gres=gpu:1

use anaconda3

conda activate gpu

python script_modelo.py

source deactivate

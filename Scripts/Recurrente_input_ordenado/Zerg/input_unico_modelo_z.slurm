#!/bin/bash
#SBATCH --mail-user=nicolas.pinob@sansano.usm.cl
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -e error.err
#SBATCH --time=72:00:00
#SBATCH --mem=25G
#SBATCH -p gpuk
#SBATCH --gres=gpu:1

use anaconda3

conda activate gpu

python script_ordenado_modelo.py

source deactivate

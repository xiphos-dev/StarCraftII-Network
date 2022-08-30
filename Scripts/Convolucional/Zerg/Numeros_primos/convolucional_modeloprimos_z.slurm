#!/bin/bash
#SBATCH --mail-user=nicolas.pinob@sansano.usm.cl
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -e error.err
#SBATCH --time=24:00:00
#SBATCH --mem=40G
#SBATCH -p gpuk
#SBATCH --gres=gpu:1

use anaconda3

conda activate gpu

python script_continuado_modelo.py

source deactivate

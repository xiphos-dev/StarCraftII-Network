#!/bin/bash
#SBATCH --mail-user=nicolas.pinob@sansano.usm.cl
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -e error.err
#SBATCH --time=1:00:00
#SBATCH --mem=40G

use anaconda3

conda activate gpu

python script_red_convolucional.py

source deactivate

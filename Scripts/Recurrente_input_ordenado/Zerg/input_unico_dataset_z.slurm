#!/bin/bash
#SBATCH --mail-user=nicolas.pinob@sansano.usm.cl
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -e error.err
#SBATCH --time=12:00:00
#SBATCH --mem=15G
use anaconda3
python script_ordenado_recurrente.py

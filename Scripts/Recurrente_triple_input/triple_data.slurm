#!/bin/bash
#SBATCH --mail-user=nicolas.pinob@sansano.usm.cl
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -e error.err
#SBATCH --time=10:00
#SBATCH --mem=20G
use anaconda
python script_tripleinput_recurrente.py

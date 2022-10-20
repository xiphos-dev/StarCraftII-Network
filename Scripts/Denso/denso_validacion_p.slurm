#!/bin/bash
#SBATCH --mail-user=nicolas.pinob@sansano.usm.cl
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -e error_validacion.err
#SBATCH --time=10:00
#SBATCH --mem=15G
use anaconda3
python script_metricas_p.py

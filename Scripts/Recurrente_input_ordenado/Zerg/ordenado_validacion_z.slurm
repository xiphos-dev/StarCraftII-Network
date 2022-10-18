#!/bin/bash
#SBATCH --mail-user=nicolas.pinob@sansano.usm.cl
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -e error_validacion.err
#SBATCH --time=2:00
#SBATCH --mem=30G
use anaconda3
python script_metricas_z.py

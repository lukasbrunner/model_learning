#!/bin/bash
# SLURM specific commands
#SBATCH --job-name=model-learning
#SBATCH --ntasks=16
#SBATCH --mem=100GB
#SBATCH --time=24:00:00
#SBATCH --mail-type=END    # first have to state the type of event to occur
#SBATCH --mail-user=l.brunner@univie.ac.at   # and then your email address

module purge
module load miniconda3
/jetfs/home/lbrunner/.conda/envs/model_learning/bin/python logreg_dataset_outsample.py

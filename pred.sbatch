#!/bin/bash
#SBATCH --job-name=predslicetokeep
#SBATCH --nodes=1
#SBATCH --partition=compute-long
#SBATCH --time=10-00:00:00
#SBATCH --output=predslicetokeep.out # any screen print out will be written to this .out file
#SBATCH --error=predslicetokeep.err

python ./predslicetokeep.py




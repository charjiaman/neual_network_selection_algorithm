#!/bin/bash
#SBATCH --job-name=genslicetokeep
#SBATCH --nodes=1
#SBATCH --partition=compute-long
#SBATCH --time=10-00:00:00
#SBATCH --output=genslicetokeep.out # any screen print out will be written to this .out file
#SBATCH --error=genslicetokeep.err

python ./genslicetokeep.py




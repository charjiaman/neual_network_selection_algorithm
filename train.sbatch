#!/bin/bash
#SBATCH --job-name=trainstackslice
#SBATCH --nodes=1
#SBATCH --partition=compute-long
#SBATCH --time=30-00:00:00
#SBATCH --output=trainstackslice.out # any screen print out will be written to this .out file
#SBATCH --error=trainstackslice.err

python ./trainstackslice.py




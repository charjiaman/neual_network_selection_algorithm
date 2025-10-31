#!/bin/bash
#SBATCH --job-name=56genvardata
#SBATCH --nodes=1
#SBATCH --partition=compute-long
#SBATCH --time=3-00:00:00
#SBATCH --output=genvardata56.out # any screen print out will be written to this .out file
#SBATCH --error=genvardata56.err

python ./genvardata56.py


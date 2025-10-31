#!/bin/bash
#SBATCH --job-name=main
#SBATCH --nodes=1
#SBATCH --partition=compute-long
#SBATCH --time=72:00:00
#SBATCH --output=main.out # any screen print out will be written to this .out file
#SBATCH --error=main.err

if [ -d ./mlsubtuner ]; then
  rm -Rf ./mlsubtuner
fi

python ./main.py


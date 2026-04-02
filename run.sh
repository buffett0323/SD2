#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 2:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --output=fk_false_output_%j.log
#SBATCH --error=fk_false_error_%j.log

# load cuda
module load cuda/12.6.1

# activate environment
nvidia-smi

cd /ocean/projects/cis250260p/bliu10/anlp_final
cd vendor/dgrammar/
source .venv/bin/activate

uv pip install -r requirements.txt
uv pip install -Ue .


python bench/run_lave_timed.py 0 10 jsonschemabench 128 0
python bench/run_igcd_timed.py 0 10 jsonschemabench 128 0


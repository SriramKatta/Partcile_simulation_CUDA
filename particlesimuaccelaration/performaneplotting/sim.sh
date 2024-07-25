#!/bin/bash -l
#SBATCH --gres=gpu:a40:1
#SBATCH --time=12:30:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

mkdir -p "$WORK/$SLURM_JOBID" || exit 1
cp -v ../executable/MDsimu "$WORK/$SLURM_JOBID"

module load python
module load cmake
module load gcc
module load cuda

rm ../build/CMakeCache.txt

cmake -S .. -B ../build -Dcutoff=ON
cmake --build ../build/ -j

ifname="inputsim3"

python3 inputfilegenerator.py 10000 0.001 5 > $ifname
cp -v $ifname "$WORK/$SLURM_JOBID"
cd  "$WORK/$SLURM_JOBID"

srun ./MDsimu ./$ifname /home/hpc/hesp/hesp101h/particlesimuaccelaration/performaneplottingoutputvtk4cutoff/

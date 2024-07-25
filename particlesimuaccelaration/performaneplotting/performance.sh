#!/bin/bash -l
#SBATCH --gres=gpu:a40:1
#SBATCH --time=12:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load cmake
module load gcc
module load cuda
module load python

value=20

mkdir newdat

for name in nbdacc cutoff
do
rm ../build/CMakeCache.txt
cmake -S .. -B ../build -D$name=ON -Dperfana=ON
cmake --build ../build/ -j
echo "#N meantimeperstep" > ./newdat/sim-dat$name
for ((i=5; i <= value; i+=2))
do 
  ifname=res_points
  echo "$i start"
  python3 inputfilegenerator.py 5000 0.05 $i > $ifname
  srun ../executable/MDsimu ./$ifname ./outputvtk/ | tail -n 1 >> ./newdat/sim-dat$name
  echo "$i done"
  rm $ifname
done
done

gnuplot plotjob2.sh

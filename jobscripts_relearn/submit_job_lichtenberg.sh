#!/bin/bash

#SBATCH -J relearn
#SBATCH -e ./stderr.relearn.%j.txt
#SBATCH -o ./stdout.relearn.%j.txt
#SBATCH -C avx512
#SBATCH -n 1
#SBATCH --mem-per-cpu=1024
#SBATCH --time=1440
#SBATCH --cpus-per-task=12
#SBATCH --account=special00001

if [ "$#" -ne 3 ]; then
	echo "Must supply 3 arguments:"
    echo "The target calcium value, the beta value, the growth rate"
	exit 1
fi

module r

echo "This is job $SLURM_JOB_ID"
echo "Testing Relearn Network"
echo $1 $2 $3

mkdir ./output/

./relearn \
--log-path ./output/ \
--steps 1000000 \
--algorithm barnes-hut \
--openmp 12 \
--theta 0.3 \
--file ./input/positions.txt \
--graph ./input/network.txt \
--synaptic-elements-lower-bound 0.3 \
--synaptic-elements-upper-bound 0.7 \
--target-ca $1 \
--beta $2 \
--growth-rate $3

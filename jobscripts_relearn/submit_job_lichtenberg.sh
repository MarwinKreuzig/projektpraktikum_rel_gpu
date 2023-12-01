#!/bin/bash

#SBATCH -J relearn_benchmark
#SBATCH -e ./stderr.relearn.%j.txt
#SBATCH -o ./stdout.relearn.%j.txt
#SBATCH -C avx512
#SBATCH -n 1
#SBATCH --mem-per-cpu=1024
#SBATCH --time=1440
#SBATCH --cpus-per-task=12
#SBATCH --account=project02279

cd ../relearn/cmake-build-debug/bin/

./relearn_benchmarks

#!/bin/bash
#SBATCH -J relearn_benchmark
#SBATCH -e ./stderr.relearn.%j.txt
#SBATCH -o ./stdout.relearn.%j.txt
#SBATCH -C avx512
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem-per-cpu=4000
#SBATCH -t 00:05:00
#SBATCH -A project02279
#SBATCH --gres=gpu

cd ../relearn/build/bin

./relearn_tests

./relearn_tests_cuda

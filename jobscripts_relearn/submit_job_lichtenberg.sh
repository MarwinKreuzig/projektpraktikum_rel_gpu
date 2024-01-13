#!/bin/bash
#SBATCH -J relearn_benchmark
#SBATCH -e /work/home/fk15toda/BP/Relearn_new/BP-RELeARN-GPU/jobscripts_relearn/stderr.relearn.%j.txt
#SBATCH -o /work/home/fk15toda/BP/Relearn_new/BP-RELeARN-GPU/jobscripts_relearn/stdout.relearn.%j.txt
#SBATCH -C avx512
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem-per-cpu=1024
#SBATCH -t 00:05:00
#SBATCH -A project02279
#SBATCH --gres=gpu

cd /work/home/fk15toda/BP/Relearn_new/BP-RELeARN-GPU/relearn/build/bin

./relearn_tests


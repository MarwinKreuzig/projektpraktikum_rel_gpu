#!/bin/bash
#SBATCH -A extension00001
#SBATCH -J relearn_procs-8_neurons-51842_steps-6000000_neuronPositions-input_positions.txt
#SBATCH -e /home/sr42myny/RELEARN/Klinikum-TUM/joboutput/relearn_procs-8_neurons-51842_steps-6000000_neuronPositions-input_positions.txt.%j.err
#SBATCH -o /home/sr42myny/RELEARN/Klinikum-TUM/joboutput/relearn_procs-8_neurons-51842_steps-6000000_neuronPositions-input_positions.txt.%j.out
#SBATCH -t 12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rinke@cs.tu-darmstadt.de
#SBATCH --exclusive
#SBATCH -N 4
#SBATCH -n 8
#SBATCH --mem-per-cpu=1600
#SBATCH --cpus-per-task=1
#SBATCH -C "mpi"
 
module purge
module add gcc/8.3.0 openmpi/4.0.1
cd /home/sr42myny/RELEARN/Klinikum-TUM/src.git/relearn/
for n in `seq 1`; do
    mpiexec -np 8 ./relearn 0.3 51842 5489 6000000 input/positions.txt
done

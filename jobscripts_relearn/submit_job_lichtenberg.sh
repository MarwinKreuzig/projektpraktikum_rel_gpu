#!/bin/bash

if test "$#" -ne 5; then
    echo "Usage: $0 <num procs> <num nodes> <neurons per proc> <simulation steps> <num repetitions> [<file with neuron positions>]"
    exit
fi

num_procs=$1
num_nodes=$2
neurons_per_proc=$3
num_sim_steps=$4
num_repetitions=$5
file_with_neuron_positions=$6

home=/home/sr42myny/RELEARN/Klinikum-TUM
bindir=$home/src.git
outputdir=$home/joboutput
prefix="relearn"

num_neurons=$((neurons_per_proc * num_procs))

name=${prefix}_procs${num_procs}_neurons${num_neurons}_steps${num_sim_steps}
job=${name}.sbatch

echo "#!/bin/bash"                                          >  $job
echo "#SBATCH -A extension00001"                            >> $job
echo "#SBATCH -J $name"                                     >> $job
echo "#SBATCH -e $outputdir/${name}.%j.err"                 >> $job
echo "#SBATCH -o $outputdir/${name}.%j.out"                 >> $job
echo "#SBATCH -t 24:00:00"                                  >> $job
echo "#SBATCH --mail-type=END,FAIL"                         >> $job
echo "#SBATCH --mail-user=rinke@cs.tu-darmstadt.de"         >> $job
echo "#SBATCH --exclusive"                                  >> $job
echo "#SBATCH -N $num_nodes"                                >> $job
echo "#SBATCH -n $num_procs"                                >> $job
echo "#SBATCH --mem-per-cpu=1600"                           >> $job
echo "#SBATCH --cpus-per-task=1"                            >> $job
echo "#SBATCH -C \"avx&mpi\""                               >> $job
#echo "#SBATCH -C \"avx&mpi&multi\""                         >> $job
echo " "                                                    >> $job
echo "module purge"                                         >> $job
echo "module add gcc/8.3.0 openmpi/4.0.1"                   >> $job
echo "cd $bindir"                                           >> $job
echo "for n in \`seq $num_repetitions\`; do"                >> $job
echo "    mpiexec -np $num_procs ./relearn 0.3 $num_neurons 5489 $num_sim_steps $file_with_neuron_positions"   >> $job
echo "done"                                                 >> $job

#sbatch $job

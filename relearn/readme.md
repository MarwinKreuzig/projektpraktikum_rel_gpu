# Usage
./relearn --steps 200000 --random-seed 0 --theta 0.3 --num-neurons 100

./relearn --steps 200000 --random-seed 0 --theta 0.3 --file ../../input/positions.txt --graph ../../input/network.txt

#### You can also prefix the commands to accomplish parallelization with MPI: 
mpiexec -n 4 ...
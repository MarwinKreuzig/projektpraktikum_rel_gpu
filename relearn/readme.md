
# Usage
The program offers multiple command line arguments. `--steps` and either `--num-neurons` or `--file` are required, the rest is optional.

Command | Shortcut | Effect | Notes
--- | --- | --- | ---
`--num-neurons <size_t>` | `-n <size_t>` | Starts a simulation with the given number of neurons | Excludes `--file`
`--file <file-path>`| `-f <file-path>` | Starts a simulation with the neurons from the specified file | Excludes `--num-neurons`
`--graph <file-path>` | `-g <file-path>` | Starts a simulation with an initial connectivity from the specified file | Requires `--file`
`--enable-interrupts <file-path>` | | Specifies the enable interrupts for a simulation | 
`--disable-interrupts <file-path>` | | Specified the disable interrupts for a simulation |
`--creation-interrupts <file-path>` | | Specified the creation interrupts for a simulation |
`--log-prefix <str>` | `-p <str>` | Prefixes all log files with the given prefix |
`--log-path <dir-path>` | `-l <dir-path>` | Specified the directory in which the log files will be created |
`--steps <size_t>` | `-s <size_t>` | Specifies the number of steps in the simulation |
`--random-seed <uint32>` | `-r <uint32>` | The seed for the random number  generator |
`--openmp <int32>` | | Sets the number of OpenMP Threads | Must be greater than 0
`--theta <double>` | `-t <double>` | Sets the acceptance criterion for the Barnes-Hut-Algorithm | Must be from [0, 0.5]
`--interactive` | `-i` | Allows an interactive run (does not stop after the number of steps is completed) |
`--synaptic-elements-lower-bound <double>` | | Sets the lower bound of initial synaptic elements | Must be from [0, $\infty$)
`--synaptic-elements-upper-bound <double>` | | Sets the upper bound of initial synaptic elements | Must be from [0, $\infty$) and not less than the lower bound
`--target-ca <double>` | | Sets the target calcium for all neurons | Must be from [0, 100]
`--growth-rate <double>` | | Specifies the growth rate of the synaptic elements | Must be from [0, 1]
`--beta` | | Specifies the amount of calcium ions that are gathered whenever a neuron spikes |


### In case you have compiled the program with MPI, you can also prefix the command with: `mpiexec -n 4 ...`

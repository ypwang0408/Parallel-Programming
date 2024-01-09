To reproduce our code, typing `make` in the `code` directory should build the code and create multiple executables in `/bin`

Type `mpicc -g -Wall mpi.c -o ./bin/mpi.out -O3` to create executable for mpi

Run `python3 ./test-*.py` to run our test script for each algorithm.

from subprocess import call
import time

for p in range(1, 9):
	for d in [960, 1920, 3840, 5760, 7680, 9600, 11520]:
		call(["mpirun",  "-np",  str(p), "./bin/mpi.out",  str(d), "100"])
		print("P=" + str(p) + "   D=" + str(d) + "\n")
		print("\n\n\n\n")
		time.sleep(2)

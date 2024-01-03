from subprocess import call
import time

with open("./result/omp_block_result.txt", "w") as file:
    file.flush()

for t in range(8, 9):
    for block_sz in [16, 32, 48, 96, 120, 160]:
        for d in [960, 1920, 3840, 5760, 7680, 9600, 11520]:
            call(["./bin/openmp_block.out", str(d), "100", str(t), str(block_sz)])
            print("Result when:  T=" + str(t) + "   D=" + str(d) + "\n")
            time.sleep(1)
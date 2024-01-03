from subprocess import call
import time

with open("./result/cuda_block_result.txt", "w") as file:
    file.flush()

for SZ in [2, 4, 8, 16, 24, 32]:
    for d in [960, 1920, 3840, 5760, 7680, 9600, 11520]:
        call(["./bin/cuda_block_" + str(SZ) + ".out", str(d), "100"])
        print("Result when:  B=" + str(SZ**2) + "   D=" + str(d) + "\n")
        time.sleep(1)
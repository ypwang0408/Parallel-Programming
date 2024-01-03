from subprocess import call
import time

with open("./result/cuda_result.txt", "w") as file:
    file.flush()

for b in [32, 64, 128, 256, 512, 1024]:
    for d in [960, 1920, 3840, 5760, 7680, 9600, 11520]:
        call(["./bin/cuda.out", str(d), "100", str(b)])
        print("Result when:  B=" + str(b) + "   D=" + str(d) + "\n")
        time.sleep(1)

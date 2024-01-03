from subprocess import call
import time

with open("./result/sequential_result.txt", "w") as file:
    file.flush()

for d in [960, 1920, 3840, 5760, 7680, 9600, 11520]:
    call(["./bin/sequential.out", str(d), "100"])
    print("Result when:  D=" + str(d) + "\n")
    time.sleep(1)

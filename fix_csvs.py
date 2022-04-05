import glob
import os
import sys

pth = sys.argv[1]
for file in sorted(glob.glob(pth + "*.csv"), key=lambda x: int(x.split("_")[-1][:-4])):
    print("file: ", file)
    with open(file) as f:
        data = f.read()
    data = data.replace("steps", "steps\n")
    lines = data.splitlines()
    new_lines_1 = lines[:2]
    new_lines_2 = lines[2:]
    second_line = new_lines_1.pop(-1)
    vals = second_line.split(",")
    second_line = ",".join(vals[:6] + [vals[6][:-1]])
    third_line = ",".join([vals[6][-1]] + vals[7:])
    new_lines_1.append(second_line)
    new_lines_1.append(third_line)

    all_lines = new_lines_1 + new_lines_2

    with open(file, "w") as f:
        f.write("\n".join(all_lines))

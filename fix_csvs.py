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
    all_lines = []
    all_lines += new_lines_1
    vals = second_line.split(",")
    for i in range(0, int((len(vals) - 1) / 6), 12):
        line = ",".join(vals[i : i + 6] + [vals[i + 6][:-1]])
        line_beg = line.split(",")[0][-1]
        line_total = line_beg + "," + ",".join(line.split(",")[1:])
        all_lines += [line_total]
    all_lines += new_lines_2
    new_dir = os.path.join("/".join(file.split("/")[:-1]), "fixed")
    new_file = os.path.join(new_dir, file.split("/")[-1])
    os.makedirs(new_dir, exist_ok=True)
    with open(new_file, "w") as f:
        f.write("\n".join(all_lines))

import numpy as np
import re

def read_data(data_path):

    with open(data_path) as f:
        lines = f.readlines()

    count = 0
    data_dict = []
    while count < len(lines):
        if lines[count].find(":") == -1:
            count += 1
            step_dict = {}
            
            if count >= len(lines): continue
            while count < len(lines) and lines[count].find(":") != -1:
                
                line = lines[count]
                data_name = line[:line.find(":")]
                data_string = (re.sub("[^e.0-9\s+\-]", "", line))
                step_dict[data_name] = np.fromstring(data_string, sep=" ")
            
                count += 1
            data_dict.append(step_dict)
        else:
            count += 1

    return data_dict

        

def main():
    data_dict = read_data('data/couple_noise.txt')

if __name__=="__main__":
    main()

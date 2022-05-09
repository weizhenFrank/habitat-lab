from torch.utils.data import Dataset
import numpy as np
import re
import torch

class VelocityDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # store the inputs and outputs
        print('Loading data from ' + path)
        data_dict = self.read_data(path)
        
        input_data = ['init ang vel', 'init lin vel', 'cmd']
        input_dim = 0
        for ke in input_data:
            input_dim += len(data_dict[0][ke])
            
        self.X = torch.zeros((len(data_dict), input_dim)) # init ang vel, init lin vel, cmd
        self.y = torch.zeros((len(data_dict), 3)) # x,y,w
        
        for i, data in enumerate(data_dict):
            
            
            input_vec = data[input_data[0]]
            for ke in input_data[1:]:
                input_vec = np.concatenate((input_vec, data[ke]), axis=0)
            
            
            self.X[i,:] = torch.from_numpy(input_vec)
            self.y[i,:] = torch.from_numpy(data['cur lin vel'])
            
        print('Data loaded. ' + str(len(data_dict)) + ' data points')
    def __len__(self):
        return len(self.X)
 
    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
    
                                
    def read_data(self, data_path):
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
                    line = line.replace('vel', '')
                    line = line.replace('Vector', '')
                    data_string = (re.sub("[^e.0-9\s+\-]", "", line))
                    step_dict[data_name] = np.fromstring(data_string, sep=" ")

                    count += 1
                data_dict.append(step_dict)
            else:
                count += 1

        return data_dict

        


import torch
import numpy as np
from model import MLP
from matplotlib import pyplot as plt
import sys

data_file = sys.argv[1]

input_size = 7
model = MLP(input_size, 32, 3)
model.load_state_dict(torch.load('models/' + data_file + '.pt'))
model.eval()


N = 100
vel_range = torch.linspace(-0.5, 0.5, 100)
x_dir_cmd = torch.zeros((100, input_size))
x_dir_cmd[:, 4] = vel_range
vel_output_x_cmd = model(x_dir_cmd)

y_dir_cmd = torch.zeros((100, input_size))
y_dir_cmd[:, 5] = vel_range
vel_output_y_cmd = model(y_dir_cmd)

fig, ax = plt.subplots(2,2)

ax[0,0].scatter(vel_range.detach().numpy(), vel_output_x_cmd[:, 0].detach().numpy(), c = 'k', s = 0.5)
ax[0,0].set_xlabel('Commanded X Velocity')
ax[0,0].set_ylabel('Realized X Velocity')
ax[0,0].grid()

ax[0,1].scatter(vel_range.detach().numpy(), vel_output_x_cmd[:, 1].detach().numpy(), c = 'k', s = 0.5)
ax[0,1].set_xlabel('Commanded X Velocity')
ax[0,1].set_ylabel('Realized Y Velocity')
ax[0,1].grid()

ax[1,0].scatter(vel_range.detach().numpy(), vel_output_y_cmd[:, 0].detach().numpy(), c = 'k', s = 0.5)
ax[1,0].set_xlabel('Commanded Y Velocity')
ax[1,0].set_ylabel('Realized X Velocity')
ax[1,0].grid()

ax[1,1].scatter(vel_range.detach().numpy(), vel_output_y_cmd[:, 1].detach().numpy(), c = 'k', s = 0.5)
ax[1,1].set_xlabel('Commanded Y Velocity')
ax[1,1].set_ylabel('Realized Y Velocity')
ax[1,1].grid()

fig.tight_layout()

plt.savefig('results/' + data_file + '.png')

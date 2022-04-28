
try:
    #import habitat_sim
    from habitat_sim.bindings import RigidState
    from habitat_sim.physics import VelocityControl
except ImportError:
    pass
import numpy as np
import magnum as mn
from read_data import read_data
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal as mvn

vel_control = VelocityControl()
vel_control.controlling_lin_vel = True
vel_control.controlling_ang_vel = True
vel_control.lin_vel_is_local = True
vel_control.ang_vel_is_local = True


data_dict = read_data('data/couple_noise.txt')

disp_errs = np.zeros((len(data_dict), 3))

for i,data in enumerate(data_dict):
    init_pos = data['init pos']
    init_quat = data['init quat xyzw']
    cmd_vel = data['cmd']

    lin_vel = cmd_vel[0]
    hor_vel = cmd_vel[1]
    ang_vel = cmd_vel[2]
    time_step = cmd_vel[3]

    vel_control.linear_velocity = np.array([hor_vel, 0.0, lin_vel])
    vel_control.angular_velocity = np.array([0.0, ang_vel, 0.0])

    current_rigid_quat = mn.Quaternion()
    current_rigid_quat.vector = mn.Vector3(init_quat[1],init_quat[2],init_quat[0])
    current_rigid_quat.scalar = init_quat[-1]
    
    current_rigid_state = RigidState(
            current_rigid_quat,
            mn.Vector3(init_pos[1],init_pos[2],init_pos[0] ),
        )
    goal_rigid_state = vel_control.integrate_transform(
        time_step, current_rigid_state
    )


    cmd_disp = np.array([   goal_rigid_state.translation.x, 
                            goal_rigid_state.translation.y, 
                            goal_rigid_state.translation.z])

    true_disp = np.array([data['cur pos'][1],
                    data['cur pos'][2],
                    data['cur pos'][0]])

    disp_errs[i, :] = true_disp - cmd_disp

plt.scatter(disp_errs[:, 0], disp_errs[:,2],c='k', s=0.5)
plt.title('Displacement Error for (x,y,w) Movement')
plt.xlabel('X Displacement Error')
plt.ylabel('Y Displacement Error')

sig = np.cov(disp_errs[:, [0,2]].T)
mu = np.mean(disp_errs[:, [0,2]], axis=0)

plt.grid()
N = 100
x1 = np.linspace(-0.5, 0.5, num=N)
x1,x2 = np.meshgrid(x1,x1)
con = mvn(mean=mu, cov=sig)
vals = np.concatenate((x1.flatten()[:,np.newaxis], x2.flatten()[:,np.newaxis]), axis=1)

f = con.pdf(vals)

plt.contour(x1, x2, np.reshape(f, (N,N)))

plt.savefig("output.png")

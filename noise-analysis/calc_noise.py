
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


collect_types = ['couple_noise', 'data_w', 'data_x', 'data_y']
file_name = collect_types[3]
data_dict = read_data('data/' + file_name  + '.txt')

disp_errs = np.zeros((len(data_dict), 3))
ang_errs = np.zeros((len(data_dict)))
pose = np.zeros((len(data_dict), 3))

for i,data in enumerate(data_dict):
    if data == {}: 
        print(i)
        break
    init_pos = data['init pos']
    init_quat = data['init quat xyzw']
    cur_quat = data['cur quat xyzw']
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

    true_final_rot = mn.Quaternion()
    true_final_rot.vector = mn.Vector3(cur_quat[1],cur_quat[2],cur_quat[0])
    true_final_rot.scalar = cur_quat[-1]

    current_rigid_state = RigidState(
            current_rigid_quat,
            mn.Vector3(init_pos[1],init_pos[2], init_pos[0] ),
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

    q0g = goal_rigid_state.rotation.scalar
    q1g = goal_rigid_state.rotation.vector.x
    q2g = goal_rigid_state.rotation.vector.y
    q3g = goal_rigid_state.rotation.vector.z

    q0t = true_final_rot.scalar
    q1t = true_final_rot.vector.x
    q2t = true_final_rot.vector.y
    q3t = true_final_rot.vector.z

    q0i = current_rigid_quat.scalar
    q1i = current_rigid_quat.vector.x
    q2i = current_rigid_quat.vector.y
    q3i = current_rigid_quat.vector.z



    psi_goal = np.arctan( (2 * (q0g * q3g + q1g * q2g)) / (1 - 2 * (q2g * q2g + q3g * q3g))) / np.pi * 180
    psi_true = np.arctan( (2 * (q0t * q3t + q1t * q2t)) / (1 - 2 * (q2t * q2t + q3t * q3t))) 

    theta_goal = np.arcsin( (2 * (q0g * q2g + q3g * q1g) ) )/ np.pi * 180
    theta_true = np.arcsin( (2 * (q0t * q2t + q3t * q1t) ) )/ np.pi * 180
    theta_init = -np.arcsin( (2 * (q0i * q2i + q3i * q1i) ) )

    phi_goal = np.arctan( (2 * (q0g * q1g + q2g * q3g)) / (1 - 2 * (q2g * q2g + q1g * q1g)))/ np.pi * 180
    phi_true = np.arctan( (2 * (q0t * q1t + q2t * q3t)) / (1 - 2 * (q2t * q2t + q1t * q1t)))/ np.pi * 180

    

    R = np.array([[np.cos(theta_init), -np.sin(theta_init)], [np.sin(theta_init), np.cos(theta_init)]])
    
    disp_errs[i, :] = true_disp - cmd_disp
    disp_errs[i, [0,2]] = R @ disp_errs[i, [0,2]]

    ang_errs[i] = theta_goal - theta_true
    pose[i, :] = cmd_disp



plt.subplot(2,1,1)

plt.scatter(disp_errs[:, 0], disp_errs[:,2],c='k', s=0.5)



plt.xlabel('X Displacement Error')
plt.ylabel('Y Displacement Error')

sig_xy = np.cov(disp_errs[:, [0,2]].T)
mu_xy = np.mean(disp_errs[:, [0,2]], axis=0)

mu_w = np.mean(ang_errs)
var_w = np.var(ang_errs)

print('Params for ' + file_name)
print('Mean (X,Y): ' + str(mu_xy))
print('Variance X: ' + str((np.abs(sig_xy))[0,0]) + ' (std: ' +  str(np.sqrt(sig_xy[0,0])) )
print('Variance Y: ' + str((np.abs(sig_xy))[1,1]) + ' (std: ' +  str(np.sqrt(sig_xy[1,1])) )

print('Mean (w degrees): ' + str(mu_w))
print('Variance (w degrees): ' + str(var_w) + ' (std: ' +  str(np.sqrt(var_w)) )



plt.title(r'Error for ' + file_name) # $\mu=$' + str(mu) + ' $\sigma_{x}=$' + str(sig[0]) + ' $\sigma_{y}=$' + str(sig[1,1]))

plt.grid()
N = 100
x1 = np.linspace(-0.1, 0.1, num=N)
x1,x2 = np.meshgrid(x1,x1)
con = mvn(mean=mu_xy, cov=sig_xy)
vals = np.concatenate((x1.flatten()[:,np.newaxis], x2.flatten()[:,np.newaxis]), axis=1)

f = con.pdf(vals)

# plt.contour(x1, x2, np.reshape(f, (N,N)))

print(disp_errs.shape)
plt.axis('equal')
plt.subplot(2,1,2)

plt.hist(ang_errs, bins=40)
plt.savefig(file_name + ".png")


# plt.subplot(1,3,1)
# plt.scatter(pose[:, 0], pose[:, 1])
# plt.axis('equal')
# plt.subplot(1,3,2)
# plt.scatter(pose[:, 0], pose[:, 2])
# plt.axis('equal')
# plt.subplot(1,3,3)
# plt.scatter(pose[:, 1], pose[:, 2])
# plt.axis('equal')
# plt.savefig(file_name + "_pose.png")

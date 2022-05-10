
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


collect_types = ['data_x', 'data_y', 'couple_noise', 'noise'] #'kinematic_real','habitat_dynamic_runs','gibson_kinematic_runs','habitat_kinematic_runs','couple_noise', 'data_w', 'data_x', 'data_y', 'noise']

# file_name = collect_types[-1]

for file_name in collect_types:
    print(file_name)
    data_dict = read_data('data/' + file_name  + '.txt')

    disp_errs = np.zeros((len(data_dict), 3))
    ang_errs = np.zeros((len(data_dict)))
    pose = np.zeros((len(data_dict), 3))
    cmd_vels = np.zeros((len(data_dict), 2))

    for i,data in enumerate(data_dict):
   
        if data == {}: 
            print(i)
            break
        if 'cmd' in list(data.keys()):
            init_pos = data['init pos']
            init_quat = data['init quat xyzw']
            cur_quat = data['cur quat xyzw']
            cmd_vel = data['cmd']
            init_vel = data['init lin vel']


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
            psi_true = np.arctan( (2 * (q0t * q3t + q1t * q2t)) / (1 - 2 * (q2t * q2t + q3t * q3t))) / np.pi * 180
            psi_init = np.arctan( (2 * (q0i * q3i + q1i * q2i)) / (1 - 2 * (q2i * q2i + q3i * q3i))) 

            theta_goal = np.arcsin( (2 * (q0g * q2g + q3g * q1g) ) )/ np.pi * 180
            theta_true = np.arcsin( (2 * (q0t * q2t + q3t * q1t) ) )/ np.pi * 180
            theta_init = -np.arcsin( (2 * (q0i * q2i + q3i * q1i) ) )

            phi_goal = np.arctan( (2 * (q0g * q1g + q2g * q3g)) / (1 - 2 * (q2g * q2g + q1g * q1g)))/ np.pi * 180
            phi_true = np.arctan( (2 * (q0t * q1t + q2t * q3t)) / (1 - 2 * (q2t * q2t + q1t * q1t)))/ np.pi * 180
            phi_init = np.arctan( (2 * (q0i * q1i + q2i * q3i)) / (1 - 2 * (q2i * q2i + q1i * q1i)))

            
            yaw = theta_init
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
       
            disp_errs[i, :] = true_disp - cmd_disp
            disp_errs[i, [0,2]] = R @ disp_errs[i, [0,2]]

            ang_errs[i] = theta_goal - theta_true
            pose[i, :] = cmd_disp
            cmd_vels[i, :] = np.array([lin_vel, hor_vel]) - init_vel[:2]
            print(np.array([lin_vel, hor_vel]))
            print(cmd_vels[i, :])
            print('-------------------')




    # Swapped direction to be consistent with velocity
    disp_errs[:, [0,2]] = disp_errs[:, [2,0]]

    # ax[0,0].subplot(2,1,1)
    plt.clf()
    colors = np.linalg.norm(cmd_vels, axis=1)
    bounds_x = np.arange(-0.5,0.51,0.1)
    bounds_y = bounds_x

    mus_x = np.zeros((bounds_x.shape[0]-1,))
    vars_x = np.zeros((bounds_x.shape[0]-1,))

    for i,lower_bound in enumerate(bounds_x[:-1]):
        colors = np.linalg.norm(cmd_vels, axis=1)
        vel_mask = (cmd_vels[:, [0]] < bounds_x[i+1]) & (cmd_vels[:, [0]] > lower_bound)
        inrange_vel = disp_errs[vel_mask[:,0], :] if  np.sum(vel_mask) > 0 else np.zeros((3,3))
        
        sig_xy = np.cov(inrange_vel[:, [0,2]].T)
        mu_xy = np.mean(inrange_vel[:, [0,2]], axis=0)
        mus_x[i] = mu_xy[0]
        vars_x[i] = sig_xy[0,0]
        #plt.scatter(inrange_vel[:, 0], inrange_vel[:,2], s=1)

    mus_y = np.zeros((bounds_y.shape[0]-1,))
    vars_y = np.zeros((bounds_y.shape[0]-1,))

    for i,lower_bound in enumerate(bounds_y[:-1]):
        colors = np.linalg.norm(cmd_vels, axis=1)
        vel_mask = (cmd_vels[:, [1]] < bounds_y[i+1]) & (cmd_vels[:, [1]] > lower_bound)
        
        inrange_vel = disp_errs[vel_mask[:,0], :] if np.sum(vel_mask) > 0 else np.zeros((3,3))
       
        sig_xy = np.cov(inrange_vel[:, [0,2]].T)
        mu_xy = np.mean(inrange_vel[:, [0,2]], axis=0)
        mus_y[i] = mu_xy[1]
        vars_y[i] = sig_xy[1,1]
        #plt.scatter(inrange_vel[:, 0], inrange_vel[:,2], s=1)
       


    fig,ax = plt.subplots(2,2)
    fig.suptitle(file_name)
    ax[0,0].set_title('Variance (x)')
    ax[0,0].plot(np.power(bounds_x[:-1], 1),vars_x)
    ax[0,0].set_xlabel('Velocity Range Squared (x)')
    ax[0,0].grid()

    ax[0,1].set_title('Mean (x)')
    ax[0,1].plot(bounds_x[:-1],mus_x)
    ax[0,1].set_xlabel('Velocity Range (x)')
    ax[0,1].grid()

    ax[1,0].plot(np.power(bounds_y[:-1], 1), vars_y)
    ax[1,0].set_title('Variance (y)')
    ax[1,0].set_xlabel('Velocity Range Squared (y)')
    ax[1,0].grid()

    ax[1,1].plot(bounds_y[:-1],mus_y)
    ax[1,1].set_title('Mean (y)')
    ax[1,1].set_xlabel('Velocity Range (y)')
    ax[1,1].grid()
        
    fig.tight_layout()
    fig.savefig('results/' + file_name + "_conditional.png")
        
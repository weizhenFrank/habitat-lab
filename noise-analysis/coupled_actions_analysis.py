
try:
    #import habitat_sim
    from habitat_sim.bindings import RigidState
    from habitat_sim.physics import VelocityControl
except ImportError:
    pass
import numpy as np
import magnum as mn
from read_data import read_data

vel_control = VelocityControl()
vel_control.controlling_lin_vel = True
vel_control.controlling_ang_vel = True
vel_control.lin_vel_is_local = True
vel_control.ang_vel_is_local = True
time_step = 1




data_dict = read_data('data/data_pos_x.txt')
print(len(data_dict))
for data in data_dict:
    init_pos = data['init pos']
    init_quat = data['init quat xyzw']
    cmd_vel = data['cmd']

    lin_vel = cmd_vel[0]
    hor_vel = cmd_vel[1]
    ang_vel = cmd_vel[2]
    time_step = cmd_vel[3]

    vel_control.linear_velocity = np.array([hor_vel*0, 0.0, -lin_vel])
    vel_control.angular_velocity = np.array([0.0, 0.0*ang_vel, 0.0])

    current_rigid_quat = mn.Quaternion()
    current_rigid_quat.vector = mn.Vector3(*init_quat[:3])
    current_rigid_quat.scalar = init_quat[-1]
    
    current_rigid_state = RigidState(
            current_rigid_quat,
            mn.Vector3(init_pos[1],init_pos[2],-init_pos[0] ),
        )
    goal_rigid_state = vel_control.integrate_transform(
        time_step, current_rigid_state
    )

    print(goal_rigid_state.translation)







# vel_control.linear_velocity = np.array([hor_vel, 0.0, -lin_vel])
# vel_control.angular_velocity = np.array([0.0, ang_vel, 0.0])

# goal_rigid_state = self.vel_control.integrate_transform(
#     time_step, current_rigid_state
# )
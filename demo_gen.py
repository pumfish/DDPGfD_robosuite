import os
import sys
import joblib
import datetime
import argparse
sys.path.append('/workspace/S/heguanhua2/robot_rl/robosuite_jimu')

import numpy as np
import robosuite as suite
import robosuite.macros as macros
from robosuite.controllers import load_controller_config
from robosuite.utils import camera_utils as CU
from robosuite.utils import transform_utils as TU
from PIL import Image
import cv2
import random

# if only use colors, set "opencv"
macros.IMAGE_CONVENTION = "opencv"


from OpenGL import GL
def ignore_gl_errors(*args, **kwargs):
    pass
GL.glCheckError = ignore_gl_errors



def imgs2video(imgs, video_dir, fps=20):
    assert len(imgs) != 0
    frame = imgs[0]
    h, w, l = frame.shape
    video = cv2.VideoWriter(video_dir, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for img in imgs:
        video.write(img)
    video.release()


def fetch_obs(env, obs):
    # original observation info
    eef_pos = obs['robot0_eef_pos']
    eef_quat = obs['robot0_eef_quat']
    gripper_qpos = obs['robot0_gripper_qpos']

    # get cube position
    achieved_goal, desired_goal = env.get_cube_pos()

    return np.r_[eef_pos, eef_quat, gripper_qpos,
                 achieved_goal, desired_goal]


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num', default=20, type=int)
parser.add_argument('-v', '--view', default='agentview', choices=['frontview', 'agentview', 'birdview'])
args = parser.parse_args()

controller_config = load_controller_config(default_controller="OSC_POSE")
controller_config['control_delta']=False
controller_config['uncouple_pos_ori']=False

# create environment instance
env = suite.make(
    #env_name="NutAssembly", # try with other tasks like "Stack" and "Door"
    #env_name="PickPlace", # try with other tasks like "Stack" and "Door"
    #env_name="Mstt", # try with other tasks like "Stack" and "Door"
    env_name="Jimu", # try with other tasks like "Stack" and "Door"
    #env_name="Sunmao", # try with other tasks like "Stack" and "Door"
    #env_name="Stack", # try with other tasks like "Stack" and "Door"
    robots="UR5e",  # try with other robots like "Sawyer" and "Jaco"
    gripper_types = "default",
    controller_configs = controller_config,

    #has_renderer=True,
    #has_offscreen_renderer=False,
    #use_camera_obs = False,
    #render_camera = "frontview",
    #control_freq = 20,

    has_renderer=False,
    has_offscreen_renderer=True,
    render_gpu_device_id=0,
    horizon=500,
    render_camera="frontview",
    use_object_obs=False,
    use_camera_obs=True,
    control_freq=6,
    camera_depths=True,
    camera_heights= 720,
    camera_widths=1280,
    reward_shaping=True,
)

obs = env.reset()

# set demo save path
demo_dir = os.path.join(os.getcwd(), 'demos')
os.makedirs(demo_dir, exist_ok=True)
current_time = datetime.datetime.now()
save_dir = os.path.join(demo_dir, current_time.strftime("%m%d%H%M%S"))
os.makedirs(save_dir, exist_ok=True)
print(f"generated demo saved in {save_dir}")

demo_list = []

# reset the environment
import time
#start_time = time.time()
obs=env.reset()
curr_state = fetch_obs(env, obs)
action_ori=TU.quat2axisangle(TU.mat2quat(env.robots[0].controller.ee_ori_mat))#TU.quat2axisangle(rotation_world)
action=np.zeros(7)
action[0:3]=obs['robot0_eef_pos']
action[3:6]=action_ori
# print("ori pos", action[0:3], "ori mat", env.robots[0].controller.ee_ori_mat, "ori angle", action[3:6])

# initial_steps = 100
# for i in range(initial_steps):
#     obs, reward, done, info = env.step(action)  # take action in the environment
#     # next_state = fetch_obs(env, obs)
#     # demo_list.append((last_state, action, reward, next_state))
#     # curr_state = next_state
#     curr_state = fetch_obs(env, obs)


reaching_steps = args.num
picking_steps = args.num
gripper_steps = args.num // 2  # twice
lifting_steps = args.num // 2  # twice
moving_steps = args.num
placing_steps = args.num


pick_pos = env.sim.data.body_xpos[env.sim.model.body_name2id(env.jimu_tgt_cubes[-1].root_body)]
cubeB_pos_ranges = env.tgt_cube_poses[-1]
cubeB_pos = [
    (cubeB_pos_ranges[0][0]+cubeB_pos_ranges[0][1])/2,
    (cubeB_pos_ranges[1][0]+cubeB_pos_ranges[1][1])/2,
    (cubeB_pos_ranges[2][0]+cubeB_pos_ranges[2][1])/2-0.02,
             ]
place_pos = np.array(cubeB_pos)


def rotation_matrix(rot_angle, axis = 'z'):
    if axis == "x":
        return TU.quat2mat(np.array([np.cos(rot_angle / 2), np.sin(rot_angle / 2), 0, 0]))
    elif axis == "y":
        return TU.quat2mat(np.array([np.cos(rot_angle / 2), 0, np.sin(rot_angle / 2), 0]))
    elif axis == "z":
        return TU.quat2mat(np.array([np.cos(rot_angle / 2), 0, 0, np.sin(rot_angle / 2)]))

final_angle = TU.quat2axisangle(TU.mat2quat(rotation_matrix(0.5*np.pi, axis="x")@rotation_matrix(0, axis="y")@rotation_matrix(0, axis='z')))


# 1.reach
colors=obs[args.view + '_image']
img=Image.fromarray(colors)
# img.save(save_dir +  '/result_before_reaching.jpg')
# reaching
action = np.zeros(7)
action[:3] = pick_pos
action[3:6] = final_angle
action[2] += 0.1
imgs = []
for i in range(reaching_steps):
    obs, reward, done, info = env.step(action)  # take action in the environment
    next_state = fetch_obs(env, obs)
    demo_list.append((curr_state, action, reward, next_state, done))
    curr_state = next_state
    img = obs[args.view + '_image']
    imgs.append(img)

# video_dir = os.path.join(save_dir, 'reaching.mp4')
# imgs2video(imgs, video_dir)

print("reaching reward", reward)
colors=obs['agentview_image']
img=Image.fromarray(colors)
# img.save(save_dir +  '/result_after_reaching.jpg')
#assert 0

# 2.picking
action[:3] = pick_pos
action[3:6] = final_angle
for i in range(picking_steps):
    obs, reward, done, info = env.step(action)  # take action in the environment
    img = obs[args.view + '_image']
    imgs.append(img)
    next_state = fetch_obs(env, obs)
    demo_list.append((curr_state, action, reward, next_state, done))
    curr_state = next_state

# video_dir = os.path.join(save_dir, 'picking.mp4')
# imgs2video(imgs, video_dir)

print("picking reward", reward)
colors=obs['agentview_image']
img=Image.fromarray(colors)
# img.save(save_dir + '/result_after_picking.jpg')

# 3.picking gripper
action[6] = 1
for i in range(gripper_steps):
    obs, reward, done, info = env.step(action)  # take action in the environment
    img = obs[args.view + '_image']
    imgs.append(img)
    next_state = fetch_obs(env, obs)
    demo_list.append((curr_state, action, reward, next_state, done))
    curr_state = next_state

# video_dir = os.path.join(save_dir, 'picking_gripper.mp4')
# imgs2video(imgs, video_dir)

print("picking gripper reward", reward)
colors=obs['agentview_image']
img=Image.fromarray(colors)
# img.save(save_dir + '/result_after_picking_gripper.jpg')

# 4.lifting
action[2] += 0.2
for i in range(lifting_steps):
    obs, reward, done, info = env.step(action)  # take action in the environment
    img = obs[args.view + '_image']
    imgs.append(img)
    next_state = fetch_obs(env, obs)
    demo_list.append((curr_state, action, reward, next_state, done))
    curr_state = next_state

# video_dir = os.path.join(save_dir, 'lifting.mp4')
# imgs2video(imgs, video_dir)

print("lifting reward", reward)
colors=obs['agentview_image']
img=Image.fromarray(colors)
# img.save(save_dir + '/result_after_lifting.jpg')

# 5.moving
action[:3] = place_pos
action[2] += 0.2
for i in range(moving_steps):
    obs, reward, done, info = env.step(action)  # take action in the environment
    img = obs[args.view + '_image']
    imgs.append(img)
    next_state = fetch_obs(env, obs)
    demo_list.append((curr_state, action, reward, next_state, done))
    curr_state = next_state

# video_dir = os.path.join(save_dir, 'moving.mp4')
# imgs2video(imgs, video_dir)

print("moving reward", reward)
colors=obs['agentview_image']
img=Image.fromarray(colors)
# img.save(save_dir + '/result_after_moving.jpg')

# 6.placing
action[2] -= 0.12
for i in range(placing_steps):
    obs, reward, done, info = env.step(action)  # take action in the environment
    img = obs[args.view + '_image']
    imgs.append(img)
    next_state = fetch_obs(env, obs)
    demo_list.append((curr_state, action, reward, next_state, done))
    curr_state = next_state

# video_dir = os.path.join(save_dir, 'placing.mp4')
# imgs2video(imgs, video_dir)

print("placing reward", reward)
colors=obs['agentview_image']
img=Image.fromarray(colors)
# img.save(save_dir + '/result_after_placing.jpg')

# 7.placing griper
action[6] = -1
for i in range(gripper_steps):
    obs, reward, done, info = env.step(action)  # take action in the environment
    img = obs[args.view + '_image']
    imgs.append(img)
    next_state = fetch_obs(env, obs)
    demo_list.append((curr_state, action, reward, next_state, done))
    curr_state = next_state

# video_dir = os.path.join(save_dir, 'placing_griper.mp4')
# imgs2video(imgs, video_dir)

print("placing gripper reward", reward)
colors=obs['agentview_image']
img=Image.fromarray(colors)
# img.save(save_dir + '/result_after_placing_gripper.jpg')


# 8.lifting_2
action[2] += 0.2
for i in range(lifting_steps):
    obs, reward, done, info = env.step(action)  # take action in the environment
    img = obs[args.view + '_image']
    imgs.append(img)
    next_state = fetch_obs(env, obs)
    demo_list.append((curr_state, action, reward, next_state, done))
    curr_state = next_state

video_dir = os.path.join(save_dir, 'demo.mp4')
imgs2video(imgs, video_dir, 5)

print("place lifting reward", reward)
colors=obs['agentview_image']
img=Image.fromarray(colors)
# img.save(save_dir + '/result_after_lifting_2.jpg')

# save demos
pkl_dir = os.path.join(save_dir, 'demo.pkl')
joblib.dump(demo_list, pkl_dir)



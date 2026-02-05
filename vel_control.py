#!/usr/bin/env python3
import time
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32MultiArray, Bool
import numpy as np
import sys
import torch
print("CUDA available:", torch.cuda.is_available())
device = None
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print("Running on:", torch.cuda.get_device_name(device))

prj_path = '/home/nvidia'
if prj_path not in sys.path:
    sys.path.append(prj_path)
from MPC.Adaptive_mpc import AdaptiveMPC
# from MPC.Adaptive_mpc_torch import TorchAdaptiveMPC

frame_width = 640
frame_height = 480
prev_linear = 0.0
prev_angular = 0.0
prev_time = time.time()
occlusion = False
run = True
person_z_prev = 0.0
flg = False
spec_flg = False
error_x_prev_meters = 0.0
error_x_prev = 0.0
max_depth = 10.0
prev_linear_pub = 0.0
prev_angular_pub = 0.0
adaptive_mpc = AdaptiveMPC()
# torchAdaptiveMPC = TorchAdaptiveMPC(device=device)
camera_params = {
            'focalLengthRGB': (-1216.632568359375, 1216.499755859375),
            'focalLengthDepth': (-774.1458740234375, 774.1458740234375),
            'principalPointRGB': (311.89801025390625, 245.68055725097656),
            'principalPointDepth': (323.0037841796875, 234.12889099121094),
            'positionRGB': None,
            'positionDepth': None
        }

def get_lat_axial_velocity_dimless(bbox, depth_val):
    global error_x_prev, prev_time, prev_angular, prev_linear, person_z_prev, spec_flg
    curr_time = time.time()
    focalLengthDepth = camera_params['focalLengthDepth']
    person_x = bbox[0] + bbox[2]/2
    error_x_curr = (person_x - frame_width/2) / (frame_width/2)
    
    if not spec_flg:
        error_x_prev = error_x_curr
        prev_time = curr_time
        spec_flg = True
        return 0.0, 0.0
    
    frame_width_meters = (frame_width/2) * person_z_prev / focalLengthDepth[0]
    V_human_lat = (error_x_curr - error_x_prev) / (curr_time - prev_time) + (prev_angular*person_z_prev)/frame_width_meters
    V_human_axial = ((depth_val - person_z_prev)/max_depth) / (curr_time - prev_time) + prev_linear
    error_x_prev = error_x_curr  
    prev_time = curr_time
    
    return V_human_lat, V_human_axial

def get_lat_axial_velocity_meters(bbox, depth_val):
    global error_x_prev_meters, prev_time, prev_angular, prev_linear, person_z_prev, spec_flg
    curr_time = time.time()
    focalLengthDepth = camera_params['focalLengthDepth']
    
    person_x = bbox[0] + bbox[2]/2
    error_x_curr_meters = (person_x - frame_width/2) * depth_val / focalLengthDepth[0]
    
    if not spec_flg:
        error_x_prev_meters = error_x_curr_meters
        prev_time = curr_time
        spec_flg = True
        return 0.0, 0.0
    
    V_human_lat = (error_x_curr_meters - error_x_prev_meters) / (curr_time - prev_time) + prev_angular*person_z_prev
    V_human_axial = (depth_val - person_z_prev) / (curr_time - prev_time) + prev_linear
    error_x_prev_meters = error_x_curr_meters   
    prev_time = curr_time
    
    return V_human_lat, V_human_axial                                                                                                                  

def time_wait(duration, linear, angular):
    """Wait for specified duration while maintaining current velocities."""
    twist = Twist()
    twist.linear.x = linear
    twist.angular.z = angular
    t = time.time()
    
    while time.time() - t < duration:
        ControlNode().pub.publish(twist)

def calculate_control_commands(bbox, depth_val):
    """Calculate control commands based on person's position in frame and depth."""
    if bbox is None:
        return 0.0, 0.0  # Stop if no position available
    
    # Get center of bounding box
    person_x = bbox[0] + bbox[2]/2
    person_y = bbox[1] + bbox[3]/2
    
    # Calculate error from center of frame
    error_x = (person_x - frame_width/2) / (frame_width/2)
    error_y = (person_y - frame_height/2) / (frame_height/2)
    
    # Use depth information if available
    distance_factor = 1.0
    if depth_val is not None:
        # Adjust control based on depth (slower when closer)
        if depth_val < 1.5:  # Closer than 2 meter
            distance_factor = 0.5  # Reduce speed
        elif depth_val > 3.0:  # Further than 3 meters
            distance_factor = 1.5  # Increase speed
    
    # Calculate control commands with smoother response and depth adjustment
    linear_vel = 0.3 * (1 - abs(error_x)) * distance_factor  # Depth-adjusted forward velocity
    angular_vel = -(0.8) * error_x  # Angular velocity
    
    return linear_vel, angular_vel

class ControlNode:
    def __init__(self):
        rospy.init_node('control_node', anonymous=True)
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.sub = rospy.Subscriber('/person_position', Float32MultiArray, self.position_callback)
        self.sub2 = rospy.Subscriber('/real_velocity', Twist, self.real_velocity_callback)
    
    def real_velocity_callback(self, msg):
        global prev_linear, prev_angular
        try:
            data = msg
            if data is not None:
                prev_linear = data.linear.x
                prev_angular = data.angular.z
                # print(f"prev_linear: {prev_linear}, prev_angular: {prev_angular}")
        except Exception as e:
            rospy.logerr(e)
    
    def position_callback(self, msg):
        global run, prev_linear_pub, prev_angular_pub, occlusion, person_z_prev, flg
        try:
            data = msg.data
            current_pos = [data[0], data[1], data[2], data[3]]
            person_z = data[4]
            if current_pos is not None:
                # Calculate and publish control commands using PID
                if person_z is not None:
                    # print("person_z is not none")
                    # linear_vel, angular_vel = get_stanley_controls(current_pos, person_z, frame.shape[1], frame.shape[0])
                    # linear_vel, angular_vel = get_controls(float(person_x), float(person_z), frame.shape[1])
                    # linear_vel, angular_vel = calculate_control_commands(current_pos, person_z)
                    V_human_lat, V_human_axial = get_lat_axial_velocity_dimless(current_pos, person_z)
                    print(f"current_pos: {current_pos}, person_z: {person_z}")
                    print(f"V_human_lat: {V_human_lat}, V_human_axial: {V_human_axial}")
                    # linear_vel, angular_vel = torchAdaptiveMPC.get_velocity(current_pos, frame_width, person_z, V_human_lat, V_human_axial)
                    linear_vel, angular_vel = adaptive_mpc.get_velocity(current_pos, frame_width, person_z, V_human_lat, V_human_axial)
                
            if person_z_prev - person_z > 2.00:
                occlusion = not occlusion
                time_wait(1, prev_linear_pub, prev_angular_pub)
                print("Occlusion detected, waiting for 1 second")
            
            if person_z <= 0.75:
                flg = True
                print(f"run is false: {person_z}")
                run = False
            else:
                if flg:
                    print(f"run is true: {person_z}")
                    flg = False
                run = True
            cmd = Twist()
            cmd.linear.x = linear_vel
            cmd.angular.z = angular_vel
            cmd.linear.x *= run
            cmd.angular.z*= run
            if not occlusion:
                print(f"linear vel: {cmd.linear.x}, angular vel: {cmd.angular.z}")
                print(" ")
                self.pub.publish(cmd) 
            
            occlusion = False
            person_z_prev = person_z
            prev_linear_pub, prev_angular_pub = linear_vel, angular_vel
            
        except Exception as e:
            rospy.logerr(e)

if __name__ == '__main__':
    ControlNode()
    rospy.spin()


# #!/usr/bin/env python3
# from collections import deque
# import time
# import rospy
# from geometry_msgs.msg import Twist
# from std_msgs.msg import Float32MultiArray, Bool

# frame_width = 640
# frame_height = 480
# prev_linear = 0.0
# prev_angular = 0.0
# prev_time = time.time()
# occlusion = False
# run = True
# person_z_prev = 0.0
# flg = False

# def time_wait(duration, linear, angular):
#     """Wait for specified duration while maintaining current velocities."""
#     twist = Twist()
#     twist.linear.x = linear
#     twist.angular.z = angular
#     t = time.time()
    
#     while time.time() - t < duration:
#         ControlNode().pub.publish(twist)

# def calculate_control_commands(bbox, frame_width, frame_height, depth_val):
#     """Calculate control commands based on person's position in frame and depth."""
#     if bbox is None:
#         return 0.0, 0.0  # Stop if no position available
    
#     # Get center of bounding box
#     person_x = bbox[0] + bbox[2]/2
#     person_y = bbox[1] + bbox[3]/2
    
#     # Calculate error from center of frame
#     error_x = (person_x - frame_width/2) / (frame_width/2)
#     error_y = (person_y - frame_height/2) / (frame_height/2)
    
#     # Use depth information if available
#     distance_factor = 1.0
#     if depth_val is not None:
#         # Adjust control based on depth (slower when closer)
#         if depth_val < 1.5:  # Closer than 2 meter
#             distance_factor = 0.5  # Reduce speed
#         elif depth_val > 3.0:  # Further than 3 meters
#             distance_factor = 1.5  # Increase speed
    
#     # Calculate control commands with smoother response and depth adjustment
#     linear_vel = 0.3 * (1 - abs(error_x)) * distance_factor  # Depth-adjusted forward velocity
#     angular_vel = -(0.8) * error_x  # Angular velocity
    
#     return linear_vel, angular_vel

# class ControlNode:
#     def __init__(self):
#         global prev_time
#         rospy.init_node('control_node', anonymous=True)
#         self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
#         self.sub = rospy.Subscriber('/person_position', Float32MultiArray, self.position_callback)
#         prev_time = time.time()
    
#     def position_callback(self, msg):
#         global run, frame_width, frame_height, prev_linear, prev_angular, occlusion, person_z_prev, prev_time, flg
#         try:
#             data = msg.data
#             current_pos = [data[0], data[1], data[2], data[3]]
#             person_z = data[4]
#             if current_pos is not None:
#                 # Calculate and publish control commands using PID
#                 if person_z is not None:
#                     # print("person_z is not none")
#                     # linear_vel, angular_vel = get_stanley_controls(current_pos, person_z, frame.shape[1], frame.shape[0])
#                     # linear_vel, angular_vel = get_controls(float(person_x), float(person_z), frame.shape[1])
#                     linear_vel, angular_vel = calculate_control_commands(current_pos, frame_width, frame_height, person_z)
                
#             if person_z_prev - person_z > 2.00:
#                 occlusion = not occlusion
#                 time_wait(1, prev_linear, prev_angular)
#                 print("Occlusion detected, waiting for 1 second")
            
#             if person_z <= 0.75:
#                 flg = True
#                 print(f"run is False: {person_z}")
#                 run = False
#             else:
#                 if flg:
#                     print(f"run is True: {person_z}")
#                 run = True
#             cmd = Twist()
#             # cmd.linear.x = 0
#             cmd.linear.x = linear_vel
#             cmd.angular.z = angular_vel
#             cmd.linear.x *= run
#             cmd.angular.z*= run
#             if not occlusion:
#             #     # print(f"linear vel: {cmd.linear.x}, angular vel: {cmd.angular.z}")
#                 # current_time = time.time()
#                 # loopSpeed = 1.0 / (current_time - prev_time)
#                 # prev_time = current_time
#                 # print(f"loopSpeed: {loopSpeed:.2f} Hz")
#                 self.pub.publish(cmd) 
            
#             occlusion = False
#             person_z_prev = person_z
#             prev_linear, prev_angular = linear_vel, angular_vel
            
#         except Exception as e:
#             rospy.logerr(e)

# if __name__ == '__main__':
#     ControlNode()
#     rospy.spin()

#!/usr/bin/env python3
from collections import deque
import time
import rospy
import cv2
import numpy as np
import sys
from std_msgs.msg import Float32MultiArray
from pal.products.qbot_platform import QBotPlatformRealSense
# from quanser.hardware import HILError

# prj_path = os.path.join(os.path.dirname(__file__), '..')
# print(prj_path)
prj_path = '/home/nvidia'
prj_path2 = '/home/nvidia/Person_follower/Stark'
if prj_path not in sys.path:
    sys.path.append(prj_path)
if prj_path2 not in sys.path:
    sys.path.append(prj_path2)

from Person_follower.Stark.lib.test.evaluation import Tracker

position_history = deque(maxlen=10)
depth_history = deque(maxlen=3)
flg = False
last_known_position = None
initial_bbox = None
tracker = None
tracker_inner = None
tracker_name = "stark_st"
tracker_param = "baseline_R101"
tracker_initialized = False
prev_linear = 0.0
prev_angular = 0.0
last_detection_time = None
conf_score = 0.
current_bbox = None
camera_params = {
            'focalLengthRGB': (-1216.632568359375, 1216.499755859375),
            'focalLengthDepth': (-774.1458740234375, 774.1458740234375),
            'principalPointRGB': (311.89801025390625, 245.68055725097656),
            'principalPointDepth': (323.0037841796875, 234.12889099121094),
            'positionRGB': None,
            'positionDepth': None
        }
THRESHOLD_CONFIDANCE = 0.0


def display_RGBD(RGB_image, depth_image):
    cv2.imshow('QBOT Camera', RGB_image)
    cv2.waitKey(1)
    
    
    # Normalize the depth image to the range 0-255 for visualization
    depth_normalized = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    depth_normalized = depth_normalized.astype('uint8')
    
    # Apply a colormap for a better depth visualization
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    cv2.imshow('QBOT Depth', depth_colormap)
    cv2.waitKey(1)

def is_valid_depth(depth_value):
    """Check if a depth value is valid."""
    # Some depth sensors return 0 or negative values for invalid measurements
    return depth_value > 0 and not np.isnan(depth_value) and not np.isinf(depth_value)

def predict_next_position():
    """Predict next position based on position history."""
    if len(position_history) < 2:
        return None
    
    # Calculate average velocity
    positions = list(position_history)
    velocities = []
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i-1][0]
        dy = positions[i][1] - positions[i-1][1]
        velocities.append([dx, dy])
    
    if not velocities:
        return None
    
    # Average velocity
    avg_vel = np.mean(velocities, axis=0)
    
    # Predict next position
    last_pos = positions[-1]
    predicted_pos = [
        last_pos[0] + avg_vel[0],
        last_pos[1] + avg_vel[1],
        last_pos[2],
        last_pos[3]
    ]
    
    return predicted_pos

def get_current_position(frame, depth_frame=None):
    """Get current position of the person using tracking and detection."""
    global tracker, tracker_inner, current_bbox, last_known_position, last_detection_time, prev_linear, prev_angular, conf_score, flg
    
    # Update tracker
    try:
        score, bbox = tracker.update_tracker(tracker_inner, frame, flg)
    except Exception as e:
        print(f"Error updating tracker: {e}")
        score, bbox = 0.0, None
    
    conf_score = score
    # print(f"Tracker confidence score: {conf_score}")
    
    if conf_score >= THRESHOLD_CONFIDANCE:
        # Tracker is working, use its result
        current_bbox = bbox
        last_known_position = bbox
        position_history.append(bbox)
        
        person_z = None
        x, y, w, h = [int(v) for v in bbox]
        if depth_frame is not None:
            # Get the 4 corner coordinates in the specified proportions
            corners = [
                (int(x + w * (1/3)), int(y + h * (1/3))),  # Top-left (1/3, 1/3)
                (int(x + w * (1/3)), int(y + h * (2/3))),  # Bottom-left (1/3, 2/3)
                (int(x + w * (2/3)), int(y + h * (1/3))),  # Top-right (2/3, 1/3)
                (int(x + w * (2/3)), int(y + h * (2/3))),  # Bottom-right (2/3, 2/3)
                (int(x + w * (1/4)), int(y + h * (1/3))),  # Bottom-right (2/3, 2/3)
                (int(x + w * (1/4)), int(y + h * (2/3))),  # Bottom-right (2/3, 2/3)
                (int(x + w * (3/4)), int(y + h * (1/3))),  # Bottom-right (2/3, 2/3)
                (int(x + w * (3/4)), int(y + h * (2/3))),  # Bottom-right (2/3, 2/3)
                (int(x + w * (1/2)), int(y + h * (1/2)))   # Center (1/2, 1/2)
            ]
            
            depth_values = []
            for x, y in corners:
                depth_val = depth_frame[y, x]
                # print(f"Depth value at corner ({x}, {y}): {depth_val}")
                if is_valid_depth(depth_val):
                    depth_values.append(depth_val)
            
            # Calculate the average depth if valid values exist
            if depth_values:
                person_z = min(depth_values)
                # person_z = sum(depth_values) / len(depth_values)
                # print(person_z)
        
        return bbox, person_z
    
    # If tracking failed try prediction
    if len(position_history) >= 2:
        predicted_pos = predict_next_position()
        if predicted_pos is not None and len(depth_history) >= 1:
            return predicted_pos, depth_history[-1]
    
    # If all else fails, return last known position
    return last_known_position

def check_camera(camera):
    """Check if camera is working properly by reading both RGB and depth frames."""
    try:
        rgb_timestamp = camera.read_RGB()
        depth_timestamp = camera.read_depth(dataMode='M')
        # Get first frame and select person to track
        start_time = time.time()
        while rgb_timestamp == -1 and (time.time() - start_time) < 5:
            rgb_timestamp = camera.read_RGB()
        while depth_timestamp == -1 and (time.time() - start_time) < 10:
            depth_timestamp = camera.read_depth(dataMode='M')
        if rgb_timestamp == -1:
            print("Error: Failed to read from camera RGB stream")
            return False
        if depth_timestamp == -1:
            print("Error: Failed to read from camera depth stream")
            return False
    except Exception as e:
        print(f"Error: Camera initialization failed: {e}")
        return False

def select_initial_bbox(frame):
    """Allow user to select initial bounding box for tracking."""
    global initial_bbox
    cv2.namedWindow('Select Person')
    print("Draw a rectangle around the person and press Enter.")
    roi = cv2.selectROI('Select Person', frame, fromCenter=False, showCrosshair=True)
    if roi != (0, 0, 0, 0):
        initial_bbox = (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3]))
    cv2.destroyWindow('Select Person')
    while initial_bbox is None:
        cv2.imshow('Select Person', frame)
        cv2.imwrite("constructed_image2.png", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    return initial_bbox

def create_tracker():
    global tracker_inner
    try:
        tracker = Tracker(tracker_name, tracker_param, "video")
        tracker_inner = tracker.create_karena()
        return tracker
    except AttributeError as e:
        print("Tracker Error")

def display(frame, bbox, person_z, fps_text):
    """Display the bounding box and depth information on the frame."""
    global conf_score, flg
    if bbox is not None:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if person_z is not None:
        depth_str = f"Distance: {float(person_z):.2f}m"
        cv2.putText(frame, depth_str, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    font_color = (0, 255, 0)
    cv2.putText(frame, f'Conf: {conf_score:.2f}', (20, 105), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                           font_color, 1)
    cv2.putText(frame, fps_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow('Person Following', frame)
    key = cv2.waitKey(1)
    if key == 13:
        flg = True
    else:
        flg = False

def get_coordinates_wrt_camera(current_pos, person_z):
    global camera_params
    focalLengthRGB = camera_params['focalLengthRGB'] 
    focalLengthDepth = camera_params['focalLengthDepth']
    principalPointRGB = camera_params['principalPointRGB']
    principalPointDepth = camera_params['principalPointDepth']
    
    # Convert pixel coordinates to camera coordinates
    y = (current_pos[0] + current_pos[2] / 2 - principalPointDepth[0]) * person_z / focalLengthDepth[0]
    z = (principalPointDepth[1] - (current_pos[1] + current_pos[3] / 2)) * person_z / focalLengthDepth[1]
    x = person_z
    coordinates_wrtCamera = np.array([x, y, z])
    return coordinates_wrtCamera

def main():
    global tracker, tracker_inner, tracker_initialized, position_history, last_known_position, last_detection_time, initial_bbox, current_bbox, camera_params
    try:
        rospy.init_node('tracker_node')
        pos_pub = rospy.Publisher('/person_position', Float32MultiArray, queue_size=1)
        try:
            camera = QBotPlatformRealSense()
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return
        
        # Wait for camera to initialize
        time.sleep(1)
        
        check_camera(camera)
        RGB_image = camera.imageBufferRGB.copy()
        initial_bbox = select_initial_bbox(RGB_image)
        if initial_bbox is None:
            print("No person selected")
            return

        print("Initializing tracker...")
        try:
            tracker = create_tracker()
            tracker.initialize_tracker(tracker_inner, RGB_image, initial_bbox)
        except Exception as e:
            print(f"Error initializing tracker: {e}")
            return

        current_bbox = initial_bbox
        last_known_position = current_bbox
        position_history.append(current_bbox)
        last_detection_time = time.time()
        prev_time = time.time()
        print("Starting person following...")

        while not rospy.is_shutdown():
            try:
                check_camera(camera)
                RGB_image = camera.imageBufferRGB.copy()
                depth_image = camera.imageBufferDepthM.copy()
                
                current_pos, person_z = get_current_position(RGB_image, depth_image)
                # coordinates_wrtCamera = get_coordinates_wrt_camera(current_pos, person_z)
                # print(f"Coordinates with respect to camera: {coordinates_wrtCamera}")
                
                current_time = time.time()
                fps = 1.0 / (current_time - prev_time)
                prev_time = current_time
                fps_text = f"FPS: {fps:.2f}"
                display(RGB_image, current_pos, person_z, fps_text)
                if current_pos is not None and person_z is not None:
                    if person_z < 0.7:
                        print(f"distance: {person_z}")
                    pos_msg = Float32MultiArray()
                    pos_msg.data = list(current_pos) + [person_z]
                    pos_pub.publish(pos_msg)
            except Exception as e:
                rospy.logerr(e)
                break
    except KeyboardInterrupt:
        print("\nUser interrupted")
    # except HILError as h:
    #     print(f"Hardware error: {h.get_error_message()}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        print("Cleaning up...")
        try:
            camera.terminate()
            cv2.destroyAllWindows()
        except:
            pass
        print("Cleanup complete")

if __name__ == '__main__':
    main()

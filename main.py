# Imports
from pal.products.qbot_platform import QBotPlatformDriver, QBotPlatformLidar
from quanser.hardware import HILError
import time
import numpy as np
import os
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan  # Import LaserScan message
from nav_msgs.msg import Odometry  # Import Odometry message
import tf

time.sleep(1)

global commands
commands =  np.array([0, 0], dtype = np.float64)

def cmd_vel_callback(msg):
    """
    Callback function that updates QBot commands from the ROS /cmd_vel topic.
    """
    global commands
    commands[0] = msg.linear.x  # Linear velocity (m/s)
    commands[1] = msg.angular.z  # Angular velocity (rad/s)
    
ip = ""
os.environ['ROS_IP'] = ip
os.environ['ROS_MASTER_URI'] = f'http://{ip}:11311'
rospy.init_node('qbot_platform_controller', anonymous=True)
rospy.Subscriber('/cmd_vel', Twist, cmd_vel_callback)
lidar_pub = rospy.Publisher('lidar_scan', LaserScan, queue_size=50)
odom_pub = rospy.Publisher('odom', Odometry, queue_size=50)
real_velocity_pub = rospy.Publisher('real_velocity', Twist, queue_size=1)
odom_broadcaster = tf.TransformBroadcaster()


os.system('quarc_run -q -Q -t tcpip://localhost:17000 *.rt-linux_qbot_platform -d /tmp')
time.sleep(5)
os.system('quarc_run -r -t tcpip://localhost:17000 qbot_platform_driver_physical.rt-linux_qbot_platform  -d /tmp -uri tcpip://localhost:17099')
time.sleep(3)
print('driver loaded')

arm = 1
x , y, yaw = 0, 0, 0

startTime = time.time()
def elapsed_time():
    return time.time() - startTime

try: 
    myQBot = QBotPlatformDriver(mode=3, ip="localhost")
    lidar = QBotPlatformLidar()

    startTime = time.time()
    time.sleep(1)
    prev_Time = time.time()

    while True:
        t = elapsed_time()

        newHIL = myQBot.read_write_std(timestamp = time.time() - startTime,
                                                arm=1,
                                                hold=1,
                                                commands = commands)
        
        wheel_radius = myQBot.WHEEL_RADIUS
        # Calculate the linear and angular velocities
        linear_velocity = (myQBot.wheelSpeeds[0] + myQBot.wheelSpeeds[1]) * wheel_radius / 2
        angular_velocity = (myQBot.wheelSpeeds[1] - myQBot.wheelSpeeds[0]) * wheel_radius / myQBot.WHEEL_BASE
        # Publish the real velocity
        real_velocity = Twist()
        real_velocity.linear.x = linear_velocity
        real_velocity.angular.z = angular_velocity
        real_velocity_pub.publish(real_velocity)

        current_time = time.time()
        time_diff = current_time - prev_Time

        # calcualte current bot position
        x = x + linear_velocity * np.cos(yaw) * time_diff
        y = y + linear_velocity * np.sin(yaw) * time_diff
        yaw = yaw + angular_velocity * time_diff

        # Publish the odometry message
        odom = Odometry()
        odom.header.stamp = rospy.Time.now()
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_link"
        odom.pose.pose.position.x = x
        odom.pose.pose.position.y = y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.x = 0.0
        odom.pose.pose.orientation.y = 0.0
        odom.pose.pose.orientation.z = np.sin(yaw / 2)
        odom.pose.pose.orientation.w = np.cos(yaw / 2)  

        odom_pub.publish(odom)

        # publish tf between odom and base_link
        odom_broadcaster.sendTransform(
            (x, y, 0),
            tf.transformations.quaternion_from_euler(0, 0, yaw),
            rospy.Time.now(),
            "base_link",
            "odom"
        )

        # publish a static trasnfrom between base_link and lidar_frame
        odom_broadcaster.sendTransform(
            (0.15, 0, 0),
            tf.transformations.quaternion_from_euler(0, 0, 0),
            rospy.Time.now(),
            "lidar_frame",
            "base_link"
        )
        prev_Time = current_time

        lidar.read()
        distances = lidar.distances
        angles = lidar.angles

        # rotate distance array by 90 degrees
        distances = np.roll(distances, int(len(distances) * 3/4))
        distances = distances[::-1]  

        # Create and publish LaserScan message
        scan_msg = LaserScan()
        scan_msg.header.stamp = rospy.Time.now()
        scan_msg.header.frame_id = "lidar_frame"
        if distances.size > 0 and angles.size > 0:
            # Assume angles are sorted; otherwise, sort them along with distances.
            scan_msg.angle_min = float(np.min(angles))
            scan_msg.angle_max = float(np.max(angles))
            if angles.size > 1:
                scan_msg.angle_increment = float((scan_msg.angle_max - scan_msg.angle_min) / (angles.size - 1))
            else:
                scan_msg.angle_increment = 0.0
            
            scan_msg.range_min = 0.1  # Adjust if your sensor has a minimum range
            scan_msg.range_max = 10.0  # Adjust to your sensor's maximum range
            
            # Convert NumPy array to list for ROS message
            scan_msg.ranges = distances.tolist()
            scan_msg.intensities = []  # Populate if available; otherwise, leave empty

            # Publish the LaserScan message
            lidar_pub.publish(scan_msg)
            

except KeyboardInterrupt:
    print('User interrupted.')
except HILError as h:
    print(h.get_error_message())
finally:
    myQBot.terminate()
    os.system('quarc_run -q -Q -t tcpip://localhost:17000 *.rt-linux_qbot_platform -d /tmp')

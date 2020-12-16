from std_msgs.msg import Int8MultiArray
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, rotation_matrix, quaternion_from_matrix
import time
from geometry_msgs.msg import Twist, Vector3

class RobotMotion():
    """Maintains the ROS publisher for simple velocity commands"""
    def __init__(self):
        #rospy.init_node('robo_motion')
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.line_vel = 0.15
        self.ang_vel = 0.25
        self.expected_sign = 52
        self.detected_sign = None

    def starter_motion(self):
        self.pub.publish(Twist(linear=Vector3(x=self.line_vel, y=0)))

    def stop(self):
        self.pub.publish(Twist(linear=Vector3(x=0, y=0), angular=Vector3(z=0)))
        time.sleep(2)

    def turn_right(self):
        self.pub.publish(Twist(linear=Vector3(x=0, y=0), angular=Vector3(z=-0.55)))
        time.sleep(3)

    def turn_left(self):
        self.pub.publish(Twist(linear=Vector3(x=0, y=0), angular=Vector3(z=0.55)))
        time.sleep(3)

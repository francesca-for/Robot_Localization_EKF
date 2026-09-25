import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
import tf_transformations
import numpy as np


class BumpAndGoController(Node):
    def __init__(self):
        super().__init__("bump_and_go_controller")

        # Publishers and Subscribers
        self.publisher=self.create_publisher(Twist,"/cmd_vel",10)
        self.laser_data_subscriber=self.create_subscription(LaserScan,"/scan", self.get_laser_data_callback, 10)
        self.odometry_subscriber=self.create_subscription(Odometry,"/odom", self.odometry_callback, 10)
        
        timer_period=1
        self.timer=self.create_timer(timer_period,self.control_loop_callback)

        # Declare parameters
        self.declare_parameter('max_ang', 0.3)
        self.declare_parameter('max_lin', 0.22)
        self.declare_parameter('min_distance_limit', 0.5)

        # Declare and initialize variables
        self.distance_limit = self.get_parameter('min_distance_limit').get_parameter_value().double_value
        self.yaw = 0.0
        self.n_sectors = 12   # number of sectors in which the space around the robot is divided
        self.laser_data_limited = np.array([0.0] * 360)     # measurements putting nan->0.0 and inf->max_range
        self.sectors_mean_distances = np.array([float('inf')] * self.n_sectors)  # mean distances for each sector
        self.desired_angle = 0.0   # Rotation angle to reach the desired direction
        self.desired_yaw = 0.0
        self.state = 'FORWARD'  # Robot possible states: FORWARD, TURN


    def get_laser_data_callback(self,laser_msg):
        # self.get_logger().info(f"Receiving: {laser_msg}")

        laser_data = np.array(laser_msg.ranges)
        # substitute nan and inf to avoid issues in mean calculation
        self.laser_data_limited = np.nan_to_num(laser_data, nan=laser_msg.range_min, posinf=laser_msg.range_max)

        # Reshape the laser data into sectors (12x30 matrix) and compute mean distances
        laser_data_reshaped = self.laser_data_limited.reshape(self.n_sectors, -1)
        self.sectors_mean_distances = laser_data_reshaped.mean(axis=1)  # mean value for each row
        

    def odometry_callback(self,odom_msg):
        quat = [odom_msg.pose.pose.orientation.x,
                odom_msg.pose.pose.orientation.y,
                odom_msg.pose.pose.orientation.z,
                odom_msg.pose.pose.orientation.w]
        _, _, self.yaw = tf_transformations.euler_from_quaternion(quat)   # get current yaw from quaternion
        self.get_logger().info(f"Current Yaw: {self.yaw}")


    def control_loop_callback(self):
        linear_speed = self.get_parameter('max_lin').get_parameter_value().double_value
        angular_speed = self.get_parameter('max_ang').get_parameter_value().double_value
        twist_msg = Twist()

        # Check for obstacles moving forward
        distFR = np.mean(self.laser_data_limited[0:20])    # [0°, -20°], front right
        distR = np.mean(self.laser_data_limited[21:40])    # [-20°, -40°], front right
        distFL = np.mean(self.laser_data_limited[-20:])    # [0°, +20°], front left
        distL = np.mean(self.laser_data_limited[-40:-21])  # [+20°, +40°], front left
        
        # robot finds an obstacle while moving forward
        if self.state != 'TURN' and (distFR < self.distance_limit or distR < self.distance_limit or distFL < self.distance_limit or distL < self.distance_limit):
            self.state = 'TURN'   # change robot state to TURN

            # Compute the angle of rotation that corresponds to the sector with the maximum mean distance
            # (robot rotate at most of a restricted angle to avoid going backward)
            sectors_per_side = 4    # num of considered sectors on each side (4*30°=120°)
            front_sectors = np.concatenate((self.sectors_mean_distances[:sectors_per_side],
                               self.sectors_mean_distances[-sectors_per_side:]))
            idx_max_front = np.argmax(front_sectors)  # index of the maximum mean distance considering only the front sectors

            if idx_max_front >= sectors_per_side: # remapping of the index on the complete sectors array
                idx_max_front = self.n_sectors - sectors_per_side + (idx_max_front - sectors_per_side)
            
            self.desired_angle = (idx_max_front + 0.5) * (2 * np.pi / self.n_sectors)   # +0.5 to consider the center of the sector

            self.get_logger().warn(f"Desired Angle: {self.desired_angle}")
            
            self.desired_yaw = self.yaw + self.desired_angle   # compute desired_yaw
            self.desired_yaw = (self.desired_yaw + np.pi) % (2 * np.pi) - np.pi  # normalize to [-pi, pi]
            self.get_logger().warn(f"Desired Yaw: {self.desired_yaw}")
            
        # robot is turning
        if self.state == 'TURN':
            # angle difference normalized in [-pi, pi]
            angle_diff = (self.desired_yaw - self.yaw + np.pi) % (2 * np.pi) - np.pi
        
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = angular_speed * np.sign(angle_diff)   # define direction of rotation

            self.get_logger().warn(f"----> Yaw diff: {angle_diff}")
            if abs(angle_diff) <= 0.15:  # check if the desired_yaw is reached within a tolerance
                self.state = 'FORWARD'  # rotation completed, change robot state to FORWARD

        # robot is moving forward
        if self.state == 'FORWARD':
            twist_msg.linear.x = linear_speed
            twist_msg.angular.z = 0.0

        self.publisher.publish(twist_msg)


def main(param=None):
    rclpy.init(args=param)
    controller_reset=BumpAndGoController()
    rclpy.spin(controller_reset)
    controller_reset.destroy_node()
    rclpy.shutdown

if __name__=="__main__":
    main()
    
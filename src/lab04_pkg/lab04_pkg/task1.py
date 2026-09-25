#librerie da importare
import rclpy
from rclpy.node import Node
from rclpy.time import Time
import numpy as np
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int
from scipy.interpolate import interp1d
import math
import tf_transformations

#funzioni da importare
from lab04_pkg.ekf import RobotEKF
from lab04_pkg.utils import eval_jacobiane_Gt_e_Vt, sample_velocity_motion_model, eval_jacobian_hux_Ht
from lab04_pkg.utils import residual_measurement

#messaggi da importare
from nav_msgs.msg import Odometry
from landmark_msgs.msg import LandmarkArray 
import pdb



class EKF_ESTIMATE(Node):
    def __init__(self):
        super().__init__('EKF')
        
        #ROBOT
        self.eval_Gt, self.eval_Vt = eval_jacobiane_Gt_e_Vt()
        self.robot=RobotEKF(dim_x=3, dim_u=2, eval_gux=sample_velocity_motion_model,eval_Gt=self.eval_Gt, eval_Vt=self.eval_Vt)
        self.sigma_z = np.array([0.3, math.pi/24])
        self.Qt = np.diag(self.sigma_z**2)

        #PARAMETRI
        self.v=0.0
        self.w=0.0
        self.dt=0.05
        self.alpha= [0.001, 0.002, 0.001, 0.002, 0.005, 0.004]
        
        self.landmarks = {
            11: {"x": -1.1, "y": -1.1, "z": 0.5 },
            12: {"x": -1.1, "y":  0.0, "z": 1.5 },
            13: {"x": -1.1, "y":  1.1, "z": 0.5 },
            21: {"x":  0.0, "y": -1.1, "z": 1.0 },
            22: {"x":  0.0, "y":  0.0, "z": 0.75 },
            23: {"x":  0.0, "y":  1.1, "z": 0.3 },
            31: {"x":  1.1, "y": -1.1, "z": 1.5 },
            32: {"x":  1.1, "y":  0.0, "z": 1.0 },
            33: {"x":  1.1, "y":  1.1, "z": 0.0 },
            }
        
        #CANALI
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        time_period = 0.05  # seconds 
        self.timer = self.create_timer(time_period, self.prediction)

        self.landmark_sub = self.create_subscription(LandmarkArray, '/landmarks', self.landmark_callback, 10)
        self.odom_publisher = self.create_publisher(Odometry, '/ekf', 10)
        self.gt_subscriber = self.create_subscription(Odometry, '/ground_truth', self.groundTruth_callback, 10)

        self.eval_hx, self.eval_Ht = eval_jacobian_hux_Ht()  # to avoid generating them at each update


    def odom_callback(self, msg: Odometry):
        self.w=msg.twist.twist.angular.z
        self.v=msg.twist.twist.linear.x

        if abs(self.w) < 1e-6:
            self.w = 1e-6


    def prediction(self):
        u=np.array([self.v, self.w])
        
        sigma_u=np.array([np.sqrt(self.alpha[0]*self.v**2+self.alpha[1]*self.w**2), 
                          np.sqrt(self.alpha[2]*self.v**2+self.alpha[3]*self.w**2)])
        
        self.robot.Mt = np.diag(sigma_u**2)

        self.robot.predict(u, sigma_u, g_extra_args=(self.dt,))


    def landmark_callback(self, measures: LandmarkArray):
        for element in measures.landmarks:
            self.get_logger().info(f'Ricevuti {len(measures.landmarks)} landmarks')

            z=np.array([[element.range], [element.bearing]])
            id=element.id

            # VERIFICA che il landmark esista nel dizionario
            if id not in self.landmarks:
                self.get_logger().warn(f'Landmark {id} not found!')
                continue

            Ht_args=(self.robot.mu[0], self.robot.mu[1],self.robot.mu[2],self.landmarks[id]["x"], self.landmarks[id]["y"])
            hx_args=Ht_args
            print(f"L_x = {self.landmarks[id]['x']} \t\tL_y= {self.landmarks[id]['y']}")

            self.robot.update(z,eval_hx=self.eval_hx, eval_Ht=self.eval_Ht, Qt=self.Qt, Ht_args=Ht_args,hx_args=hx_args, residual=residual_measurement)

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"

        x,y,theta=self.robot.mu
        self.get_logger().info(f'\nekf mu:\n{x,y,theta}')
        msg.pose.pose.position.x=x
        msg.pose.pose.position.y=y
        msg.pose.pose.position.z=0.0 
        
        quat = tf_transformations.quaternion_from_euler(0.0, 0.0, theta)
        msg.pose.pose.orientation.x = quat[0]
        msg.pose.pose.orientation.y = quat[1]
        msg.pose.pose.orientation.z = quat[2]
        msg.pose.pose.orientation.w = quat[3]

        # Covarianza della posa (solo diagonale x,y,yaw)
        msg.pose.covariance[0]  = self.robot.Sigma[0, 0]   # var(x)
        msg.pose.covariance[7]  = self.robot.Sigma[1, 1]   # var(y)
        msg.pose.covariance[35] = self.robot.Sigma[2, 2]   # var(yaw)

        self.odom_publisher.publish(msg)


    def groundTruth_callback(self, msg: Odometry):
        x_gt = msg.pose.pose.position.x
        y_gt = msg.pose.pose.position.y

        orientation_q = msg.pose.pose.orientation
        (_, _, theta_gt) = tf_transformations.euler_from_quaternion(
            [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        )


def main(args=None):
    rclpy.init(args=args)
    EKF = EKF_ESTIMATE()
    rclpy.spin(EKF)
    EKF.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
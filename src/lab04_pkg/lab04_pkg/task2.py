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
from lab04_pkg.ekf_2 import RobotEKF
from lab04_pkg.utils_2 import eval_jacobiane_Gt_e_Vt, sample_velocity_motion_model, eval_jacobian_hux_Ht
from lab04_pkg.utils_2 import residual_measurement, hx_odom, Ht_odom, hx_imu, Ht_imu

#messaggi da importare
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from landmark_msgs.msg import LandmarkArray 
import pdb



class EKF_ESTIMATE(Node):
    def __init__(self):
        super().__init__('EKF')
        
        #ROBOT
        self.eval_Gt, self.eval_Vt = eval_jacobiane_Gt_e_Vt()
        self.robot=RobotEKF(dim_x=5, dim_u=2, eval_gux=sample_velocity_motion_model, eval_Gt=self.eval_Gt, eval_Vt=self.eval_Vt)
        self.sigma_z = np.array([0.3, math.pi/24])
        self.Qt = np.diag(self.sigma_z**2)

        self.initialized = False

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
        self.imu_subscriber = self.create_subscription(Imu, '/imu', self.imu_callback, 10)

        self.eval_hx, self.eval_Ht = eval_jacobian_hux_Ht()  # per evitare di rigenerarle ad ogni update

    #MODIFICATA, HO CANCELLATO LA VECCHIA
    def odom_callback(self, msg: Odometry):
        # 1. Estrai la misura
        z_v = msg.twist.twist.linear.x
        z_w = msg.twist.twist.angular.z
        z = np.array([z_v, z_w])

        # 2. Definisci incertezza odometria (Qt_odom)
        # Fidati abbastanza della velocità lineare, meno di quella angolare
        Qt_odom = np.diag([0.01**2, 0.05**2]) 

        # 4. Preparo args (tutto lo stato)
        # Nota: passo i 5 elementi dello stato
        args = (self.robot.mu[0], self.robot.mu[1], self.robot.mu[2], self.robot.mu[3], self.robot.mu[4])
        
        # 5. FACCIO L'UPDATE!
        self.robot.update(z, 
                          eval_hx=hx_odom, 
                          eval_Ht=Ht_odom, 
                          Qt=Qt_odom, 
                          Ht_args=args, 
                          hx_args=args, 
                          residual=np.subtract) # Sottrazione normale (no angoli)


    def imu_callback(self, msg: Imu):
        # 1. Estrai la misura (velocità angolare asse Z)
        z_w = msg.angular_velocity.z
        z = np.array([z_w])

        # 2. Incertezza IMU (Qt_imu)
        # Di solito l'IMU è molto preciso sulla rotazione
        Qt_imu = np.diag([0.005**2]) 
        args = (self.robot.mu[0], self.robot.mu[1], self.robot.mu[2], self.robot.mu[3], self.robot.mu[4])

        # 4. UPDATE
        self.robot.update(z, 
                          eval_hx=hx_imu, 
                          eval_Ht=Ht_imu, 
                          Qt=Qt_imu, 
                          Ht_args=args, 
                          hx_args=args, 
                          residual=np.subtract)



    def prediction(self):
        #u=np.array([self.v, self.w])
        u = np.array([0.0, 0.0])
        
        v_curr = self.robot.mu[3]
        w_curr = self.robot.mu[4]

        # 3. CORREZIONE: Calcola sigma usando v_curr e AGGIUNGI UN RUMORE BASE
        # Quel "+ 0.05" è fondamentale per permettere al robot di "sbloccarsi" da zero.
        sigma_v = np.sqrt(self.alpha[0]*v_curr**2 + self.alpha[1]*w_curr**2) + 0.05
        sigma_w = np.sqrt(self.alpha[2]*v_curr**2 + self.alpha[3]*w_curr**2) + 0.05
        
        sigma_u = np.array([sigma_v, sigma_w])       

        self.robot.Mt = np.diag(sigma_u**2)

        self.robot.predict(u, sigma_u, g_extra_args=(self.dt,))
        #self.get_logger().info(f'\nekf mu:\n{self.robot.mu}')
        #self.get_logger().info(f'\nsigma :\n{self.robot.Sigma}')


    def landmark_callback(self, measures: LandmarkArray):
        for element in measures.landmarks:
            self.get_logger().info(f'Ricevuti {len(measures.landmarks)} landmarks')

            z=np.array([[element.range], [element.bearing]])
            id=element.id

            # VERIFICA che il landmark esista nel dizionario
            if id not in self.landmarks:
                self.get_logger().warn(f'Landmark {id} not found!')
                continue

            # eval_hx, eval_Ht = eval_jacobian_hux_Ht()  # mettendo qui le funzioni vemngono generate ogni volta e poi eliminate, poco efficiente
            # self.get_logger().info(f'\nUpdate DONE!')
            Ht_args=(self.robot.mu[0], self.robot.mu[1], self.robot.mu[2], self.robot.mu[3], self.robot.mu[4], self.landmarks[id]["x"], self.landmarks[id]["y"])
            hx_args=Ht_args
            print(f"L_x = {self.landmarks[id]['x']} \t\tL_y= {self.landmarks[id]['y']}")

            #pdb.set_trace()
            self.robot.update(z,eval_hx=self.eval_hx, eval_Ht=self.eval_Ht, Qt=self.Qt, Ht_args=Ht_args,hx_args=hx_args, residual=residual_measurement)

        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"

        x,y,theta,v,w=self.robot.mu
        self.get_logger().info(f'\nekf mu:\n{x,y,theta,v,w}')
        msg.pose.pose.position.x=x
        msg.pose.pose.position.y=y
        msg.pose.pose.position.z=0.0 
        msg.twist.twist.linear.x=v
        msg.twist.twist.angular.z=w
        
        # self.get_logger().info(f'\nekf mu:\n{self.robot.mu}')

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
        # Estrai posizione e orientamento reali
        x_gt = msg.pose.pose.position.x
        y_gt = msg.pose.pose.position.y
        
        orientation_q = msg.pose.pose.orientation
        (_, _, theta_gt) = tf_transformations.euler_from_quaternion(
            [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        )

        # --- AGGIUNGI QUESTO BLOCCO ---
        if not self.initialized:
            # Inizializza lo stato: [x, y, theta, v, w]
            # Prendiamo posizione dalla GT, velocità a zero
            self.robot.mu = np.array([x_gt, y_gt, theta_gt, 0.0, 0.0])
            
            # Opzionale: Se vuoi azzerare anche l'incertezza iniziale perché ti fidi della GT
            # self.robot.Sigma = np.eye(5) * 0.001 
            
            self.initialized = True
            self.get_logger().info(f"EKF Inizializzato su GT: {self.robot.mu}")
            return



def main(args=None):
    rclpy.init(args=args)
    EKF = EKF_ESTIMATE()
    rclpy.spin(EKF)
    EKF.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
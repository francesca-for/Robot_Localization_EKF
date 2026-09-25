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
# classe da importare per leggere i rosbag
from rosbag2_reader_py import Rosbag2Reader
# funzioni per il calcolo delle metriche
from Gaussian_Filters.utils import rmse, mae


def get_yaw(orientation):
    """
    Get yaw (theta) from quaternion manually to avoid problems with tf_transformations
    """
    x = orientation.x
    y = orientation.y
    z = orientation.z
    w = orientation.w
    
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    return yaw


def normalize_angle(angle):
    """Normalizza l'angolo tra -pi e pi."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def get_data_from_rosbag(rosbag_path, topics):
    reader = Rosbag2Reader(rosbag_path)

    gt_time = []
    gt_data = []
    ekf_time = []
    ekf_data = []
    odom_time = []
    odom_data = []

    reader.set_filter(topics)

    for topic_name, msg, t in reader:
        # extract timestamp
        time_ns = Time.from_msg(msg.header.stamp).nanoseconds

        # extract position and orientation
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = get_yaw(msg.pose.pose.orientation)

        if topic_name == "/ground_truth":
            gt_time.append(time_ns)
            gt_data.append([x, y, theta])
        elif topic_name == "/ekf":
            ekf_time.append(time_ns)
            ekf_data.append([x, y, theta])
        elif topic_name == "/odom":
            odom_time.append(time_ns)
            odom_data.append([x, y, theta])

    return (np.array(gt_time), np.array(gt_data), 
            np.array(ekf_time), np.array(ekf_data), 
            np.array(odom_time), np.array(odom_data))


def compute_metrics(gt_time, gt_data, topic_time, topic_data):
    # compute Root Mean Square  Error  (RMSE)  and  Mean  Absolute  Error  (MAE) w.r.t. ground truth
    
    if len(topic_data) == 0 or len(gt_data) == 0:
        return None

    # prepare ground truth for interpolation
    gt_data_unwrapped = np.copy(gt_data)
    gt_data_unwrapped[:, 2] = np.unwrap(gt_data[:, 2])

    # interpolate
    gt_interpol_func = interp1d(gt_time, gt_data_unwrapped, axis=0, fill_value="extrapolate", kind="nearest")
    gt_data_interp = gt_interpol_func(topic_time)

    # normalize theta after interpolation
    gt_data_interp[:, 2] = normalize_angle(gt_data_interp[:, 2])

    # error computation
    error_x = topic_data[:, 0] - gt_data_interp[:, 0]
    error_y = topic_data[:, 1] - gt_data_interp[:, 1]

    # compute theta error with normalization
    raw_error_theta = topic_data[:, 2] - gt_data_interp[:, 2]
    error_theta = normalize_angle(raw_error_theta)

    # compute metrics
    metrics = {        
        'rmse_x': rmse(gt_data_interp[:, 0], topic_data[:, 0]),
        'mae_x':  mae(error_x),
        
        'rmse_y': rmse(gt_data_interp[:, 1], topic_data[:, 1]),
        'mae_y':  mae(error_y),

        # for theta, we pecomputed the errors to handle angle wrapping, so we pass zeros as second parameter
        'rmse_th': rmse(error_theta, np.zeros_like(error_theta)),
        'mae_th':  mae(error_theta)
    }
    
    return metrics


def print_metrics(name, metrics):
    if metrics is None:
        return
    print(f"\n----- Metrics for {name} -----")
    print(f"X     -> RMSE: {metrics['rmse_x']:.4f} m   | MAE: {metrics['mae_x']:.4f} m")
    print(f"Y     -> RMSE: {metrics['rmse_y']:.4f} m   | MAE: {metrics['mae_y']:.4f} m")
    print(f"Theta -> RMSE: {metrics['rmse_th']:.4f} rad | MAE: {metrics['mae_th']:.4f} rad")


def main():
    rosbag_path_t1 = "/home/ubuntu/ros2_ws/src/lab04_pkg/simulated_robot_task1" 
    rosbag_path_t2 = "/home/ubuntu/ros2_ws/src/lab04_pkg/simulated_robot_task2"
    useful_topics = ["/ground_truth", "/ekf", "/odom"]

    # get data
    while True:
        case = input("Compute metrics for task1 or task2? (enter 1 or 2): ")
        if case == '1':
            rosbag_path = rosbag_path_t1
            break
        elif case == '2':
            rosbag_path = rosbag_path_t2
            break

    gt_time, gt_data, ekf_time, ekf_data, odom_time, odom_data = get_data_from_rosbag(rosbag_path, useful_topics)

    print("Computing metrics with respect to ground truth:")
    
    # metrics for /odom
    if len(odom_data) == 0:
        print(f"No data found for {useful_topics[2]}")
    else:
        metrics_odom = compute_metrics(gt_time, gt_data, odom_time, odom_data)
        print_metrics("ODOMETRY", metrics_odom)

    # metrics for /ekf
    if len(ekf_data) == 0:
        print(f"No data found for {useful_topics[1]}")
    else:
        metrics_ekf = compute_metrics(gt_time, gt_data, ekf_time, ekf_data)
        print_metrics("EKF", metrics_ekf)


if __name__ == "__main__":
    main()
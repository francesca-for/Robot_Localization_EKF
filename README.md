# ROS 2 Mobile Robotics: Navigation, Perception and EKF Localization

![ROS 2](https://img.shields.io/badge/ros2-humble-blue?logo=ros)
![Python](https://img.shields.io/badge/python-3.10-blue?logo=python)

## Overview
This repository contains a collection of ROS 2 packages and Python implementations focused on mobile robotics (TurtleBot3), probabilistic localization, and autonomous navigation. 

The core focus of the project (`lab04_pkg`) is a custom **Extended Kalman Filter (EKF)** for 2D robot localization, fusing internal odometry (velocity commands) with external observations (range and bearing from landmarks) to estimate the robot's pose in real-time, both in simulation and on real-world rosbag datasets.

## Repository Structure

* **`src/lab04_pkg` (Main Project - EKF Localization):** Custom Extended Kalman Filter implementation, velocity motion model sampling (`task0.py`), and sensor fusion nodes (`task1.py`, `task2.py`, `task3.py`).
* **`src/lab01_pkg`:** Foundational ROS 2 nodes for basic robot control, pose reset, and odometry tracking.
* **`src/lab02_pkg` & `src/lab03_pkg`:** Reactive navigation controllers implementing obstacle avoidance algorithms (*Bump and Go*).
* **`212_Report01.pdf` & `212_Report02.pdf`:** Detailed technical reports analyzing the mathematical models, experimental setups, and localization performance plots.

## Key Features (EKF Localization)
* **Probabilistic Velocity Motion Model:** Implements sampling algorithms to visualize non-linear error propagation (the classic "banana distribution" effect) under high angular and linear noise.
* **Landmark Sensor Fusion:** Fuses range and bearing measurements from known landmarks, computing the likelihood of measurements.
* **Dynamic Symbolic Jacobians:** Uses `SymPy` to mathematically derive Jacobians w.r.t. state ($H_t$) and control ($G_t$). The matrices are lambdified and evaluated analytically at node initialization to ensure real-time ROS 2 performance without computational lag.
* **Stable EKF Updates:** Handles edge cases such as angle wrapping ($-\pi$ to $\pi$ boundaries) during the measurement residual calculation to prevent filter divergence.

## Prerequisites and External Dependencies
To run the code in this workspace, the following environment is required:
* **ROS 2** (Humble/Iron)
* **Python 3.10+**
* Python libraries: `numpy`, `scipy`, `matplotlib`, `sympy`, `jupyter`

```bash
pip install numpy scipy matplotlib sympy jupyter
```

**Note on External Packages:** To keep this repository clean and focused on custom implementations, third-party simulation, perception, and course dependencies (`turtlebot3_simulations`, `turtlebot3_perception`, `landmark_msgs`, `planning_control_method`, `probabilistic-robotics-python-examples`) are excluded via `.gitignore`. They must be cloned separately into the `src/` directory to run the full Gazebo/Ignition simulation environment.

## Build and Run

1. **Clone the repository:**
```bash
git clone https://github.com/francesca-for/SESASR_ros2_ws.git ~/ros2_ws
cd ~/ros2_ws
```

2. **Build the packages:**
```bash
colcon build
source install/setup.bash
```

3. **Run the standalone Motion Model Sampling script:**
```bash
python3 src/lab04_pkg/lab04_pkg/task0.py
```

4. **Run the ROS 2 EKF Localization nodes:**
```bash
ros2 run lab04_pkg task1
# Alternatively: task2 or task3 depending on the configuration
```

## Technical Challenges Solved
1. **Real-Time Computational Bottlenecks:** Evaluating `SymPy` symbolic Jacobians inside the high-frequency ROS 2 landmark callback initially caused message queuing and pose "teleportation". This was solved by pre-computing and lambdifying the symbolic expressions once during the node `__init__` phase.
2. **Angle Wrapping Divergence:** Addressed filter divergence during sharp rotations by implementing a custom modular arithmetic residual function that strictly normalizes angular differences within $[-\pi, \pi]$.
import time
from launch import LaunchDescription
from launch.actions import ExecuteProcess

def generate_launch_description():

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    bag_path = f"/home/s344605/ros2_ws/bags/session_{timestamp}"

    # Nodo controllore
    controllore_terminal = ExecuteProcess(
        cmd=["xterm", "-fa", "Monospace", "-fs", "12", "-hold", "-e",
             "ros2 run lab01_pkg controller_reset"],
        output="screen"
    )

    # Nodo localizzatore
    localizzatore_terminal = ExecuteProcess(
        cmd=["xterm", "-fa", "Monospace", "-fs", "12", "-hold", "-e",
             "ros2 run lab01_pkg localization_reset"],
        output="screen"
    )

    # Nodo reset (quello che hai mostrato tu)
    reset_terminal = ExecuteProcess(
        cmd=["xterm", "-fa", "Monospace", "-fs", "12", "-hold", "-e",
             "ros2 run lab01_pkg reset_node"],
        output="screen"
    )

    return LaunchDescription([
        controllore_terminal,
        localizzatore_terminal,
        reset_terminal,
    ])

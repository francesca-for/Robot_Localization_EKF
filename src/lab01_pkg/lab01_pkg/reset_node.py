import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Bool
import math

class Listen(Node):
    def __init__(self):
        super().__init__("reset_node")
        self.subscriber=self.create_subscription(Pose,"/pose", self.listen_callback, 10)
        self.publisher=self.create_publisher(Bool,"/reset",10)
    
    def listen_callback(self,msg):
        rst_msg=Bool()
        distance=math.sqrt(msg.position.x**2+msg.position.y**2) 
        if distance > 6:
            rst_msg.data=True
            self.publisher.publish(rst_msg)
            self.get_logger().warn("Distance > 6 m! Publishing reset=True")
        if distance < 6:
            rst_msg.data=False


def main(param=None):
    rclpy.init(args=param)
    reset_node=Listen()
    rclpy.spin(reset_node)
    reset_node.destroy_node
    rclpy.shutdown


if __name__=="__main__":
    main()
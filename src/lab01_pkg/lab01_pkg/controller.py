import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist,Pose
from std_msgs.msg import Bool

class Minimalpublisher(Node):
    def __init__(self):
        super().__init__("controller")
        self.publisher=self.create_publisher(Twist, "/cmd_vel", 10)
        #self.subscriber=self.create_subscription(Bool, "/reset", self.listen_callback, 10)
        timer_period=1
        self.timer=self.create_timer(timer_period,self.callback_function)
        self.i=1  # seconds until being back at origin position
        self.m=4  # perimeter length 
        self.c=1  # cycle counter
    

    def callback_function(self):
        msg=Twist()
        change_dir=self.m/4

        if self.i <= change_dir:
            msg.linear.x=1.0
        elif self.i <= 2*change_dir:
            msg.linear.y =1.0
        elif self.i <= 3*change_dir:
            msg.linear.x=-1.0
        elif self.i <= 4*change_dir:
            msg.linear.y=-1.0
        
        self.i+=1
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: x_dot={msg.linear.x}, y_dot={msg.linear.y}')

        if self.i > self.m:
            self.i = 1
            self.c+=1
            self.m=self.m*self.c
            self.get_logger().info(f'Completed cycle! Now p={self.m} and the robot will change the direction every {self.m/4} seconds\n')

    

def main(param=None):
    rclpy.init(args=param)
    controller=Minimalpublisher()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown

if __name__=="__main__":
    main()

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose


class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__("localization")
        self.subscriber=self.create_subscription(Twist,"/cmd_vel", self.sub_callback_func, 10)
        self.pose_publisher=self.create_publisher(Pose,"/pose",10)
        timer_period=1
        self.timer=self.create_timer(timer_period,self.timer_callback)

        self.x=0.0
        self.y=0.0
        self.theta=0.0


    def sub_callback_func(self,msg):
        #self.get_logger().info(f'Receiving from topic /cmd_vel velocity data: {msg}')
        
        dt=1.0
        self.x=self.x+msg.linear.x*dt
        self.y=self.y+msg.linear.y*dt
        #self.theta=self.theta+msg.angular.z*dt
        if self.x == 0 and self.y ==0:
            self.get_logger().info(f" Completed a square! \n")
    

    def timer_callback(self):
        pose_msg=Pose()
        pose_msg.position.x=self.x
        pose_msg.position.y=self.y
        pose_msg.orientation.z=self.theta
        self.pose_publisher.publish(pose_msg)
        self.get_logger().info(f'Publishing pose:  x={pose_msg.position.x}, y={pose_msg.position.y}')

        #theta={pose_msg.orientation.z}'




def main(param=None):
    rclpy.init(args=param)
    localization_node=MinimalSubscriber()
    rclpy.spin(localization_node)
    localization_node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()
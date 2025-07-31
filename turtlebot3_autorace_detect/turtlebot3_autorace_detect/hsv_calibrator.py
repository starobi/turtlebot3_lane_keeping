import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class HSVCalibrator(Node):
    def __init__(self):
        super().__init__('hsv_calibrator')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  # ajusta según tu topic
            self.image_callback,
            10)
        self.cv_bridge = CvBridge()

        # Ventana y sliders
        cv2.namedWindow("HSV Calibrator")
        cv2.createTrackbar("H Low", "HSV Calibrator", 0, 179, lambda x: None)
        cv2.createTrackbar("H High", "HSV Calibrator", 179, 179, lambda x: None)
        cv2.createTrackbar("S Low", "HSV Calibrator", 0, 255, lambda x: None)
        cv2.createTrackbar("S High", "HSV Calibrator", 255, 255, lambda x: None)
        cv2.createTrackbar("V Low", "HSV Calibrator", 0, 255, lambda x: None)
        cv2.createTrackbar("V High", "HSV Calibrator", 255, 255, lambda x: None)

    def image_callback(self, msg):
        frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h_l = cv2.getTrackbarPos("H Low", "HSV Calibrator")
        h_h = cv2.getTrackbarPos("H High", "HSV Calibrator")
        s_l = cv2.getTrackbarPos("S Low", "HSV Calibrator")
        s_h = cv2.getTrackbarPos("S High", "HSV Calibrator")
        v_l = cv2.getTrackbarPos("V Low", "HSV Calibrator")
        v_h = cv2.getTrackbarPos("V High", "HSV Calibrator")

        lower = np.array([h_l, s_l, v_l])
        upper = np.array([h_h, s_h, v_h])

        mask = cv2.inRange(hsv, lower, upper)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow("HSV Calibrator", result)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = HSVCalibrator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


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

        # Crear ventana y sliders para dos rangos (rojo bajo y rojo alto)
        cv2.namedWindow("HSV Calibrator")

        # Rojo bajo (0–20)
        cv2.createTrackbar("H Low 1", "HSV Calibrator", 0, 179, lambda x: None)
        cv2.createTrackbar("H High 1", "HSV Calibrator", 20, 179, lambda x: None)
        cv2.createTrackbar("S Low 1", "HSV Calibrator", 100, 255, lambda x: None)
        cv2.createTrackbar("S High 1", "HSV Calibrator", 255, 255, lambda x: None)
        cv2.createTrackbar("V Low 1", "HSV Calibrator", 100, 255, lambda x: None)
        cv2.createTrackbar("V High 1", "HSV Calibrator", 255, 255, lambda x: None)

        # Rojo alto (160–179)
        cv2.createTrackbar("H Low 2", "HSV Calibrator", 160, 179, lambda x: None)
        cv2.createTrackbar("H High 2", "HSV Calibrator", 179, 179, lambda x: None)
        cv2.createTrackbar("S Low 2", "HSV Calibrator", 100, 255, lambda x: None)
        cv2.createTrackbar("S High 2", "HSV Calibrator", 255, 255, lambda x: None)
        cv2.createTrackbar("V Low 2", "HSV Calibrator", 100, 255, lambda x: None)
        cv2.createTrackbar("V High 2", "HSV Calibrator", 255, 255, lambda x: None)

    def image_callback(self, msg):
        frame = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Primer rango (rojo bajo)
        h_l1 = cv2.getTrackbarPos("H Low 1", "HSV Calibrator")
        h_h1 = cv2.getTrackbarPos("H High 1", "HSV Calibrator")
        s_l1 = cv2.getTrackbarPos("S Low 1", "HSV Calibrator")
        s_h1 = cv2.getTrackbarPos("S High 1", "HSV Calibrator")
        v_l1 = cv2.getTrackbarPos("V Low 1", "HSV Calibrator")
        v_h1 = cv2.getTrackbarPos("V High 1", "HSV Calibrator")

        # Segundo rango (rojo alto)
        h_l2 = cv2.getTrackbarPos("H Low 2", "HSV Calibrator")
        h_h2 = cv2.getTrackbarPos("H High 2", "HSV Calibrator")
        s_l2 = cv2.getTrackbarPos("S Low 2", "HSV Calibrator")
        s_h2 = cv2.getTrackbarPos("S High 2", "HSV Calibrator")
        v_l2 = cv2.getTrackbarPos("V Low 2", "HSV Calibrator")
        v_h2 = cv2.getTrackbarPos("V High 2", "HSV Calibrator")

        lower_red1 = np.array([h_l1, s_l1, v_l1])
        upper_red1 = np.array([h_h1, s_h1, v_h1])
        lower_red2 = np.array([h_l2, s_l2, v_l2])
        upper_red2 = np.array([h_h2, s_h2, v_h2])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

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

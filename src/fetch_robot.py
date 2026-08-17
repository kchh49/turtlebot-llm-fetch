import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO


class FetchRobot(Node):
    def __init__(self, target_class):
        super().__init__('fetch_robot')
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')
        self.target_class = target_class

        self.subscription = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self.image_callback, 10
        )
        self.odom_subscription = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.current_x = 0.0
        self.current_y = 0.0

        # 가상 경계 (1.5m x 1.5m, 시작 위치가 중앙이라고 가정)
        self.boundary_min_x = -0.75
        self.boundary_max_x = 0.75
        self.boundary_min_y = -0.75
        self.boundary_max_y = 0.75

        self.state = 'searching'  # 'searching' / 'approaching' / 'arrived'
        self.miss_count = 0
        self.miss_threshold = 10

        self.emergency_stop = False

    def odom_callback(self, msg):
        # 맨 처음 받은 오도메트리 값을 기준점(원점)으로 저장
        if not hasattr(self, 'origin_x'):
            self.origin_x = msg.pose.pose.position.x
            self.origin_y = msg.pose.pose.position.y
            self.get_logger().info(
                f'시작 위치를 원점으로 설정: ({self.origin_x:.2f}, {self.origin_y:.2f})'
            )

        self.current_x = msg.pose.pose.position.x - self.origin_x
        self.current_y = msg.pose.pose.position.y - self.origin_y

    def is_out_of_boundary(self):
        return (self.current_x < self.boundary_min_x or self.current_x > self.boundary_max_x or
                self.current_y < self.boundary_min_y or self.current_y > self.boundary_max_y)

    def image_callback(self, msg):
        frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        h, w, _ = frame.shape
        frame_center_x = w / 2

        # 비상 정지
        if self.emergency_stop:
            twist = Twist()
            self.cmd_pub.publish(twist)
            cv2.putText(frame, 'EMERGENCY STOP - press R to resume', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow('Fetch Robot', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self.emergency_stop = False
                self.get_logger().info('비상 정지 해제, 재개')
            return

        # 가상 경계 이탈 체크
        if self.is_out_of_boundary() and self.state != 'arrived':
            twist = Twist()
            twist.linear.x = -0.05
            twist.angular.z = 0.3
            self.get_logger().info(
                f'가상 경계 이탈 위험! 안쪽으로 이동 (x={self.current_x:.2f}, y={self.current_y:.2f})'
            )
            self.cmd_pub.publish(twist)
            cv2.imshow('Fetch Robot', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                self.emergency_stop = True
            return

        results = self.model(frame, verbose=False)
        boxes = results[0].boxes

        target_box = None
        best_conf = 0

        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0])

            if cls_name == self.target_class and conf > best_conf:
                best_conf = conf
                target_box = box

        twist = Twist()

        # 타겟 발견됨 
        if target_box is not None:
            self.miss_count = 0
            x1, y1, x2, y2 = target_box.xyxy[0].tolist()
            center_x = (x1 + x2) / 2
            box_height = y2 - y1
            height_ratio = box_height / h

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            cv2.putText(frame, f'{self.state}', (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if height_ratio > 0.4:
                self.state = 'arrived'
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.get_logger().info(f'도착! {self.target_class} 찾았습니다. 정지')
                cv2.putText(frame, "press 'n' to search a new item", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                self.state = 'approaching'
                error_x = center_x - frame_center_x
                twist.angular.z = -0.002 * error_x
                speed = 0.1 * (1 - height_ratio / 0.4)
                twist.linear.x = max(speed, 0.03)
                self.get_logger().info(
                    f'접근 중: error_x={error_x:.0f}, height_ratio={height_ratio:.2f}, speed={twist.linear.x:.3f}'
                )

            self.cmd_pub.publish(twist)

        # 타겟 없음
        else:
            self.miss_count += 1
            if self.miss_count < self.miss_threshold:
                self.get_logger().info(f'일시적 미검출 ({self.miss_count}/{self.miss_threshold})')
            else:
                self.state = 'searching'
                twist.linear.x = 0.0
                twist.angular.z = 0.3
                self.get_logger().info('탐색 중... (제자리 회전)')
                self.cmd_pub.publish(twist)

        cv2.imshow('Fetch Robot', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            self.emergency_stop = True
            self.get_logger().info('!!! 비상 정지 발동 !!!')
        elif key == ord('n') and self.state == 'arrived':
            print('\n=== 새 타겟을 다시 찾을 수 있습니다 ===')
            new_target = choose_target_class(self.model)
            self.target_class = new_target
            self.state = 'searching'
            self.miss_count = 0
            self.get_logger().info(f'새 타겟 설정: {new_target} / 재탐색 시작')


def choose_target_class(model):
    class_names = list(model.names.values())
    print('=== 찾을 수 있는 물체 목록 (COCO 80 클래스) ===')
    for i, name in enumerate(class_names):
        print(f'{i}: {name}', end='  ')
        if (i + 1) % 5 == 0:
            print()
    print('\n')

    while True:
        user_input = input('찾을 물체 이름을 입력하세요 (예: cup): ').strip().lower()
        if user_input in class_names:
            return user_input
        else:
            print(f'"{user_input}"은(는) 목록에 없는 클래스입니다. 다시 입력해주세요.')


def main():
    print('YOLO 모델 로딩 중...')
    model = YOLO('yolov8n.pt')
    target_class = choose_target_class(model)
    print(f'타겟: {target_class} / 로봇을 시작 위치(구역 중앙)에 놓고 시작합니다.')

    rclpy.init()
    node = FetchRobot(target_class)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_twist = Twist()
        node.cmd_pub.publish(stop_twist)
        node.get_logger().info('종료: 정지 명령 발행')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
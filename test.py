import pybullet as p
import time

# Kết nối đến Bullet (sử dụng GUI để xem)
p.connect(p.DIRECT)

# Load mô hình robot từ URDF
robot_id = p.loadURDF("./assets/urdf/ur5_robotiq_85.urdf", useFixedBase=True)

# Lấy số lượng khớp của robot
num_joints = p.getNumJoints(robot_id)

# Lặp qua từng khớp và lấy thông tin giới hạn
for joint_index in range(num_joints):
    joint_info = p.getJointInfo(robot_id, joint_index)

    # Giới hạn góc của khớp
    lower_limit = joint_info[8]  # Lower limit of the joint
    upper_limit = joint_info[9]  # Upper limit of the joint

    print(
        f"Khớp {joint_index}: Giới hạn góc - Lower: {lower_limit}, Upper: {upper_limit}"
    )

# Giữ Bullet simulation chạy một lúc để xem kết quả
time.sleep(10)

# Ngắt kết nối
p.disconnect()

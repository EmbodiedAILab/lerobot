import argparse
import time
from pathlib import Path
import logging
import threading
import traceback

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float64

from lerobot.common.utils.utils import init_logging
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
)

class ActionPub(Node):
    def __init__(self, action_frames):
        super().__init__('action_pub')
        self.action_frames = action_frames
        self.right_pub = self.create_publisher(Float64MultiArray, "/right_arm_position_controller/commands", 10)
        self.left_pub = self.create_publisher(Float64MultiArray, "/left_arm_position_controller/commands", 10)
        
        self.left_gripper_pub = self.create_publisher(Float64MultiArray, "/left_gripper_position_controller/commands", 10)
        self.right_gripper_pub = self.create_publisher(Float64MultiArray, "/right_gripper_position_controller/commands", 10)

        self.timer = self.create_timer(1/30.0, self.timer_callback)
        self.step_count = 0
        
    def timer_callback(self):
        print('tick once')
        if self.step_count >= len(self.action_frames):
            print("episodes done, skip pub")
            return
        
        print(f"=== frames: {self.action_frames[self.step_count]}")
        action = self.action_frames[self.step_count].numpy().tolist()
        logging.info(f'action: {action}')
        left_arm = action[0:6]
        left_gripper = action[6]
        left_grippers = [left_gripper/2.0, left_gripper/2.0]
        
        right_arm = action[7:13]
        right_gripper = action[13]
        right_grippers = [right_gripper/2.0, right_gripper/2.0]
        
        left_msg = Float64MultiArray()
        left_msg.data = left_arm 
        self.left_pub.publish(left_msg)

        right_msg = Float64MultiArray()
        right_msg.data = right_arm
        self.right_pub.publish(right_msg)

        left_gripper_msg = Float64MultiArray()
        left_gripper_msg.data = left_grippers
        self.left_gripper_pub.publish(left_gripper_msg)

        right_gripper_msg = Float64MultiArray()
        right_gripper_msg.data = right_grippers
        self.right_gripper_pub.publish(right_gripper_msg)

        self.step_count = self.step_count + 1
        logging.info("pub once")


def parse_args():
    parser = argparse.ArgumentParser(description='Robot Control with Policy Selection')
    parser.add_argument('--inference_time', type=int, default=600,
                        help='Inference time in seconds')
    parser.add_argument('--rate', type=float, default=30.0,
                        help='Publishing rate in Hz')
    parser.add_argument('--repo_id', type=str,
                        help='Record repo id')
    parser.add_argument('--root', type=str,
                        help='Record root directory')
    parser.add_argument('--episode_index', type=int,
                        help='episode index')
    return parser.parse_args()


def main():
    args = parse_args()
    logging.info(f"args: {args}")
    rclpy.init()
    init_logging()
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.root,
        # episodes=[args.episode_index],
        episodes=None,
        # revision='v20',
        local_files_only=True,
    )
    
    print("data loaded")
    from_id = dataset.episode_data_index['from'][0].item()
    to_id = dataset.episode_data_index['to'][0].item()
    print("id acquired", from_id, to_id)
    action_frames = [dataset[idx]['observation.state'] for idx in range(from_id, to_id)]
    print("frames got")
    # Main control loop
    total_steps = len(action_frames)
    step_count = 0
    print(f"episode #{args.episode_index} has frames {total_steps}")
    
    node = ActionPub(action_frames)
    print("node inited")
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        logging.warning("Shutting down gracefully...")
    except Exception as e:
        print(traceback.format_exc())
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()

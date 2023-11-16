import logging
import os
import pickle
from enum import Enum
from typing import Any, Dict

import numpy as np
import rlbench.backend.observation
import rlbench.demo
import rlbench.utils
import torch
import torch.utils.data as data
import tree
from pyrep.backend import sim
from rlbench.observation_config import CameraConfig, ObservationConfig
from rlbench.tasks import (
    InsertOntoSquarePeg,
    InsertUsbInComputer,
    PhoneOnBase,
    PutToiletRollOnStand,
    StackWine,
)
from scipy.spatial.transform import Rotation as R


class StackWinePhase(str, Enum):
    GRASP = "grasp"
    PLACE = "place"


# _TASK_HANDLES = {
#     "stack_wine": {
#         "task_class": StackWine,
#         "phase": {
#             "grasp": {
#                 "action_obj_handles": [221, 222, 225],
#                 "anchor_obj_handles": [160],
#             },
#             "place": {
#                 "action_obj_handles": [160],
#                 "anchor_obj_handles": [152, 154],
#             },
#         },
#     },
# }

TASK_DICT = {
    "stack_wine": {
        "task_class": StackWine,
        "phase": {
            "grasp": {
                "action_obj_names": [
                    "Panda_leftfinger_visual",
                    "Panda_rightfinger_visual",
                    "Panda_gripper_visual",
                ],
                "anchor_obj_names": ["wine_bottle_visual"],
                "action_pose_name": "gripper",
            },
            "place": {
                "action_obj_names": ["wine_bottle_visual"],
                "anchor_obj_names": ["rack_bottom_visual", "rack_top_visual"],
                "action_pose_name": "wine_bottle",
            },
        },
    },
    "insert_onto_square_peg": {
        "task_class": InsertOntoSquarePeg,
        "phase": {
            "grasp": {
                "action_obj_names": [
                    "Panda_leftfinger_visual",
                    "Panda_rightfinger_visual",
                    "Panda_gripper_visual",
                ],
                "anchor_obj_names": ["square_ring"],
                "action_pose_name": "gripper",
            },
            "place": {
                "action_obj_names": ["square_ring"],
                "anchor_obj_names": ["square_base", "pillar0", "pillar1", "pillar2"],
                "action_pose_name": "square_ring",
            },
        },
    },
    # THIS ONE SEEMS TO BE BROKEN
    "insert_usb_in_computer": {
        "task_class": InsertUsbInComputer,
        "phase": {
            "grasp": {
                "action_obj_names": [
                    "Panda_leftfinger_visual",
                    "Panda_rightfinger_visual",
                    "Panda_gripper_visual",
                ],
                "anchor_obj_names": ["usb", "usb_visual0", "usb_visual1", "tip"],
            },
            "place": {
                "action_obj_names": ["usb", "usb_visual0", "usb_visual1", "tip"],
                "anchor_obj_names": ["computer", "computer_visual"],
            },
        },
    },
    "phone_on_base": {
        "task_class": PhoneOnBase,
        "phase": {
            "grasp": {
                "action_obj_names": [
                    "Panda_leftfinger_visual",
                    "Panda_rightfinger_visual",
                    "Panda_gripper_visual",
                ],
                "anchor_obj_names": ["phone", "phone_visual"],
            },
            "place": {
                "action_obj_names": ["phone", "phone_visual"],
                "anchor_obj_names": ["phone_case", "phone_case_visual"],
            },
        },
    },
    "put_toilet_roll_on_stand": {
        "task_class": PutToiletRollOnStand,
        "phase": {
            "grasp": {
                "action_obj_names": [
                    "Panda_leftfinger_visual",
                    "Panda_rightfinger_visual",
                    "Panda_gripper_visual",
                ],
                "anchor_obj_names": ["toilet_roll_visual"],
            },
            "place": {
                "action_obj_names": ["toilet_roll_visual"],
                "anchor_obj_names": ["holder_visual", "stand_base"],
            },
        },
    },
}


def get_rgb_point_cloud_by_object_handles(rgb, point_cloud, seg_labels, handles):
    indices = np.isin(seg_labels, handles).reshape((-1))
    rgb = rgb[indices]
    point_cloud = point_cloud[indices]
    return rgb, point_cloud


# Get rgb and point cloud for all points whose mask matches the given handles.
def get_rgb_point_cloud_by_object_names(rgb, point_cloud, seg_labels, names):
    handles = []
    for name in names:
        try:
            handles.append(sim.simGetObjectHandle(name))
        except RuntimeError:
            logging.warning(f"Object {name} not found in scene.")

    return get_rgb_point_cloud_by_object_handles(rgb, point_cloud, seg_labels, handles)


def obs_to_rgb_point_cloud(obs):
    # Get the overhead, left, front, and right RGB images.
    overhead_rgb = obs.overhead_rgb
    left_rgb = obs.left_shoulder_rgb
    right_rgb = obs.right_shoulder_rgb
    front_rgb = obs.front_rgb
    wrist_rgb = obs.wrist_rgb

    # Get the overhead, left, front, and right point clouds. The point clouds are
    # in the same shape as the images.
    overhead_point_cloud = obs.overhead_point_cloud
    left_point_cloud = obs.left_shoulder_point_cloud
    right_point_cloud = obs.right_shoulder_point_cloud
    front_point_cloud = obs.front_point_cloud
    wrist_point_cloud = obs.wrist_point_cloud

    # Get masks.
    overhead_mask = obs.overhead_mask
    left_mask = obs.left_shoulder_mask
    right_mask = obs.right_shoulder_mask
    front_mask = obs.front_mask
    wrist_mask = obs.wrist_mask

    # Flatten RGB and point cloud images into Nx3 arrays
    overhead_rgb = overhead_rgb.reshape((-1, 3))
    left_rgb = left_rgb.reshape((-1, 3))
    right_rgb = right_rgb.reshape((-1, 3))
    front_rgb = front_rgb.reshape((-1, 3))
    wrist_rgb = wrist_rgb.reshape((-1, 3))

    overhead_point_cloud = overhead_point_cloud.reshape((-1, 3))
    left_point_cloud = left_point_cloud.reshape((-1, 3))
    right_point_cloud = right_point_cloud.reshape((-1, 3))
    front_point_cloud = front_point_cloud.reshape((-1, 3))
    wrist_point_cloud = wrist_point_cloud.reshape((-1, 3))

    # # Transform each point cloud into the world frame.
    # def transform_point_cloud(point_cloud, camera_extrinsics):
    #     point_cloud = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
    #     point_cloud = np.matmul(np.linalg.inv(camera_extrinsics), point_cloud.T).T
    #     return point_cloud[:, :3]

    # overhead_point_cloud = transform_point_cloud(
    #     overhead_point_cloud, obs.misc["overhead_camera_extrinsics"]
    # )
    # left_point_cloud = transform_point_cloud(
    #     left_point_cloud, obs.misc["left_shoulder_camera_extrinsics"]
    # )
    # right_point_cloud = transform_point_cloud(
    #     right_point_cloud, obs.misc["right_shoulder_camera_extrinsics"]
    # )
    # front_point_cloud = transform_point_cloud(
    #     front_point_cloud, obs.misc["front_camera_extrinsics"]
    # )
    # wrist_point_cloud = transform_point_cloud(
    #     wrist_point_cloud, obs.misc["wrist_camera_extrinsics"]
    # )

    # Reshape the masks into Nx1 arrays.
    overhead_mask = overhead_mask.reshape((-1, 1))
    left_mask = left_mask.reshape((-1, 1))
    right_mask = right_mask.reshape((-1, 1))
    front_mask = front_mask.reshape((-1, 1))
    wrist_mask = wrist_mask.reshape((-1, 1))

    # Stack the RGB and point cloud images together.
    rgb = np.vstack(
        (
            overhead_rgb,
            left_rgb,
            right_rgb,
            front_rgb,
            # wrist_rgb,
        )
    )
    point_cloud = np.vstack(
        (
            overhead_point_cloud,
            left_point_cloud,
            right_point_cloud,
            front_point_cloud,
            # wrist_point_cloud,
        )
    )
    mask = np.vstack(
        (
            overhead_mask,
            left_mask,
            right_mask,
            front_mask,
            # wrist_mask,
        )
    )

    return rgb, point_cloud, mask


def load_handle_mapping(
    dataset_root: str, task_name: str, variation: int
) -> Dict[str, int]:
    handle_path = os.path.join(
        dataset_root, task_name, f"variation{variation}", "handles.pkl"
    )

    with open(handle_path, "rb") as f:
        names_to_handles = pickle.load(f)

    return names_to_handles


def load_state_pos_dict(
    dataset_root: str, task_name: str, variation: int, episode: int
) -> Dict[str, np.ndarray]:
    state_path = os.path.join(
        dataset_root,
        task_name,
        f"variation{variation}",
        "episodes",
        f"episode{episode}",
        "raw_state_pos_dict.pkl",
    )

    with open(state_path, "rb") as f:
        state_pos_dict = pickle.load(f)

    return state_pos_dict


class RLBenchPlacementDataset(data.Dataset):
    def __init__(
        self,
        dataset_root: str,
        task_name: str = "stack_wine",
        n_demos: int = 10,
        phase: StackWinePhase = StackWinePhase.GRASP,
    ) -> None:
        """Dataset for RL-Bench placement tasks.

        Args:
            dataset_root (str): The root of where the RLBench demonstrations were generated.
        """
        super().__init__()

        self.dataset_root = dataset_root
        self.task_name = task_name
        self.n_demos = n_demos
        self.phase = phase

        if self.task_name not in TASK_DICT:
            raise ValueError(f"Task name {self.task_name} not supported.")

        logging.info(f"Loading {self.n_demos} demos for task {self.task_name}...")
        self.demos, self.names_to_handles = self._load_demos()
        logging.info("Demos loaded!")

    def _load_demos(self) -> Any:
        ## This is overkill! We don't actually need to make the environment...
        # action_mode = MoveArmThenGripper(
        #     arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
        # )

        # env = Environment(action_mode, self.dataset_root, headless=True)
        # env.launch()

        # task_env = env.get_task(TASK_DICT[self.task_name]["task_class"])
        # task_env.set_variation(0)
        # descriptions, _ = task_env.reset()

        # demos = task_env.get_demos(
        #     self.n_demos, live_demos=False, random_selection=False
        # )
        variation = 0

        names_to_handles = load_handle_mapping(
            self.dataset_root, self.task_name, variation
        )

        def leaf_fn(path, x):
            if path[1] == "action_obj_names" or path[1] == "anchor_obj_names":
                return names_to_handles[x]
            else:
                return x

        obj_handles = tree.map_structure_with_path(
            leaf_fn, TASK_DICT[self.task_name]["phase"]
        )

        demos = rlbench.utils.get_stored_demos(
            amount=self.n_demos,
            image_paths=False,
            dataset_root=self.dataset_root,
            variation_number=variation,
            task_name=self.task_name,
            obs_config=ObservationConfig(
                left_shoulder_camera=CameraConfig(image_size=(256, 256)),
                right_shoulder_camera=CameraConfig(image_size=(256, 256)),
                front_camera=CameraConfig(image_size=(256, 256)),
                wrist_camera=CameraConfig(image_size=(256, 256)),
                overhead_camera=CameraConfig(image_size=(256, 256)),
                task_low_dim_state=True,
            ),
            random_selection=False,
            from_episode_number=0,
        )

        # Get a mapping from object names to handles.
        # TODO: rename the keys from "*_obj_names" to "*_obj_handles".
        # Also have to change below.
        # for handle in sim.simGetObjects(sim.sim_handle_all):
        #     print(handle, sim.simGetObjectName(handle))

        # handle_path = os.path.join(
        #     self.dataset_root, self.task_name, f"variation{variation}", "handles.pkl"
        # )

        # with open(handle_path, "rb") as f:
        #     names_to_handles = pickle.load(f)

        # env.shutdown()

        return demos, obj_handles

    # @functools.cache
    def get_item(self, index: int) -> Dict[str, torch.Tensor]:
        demo: rlbench.demo.Demo = self.demos[index]

        # Each demonstration has a list of poses, which are the states of the
        low_dim_state_dict = load_state_pos_dict(
            self.dataset_root, self.task_name, 0, index
        )

        initial_obs = demo[0]

        # Find the first grasp instance
        if self.phase == "grasp":
            first_grasp_ix = list(
                filter(lambda t: t[1].gripper_open == 0.0, enumerate(demo))
            )[0][0]

            last_open_ix = first_grasp_ix - 1
            assert last_open_ix >= 0

            key_obs = demo[last_open_ix]
        elif self.phase == "place":
            key_obs = demo[-1]

        # Merge all the initial point clouds and masks into one.
        init_rgb, init_point_cloud, init_mask = obs_to_rgb_point_cloud(initial_obs)

        # Split the initial point cloud and rgb into action and anchor.
        (
            init_action_rgb,
            init_action_point_cloud,
        ) = get_rgb_point_cloud_by_object_handles(
            init_rgb,
            init_point_cloud,
            init_mask,
            self.names_to_handles[self.phase]["action_obj_names"],
        )
        (
            init_anchor_rgb,
            init_anchor_point_cloud,
        ) = get_rgb_point_cloud_by_object_handles(
            init_rgb,
            init_point_cloud,
            init_mask,
            self.names_to_handles[self.phase]["anchor_obj_names"],
        )

        # Merge all the key point clouds and masks into one.
        key_rgb, key_point_cloud, key_mask = obs_to_rgb_point_cloud(key_obs)

        # Split the key point cloud and rgb into action and anchor.
        key_action_rgb, key_action_point_cloud = get_rgb_point_cloud_by_object_handles(
            key_rgb,
            key_point_cloud,
            key_mask,
            self.names_to_handles[self.phase]["action_obj_names"],
        )
        key_anchor_rgb, key_anchor_point_cloud = get_rgb_point_cloud_by_object_handles(
            key_rgb,
            key_point_cloud,
            key_mask,
            self.names_to_handles[self.phase]["anchor_obj_names"],
        )

        def extract_action_pose(obs):
            # Extract the positions.
            action_pose_name = TASK_DICT[self.task_name]["phase"][self.phase][
                "action_pose_name"
            ]
            if action_pose_name == "gripper":
                action_pq = obs.gripper_pose
            else:
                start = low_dim_state_dict[action_pose_name]
                end = start + 7
                action_pq = obs.task_low_dim_state[start:end]

            # Convert to a 4x4 matrix.
            T_action_world = np.eye(4)
            T_action_world[:3, :3] = R.from_quat(action_pq[3:]).as_matrix()
            T_action_world[:3, 3] = action_pq[:3]

            return T_action_world

        # Get initial, key, and relative.
        T_action_init_world = extract_action_pose(initial_obs)
        T_action_key_world = extract_action_pose(key_obs)
        T_init_key = T_action_key_world @ np.linalg.inv(T_action_init_world)

        return {
            "init_action_rgb": torch.from_numpy(init_action_rgb),
            "init_action_pc": torch.from_numpy(init_action_point_cloud),
            "init_anchor_rgb": torch.from_numpy(init_anchor_rgb),
            "init_anchor_pc": torch.from_numpy(init_anchor_point_cloud),
            "key_action_rgb": torch.from_numpy(key_action_rgb),
            "key_action_pc": torch.from_numpy(key_action_point_cloud),
            "key_anchor_rgb": torch.from_numpy(key_anchor_rgb),
            "key_anchor_pc": torch.from_numpy(key_anchor_point_cloud),
            "T_action_init_world": torch.from_numpy(T_action_init_world),
            "T_action_key_world": torch.from_numpy(T_action_key_world),
            "T_init_key": torch.from_numpy(T_init_key),
        }

    def __len__(self) -> int:
        return len(self.demos)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.get_item(index)


# Write a class which inheits from the above dataset, but accepts an
# argument which repeats the dataset K times.
class RLBenchPlacementDatasetRepeat(RLBenchPlacementDataset):
    def __init__(
        self,
        dataset_root: str,
        task_name: str = "stack_wine",
        n_demos: int = 10,
        repeat: int = 1,
    ) -> None:
        super().__init__(dataset_root, task_name, n_demos)
        self.repeat = repeat

    def __len__(self) -> int:
        return self.repeat * len(self.demos)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.get_item(index % len(self.demos))

    def __len__(self) -> int:
        return self.repeat * len(self.demos)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.get_item(index % len(self.demos))

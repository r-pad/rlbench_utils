import functools
import logging
from typing import Dict

import numpy as np
import torch
import torch.utils.data as data
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.tasks import StackWine

TASK_DICT = {
    "stack_wine": {
        "task_class": StackWine,
        "action_obj_ids": [160],
        "anchor_obj_ids": [152, 154],
    },
}


# Get get rgb and point cloud for all points whose mask matches in list of ids
def get_rgb_point_cloud_by_mask(rgb, point_cloud, mask, ids):
    # Get the indices of the points which match the ids.
    indices = np.isin(mask, ids).reshape((-1))
    # Get the rgb and point cloud for the indices.
    rgb = rgb[indices]
    point_cloud = point_cloud[indices]
    return rgb, point_cloud


def obs_to_rgb_point_cloud(obs):
    # Get the overhead, left, front, and right RGB images.
    overhead_rgb = obs.overhead_rgb
    left_rgb = obs.left_shoulder_rgb
    right_rgb = obs.right_shoulder_rgb
    front_rgb = obs.front_rgb

    # Get the overhead, left, front, and right point clouds. The point clouds are
    # in the same shape as the images.
    overhead_point_cloud = obs.overhead_point_cloud
    left_point_cloud = obs.left_shoulder_point_cloud
    right_point_cloud = obs.right_shoulder_point_cloud
    front_point_cloud = obs.front_point_cloud

    # Get masks.
    overhead_mask = obs.overhead_mask
    left_mask = obs.left_shoulder_mask
    right_mask = obs.right_shoulder_mask
    front_mask = obs.front_mask

    # Flatten RGB and point cloud images into Nx3 arrays
    overhead_rgb = overhead_rgb.reshape((-1, 3))
    left_rgb = left_rgb.reshape((-1, 3))
    right_rgb = right_rgb.reshape((-1, 3))
    front_rgb = front_rgb.reshape((-1, 3))

    overhead_point_cloud = overhead_point_cloud.reshape((-1, 3))
    left_point_cloud = left_point_cloud.reshape((-1, 3))
    right_point_cloud = right_point_cloud.reshape((-1, 3))
    front_point_cloud = front_point_cloud.reshape((-1, 3))

    # Reshape the masks into Nx1 arrays.
    overhead_mask = overhead_mask.reshape((-1, 1))
    left_mask = left_mask.reshape((-1, 1))
    right_mask = right_mask.reshape((-1, 1))
    front_mask = front_mask.reshape((-1, 1))

    # Stack the RGB and point cloud images together.
    rgb = np.vstack((overhead_rgb, left_rgb, right_rgb, front_rgb))
    point_cloud = np.vstack(
        (overhead_point_cloud, left_point_cloud, right_point_cloud, front_point_cloud)
    )
    mask = np.vstack((overhead_mask, left_mask, right_mask, front_mask))

    return rgb, point_cloud, mask


class RLBenchPlacementDataset(data.Dataset):
    def __init__(
        self, dataset_root: str, task_name: str = "stack_wine", n_demos: int = 10
    ) -> None:
        """Dataset for RL-Bench placement tasks.

        Args:
            dataset_root (str): The root of where the RLBench demonstrations were generated.
        """
        super().__init__()

        self.dataset_root = dataset_root
        self.task_name = task_name
        self.n_demos = n_demos

        if self.task_name not in TASK_DICT:
            raise ValueError(f"Task name {self.task_name} not supported.")

        logging.info(f"Loading {self.n_demos} demos for task {self.task_name}...")
        self.demos = self._load_demos()
        logging.info("Demos loaded!")

    def _load_demos(self) -> None:
        action_mode = MoveArmThenGripper(
            arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()
        )

        env = Environment(action_mode, self.dataset_root, headless=True)
        env.launch()

        task = env.get_task(TASK_DICT[self.task_name]["task_class"])

        demos = task.get_demos(self.n_demos, live_demos=False, random_selection=False)
        env.shutdown()

        return demos

    @functools.cache
    def get_item(self, index: int) -> Dict[str, torch.Tensor]:
        demo = self.demos[index]
        final_obs = demo[-1]

        # Merge all the point clouds and masks into one.
        rgb, point_cloud, mask = obs_to_rgb_point_cloud(final_obs)

        # Filter the rgb and point cloud for the action and anchor objects.
        action_rgb, action_point_cloud = get_rgb_point_cloud_by_mask(
            rgb, point_cloud, mask, TASK_DICT[self.task_name]["action_obj_ids"]
        )

        # Get the rgb and point cloud for the anchor objects.
        anchor_rgb, anchor_point_cloud = get_rgb_point_cloud_by_mask(
            rgb, point_cloud, mask, TASK_DICT[self.task_name]["anchor_obj_ids"]
        )

        return {
            "action_rgb": torch.from_numpy(action_rgb),
            "action_pc": torch.from_numpy(action_point_cloud),
            "anchor_rgb": torch.from_numpy(anchor_rgb),
            "anchor_pc": torch.from_numpy(anchor_point_cloud),
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

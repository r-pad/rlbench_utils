import logging
import os
import pickle
from enum import Enum
from typing import Dict, List, Literal, Union, cast

import numpy as np
import rlbench.backend.observation
import rlbench.demo
import rlbench.utils
import torch
import torch.utils.data as data
import tree
from joblib import Memory
from pyrep.backend import sim
from rlbench.observation_config import CameraConfig, ObservationConfig
from scipy.spatial.transform import Rotation as R

from rpad.rlbench_utils.keyframing_pregrasp import keypoint_discovery_pregrasp
from rpad.rlbench_utils.task_info import GRIPPER_OBJ_NAMES, GRIPPER_POSE_NAME, TASK_DICT


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

    return cast(Dict[str, int], names_to_handles)


def load_state_pos_dict(
    dataset_root: str, task_name: str, variation: int, episode: int
) -> Dict[str, int]:
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

    return cast(Dict[str, int], state_pos_dict)


BACKGROUND_NAMES = [
    "ResizableFloor_5_25_visibleElement",
    "Wall3",
    "diningTable_visible",
    "workspace",
]

ROBOT_NONGRIPPER_NAMES = [
    "Panda_link0_visual",
    "Panda_link1_visual",
    "Panda_link2_visual",
    "Panda_link3_visual",
    "Panda_link4_visual",
    "Panda_link5_visual",
    "Panda_link6_visual",
    # "Panda_link7_visual",
]


def filter_out_names(rgb, point_cloud, mask, handlemapping, names=BACKGROUND_NAMES):
    # Get the indices of the background.
    background_handles = [handlemapping[name] for name in names]
    background_indices = np.isin(mask, background_handles).reshape((-1))

    # Get the indices of the foreground.
    foreground_indices = ~background_indices

    # Get the foreground rgb and point cloud.
    foreground_rgb = rgb[foreground_indices]
    foreground_point_cloud = point_cloud[foreground_indices]

    return foreground_rgb, foreground_point_cloud


class ActionMode(str, Enum):
    GRIPPER_AND_OBJECT = "gripper_and_object"
    OBJECT = "object"


class AnchorMode(str, Enum):
    RAW = "raw"
    BACKGROUND_REMOVED = "background_removed"
    BACKGROUND_ROBOT_REMOVED = "background_robot_removed"
    SINGLE_OBJECT = "single_object"


def get_anchor_points(
    anchor_mode: AnchorMode,
    rgb,
    point_cloud,
    mask,
    task_name,
    phase,
    use_from_simulator=False,
    handle_mapping=None,
    names_to_handles=None,
):
    if anchor_mode == AnchorMode.RAW:
        return rgb, point_cloud
    elif anchor_mode == AnchorMode.BACKGROUND_REMOVED:
        return filter_out_names(
            rgb, point_cloud, mask, handle_mapping, BACKGROUND_NAMES
        )
    elif anchor_mode == AnchorMode.BACKGROUND_ROBOT_REMOVED:
        return filter_out_names(
            rgb,
            point_cloud,
            mask,
            handle_mapping,
            BACKGROUND_NAMES + ROBOT_NONGRIPPER_NAMES,
        )
    elif anchor_mode == AnchorMode.SINGLE_OBJECT:
        if use_from_simulator:
            return get_rgb_point_cloud_by_object_names(
                rgb,
                point_cloud,
                mask,
                TASK_DICT[task_name]["phase"][phase]["anchor_obj_names"],
            )
        else:
            return get_rgb_point_cloud_by_object_handles(
                rgb,
                point_cloud,
                mask,
                names_to_handles[phase]["anchor_obj_names"],
            )
    else:
        raise ValueError("Anchor mode must be one of the AnchorMode enum values.")


def get_action_points(
    action_mode: ActionMode,
    rgb,
    point_cloud,
    mask,
    task_name,
    phase,
    use_from_simulator=False,
    action_handles=None,
    gripper_handles=None,
):
    if use_from_simulator:
        action_names = TASK_DICT[task_name]["phase"][phase]["action_obj_names"]
        gripper_names = GRIPPER_OBJ_NAMES
        action_handles = [sim.simGetObjectHandle(name) for name in action_names]
        gripper_handles = [sim.simGetObjectHandle(name) for name in gripper_names]

    if action_mode == ActionMode.GRIPPER_AND_OBJECT:
        action_handles = action_handles + gripper_handles
    elif action_mode == ActionMode.OBJECT:
        pass
    else:
        raise ValueError("Action mode must be one of the ActionMode enum values.")

    action_rgb, action_point_cloud = get_rgb_point_cloud_by_object_handles(
        rgb, point_cloud, mask, action_handles
    )

    return action_rgb, action_point_cloud


class RLBenchPlacementDataset(data.Dataset):
    def __init__(
        self,
        dataset_root: str,
        task_name: str = "stack_wine",
        demos: Union[Literal["all"], List[int]] = "all",
        phase: str = "grasp",
        use_first_as_init_keyframe: bool = True,
        cache: bool = True,
        debugging: bool = False,
        anchor_mode: AnchorMode = AnchorMode.SINGLE_OBJECT,
        action_mode: ActionMode = ActionMode.OBJECT,
    ) -> None:
        """Dataset for RL-Bench placement tasks.

        Args:
            dataset_root (str): The root of where the RLBench demonstrations were generated.
        """
        super().__init__()

        self.dataset_root = dataset_root
        self.task_name = task_name
        if demos == "all":
            self.n_demos = len(
                os.listdir(
                    os.path.join(dataset_root, task_name, f"variation0", "episodes")
                )
            )
            self.demos = list(range(self.n_demos))
        else:
            self.n_demos = len(demos)
            self.demos = demos

        self.phase = phase
        self.variation = 0
        self.debugging = debugging
        self.use_first_as_init_keyframe = use_first_as_init_keyframe

        if self.task_name not in TASK_DICT:
            raise ValueError(f"Task name {self.task_name} not supported.")

        # Assert that the phase is in the task names.
        if phase != "all" and phase not in TASK_DICT[task_name]["phase"]:
            raise ValueError(f"Phase {phase} not supported for task {task_name}.")

        handle_mapping = load_handle_mapping(
            self.dataset_root, self.task_name, self.variation
        )
        self.handle_mapping = handle_mapping

        def leaf_fn(path, x):
            if path[1] == "action_obj_names" or path[1] == "anchor_obj_names":
                return handle_mapping[x]
            else:
                return x

        # Get a mapping from object names to handles.
        # TODO: rename the keys from "*_obj_names" to "*_obj_handles".
        self.names_to_handles = tree.map_structure_with_path(
            leaf_fn, TASK_DICT[self.task_name]["phase"]
        )

        self.gripper_handles = [handle_mapping[name] for name in GRIPPER_OBJ_NAMES]

        if isinstance(anchor_mode, bool):
            raise ValueError("Anchor mode must be one of the AnchorMode enum values.")
        self.action_mode = action_mode
        self.anchor_mode = anchor_mode

        if cache:
            self.memory = Memory(
                location=os.path.join(dataset_root, f".cache/{task_name}")
            )
        else:
            self.memory = None

    def __len__(self) -> int:
        if self.phase == "all":
            return self.n_demos * len(TASK_DICT[self.task_name]["phase_order"])
        else:
            return self.n_demos

    @staticmethod
    def _load_keyframes(
        dataset_root, variation, task_name, episode_index: int
    ) -> List[int]:
        demo = rlbench.utils.get_stored_demos(
            amount=1,
            image_paths=False,
            dataset_root=dataset_root,
            variation_number=variation,
            task_name=task_name,
            obs_config=ObservationConfig(
                left_shoulder_camera=CameraConfig(image_size=(256, 256)),
                right_shoulder_camera=CameraConfig(image_size=(256, 256)),
                front_camera=CameraConfig(image_size=(256, 256)),
                wrist_camera=CameraConfig(image_size=(256, 256)),
                overhead_camera=CameraConfig(image_size=(256, 256)),
                task_low_dim_state=True,
            ),
            random_selection=False,
            from_episode_number=episode_index,
        )[0]

        keyframe_ixs = keypoint_discovery_pregrasp(demo)

        keyframes = [demo[ix] for ix in keyframe_ixs]

        return keyframes, demo[0]  # type: ignore

    # We also cache in memory, since all the transformations are the same.
    # Saves a lot of time when loading the dataset, but don't have to worry
    # about logic changes after the fact.
    # @functools.lru_cache(maxsize=100)
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # If we're using "all" as the phase, then we'll just sequence the dataset
        # back to back.
        if self.phase == "all":
            og_index = index
            index = index % self.n_demos

        # NOTE: We are caching the outputs since it's a royal pain to load the
        # demonstrations from disk. But this means that we'll have to be careful
        # whenever we re-generate the demonstrations to delete the cache.
        if self.memory is not None:
            load_keyframes_fn = self.memory.cache(self._load_keyframes)
        else:
            load_keyframes_fn = self._load_keyframes

        keyframes, first_frame = load_keyframes_fn(
            self.dataset_root, self.variation, self.task_name, self.demos[index]
        )

        # breakpoint()

        # Get the index of the phase into keypoints.
        if self.phase == "all":
            phase_ix = og_index // self.n_demos
            phase = TASK_DICT[self.task_name]["phase_order"][phase_ix]

        else:
            phase_ix = TASK_DICT[self.task_name]["phase_order"].index(self.phase)
            phase = self.phase

        phase_onehot = np.zeros(len(TASK_DICT[self.task_name]["phase_order"]))
        phase_onehot[phase_ix] = 1

        # Select an observation to use as the initial observation.
        if self.use_first_as_init_keyframe or phase_ix == 0:
            initial_obs = first_frame
        else:
            initial_obs = keyframes[phase_ix - 1]

        # Find the first grasp instance
        key_obs = keyframes[phase_ix]

        action_handles = self.names_to_handles[phase]["action_obj_names"]

        def _select_action_vals(rgb, point_cloud, mask):
            return get_action_points(
                self.action_mode,
                rgb,
                point_cloud,
                mask,
                action_handles,
                self.gripper_handles,
            )

        def _select_anchor_vals(rgb, point_cloud, mask):
            return get_anchor_points(
                self.anchor_mode,
                rgb,
                point_cloud,
                mask,
                self.task_name,
                phase,
                use_from_simulator=False,
                handle_mapping=self.handle_mapping,
                names_to_handles=self.names_to_handles,
            )

        # Merge all the initial point clouds and masks into one.
        init_rgb, init_point_cloud, init_mask = obs_to_rgb_point_cloud(initial_obs)

        init_action_rgb, init_action_point_cloud = _select_action_vals(
            init_rgb, init_point_cloud, init_mask
        )

        init_anchor_rgb, init_anchor_point_cloud = _select_anchor_vals(
            init_rgb, init_point_cloud, init_mask
        )

        # Merge all the key point clouds and masks into one.
        key_rgb, key_point_cloud, key_mask = obs_to_rgb_point_cloud(key_obs)

        # Split the key point cloud and rgb into action and anchor.
        key_action_rgb, key_action_point_cloud = _select_action_vals(
            key_rgb, key_point_cloud, key_mask
        )
        key_anchor_rgb, key_anchor_point_cloud = _select_anchor_vals(
            key_rgb, key_point_cloud, key_mask
        )

        # Each demonstration has a list of poses, which are the states of the various objects.
        low_dim_state_dict = load_state_pos_dict(
            self.dataset_root, self.task_name, self.variation, index
        )

        def extract_pose(obs, key):
            # Extract the positions.
            pose_name = TASK_DICT[self.task_name]["phase"][phase][key]
            if pose_name == GRIPPER_POSE_NAME:
                action_pq = obs.gripper_pose
            else:
                # TODO: This is a bit of a hack to handle the fact that the demos don't
                # currently output the same stuff.
                if "custom_lowdim" in TASK_DICT[self.task_name]:
                    raise NotImplementedError("i thought i fixed this")
                    start, v_len = TASK_DICT[self.task_name]["custom_lowdim"][pose_name]
                    end = start + v_len

                    if v_len == 3:
                        action_p = obs.task_low_dim_state[start:end]
                        action_q = np.array([1, 0, 0, 0])
                        action_pq = np.concatenate((action_p, action_q))
                    else:
                        action_pq = obs.task_low_dim_state[start:end]
                else:
                    start = low_dim_state_dict[pose_name]
                    end = start + 7
                    action_pq = obs.task_low_dim_state[start:end]

            # Convert to a 4x4 matrix.
            T_action_world = np.eye(4)
            T_action_world[:3, :3] = R.from_quat(action_pq[3:]).as_matrix()
            T_action_world[:3, 3] = action_pq[:3]

            return T_action_world

        # Get initial, key, and relative.
        T_action_init_world = extract_pose(initial_obs, "action_pose_name")
        T_action_key_world = extract_pose(key_obs, "action_pose_name")
        T_init_key = T_action_key_world @ np.linalg.inv(T_action_init_world)
        T_anchor_key_world = extract_pose(key_obs, "anchor_pose_name")

        if hasattr(initial_obs, "ignore_collisions"):
            ignore_collisions = initial_obs.ignore_collisions
            ignore_collisions = torch.from_numpy(ignore_collisions.astype(np.int32))
        else:
            ignore_collisions = None

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
            "T_anchor_key_world": torch.from_numpy(T_anchor_key_world),
            "T_init_key": torch.from_numpy(T_init_key),
            # Also return some rgb images for visualization.
            "init_front_rgb": torch.from_numpy(initial_obs.front_rgb),
            "key_front_rgb": torch.from_numpy(key_obs.front_rgb),
            "init_front_mask": torch.from_numpy(
                initial_obs.front_mask.astype(np.int32)
            ),
            "key_front_mask": torch.from_numpy(key_obs.front_mask.astype(np.int32)),
            "phase": phase,
            "phase_onehot": torch.from_numpy(phase_onehot),
            "ignore_collisions": ignore_collisions,
        }

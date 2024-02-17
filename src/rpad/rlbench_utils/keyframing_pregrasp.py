from typing import List

from rlbench.demo import Demo

from rpad.rlbench_utils.keyframing import _is_stopped


def keypoint_discovery_pregrasp(demo: Demo, stopping_delta=0.1) -> List[int]:
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0

    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer, stopping_delta)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # If change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (last or stopped):
            episode_keypoints.append(i)
        elif i != 0 and (obs.gripper_open != prev_gripper_open):
            # Append previous keyframe to get pregrasp!
            episode_keypoints.append(i - 1)
            # episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open

    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)

    return episode_keypoints

from typing import List, Optional, Tuple

import numpy as np

from ..data import (
    filter_agents_by_labels,
    filter_tl_faces_by_frames,
    get_agents_slice_from_frames,
    get_tl_faces_slice_from_frames,
)
from ..data.filter import filter_agents_by_frames, filter_agents_by_track_id, is_valid_agent_by_labels
from ..geometry import (
    angular_distance,
    compute_agent_pose,
    inv_agent_pose,
    rotate_agent_from_world,
    rotation33_as_yaw,
    transform_point,
    transform_points_fast,
    transform_world_to_agent,
)
from ..kinematic import Perturbation
from ..rasterization import EGO_EXTENT_HEIGHT, EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, Rasterizer, RenderContext
from .slicing import get_future_slice, get_history_slice


def generate_agent_sample(
    state_index: int,
    frames: np.ndarray,
    agents: np.ndarray,
    tl_faces: np.ndarray,
    selected_agent_id: int,
    render_context: RenderContext,
    history_num_frames: int,
    history_step_size: int,
    history_step_time: float,
    future_num_frames: int,
    future_step_size: int,
    future_step_time: float,
    filter_agents_threshold: float,
    rasterizer: Optional[Rasterizer] = None,
    perturbation: Optional[Perturbation] = None,
    velocity_corrected_yaw: bool = False,
) -> dict:
    """Generates the inputs and targets to train a deep prediction model. A deep prediction model takes as input
    the state of the world (here: an image we will call the "raster"), and outputs where that agent will be some
    seconds into the future.

    This function has a lot of arguments and is intended for internal use, you should try to use higher level classes
    and partials that use this function.

    Args:
        state_index (int): The anchor frame index, i.e. the "current" timestep in the scene
        frames (np.ndarray): The scene frames array, can be numpy array or a zarr array
        agents (np.ndarray): The full agents array, can be numpy array or a zarr array
        tl_faces (np.ndarray): The full traffic light faces array, can be numpy array or a zarr array
        # selected_track_id (Optional[int]): Either None for AV, or the ID of an agent that you want to
        # predict the future of. This agent is centered in the raster and the returned targets are derived from
        # their future states.
        selected_agent_id (int): Either -1 for AV, or the total_agent_id of an agent that you want to
        predict the future of. This agent is centered in the raster and the returned targets are derived from
        their future states.
        raster_size (Tuple[int, int]): Desired output raster dimensions
        pixel_size (np.ndarray): Size of one pixel in the real world
        ego_center (np.ndarray): Where in the raster to draw the ego, [0.5,0.5] would be the center
        history_num_frames (int): Amount of history frames to draw into the rasters
        history_step_size (int): Steps to take between frames, can be used to subsample history frames
        future_num_frames (int): Amount of history frames to draw into the rasters
        future_step_size (int): Steps to take between targets into the future
        filter_agents_threshold (float): Value between 0 and 1 to use as cutoff value for agent filtering
        based on their probability of being a relevant agent
        rasterizer (Optional[Rasterizer]): Rasterizer of some sort that draws a map image
        perturbation (Optional[Perturbation]): Object that perturbs the input and targets, used
to train models that can recover from slight divergence from training set data

    Raises:
        ValueError: A ValueError is returned if the specified ``selected_track_id`` is not present in the scene
        or was filtered by applying the ``filter_agent_threshold`` probability filtering.

    Returns:
        dict: a dict object with the raster array, the future offset coordinates (meters),
        the future yaw angular offset, the future_availability as a binary mask
    """
    #  the history slice is ordered starting from the latest frame and goes backward in time., ex. slice(100, 91, -2)
    history_slice = get_history_slice(state_index, history_num_frames, history_step_size, include_current_state=True)
    future_slice = get_future_slice(state_index, future_num_frames, future_step_size)

    history_frames = frames[history_slice].copy()  # copy() required if the object is a np.ndarray
    future_frames = frames[future_slice].copy()

    # sorted_frames = np.concatenate((history_frames[::-1], future_frames))  # from past to future

    # get agents (past and future)
    # agent_slice = get_agents_slice_from_frames(sorted_frames[0], sorted_frames[-1])
    # agent_slice = get_agents_slice_from_frames(history_frames[-1], future_frames[-1])
    agent_slice = slice(
        history_frames[-1]["agent_index_interval"][0],
        (future_frames[-1] if len(future_frames) > 0
         else history_frames[0])["agent_index_interval"][1]
    )
    # read the slice of agents from disk
    agents = agents[agent_slice].copy()  # this is the minimum slice of agents we need
    history_frames["agent_index_interval"] -= agent_slice.start  # sync interval with the agents array
    future_frames["agent_index_interval"] -= agent_slice.start  # sync interval with the agents array
    local_agent_id = selected_agent_id - agent_slice.start
    history_agents = filter_agents_by_frames(history_frames, agents)
    future_agents = filter_agents_by_frames(future_frames, agents)

    tl_slice = get_tl_faces_slice_from_frames(history_frames[-1], history_frames[0])  # -1 is the farthest
    # sync interval with the traffic light faces array
    history_frames["traffic_light_faces_index_interval"] -= tl_slice.start
    history_tl_faces = filter_tl_faces_by_frames(history_frames, tl_faces[tl_slice].copy())

    if perturbation is not None:
        history_frames, future_frames = perturbation.perturb(
            history_frames=history_frames, future_frames=future_frames
        )

    # # State you want to predict the future of.
    # cur_frame = history_frames[0]
    # cur_agents = history_agents[0]

    # if selected_track_id is None:
    if selected_agent_id < 0:
        # For Ego
        # State you want to predict the future of.
        cur_frame = history_frames[0]
        agent_centroid_m = cur_frame["ego_translation"][:2]
        agent_yaw_rad = rotation33_as_yaw(cur_frame["ego_rotation"])
        agent_extent_m = np.asarray((EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, EGO_EXTENT_HEIGHT))
        selected_track_id = None
        selected_agent = None
    else:
        # For the target Agent
        # (simply recover the selected agent information in the agent dataset given track id)
        # State you want to predict the future of.
        # cur_agents = history_agents[0]
        # this will raise IndexError if the agent is not in the frame or under agent-threshold
        # this is a strict error, we cannot recover from this situation
        selected_agent = agents[local_agent_id]
        selected_track_id = selected_agent['track_id']
        if not is_valid_agent_by_labels(selected_agent):
            raise ValueError(f" track_id {selected_track_id} not in frame or below threshold")
        # try:
        #     agent = filter_agents_by_track_id(
        #         filter_agents_by_labels(cur_agents, filter_agents_threshold), selected_track_id
        #     )[0]
        # except IndexError:
        #     raise ValueError(f" track_id {selected_track_id} not in frame or below threshold")
        agent_centroid_m = selected_agent["centroid"]
        agent_yaw_rad = float(selected_agent["yaw"])
        agent_extent_m = selected_agent["extent"]
        # selected_agent = agent

        agent_yaw_rad, agent_velocity = _estimate_target_yaw_and_velocity(
            history_agents, selected_track_id, history_step_time, agent_yaw_rad,
        )
        selected_agent['velocity'] = agent_velocity  # not sure why this is not being used
        if velocity_corrected_yaw:
            # agent_yaw_rad = _estimate_target_yaw(
            selected_agent["yaw"] = agent_yaw_rad  # for rasterizer, this also affect agents list

    # Generate the image (most computation intense part)
    input_im = (
        None
        if not rasterizer
        else rasterizer.rasterize(history_frames, history_agents, history_tl_faces, selected_agent)
    )

    # Compute other meta data
    world_from_agent = compute_agent_pose(agent_centroid_m, agent_yaw_rad)
    # agent_from_world = np.linalg.inv(world_from_agent)
    agent_from_world = inv_agent_pose(agent_centroid_m, agent_yaw_rad)
    rotation_agent_from_world = rotate_agent_from_world(agent_yaw_rad)
    raster_from_world = render_context.raster_from_world(agent_centroid_m, agent_yaw_rad)

    future_positions_m, future_yaws_rad, future_availabilities = _create_targets_for_deep_prediction(
        future_num_frames, future_frames, selected_track_id, future_agents, rotation_agent_from_world, agent_yaw_rad,
        agent_centroid_m,
    )

    # history_num_frames + 1 because it also includes the current frame
    history_positions_m, history_yaws_rad, history_availabilities = _create_targets_for_deep_prediction(
        history_num_frames + 1, history_frames, selected_track_id, history_agents, rotation_agent_from_world, agent_yaw_rad,
        agent_centroid_m,
    )

    # compute estimated velocities by finite differentiatin on future positions
    # estimate velocity at T with (pos(T+t) - pos(T))/t
    # this gives < 0.5% velocity difference to (pos(T+t) - pos(T-t))/2t on v1.1/sample.zarr.tar

    # [future_num_frames, 2]
    future_positions_diff_m = np.concatenate((future_positions_m[:1], np.diff(future_positions_m, axis=0)))
    # [future_num_frames, 2]
    future_vels_mps = np.float32(future_positions_diff_m / future_step_time)

    # current position is included in history positions
    # [history_num_frames, 2]
    history_positions_diff_m = np.diff(history_positions_m, axis=0)
    # [history_num_frames, 2]
    history_vels_mps = np.float32(history_positions_diff_m / history_step_time)

    return {
        "image": input_im,
        "target_positions": future_positions_m,
        "target_yaws": future_yaws_rad,
        "target_velocities": future_vels_mps,
        "target_availabilities": future_availabilities,
        "history_positions": history_positions_m,
        "history_yaws": history_yaws_rad,
        "history_velocities": history_vels_mps,
        "history_availabilities": history_availabilities,
        "world_to_image": raster_from_world,  # TODO deprecate
        "raster_from_agent": raster_from_world @ world_from_agent,
        "raster_from_world": raster_from_world,
        "agent_from_world": agent_from_world,
        "world_from_agent": world_from_agent,
        "centroid": agent_centroid_m,
        "yaw": agent_yaw_rad,
        "speed": np.linalg.norm(future_vels_mps[0]),  # Louis: can be deleted
        "extent": agent_extent_m,
        "track_id": np.int64(-1 if selected_track_id is None else selected_track_id),
    }


def _create_targets_for_deep_prediction(
    num_frames: int,
    frames: np.ndarray,
    selected_track_id: Optional[int],
    agents: List[np.ndarray],
    # agent_from_world: np.ndarray,
    rotation_agent_from_world: np.ndarray,
    current_agent_yaw: float,
    current_agent_centroid: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Internal function that creates the targets and availability masks for deep prediction-type models.
    The futures/history offset (in meters) are computed. When no info is available (e.g. agent not in frame)
    a 0 is set in the availability array (1 otherwise).

    Args:
        num_frames (int): number of offset we want in the future/history
        frames (np.ndarray): available frames. This may be less than num_frames
        selected_track_id (Optional[int]): agent track_id or AV (None)
        agents (List[np.ndarray]): list of agents arrays (same len of frames)
        agent_from_world (np.ndarray): local from world matrix
        current_agent_yaw (float): angle of the agent at timestep 0

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: position offsets, angle offsets, availabilities

    """
    # How much the coordinates differ from the current state in meters.
    # positions_m = np.zeros((num_frames, 2), dtype=np.float32)
    yaws_rad = np.zeros((num_frames, 1), dtype=np.float32)
    # availabilities = np.zeros((num_frames,), dtype=np.float32)

    # for i, (frame, frame_agents) in enumerate(zip(frames, agents)):
    #     if selected_track_id is None:
    #         agent_centroid_m = frame["ego_translation"][:2]
    #         agent_yaw_rad = rotation33_as_yaw(frame["ego_rotation"])
    #     else:
    #         # it's not guaranteed the target will be in every frame
    #         try:
    #             agent = filter_agents_by_track_id(frame_agents, selected_track_id)[0]
    #             agent_centroid_m = agent["centroid"]
    #             agent_yaw_rad = agent["yaw"]
    #         except IndexError:
    #             availabilities[i] = 0.0  # keep track of invalid futures/history
    #             continue

    #     positions_m[i] = transform_point(agent_centroid_m, agent_from_world)
    #     yaws_rad[i] = angular_distance(agent_yaw_rad, current_agent_yaw)
    #     availabilities[i] = 1.0

    if len(frames) == 0:
        return np.zeros((num_frames, 2), dtype=np.float32), yaws_rad, np.zeros((num_frames,), dtype=np.float32)

    if selected_track_id is None:
        # For Ego
        positions_m = np.zeros((num_frames, 2), dtype=np.float32)
        availabilities = np.zeros((num_frames,), dtype=np.float32)
        for i, frame in enumerate(frames):
            agent_centroid_m = frame["ego_translation"][:2]
            agent_yaw_rad = rotation33_as_yaw(frame["ego_rotation"])

            # positions_m[i] = transform_point(agent_centroid_m, agent_from_world)
            positions_m[i] = transform_world_to_agent(agent_centroid_m, current_agent_centroid, rotation_agent_from_world)
            yaws_rad[i] = angular_distance(agent_yaw_rad, current_agent_yaw)
            availabilities[i] = 1.0
        return positions_m, yaws_rad, availabilities

    # For Agent
    agents = [
        frame_agents[frame_agents['track_id'] == selected_track_id]
        for frame_agents in agents
    ]
    positions_m = np.zeros((num_frames, 2), dtype=np.float64)
    availabilities = np.zeros((num_frames,), dtype=np.bool)
    availabilities[:len(agents)] = [len(a) > 0 for a in agents]
    agents_av = [a for a, av in zip(agents, availabilities) if av]  # note len(agents) >= len(availabilities)
    if len(agents_av) > 0:
        agents_av = np.concatenate(agents_av)
        positions_m[availabilities] = agents_av['centroid']
        yaws_rad[availabilities] = agents_av['yaw'].reshape(-1, 1)
        positions_m = transform_world_to_agent(positions_m, current_agent_centroid, rotation_agent_from_world)
        yaws_rad = angular_distance(yaws_rad, current_agent_yaw)
        positions_m[~availabilities] = 0
        yaws_rad[~availabilities] = 0
    availabilities = availabilities.astype(np.float32)

    return positions_m.astype(np.float32), yaws_rad, availabilities


def _estimate_target_yaw(
    history_agents, selected_track_id: int, history_step_time: float,
    agent_yaw_rad: float,
):
    selected_agents = [
        frame_agents[frame_agents['track_id'] == selected_track_id]
        for frame_agents in history_agents
    ]
    # availabilities = np.array([len(a) > 0 for a in selected_agents], dtype=np.bool)
    world_coords = np.array([a[0]['centroid'] for a in selected_agents if len(a) > 0])
    if len(world_coords) <= 1:
        return agent_yaw_rad
    # assume 0.1 s/frame
    rel_world_coords = world_coords - world_coords[0]
    avg_velocities = (
        -rel_world_coords[1:] / history_step_time / np.arange(1, len(rel_world_coords))[:, None]
    )
    # weighted_avg_velocities
    weighted_avg_velocities = avg_velocities.mean(axis=0)
    # speed threshold
    speed_square = (weighted_avg_velocities**2).sum()
    speed_threshold = speed_square > 1.0  # filter out speed^2 < (1.0m/s)^2 cases
    if not speed_threshold:
        return agent_yaw_rad
    est_yaw = float(np.angle(weighted_avg_velocities[0] + weighted_avg_velocities[1] * (1.0j)))
    # limit to the correction to maximum of 5 degree ~= 0.1 radian (about 1m in y per 10m in x)
    return est_yaw if abs(est_yaw - agent_yaw_rad) <= 0.1 else agent_yaw_rad


def _estimate_target_yaw_and_velocity(
    history_agents, selected_track_id: int, history_step_time: float,
    agent_yaw_rad: float,
):
    selected_agents = [
        frame_agents[frame_agents['track_id'] == selected_track_id]
        for frame_agents in history_agents
    ]
    # availabilities = np.array([len(a) > 0 for a in selected_agents], dtype=np.bool)
    world_coords = np.array([a[0]['centroid'] for a in selected_agents if len(a) > 0])
    if len(world_coords) <= 1:
        return agent_yaw_rad, np.zeros(2)
    # assume 0.1 s/frame
    rel_world_coords = world_coords - world_coords[0]
    avg_velocities = (
        -rel_world_coords[1:] / history_step_time / np.arange(1, len(rel_world_coords))[:, None]
    )
    # weighted_avg_velocities
    weighted_avg_velocities = avg_velocities.mean(axis=0)
    # speed threshold
    speed_square = (weighted_avg_velocities**2).sum()
    speed_threshold = speed_square > 1.0  # filter out speed^2 < (1.0m/s)^2 cases
    if not speed_threshold:
        return agent_yaw_rad, np.zeros(2)
    est_yaw = float(np.angle(weighted_avg_velocities[0] + weighted_avg_velocities[1] * (1.0j)))
    # limit to the correction to maximum of 5 degree ~= 0.1 radian (about 1m in y per 10m in x)
    return (
        est_yaw if abs(est_yaw - agent_yaw_rad) <= 0.1 else agent_yaw_rad,
        weighted_avg_velocities
    )


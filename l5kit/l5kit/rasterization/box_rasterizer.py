from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from l5kit.data.zarr_dataset import AGENT_DTYPE

from ..data.filter import filter_agents_by_labels, filter_agents_by_track_id
from ..geometry import rotation33_as_yaw, transform_points, transform_points_fast
from ..geometry.transform import yaw_as_rotation33
from .rasterizer import EGO_EXTENT_HEIGHT, EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, Rasterizer
from .render_context import RenderContext
from .semantic_rasterizer import CV2_SHIFT, cv2_subpixel


def get_ego_as_agent(frame: np.ndarray) -> np.ndarray:  # TODO this can be useful to have around
    """
    Get a valid agent with information from the frame AV. Ford Fusion extent is used

    Args:
        frame (np.ndarray): the frame we're interested in

    Returns: an agent np.ndarray of the AV

    """
    ego_agent = np.zeros(1, dtype=AGENT_DTYPE)
    ego_agent[0]["centroid"] = frame["ego_translation"][:2]
    ego_agent[0]["yaw"] = rotation33_as_yaw(frame["ego_rotation"])
    ego_agent[0]["extent"] = np.asarray((EGO_EXTENT_LENGTH, EGO_EXTENT_WIDTH, EGO_EXTENT_HEIGHT))
    return ego_agent


_MIN_EXTENT = 0.5


def draw_boxes(
    raster_size: Tuple[int, int],
    raster_from_world: np.ndarray,
    agents: np.ndarray,
    color: Union[int, Tuple[int, int, int]],
) -> np.ndarray:
    """
    Draw multiple boxes in one sweep over the image.
    Boxes corners are extracted from agents, and the coordinates are projected in the image plane.
    Finally, cv2 draws the boxes.

    Args:
        raster_size (Tuple[int, int]): Desired output image size
        world_to_image_space (np.ndarray): 3x3 matrix to convert from world to image coordinated
        agents (np.ndarray): array of agents to be drawn
        color (Union[int, Tuple[int, int, int]]): single int or RGB color

    Returns:
        np.ndarray: the image with agents rendered. RGB if color RGB, otherwise GRAY
    """
    if isinstance(color, int):
        im = np.zeros((raster_size[1], raster_size[0]), dtype=np.uint8)
    else:
        im = np.zeros((raster_size[1], raster_size[0], 3), dtype=np.uint8)

    # box_world_coords = np.zeros((len(agents), 4, 2))
    # corners_base_coords = np.asarray([[-1, -1], [-1, 1], [1, 1], [1, -1]])
    _corners_base_coords = (np.asarray([[-1, -1], [-1, 1], [1, 1], [1, -1]]) / 2)[None, :, :]

    # compute the corner in world-space (start in origin, rotate and then translate)
    # for idx, agent in enumerate(agents):
    #     corners = corners_base_coords * agent["extent"][:2] / 2  # corners in zero
    #     r_m = yaw_as_rotation33(agent["yaw"])
    #     box_world_coords[idx] = transform_points(corners, r_m) + agent["centroid"][:2]
    # shape: (N agents)x(4 corners)x(2D coords)
    corners_m = _corners_base_coords * np.maximum(agents["extent"], _MIN_EXTENT)[:, None, :2]  # corners in zero
    # corners_m = _corners_base_coords * agents["extent"][:, None, :2]  # corners in zero
    s = np.sin(agents['yaw'])
    c = np.cos(agents['yaw'])
    rotation_m = np.moveaxis(np.array([[c, -s], [s, c]]), 2, 0)
    box_world_coords = np.einsum('bti,bji->btj', corners_m, rotation_m) + agents['centroid'][:, None, :2]
    box_raster_coords = transform_points_fast(box_world_coords.reshape((-1, 2)), raster_from_world)

    # fillPoly wants polys in a sequence with points inside as (x,y)
    box_raster_coords = cv2_subpixel(box_raster_coords.reshape((-1, 4, 2)))
    cv2.fillPoly(im, box_raster_coords, color=color, lineType=cv2.LINE_AA, shift=CV2_SHIFT)
    return im


def draw_agent_velocity(
    im: np.ndarray,
    raster_size: Tuple[int, int],
    raster_from_world: np.ndarray,
    agent: np.ndarray,
):
    # (frame) * (1 for 2D coord later) assume 50 frams and 0.1s per frame
    _extrapolate_time_base = np.arange(1, 51)[:, None] * 0.1

    velocity = agent['velocity']
    extrapolate_positions = (
        velocity[None, :] * _extrapolate_time_base + agent['centroid'][None, :2]
    )
    raster_coords = transform_points_fast(extrapolate_positions, raster_from_world)
    raster_coords = np.round(raster_coords).astype('int32')
    mask = (
        (0 <= raster_coords[:, 0]) & (raster_coords[:, 0] < raster_size[0]) &
        (0 <= raster_coords[:, 1]) & (raster_coords[:, 1] < raster_size[1])
    )
    raster_coords = raster_coords[mask]
    # plot onto the image
    # Note the first dimension in the image is y, then x
    im[tuple(raster_coords[:, ::-1].T)] = 250  # any color other than 255 for distinction
    return im


class BoxRasterizer(Rasterizer):
    def __init__(
        self, render_context: RenderContext, filter_agents_threshold: float, history_num_frames: int,
    ):
        """

        Args:
            render_context (RenderContext): Render context
            filter_agents_threshold (float): Value between 0 and 1 used to filter uncertain agent detections
            history_num_frames (int): Number of frames to rasterise in the past
        """
        super(BoxRasterizer, self).__init__()
        self.render_context = render_context
        self.raster_size = render_context.raster_size_px
        self.filter_agents_threshold = filter_agents_threshold
        self.history_num_frames = history_num_frames

    def rasterize(
        self,
        history_frames: np.ndarray,
        history_agents: List[np.ndarray],
        history_tl_faces: List[np.ndarray],
        agent: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        # all frames are drawn relative to this one
        frame = history_frames[0]
        if agent is None:
            ego_translation_m = history_frames[0]["ego_translation"]
            ego_yaw_rad = rotation33_as_yaw(frame["ego_rotation"])
        else:
            ego_translation_m = np.append(agent["centroid"], history_frames[0]["ego_translation"][-1])
            ego_yaw_rad = agent["yaw"]

        raster_from_world = self.render_context.raster_from_world(ego_translation_m, ego_yaw_rad)

        # this ensures we always end up with fixed size arrays, +1 is because current time is also in the history
        out_shape = (self.history_num_frames + 1, self.raster_size[1], self.raster_size[0])
        agents_images = np.zeros(out_shape, dtype=np.uint8)
        ego_images = np.zeros(out_shape, dtype=np.uint8)

        for i, (frame, agents) in enumerate(zip(history_frames, history_agents)):
            agents = filter_agents_by_labels(agents, self.filter_agents_threshold)
            # note the cast is for legacy support of dataset before April 2020
            av_agent = get_ego_as_agent(frame).astype(agents.dtype)

            if agent is None:
                agents_image = draw_boxes(self.raster_size, raster_from_world, agents, 255)
                ego_image = draw_boxes(self.raster_size, raster_from_world, av_agent, 255)
            else:
                agent_ego = filter_agents_by_track_id(agents, agent["track_id"])
                if len(agent_ego) == 0:  # agent not in this history frame
                    agents_image = draw_boxes(self.raster_size, raster_from_world, np.append(agents, av_agent), 255)
                    ego_image = np.zeros_like(agents_image)
                else:  # add av to agents and remove the agent from agents
                    agents = agents[agents != agent_ego[0]]
                    agents_image = draw_boxes(self.raster_size, raster_from_world, np.append(agents, av_agent), 255)
                    ego_image = draw_boxes(self.raster_size, raster_from_world, agent_ego, 255)

            agents_images[i] = agents_image
            ego_images[i] = ego_image

        # if agent is not None:
        #     # add extrapolation dots from velocity to the image at 0
        #     ego_images[0] = draw_agent_velocity(
        #         ego_images[0], self.raster_size, raster_from_world, agent
        #     )

        # combine such that the image consists of [agent_t, agent_t-1, agent_t-2, ego_t, ego_t-1, ego_t-2]
        out_im = np.concatenate((agents_images, ego_images), 0)

        return out_im.astype(np.float32) / 255

    def to_rgb(self, in_im: np.ndarray, **kwargs: dict) -> np.ndarray:
        """
        get an rgb image where agents further in the past have faded colors

        Args:
            in_im: the output of the rasterize function
            kwargs: this can be used for additional customization (such as colors)

        Returns: an RGB image with agents and ego coloured with fading colors
        """
        hist_frames = in_im.shape[-1] // 2
        in_im = np.transpose(in_im, (2, 0, 1))

        # this is similar to the draw history code
        out_im_agent = np.zeros((self.raster_size[1], self.raster_size[0], 3), dtype=np.float32)
        agent_chs = in_im[:hist_frames][::-1]  # reverse to start from the furthest one
        agent_color = (0, 0, 1) if "agent_color" not in kwargs else kwargs["agent_color"]
        for ch in agent_chs:
            out_im_agent *= 0.85  # magic fading constant for the past
            out_im_agent[ch > 0] = agent_color

        out_im_ego = np.zeros((self.raster_size[1], self.raster_size[0], 3), dtype=np.float32)
        ego_chs = in_im[hist_frames:][::-1]
        ego_color = (0, 1, 0) if "ego_color" not in kwargs else kwargs["ego_color"]
        for ch in ego_chs:
            out_im_ego *= 0.85
            out_im_ego[ch > 0] = ego_color

        out_im = (np.clip(out_im_agent + out_im_ego, 0, 1) * 255).astype(np.uint8)
        return out_im

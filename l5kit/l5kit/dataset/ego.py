import bisect
import warnings
from functools import partial
from typing import Optional

import numpy as np
# from torch.utils.data import Dataset  # to avoid loading the entire pytorch package

from ..data import (
    ChunkedDataset,
    get_agents_slice_from_frames,
    get_frames_slice_from_scenes,
    get_tl_faces_slice_from_frames,
)
from ..kinematic import Perturbation
from ..rasterization import Rasterizer, RenderContext
from ..sampling import generate_agent_sample


class Dataset:
    r"""An abstract class representing a :class:`Dataset`.

    All datasets that represent a map from keys to data samples should subclass
    it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses could also optionally overwrite
    :meth:`__len__`, which is expected to return the size of the dataset by many
    :class:`~torch.utils.data.Sampler` implementations and the default options
    of :class:`~torch.utils.data.DataLoader`.

    .. note::
      :class:`~torch.utils.data.DataLoader` by default constructs a index
      sampler that yields integral indices.  To make it work with a map-style
      dataset with non-integral indices/keys, a custom sampler must be provided.
    """

    def __getitem__(self, index):
        raise NotImplementedError

    # def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
    #     return ConcatDataset([self, other])

    # No `def __len__(self)` default?
    # See NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    # in pytorch/torch/utils/data/sampler.py


class EgoDataset(Dataset):
    def __init__(
        self,
        cfg: dict,
        zarr_dataset: ChunkedDataset,
        rasterizer: Rasterizer,
        perturbation: Optional[Perturbation] = None,
    ):
        """
        Get a PyTorch dataset object that can be used to train DNN

        Args:
            cfg (dict): configuration file
            zarr_dataset (ChunkedDataset): the raw zarr dataset
            rasterizer (Rasterizer): an object that support rasterisation around an agent (AV or not)
            perturbation (Optional[Perturbation]): an object that takes care of applying trajectory perturbations.
None if not desired
        """
        self.perturbation = perturbation
        self.cfg = cfg
        self.dataset = zarr_dataset
        self.rasterizer = rasterizer

        # scene ending frame_id+1
        # map: scene_id -> end_frame_id+1
        self.cumulative_sizes = self.dataset.scenes["frame_index_interval"][:, 1]
        # map: scene_id -> start_frame_id
        self.scene_start_frame_id = self.dataset.scenes["frame_index_interval"][:, 0]

        render_context = RenderContext(
            raster_size_px=np.array(cfg["raster_params"]["raster_size"]),
            pixel_size_m=np.array(cfg["raster_params"]["pixel_size"]),
            center_in_raster_ratio=np.array(cfg["raster_params"]["ego_center"]),
        )

        velocity_corrected_yaw = (
            cfg['model_params']['velocity_corrected_yaw'] 
            if 'velocity_corrected_yaw' in cfg['model_params']
            else False
        )

        # build a partial so we don't have to access cfg each time
        self.sample_function = partial(
            generate_agent_sample,
            render_context=render_context,
            history_num_frames=cfg["model_params"]["history_num_frames"],
            history_step_size=cfg["model_params"]["history_step_size"],
            history_step_time=cfg["model_params"]["history_delta_time"] * cfg["model_params"]["history_step_size"],
            future_num_frames=cfg["model_params"]["future_num_frames"],
            future_step_size=cfg["model_params"]["future_step_size"],
            future_step_time=cfg["model_params"]["future_delta_time"] * cfg["model_params"]["future_step_size"],
            filter_agents_threshold=cfg["raster_params"]["filter_agents_threshold"],
            rasterizer=rasterizer,
            perturbation=perturbation,
            velocity_corrected_yaw=velocity_corrected_yaw,
        )

    def __len__(self) -> int:
        """
        Get the number of available AV frames

        Returns:
            int: the number of elements in the dataset
        """
        return len(self.dataset.frames)

    # def get_frame(self, scene_index: int, state_index: int, track_id: Optional[int] = None) -> dict:
    def get_frame(self, scene_index: int, state_index: int, total_agent_id: int) -> dict:
        """
        A utility function to get the rasterisation and trajectory target for a given agent in a given frame

        Args:
            scene_index (int): the index of the scene in the zarr
            state_index (int): a relative frame index in the scene
            # track_id (Optional[int]): the agent to rasterize or None for the AV
            total_agent_id (int): the total_agent_id of the agent to rasterize or -1 for the AV
        Returns:
            dict: the rasterised image in (Cx0x1) if the rast is not None, the target trajectory
            (position and yaw) along with their availability, the 2D matrix to center that agent,
            the agent track (-1 if ego) and the timestamp

        """
        frames = self.dataset.frames[
            slice(*self.dataset.scenes[scene_index]["frame_index_interval"])
        ]
        # frames = self.dataset.frames[get_frames_slice_from_scenes(self.dataset.scenes[scene_index])]

        tl_faces = self.dataset.tl_faces
        # try:
        #     if self.cfg["raster_params"]["disable_traffic_light_faces"]:
        #         tl_faces = np.empty(0, dtype=self.dataset.tl_faces.dtype)  # completely disable traffic light faces
        # except KeyError:
        #     warnings.warn(
        #         "disable_traffic_light_faces not found in config, this will raise an error in the future",
        #         RuntimeWarning,
        #         stacklevel=2,
        #     )
        data = self.sample_function(
            state_index, frames, self.dataset.agents, tl_faces, selected_agent_id=total_agent_id,
        )
        # data = self.sample_function(state_index, frames, self.dataset.agents, tl_faces, track_id)

        # add information only, so that all data keys are always preserved
        # data["host_id"] = self.dataset.scenes[scene_index]["host"]
        data["timestamp"] = frames[state_index]["timestamp"]
        # data["track_id"] = np.int64(-1 if track_id is None else track_id)  # always a number to avoid crashing torch
        # data["world_to_image"] = data["raster_from_world"]  # TODO deprecate

        # when rast is None, image could be None. In that case we remove the key
        # if data["image"] is not None:
        #     data["image"] = data["image"].transpose(2, 0, 1)  # 0,1,C -> C,0,1
        # else:
        if data["image"] is None:
            del data["image"]

        return data

    def __getitem__(self, index: int) -> dict:
        """
        Note the ego dataset is indexed by frame
        Args:
            index (int): frame_id

        Returns: please look get_frame signature and docstring

        """
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index

        scene_index = bisect.bisect_right(self.cumulative_sizes, index)
        state_index = index - self.scene_start_frame_id[scene_index]
        # if scene_index == 0:
        #     state_index = index
        # else:
        #     state_index = index - self.cumulative_sizes[scene_index - 1]
        return self.get_frame(scene_index, state_index, total_agent_id=-1)

    def get_scene_dataset(self, scene_index: int) -> "EgoDataset":
        """
        Returns another EgoDataset dataset where the underlying data can be modified.
        This is possible because, even if it supports the same interface, this dataset is np.ndarray based.

        Args:
            scene_index (int): the scene index of the new dataset

        Returns:
            EgoDataset: A valid EgoDataset dataset with a copy of the data

        """
        # copy everything to avoid references (scene is already detached from zarr if get_combined_scene was called)
        scenes = self.dataset.scenes[scene_index : scene_index + 1].copy()
        frame_slice = get_frames_slice_from_scenes(*scenes)
        frames = self.dataset.frames[frame_slice].copy()
        agent_slice = get_agents_slice_from_frames(*frames[[0, -1]])
        tl_slice = get_tl_faces_slice_from_frames(*frames[[0, -1]])

        agents = self.dataset.agents[agent_slice].copy()
        tl_faces = self.dataset.tl_faces[tl_slice].copy()

        frames["agent_index_interval"] -= agent_slice.start
        frames["traffic_light_faces_index_interval"] -= tl_slice.start
        scenes["frame_index_interval"] -= frame_slice.start

        dataset = ChunkedDataset("")
        dataset.agents = agents
        dataset.tl_faces = tl_faces
        dataset.frames = frames
        dataset.scenes = scenes

        return EgoDataset(self.cfg, dataset, self.rasterizer, self.perturbation)

    def get_scene_indices(self, scene_idx: int) -> np.ndarray:
        """
        Get indices for the given scene. EgoDataset iterates over frames, so this is just a matter
        of finding the scene boundaries.
        Args:
            scene_idx (int): index of the scene

        Returns:
            np.ndarray: indices that can be used for indexing with __getitem__
        """
        scenes = self.dataset.scenes
        assert scene_idx < len(scenes), f"scene_idx {scene_idx} is over len {len(scenes)}"
        return np.arange(*scenes[scene_idx]["frame_index_interval"])

    def get_frame_indices(self, frame_idx: int) -> np.ndarray:
        """
        Get indices for the given frame. EgoDataset iterates over frames, so this will be a single element
        Args:
            frame_idx (int): index of the scene

        Returns:
            np.ndarray: indices that can be used for indexing with __getitem__
        """
        frames = self.dataset.frames
        assert frame_idx < len(frames), f"frame_idx {frame_idx} is over len {len(frames)}"
        return np.asarray((frame_idx,), dtype=np.int64)

    def __str__(self) -> str:
        return self.dataset.__str__()

import numpy as np


class RenderContext:
    def __init__(
        self, raster_size_px: np.ndarray, pixel_size_m: np.ndarray, center_in_raster_ratio: np.ndarray,
    ) -> None:
        """
        This class stores render context information (raster size, pixel size, raster center / principle point) and
        it computes a transformation matrix (raster_from_local) to transform a local coordinates into raster ones.
        (0,0) in local will end up in the center of the raster (specified by combining `raster_size_px` and
        `center_in_raster_ratio`).

        Args:
            raster_size_px (Tuple[int, int]): Raster size in pixels
            pixel_size_m (np.ndarray): Size of one pixel in the real world, meter per pixel
            center_in_raster_ratio (np.ndarray): Where to center the local pose in the raster. [0.5,0.5] would be in
                the raster center, [0, 0] is bottom left.
        """

        if pixel_size_m[0] != pixel_size_m[1]:
            raise NotImplementedError("No support for non squared pixels yet")

        self.raster_size_px = raster_size_px
        self.pixel_size_m = pixel_size_m
        self.center_in_raster_ratio = center_in_raster_ratio

        scaling = 1.0 / pixel_size_m  # scaling factor from world to raster space [pixels per meter]
        center_in_raster_px = center_in_raster_ratio * raster_size_px
        self.raster_from_local = np.array([
            [scaling[0], 0., center_in_raster_px[0]],
            [0., scaling[1], center_in_raster_px[1]],
            [0., 0., 1.],
        ])
        self.inv_raster_from_local = np.array([
            [1. / scaling[0], 0., -center_in_raster_px[0] / scaling[0]],
            [0., 1. / scaling[1], -center_in_raster_px[1] / scaling[1]],
            [0., 0., 1.],
        ])

    def raster_from_world(self, position_m: np.ndarray, angle_rad: float) -> np.ndarray:
        """
        Return a matrix to convert a pose in world coordinates into raster coordinates

        Args:
            render_context (RenderContext): the context for rasterisation
            position_m (np.ndarray): XY position in world coordinates
            angle_rad (float): rotation angle in world coordinates

        Returns:
            (np.ndarray): a transformation matrix from world coordinates to raster coordinates
        """
        # Compute pose from its position and rotation
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        # pose_in_world = np.array([
        #     [c, -s, position_m[0]],
        #     [s, c, position_m[1]],
        #     [0, 0, 1],
        # ])
        pp = np.array([[c, s], [-s, c]]) @ position_m[:2]
        pose_from_world = np.array([
            [c, s, -pp[0]],
            [-s, c, -pp[1]],
            [0., 0., 1.],
        ])

        # pose_from_world = np.linalg.inv(pose_in_world)
        # raster_from_world = self.raster_from_local @ pose_from_world
        # return raster_from_world
        return self.raster_from_local @ pose_from_world

    def world_from_raster(self, position_m: np.ndarray, angle_rad: float) -> np.ndarray:
        """
        Return a matrix to convert a pose in raster coordinates into world coordinates

        Args:
            render_context (RenderContext): the context for rasterisation
            position_m (np.ndarray): XY position in world coordinates
            angle_rad (float): rotation angle in world coordinates

        Returns:
            (np.ndarray): a transformation matrix from raster coordinates to world coordinates
        """
        # Compute pose from its position and rotation
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        pose_in_world = np.array([
            [c, -s, position_m[0]],
            [s, c, position_m[1]],
            [0., 0., 1.],
        ])

        world_from_raster = pose_in_world @ self.inv_raster_from_local
        return world_from_raster

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for random assets importer."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.assets import AssetBaseCfg
from dataclasses import MISSING

from .random_assets import RandomAssetsImporter


@configclass
class RandomAssetCfg:
    """Configuration for a single random asset type."""

    # Asset type configuration
    asset_type: str = "rigid_object"
    """Type of asset to create: 'rigid_object' for physics simulation or 'static' for static geometry."""

    rigid_object_cfg: RigidObjectCfg | None = None
    """RigidObjectCfg for this asset type. Only used when asset_type='rigid_object'."""
    
    static_asset_cfg: AssetBaseCfg | None = None
    """AssetBaseCfg for static assets. Only used when asset_type='static'."""

    proportion: float = 1.0
    """Proportion of this asset type in the total mix. Used when use_proportions=True."""

    pos_range_x: tuple[float, float] = (-10.0, 10.0)
    """Range for random X position in world frame."""

    pos_range_y: tuple[float, float] = (-10.0, 10.0)
    """Range for random Y position in world frame."""

    pos_range_z: tuple[float, float] = (0.0, 5.0)
    """Range for random Z position in world frame."""

    rot_range_x: tuple[float, float] = (0.0, 0.0)
    """Range for random X rotation in radians (roll)."""

    rot_range_y: tuple[float, float] = (0.0, 0.0)
    """Range for random Y rotation in radians (pitch)."""

    rot_range_z: tuple[float, float] = (0.0, 6.28318)
    """Range for random Z rotation in radians (yaw - 0 to 2*pi for full rotation)."""

    scale_range: tuple[float, float] = (0.8, 1.2)
    """Range for random uniform scaling."""

    prim_path_template: str = "/World/{ASSET_NAME}/{ASSET_NAME}_{ASSET_INDEX}"
    """Template for prim path. {ASSET_NAME} and {ASSET_INDEX} will be replaced."""

    # Velocity control configuration (only for rigid_object type)
    enable_velocity_control: bool = False
    """Whether to enable continuous velocity control for this asset type. Only works with asset_type='rigid_object'."""
    
    linear_velocity_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = ((-1.0, 1.0), (-1.0, 1.0), (-0.5, 0.5))
    """Range for random linear velocities in world frame: ((x_min, x_max), (y_min, y_max), (z_min, z_max)). Only works with asset_type='rigid_object'."""
    
    angular_velocity_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = ((-0.5, 0.5), (-0.5, 0.5), (-1.0, 1.0))
    """Range for random angular velocities in world frame: ((x_min, x_max), (y_min, y_max), (z_min, z_max)). Only works with asset_type='rigid_object'."""
    
    velocity_update_frequency: float = 1.0
    """Frequency (in Hz) to update random velocities. Lower values change velocity less often. Only works with asset_type='rigid_object'."""

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Validate asset type
        if self.asset_type not in ["rigid_object", "static"]:
            raise ValueError(f"asset_type must be 'rigid_object' or 'static', got '{self.asset_type}'")
        
        # Initialize the appropriate configuration based on asset type
        if self.asset_type == "rigid_object":
            if self.rigid_object_cfg is None:
                self.rigid_object_cfg = RigidObjectCfg()
            # Set static_asset_cfg to None to avoid validation issues
            self.static_asset_cfg = None
        elif self.asset_type == "static":
            if self.static_asset_cfg is None:
                self.static_asset_cfg = AssetBaseCfg()
            # Set rigid_object_cfg to None to avoid validation issues
            self.rigid_object_cfg = None
        
        # Disable velocity control for static assets
        if self.asset_type == "static" and self.enable_velocity_control:
            print(f"Warning: Velocity control is not supported for static assets. Disabling velocity control.")
            self.enable_velocity_control = False


@configclass
class RandomAssetsImporterCfg:
    """Configuration for random assets importer.
    
    This class manages the spawning of multiple random assets across different types.
    Assets are spawned globally in /World/ space, not per environment.
    Each asset type creates a RigidObject that can be controlled and queried.
    """

    ##
    # Initialize configurations.
    ##

    class_type: type = RandomAssetsImporter

    # Random asset specific configurations
    
    assets: dict[str, RandomAssetCfg] = {}
    """Dictionary of asset configurations. Key is the asset name, value is the configuration."""

    total_global_assets: int = 100
    """Total number of assets to spawn across all types globally."""

    use_proportions: bool = True
    """Whether to use proportions to determine how many of each asset type to spawn.
    If False, spawns equal numbers of each asset type."""

    seed: int | None = None
    """Random seed for reproducible results. If None, uses random seed."""

    enable_collision_avoidance: bool = False
    """Whether to enable simple collision avoidance between spawned assets."""

    min_distance_between_assets: float = 1.0
    """Minimum distance between assets when collision avoidance is enabled."""

    debug_vis: bool = False
    """Whether to enable debug visualization showing spawn ranges and locations."""

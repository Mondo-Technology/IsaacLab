# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Random assets importer for spawning multiple random assets globally in the scene."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import torch
import numpy as np

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets.rigid_object import RigidObject, RigidObjectCfg
from isaaclab.assets import AssetBaseCfg

if TYPE_CHECKING:
    from .random_assets_cfg import RandomAssetsImporterCfg


class RandomAssetsImporter:
    """Random assets importer for spawning multiple random assets globally in the scene.
    
    This class manages the spawning of multiple random assets of different types across the scene.
    Unlike other assets, these are spawned globally in /World/ space, not per environment.
    
    Supports two asset types:
    - RigidObjects: Full physics simulation with control APIs (velocity control, forces, etc.)
    - Static Assets: Geometry-only assets for decoration/terrain (no physics simulation)
    
    The class handles:
    - Random placement of assets based on configured ranges
    - Proportional distribution of different asset types
    - Optional collision avoidance between assets
    - Seeded random generation for reproducibility
    - Full RigidObject API access for physics-enabled assets
    - Velocity control for RigidObjects (not available for static assets)
    """

    cfg: RandomAssetsImporterCfg
    """Configuration for the random assets importer."""

    rigid_objects: dict[str, list[RigidObject]]
    """Dictionary of RigidObject lists for each rigid asset type."""
    
    static_assets: dict[str, list[dict]]
    """Dictionary of static asset data for each static asset type."""

    _spawned_positions: list[tuple[float, float, float]]
    """List of spawned asset positions for collision avoidance."""

    # Velocity control attributes
    _velocity_timer: float
    """Timer for velocity updates."""
    
    _asset_velocities: dict[str, torch.Tensor]
    """Current velocities for each asset type. Format: {asset_name: tensor[num_assets, 6]}"""

    def __init__(self, cfg: RandomAssetsImporterCfg):
        """Initialize the random assets importer.

        Args:
            cfg: Configuration for the random assets importer.
        """
        # Store configuration
        self.cfg = cfg
        
        # Initialize RigidObject collections
        self.rigid_objects = {}
        self.static_assets = {}
        
        # Store the spawned asset positions for collision avoidance
        self._spawned_positions = []
        
        # Initialize velocity control
        self._velocity_timer = 0.0
        self._asset_velocities = {}
        
        # Set random seed if provided
        if self.cfg.seed is not None:
            random.seed(self.cfg.seed)
            np.random.seed(self.cfg.seed)
            torch.manual_seed(self.cfg.seed)
        
        # Spawn the assets by creating RigidObjects
        self._spawn_random_assets()

    def _calculate_asset_counts(self) -> dict[str, int]:
        """Calculate how many of each asset type to spawn based on configuration.
        
        Returns:
            Dictionary mapping asset names to the number of instances to spawn.
        """
        asset_counts = {}
        
        if not self.cfg.assets:
            return asset_counts
            
        if self.cfg.use_proportions:
            # Calculate based on proportions
            total_proportion = sum(asset_cfg.proportion for asset_cfg in self.cfg.assets.values())
            
            for asset_name, asset_cfg in self.cfg.assets.items():
                count = int((asset_cfg.proportion / total_proportion) * self.cfg.total_global_assets)
                asset_counts[asset_name] = max(1, count)  # Ensure at least 1 of each type
        else:
            # Equal distribution
            count_per_type = max(1, self.cfg.total_global_assets // len(self.cfg.assets))
            for asset_name in self.cfg.assets.keys():
                asset_counts[asset_name] = count_per_type
        
        # Adjust if we've gone over the total
        total_assigned = sum(asset_counts.values())
        if total_assigned > self.cfg.total_global_assets:
            # Reduce counts proportionally
            scale = self.cfg.total_global_assets / total_assigned
            for asset_name in asset_counts:
                asset_counts[asset_name] = max(1, int(asset_counts[asset_name] * scale))
        
        return asset_counts

    def _generate_random_positions(self, asset_cfg, count: int) -> torch.Tensor:
        """Generate random positions for multiple assets.
        
        Args:
            asset_cfg: Configuration for the specific asset type.
            count: Number of positions to generate.
            
        Returns:
            Tensor of positions [count, 3] in world frame.
        """
        positions = []
        
        for _ in range(count):
            max_attempts = 100 if self.cfg.enable_collision_avoidance else 1
            
            for attempt in range(max_attempts):
                # Generate random position
                pos_x = random.uniform(asset_cfg.pos_range_x[0], asset_cfg.pos_range_x[1])
                pos_y = random.uniform(asset_cfg.pos_range_y[0], asset_cfg.pos_range_y[1])
                pos_z = random.uniform(asset_cfg.pos_range_z[0], asset_cfg.pos_range_z[1])
                position = (pos_x, pos_y, pos_z)
                
                if self._check_collision_avoidance(position):
                    positions.append(position)
                    self._spawned_positions.append(position)
                    break
            else:
                # Could not find a valid position, use last attempt
                print(f"Warning: Could not find collision-free position after {max_attempts} attempts")
                positions.append(position)
                self._spawned_positions.append(position)
        
        return torch.tensor(positions, dtype=torch.float32)

    def _generate_random_rotations(self, asset_cfg, count: int) -> torch.Tensor:
        """Generate random rotations for multiple assets.
        
        Args:
            asset_cfg: Configuration for the specific asset type.
            count: Number of rotations to generate.
            
        Returns:
            Tensor of quaternions [count, 4] (w, x, y, z).
        """
        rotations = []
        
        for _ in range(count):
            # Generate random rotations around all three axes
            rot_x = random.uniform(asset_cfg.rot_range_x[0], asset_cfg.rot_range_x[1])  # Roll
            rot_y = random.uniform(asset_cfg.rot_range_y[0], asset_cfg.rot_range_y[1])  # Pitch  
            rot_z = random.uniform(asset_cfg.rot_range_z[0], asset_cfg.rot_range_z[1])  # Yaw
            
            # Convert Euler angles (XYZ order) to quaternion
            # Using the standard formula for XYZ Euler to quaternion conversion
            cx = math.cos(rot_x / 2)
            sx = math.sin(rot_x / 2)
            cy = math.cos(rot_y / 2)
            sy = math.sin(rot_y / 2)
            cz = math.cos(rot_z / 2)
            sz = math.sin(rot_z / 2)
            
            # Quaternion multiplication for XYZ order: qz * qy * qx
            w = cx * cy * cz + sx * sy * sz
            x = sx * cy * cz - cx * sy * sz
            y = cx * sy * cz + sx * cy * sz
            z = cx * cy * sz - sx * sy * cz
            
            rotations.append([w, x, y, z])
        
        return torch.tensor(rotations, dtype=torch.float32)

    def _generate_random_velocities(self, asset_cfg, count: int) -> torch.Tensor:
        """Generate random velocities for multiple assets.
        
        Args:
            asset_cfg: Configuration for the specific asset type.
            count: Number of velocity vectors to generate.
            
        Returns:
            Tensor of velocities [count, 6] where columns are [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z].
        """
        velocities = []
        
        for _ in range(count):
            # Generate random linear velocities
            lin_x = random.uniform(asset_cfg.linear_velocity_range[0][0], asset_cfg.linear_velocity_range[0][1])
            lin_y = random.uniform(asset_cfg.linear_velocity_range[1][0], asset_cfg.linear_velocity_range[1][1])
            lin_z = random.uniform(asset_cfg.linear_velocity_range[2][0], asset_cfg.linear_velocity_range[2][1])
            
            # Generate random angular velocities
            ang_x = random.uniform(asset_cfg.angular_velocity_range[0][0], asset_cfg.angular_velocity_range[0][1])
            ang_y = random.uniform(asset_cfg.angular_velocity_range[1][0], asset_cfg.angular_velocity_range[1][1])
            ang_z = random.uniform(asset_cfg.angular_velocity_range[2][0], asset_cfg.angular_velocity_range[2][1])
            
            velocities.append([lin_x, lin_y, lin_z, ang_x, ang_y, ang_z])
        
        return torch.tensor(velocities, dtype=torch.float32)

    def _check_collision_avoidance(self, new_pos: tuple[float, float, float]) -> bool:
        """Check if a new position would violate collision avoidance constraints.
        
        Args:
            new_pos: Position to check.
            
        Returns:
            True if position is valid (no collisions), False otherwise.
        """
        if not self.cfg.enable_collision_avoidance:
            return True
            
        min_dist_sq = self.cfg.min_distance_between_assets ** 2
        
        for existing_pos in self._spawned_positions:
            dist_sq = sum((new_pos[i] - existing_pos[i]) ** 2 for i in range(3))
            if dist_sq < min_dist_sq:
                return False
        
        return True

    def _spawn_random_assets(self):
        """Spawn all random assets according to the configuration using RigidObjects or static assets."""
        if not self.cfg.assets:
            print("Warning: No assets configured for RandomAssetsImporter")
            return
        
        # Calculate how many of each asset type to spawn
        asset_counts = self._calculate_asset_counts()
        
        print(f"RandomAssetsImporter: Spawning {sum(asset_counts.values())} total assets:")
        for asset_name, count in asset_counts.items():
            asset_cfg = self.cfg.assets[asset_name]
            print(f"  - {asset_name}: {count} instances ({asset_cfg.asset_type})")
        
        # Create assets for each asset type
        for asset_name, count in asset_counts.items():
            asset_cfg = self.cfg.assets[asset_name]
            
            if count <= 0:
                continue
            
            # Generate random initial positions and rotations
            positions = self._generate_random_positions(asset_cfg, count)
            rotations = self._generate_random_rotations(asset_cfg, count)
            
            if asset_cfg.asset_type == "rigid_object":
                self._spawn_rigid_objects(asset_name, asset_cfg, count, positions, rotations)
            elif asset_cfg.asset_type == "static":
                self._spawn_static_assets(asset_name, asset_cfg, count, positions, rotations)
            else:
                print(f"Warning: Unknown asset type '{asset_cfg.asset_type}' for {asset_name}")

    def _spawn_rigid_objects(self, asset_name: str, asset_cfg, count: int, positions: torch.Tensor, rotations: torch.Tensor):
        """Spawn RigidObjects for a specific asset type."""
        if asset_cfg.rigid_object_cfg is None:
            print(f"Error: No rigid_object_cfg provided for rigid_object asset type {asset_name}")
            return
            
        asset_rigid_objects = []
        
        # Create individual RigidObject for each instance
        for i in range(count):
            # Generate specific prim path for this instance
            prim_path = asset_cfg.prim_path_template.replace("{ASSET_NAME}", asset_name).replace("{ASSET_INDEX}", str(i))
            
            # Get position and rotation for this instance
            pos = positions[i].tolist()
            rot = rotations[i].tolist()  # [w, x, y, z]
            
            # Generate random scale
            scale = random.uniform(asset_cfg.scale_range[0], asset_cfg.scale_range[1])
            
            try:
                # Create a copy of the RigidObjectCfg for this specific instance
                rigid_cfg = RigidObjectCfg()
                for key, value in asset_cfg.rigid_object_cfg.__dict__.items():
                    if hasattr(rigid_cfg, key):
                        setattr(rigid_cfg, key, value)
                
                # Set the specific prim path (no wildcards)
                rigid_cfg.prim_path = prim_path
                
                # Set initial state
                if hasattr(rigid_cfg.init_state, 'pos'):
                    rigid_cfg.init_state.pos = tuple(pos)
                if hasattr(rigid_cfg.init_state, 'rot'):
                    rigid_cfg.init_state.rot = tuple(rot)
                
                # Apply scaling by removing it for now - will implement later
                # TODO: Implement proper scaling support
                
                # Create the individual RigidObject
                rigid_object = RigidObject(cfg=rigid_cfg)
                asset_rigid_objects.append(rigid_object)
                
            except Exception as e:
                print(f"Error creating RigidObject for {asset_name}_{i}: {e}")
                continue
        
        if asset_rigid_objects:
            # Store the list of RigidObjects for this asset type
            self.rigid_objects[asset_name] = asset_rigid_objects
            print(f"Successfully created {len(asset_rigid_objects)} RigidObjects for {asset_name}")
            
            # Initialize velocities for velocity-controlled assets
            if asset_cfg.enable_velocity_control:
                initial_velocities = self._generate_random_velocities(asset_cfg, len(asset_rigid_objects))
                self._asset_velocities[asset_name] = initial_velocities
                print(f"Initialized velocity control for {asset_name} with {len(asset_rigid_objects)} objects")
        else:
            print(f"Warning: No RigidObjects created for asset type {asset_name}")

    def _spawn_static_assets(self, asset_name: str, asset_cfg, count: int, positions: torch.Tensor, rotations: torch.Tensor):
        """Spawn static assets for a specific asset type using proper asset spawning."""
        if asset_cfg.static_asset_cfg is None:
            print(f"Error: No static_asset_cfg provided for static asset type {asset_name}")
            return
            
        asset_static_objects = []
        
        # Create individual static assets for each instance
        for i in range(count):
            # Generate specific prim path for this instance
            prim_path = asset_cfg.prim_path_template.replace("{ASSET_NAME}", asset_name).replace("{ASSET_INDEX}", str(i))
            
            # Get position and rotation for this instance
            pos = positions[i].tolist()
            rot = rotations[i].tolist()  # [w, x, y, z]
            
            # Generate random scale
            scale = random.uniform(asset_cfg.scale_range[0], asset_cfg.scale_range[1])
            
            try:
                # Get the spawn configuration from static_asset_cfg
                static_asset_cfg = asset_cfg.static_asset_cfg
                
                # Use the proper spawning pattern like InteractiveScene
                if static_asset_cfg.spawn is not None:
                    static_asset_cfg.spawn.func(
                        prim_path,
                        static_asset_cfg.spawn,
                        translation=pos,
                        orientation=rot,
                    )
                
                # Store information about the spawned static asset
                static_asset_data = {
                    "prim_path": prim_path,
                    "spawn_cfg": static_asset_cfg.spawn,
                    "position": pos,
                    "rotation": rot,
                    "scale": scale,
                }
                asset_static_objects.append(static_asset_data)
                
            except Exception as e:
                print(f"Error creating static asset for {asset_name}_{i}: {e}")
                continue
        
        if asset_static_objects:
            # Store the list of static asset data for this asset type
            self.static_assets[asset_name] = asset_static_objects
            print(f"Successfully created {len(asset_static_objects)} static assets for {asset_name}")
        else:
            print(f"Warning: No static assets created for asset type {asset_name}")

    @property
    def num_assets(self) -> int:
        """Total number of successfully spawned asset instances across all types."""
        rigid_count = sum(len(obj_list) for obj_list in self.rigid_objects.values())
        static_count = sum(len(obj_list) for obj_list in self.static_assets.values())
        return rigid_count + static_count

    def get_rigid_object(self, asset_name: str) -> list[RigidObject] | None:
        """Get the RigidObjects for a specific asset type.
        
        Args:
            asset_name: Name of the asset type.
            
        Returns:
            List of RigidObject instances or None if not found.
        """
        return self.rigid_objects.get(asset_name)

    def get_static_asset(self, asset_name: str) -> list[dict] | None:
        """Get the static assets for a specific asset type.
        
        Args:
            asset_name: Name of the asset type.
            
        Returns:
            List of static asset data or None if not found.
        """
        return self.static_assets.get(asset_name)

    def get_all_rigid_objects(self) -> dict[str, list[RigidObject]]:
        """Get all RigidObject instances.
        
        Returns:
            Dictionary mapping asset names to lists of RigidObject instances.
        """
        return self.rigid_objects.copy()

    def get_all_static_assets(self) -> dict[str, list[dict]]:
        """Get all static asset instances.
        
        Returns:
            Dictionary mapping asset names to lists of static asset data.
        """
        return self.static_assets.copy()

    def get_asset_info(self) -> dict[str, dict]:
        """Get information about all assets.
        
        Returns:
            Dictionary with asset information including type and count.
        """
        info = {}
        for asset_name, asset_cfg in self.cfg.assets.items():
            rigid_objects = self.rigid_objects.get(asset_name, [])
            static_assets = self.static_assets.get(asset_name, [])
            
            info[asset_name] = {
                "asset_type": asset_cfg.asset_type,
                "rigid_object_count": len(rigid_objects),
                "static_asset_count": len(static_assets),
                "total_count": len(rigid_objects) + len(static_assets),
                "velocity_control_enabled": asset_cfg.enable_velocity_control if asset_cfg.asset_type == "rigid_object" else False,
            }
        return info

    def write_data_to_sim(self):
        """Write data to simulation for RigidObjects. Static assets don't need this."""
        for rigid_object_list in self.rigid_objects.values():
            for rigid_object in rigid_object_list:
                rigid_object.write_data_to_sim()

    def update(self, dt: float):
        """Update RigidObjects and apply velocity control. Static assets don't need updates.
        
        Args:
            dt: Simulation time step.
        """
        # Update velocity timer
        self._velocity_timer += dt
        
        # Update velocities and apply to RigidObjects
        self._update_and_apply_velocities(dt)
        
        # Update all RigidObjects (static assets don't need updates)
        for rigid_object_list in self.rigid_objects.values():
            for rigid_object in rigid_object_list:
                rigid_object.update(dt)

    def _update_and_apply_velocities(self, dt: float):
        """Update and apply velocities to velocity-controlled rigid object assets.
        
        Args:
            dt: Simulation time step.
        """
        for asset_name, rigid_object_list in self.rigid_objects.items():
            asset_cfg = self.cfg.assets[asset_name]
            
            # Skip if not a rigid object or velocity control is not enabled
            if asset_cfg.asset_type != "rigid_object" or not asset_cfg.enable_velocity_control:
                continue
                
            # Check if it's time to update velocities based on frequency
            update_interval = 1.0 / asset_cfg.velocity_update_frequency
            if self._velocity_timer >= update_interval:
                # Generate new random velocities for all objects of this type
                new_velocities = self._generate_random_velocities(asset_cfg, len(rigid_object_list))
                self._asset_velocities[asset_name] = new_velocities
            
            # Apply current velocities to all RigidObjects of this type
            current_velocities = self._asset_velocities.get(asset_name)
            if current_velocities is not None:
                for i, rigid_object in enumerate(rigid_object_list):
                    if i < len(current_velocities):
                        try:
                            # Get velocity for this specific object
                            velocity = current_velocities[i].unsqueeze(0)  # Add batch dimension for single object
                            # Apply velocity to the RigidObject
                            rigid_object.write_root_velocity_to_sim(velocity, env_ids=None)
                        except Exception as e:
                            print(f"Error applying velocity to {asset_name}_{i}: {e}")
        
        # Reset velocity timer if it exceeded the update interval for rigid objects
        min_update_interval = float('inf')
        for asset_name in self.rigid_objects.keys():
            asset_cfg = self.cfg.assets[asset_name]
            if asset_cfg.asset_type == "rigid_object" and asset_cfg.enable_velocity_control:
                update_interval = 1.0 / asset_cfg.velocity_update_frequency
                min_update_interval = min(min_update_interval, update_interval)
        
        if min_update_interval != float('inf') and self._velocity_timer >= min_update_interval:
            self._velocity_timer = 0.0

    def reset(self, env_ids: list[int] | None = None):
        """Reset RigidObjects. Static assets don't need reset.
        
        Args:
            env_ids: Environment IDs to reset. Since these are global assets, this is ignored.
        """
        for rigid_object_list in self.rigid_objects.values():
            for rigid_object in rigid_object_list:
                rigid_object.reset()

    def get_velocity_control_info(self) -> dict[str, dict]:
        """Get information about velocity control for rigid object asset types.
        
        Returns:
            Dictionary with velocity control information for each rigid object asset type.
        """
        info = {}
        for asset_name, asset_cfg in self.cfg.assets.items():
            if asset_cfg.asset_type == "rigid_object":
                info[asset_name] = {
                    "velocity_control_enabled": asset_cfg.enable_velocity_control,
                    "num_objects": len(self.rigid_objects.get(asset_name, [])),
                    "linear_velocity_range": asset_cfg.linear_velocity_range,
                    "angular_velocity_range": asset_cfg.angular_velocity_range,
                    "update_frequency": asset_cfg.velocity_update_frequency,
                    "current_velocities": self._asset_velocities.get(asset_name)
                }
        return info

    def set_asset_velocities(self, asset_name: str, velocities: torch.Tensor):
        """Manually set velocities for a specific rigid object asset type.
        
        Args:
            asset_name: Name of the asset type.
            velocities: Velocity tensor [num_objects, 6] where columns are [lin_x, lin_y, lin_z, ang_x, ang_y, ang_z].
        """
        if asset_name in self.rigid_objects and asset_name in self.cfg.assets:
            asset_cfg = self.cfg.assets[asset_name]
            if asset_cfg.asset_type == "rigid_object" and asset_cfg.enable_velocity_control:
                self._asset_velocities[asset_name] = velocities
                print(f"Set custom velocities for {asset_name}")
            elif asset_cfg.asset_type != "rigid_object":
                print(f"Warning: Velocity control is only supported for rigid_object assets, {asset_name} is {asset_cfg.asset_type}")
            else:
                print(f"Warning: Velocity control is not enabled for {asset_name}")
        else:
            print(f"Warning: Asset type {asset_name} not found")

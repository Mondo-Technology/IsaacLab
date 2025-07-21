#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Test script for random assets with both RigidObject and static asset types.

This script demonstrates:
1. How to configure assets as either RigidObjects (with physics) or static assets (geometry only)
2. Velocity control on RigidObjects while static assets remain stationary
3. Mixed asset types in the same scene

.. code-block:: bash

    # Run the script
    ./isaaclab.sh -p scripts/demos/test_mixed_asset_types.py --num_envs 1

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Test mixed asset types (rigid objects vs static assets).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.random_assets import RandomAssetsImporterCfg, RandomAssetCfg
from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass


def main():
    """Main function."""
    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, render_interval=1)
    sim_context = SimulationContext(cfg=sim_cfg)

    # Design the scene
    @configclass
    class TestSceneCfg(InteractiveSceneCfg):
        """Configuration for the test scene with mixed asset types."""

        # Ground plane
        ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=sim_utils.GroundPlaneCfg(size=(50.0, 50.0)),
        )

        # Mixed asset types
        mixed_assets = RandomAssetsImporterCfg(
            assets={
                # Moving rigid objects with physics and velocity control
                "moving_cubes": RandomAssetCfg(
                    asset_type="rigid_object",  # Physics simulation enabled
                    rigid_object_cfg=RigidObjectCfg(
                        prim_path="/World/moving_cubes",  # This will be modified by the template
                        spawn=sim_utils.CuboidCfg(
                            size=(0.5, 0.5, 0.5),
                            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                            collision_props=sim_utils.CollisionPropertiesCfg(),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.2)),
                        ),
                    ),
                    proportion=0.3,
                    pos_range_x=(-8.0, -3.0),
                    pos_range_y=(-5.0, 5.0), 
                    pos_range_z=(0.5, 3.0),
                    # Enable velocity control for continuous movement
                    enable_velocity_control=True,
                    linear_velocity_range=((-1.0, 1.0), (-1.0, 1.0), (-0.3, 0.3)),
                    angular_velocity_range=((-0.5, 0.5), (-0.5, 0.5), (-1.0, 1.0)),
                    velocity_update_frequency=0.5,  # Change velocities every 2 seconds
                ),
                
                # Static terrain/decoration objects (no physics)
                "terrain_rocks": RandomAssetCfg(
                    asset_type="static",  # Static geometry only, no physics
                    static_asset_cfg=AssetBaseCfg(
                        prim_path="/World/terrain_rocks",  # This will be modified by the template
                        spawn=sim_utils.SphereCfg(
                            radius=0.4,
                            # No rigid_props or collision_props = static geometry
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.3, 0.2)),
                        ),
                    ),
                    proportion=0.4,
                    pos_range_x=(-2.0, 2.0),
                    pos_range_y=(-8.0, 8.0),
                    pos_range_z=(0.2, 1.0),
                    # Velocity control is automatically disabled for static assets
                    enable_velocity_control=False,  # This will be ignored anyway
                ),
                
                # More moving rigid objects
                "moving_spheres": RandomAssetCfg(
                    asset_type="rigid_object",  # Physics simulation enabled
                    rigid_object_cfg=RigidObjectCfg(
                        prim_path="/World/moving_spheres",  # This will be modified by the template
                        spawn=sim_utils.SphereCfg(
                            radius=0.3,
                            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                            collision_props=sim_utils.CollisionPropertiesCfg(),
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 1.0, 0.2)),
                        ),
                    ),
                    proportion=0.3,
                    pos_range_x=(3.0, 8.0),
                    pos_range_y=(-5.0, 5.0),
                    pos_range_z=(0.5, 4.0),
                    # Enable different velocity pattern
                    enable_velocity_control=True,
                    linear_velocity_range=((-1.5, 1.5), (-1.5, 1.5), (-0.5, 0.5)),
                    angular_velocity_range=((-1.0, 1.0), (-1.0, 1.0), (-2.0, 2.0)),
                    velocity_update_frequency=1.0,  # Change velocities every second
                ),
            },
            total_global_assets=40,
            use_proportions=True,
            seed=42,
            enable_collision_avoidance=True,
            min_distance_between_assets=1.2,
        )

        # Light
        light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0)),
        )

    # Create scene
    scene_cfg = TestSceneCfg(num_envs=args_cli.num_envs, env_spacing=10.0)
    scene = InteractiveScene(scene_cfg)

    # Play the simulation
    sim_context.play()

    # Print asset information
    random_assets = scene.random_assets["mixed_assets"]
    print("\n" + "="*80)
    print("MIXED ASSET TYPES DEMONSTRATION")
    print("="*80)
    print(f"Total assets spawned: {random_assets.num_assets}")
    print()
    
    # Print general asset info
    asset_info = random_assets.get_asset_info()
    for asset_name, info in asset_info.items():
        print(f"Asset Type: {asset_name}")
        print(f"  - Type: {info['asset_type']}")
        print(f"  - Rigid Objects: {info['rigid_object_count']}")
        print(f"  - Static Assets: {info['static_asset_count']}")
        print(f"  - Total Count: {info['total_count']}")
        print(f"  - Velocity Control: {'Enabled' if info['velocity_control_enabled'] else 'Disabled'}")
        print()
    
    # Print velocity control info for rigid objects only
    velocity_info = random_assets.get_velocity_control_info()
    if velocity_info:
        print("Velocity Control Details (Rigid Objects Only):")
        for asset_name, info in velocity_info.items():
            print(f"  {asset_name}:")
            print(f"    - Linear Velocity Range: {info['linear_velocity_range']}")
            print(f"    - Angular Velocity Range: {info['angular_velocity_range']}")
            print(f"    - Update Frequency: {info['update_frequency']} Hz")
        print()

    print("Observation Guide:")
    print("- RED CUBES (left): RigidObjects with physics and velocity control - they move around")
    print("- BROWN SPHERES (center): Static assets - no physics, stay in place as decoration")
    print("- GREEN SPHERES (right): RigidObjects with physics and velocity control - they move around")
    print()
    print("Notice how static assets don't move or fall, while rigid objects have physics!")
    print("="*80)

    # Simulation loop
    step_count = 0
    last_info_time = 0.0
    
    while sim_context.is_playing():
        # Step the simulation
        sim_context.step()
        scene.update(sim_context.get_physics_dt())
        
        step_count += 1
        current_time = step_count * sim_context.get_physics_dt()
        
        # Print info every 10 seconds
        if current_time - last_info_time >= 10.0:
            print(f"\nTime: {current_time:.1f}s")
            print("- Rigid objects continue moving with physics simulation")
            print("- Static assets remain stationary (no physics simulation)")
            
            # Example: Get current velocities for moving cubes
            velocity_info = random_assets.get_velocity_control_info()
            moving_cube_velocities = velocity_info.get("moving_cubes", {}).get("current_velocities")
            if moving_cube_velocities is not None and len(moving_cube_velocities) > 0:
                first_cube_vel = moving_cube_velocities[0]
                print(f"Sample moving cube velocity: linear=({first_cube_vel[0]:.2f}, {first_cube_vel[1]:.2f}, {first_cube_vel[2]:.2f})")
            
            last_info_time = current_time
        
        # Example: Demonstrate manual velocity control for rigid objects only
        if step_count == 600:  # After 10 seconds
            print("\n>>> MANUAL VELOCITY CONTROL DEMO (RIGID OBJECTS ONLY) <<<")
            print("Setting all moving cubes to coordinated motion...")
            
            moving_cubes = random_assets.get_rigid_object("moving_cubes")
            if moving_cubes:
                import math
                
                num_cubes = len(moving_cubes)
                # Create coordinated velocities
                coordinated_velocities = []
                for i in range(num_cubes):
                    angle = (i / num_cubes) * 2 * math.pi
                    lin_x = 0.8 * math.cos(angle)
                    lin_y = 0.8 * math.sin(angle)
                    lin_z = 0.0
                    ang_x, ang_y, ang_z = 0.0, 0.0, 0.5
                    coordinated_velocities.append([lin_x, lin_y, lin_z, ang_x, ang_y, ang_z])
                
                vel_tensor = torch.tensor(coordinated_velocities, dtype=torch.float32)
                random_assets.set_asset_velocities("moving_cubes", vel_tensor)
                print("Moving cubes should now move in coordinated circular patterns!")
                print("(Static terrain rocks remain unaffected)")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

#!/usr/bin/env python3
"""
Extract detailed geometry from Flamingo USD file.
This will help us understand the correct dimensions for the MuJoCo XML.
"""

from pxr import Usd, UsdGeom, UsdPhysics, Gf
from pathlib import Path
import numpy as np


def extract_geometry():
    """Extract geometry information from USD file."""
    
    # Get paths
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    usd_path = str(project_root / "lab/flamingo/assets/data/Robots/Flamingo/flamingo_rev03_1_1/flamingo_rev03_1_1_merge_joints.usd")
    
    print("="*80)
    print("FLAMINGO USD GEOMETRY EXTRACTION")
    print("="*80)
    print(f"\nUSD file: {usd_path}\n")
    
    # Open the USD stage
    stage = Usd.Stage.Open(usd_path)
    
    if not stage:
        print("❌ Failed to open USD stage")
        return
    
    print("✅ USD stage loaded successfully\n")
    
    # Dictionary to store link information
    links = {}
    joints_info = {}
    
    # Traverse all prims
    for prim in stage.Traverse():
        name = prim.GetName()
        path = str(prim.GetPath())
        
        # Check if it's a link (has RigidBody)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            xformable = UsdGeom.Xformable(prim)
            
            # Get mass and inertia
            mass = 0.0
            inertia = (0.0, 0.0, 0.0)
            if prim.HasAPI(UsdPhysics.MassAPI):
                mass_api = UsdPhysics.MassAPI(prim)
                mass = mass_api.GetMassAttr().Get()
                inertia_attr = mass_api.GetDiagonalInertiaAttr().Get()
                if inertia_attr:
                    inertia = (inertia_attr[0], inertia_attr[1], inertia_attr[2])
            
            # Get transform
            local_xform = xformable.GetLocalTransformation()
            world_xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            local_trans = local_xform.ExtractTranslation()
            world_trans = world_xform.ExtractTranslation()
            
            links[name] = {
                'path': path,
                'local_pos': (local_trans[0], local_trans[1], local_trans[2]),
                'world_pos': (world_trans[0], world_trans[1], world_trans[2]),
                'mass': mass,
                'inertia': inertia,
            }
            
            # Try to find collision geometry
            collisions = []
            for child in prim.GetAllChildren():
                if 'collision' in child.GetName().lower():
                    for geom_child in child.GetAllChildren():
                        if geom_child.IsA(UsdGeom.Gprim):
                            geom_type = geom_child.GetTypeName()
                            collisions.append({
                                'type': geom_type,
                                'name': geom_child.GetName()
                            })
                            
                            # Extract specific geometry parameters
                            if geom_child.IsA(UsdGeom.Cylinder):
                                cylinder = UsdGeom.Cylinder(geom_child)
                                radius = cylinder.GetRadiusAttr().Get()
                                height = cylinder.GetHeightAttr().Get()
                                collisions[-1]['radius'] = radius
                                collisions[-1]['height'] = height
                            elif geom_child.IsA(UsdGeom.Sphere):
                                sphere = UsdGeom.Sphere(geom_child)
                                radius = sphere.GetRadiusAttr().Get()
                                collisions[-1]['radius'] = radius
                            elif geom_child.IsA(UsdGeom.Cube):
                                cube = UsdGeom.Cube(geom_child)
                                size = cube.GetSizeAttr().Get()
                                collisions[-1]['size'] = size
            
            links[name]['collisions'] = collisions
        
        # Check if it's a joint
        if prim.IsA(UsdPhysics.Joint):
            joint_name = name
            
            # Get joint properties
            joint = UsdPhysics.Joint(prim)
            body0_rel = joint.GetBody0Rel()
            body1_rel = joint.GetBody1Rel()
            
            body0 = body0_rel.GetTargets()[0] if body0_rel.GetTargets() else None
            body1 = body1_rel.GetTargets()[0] if body1_rel.GetTargets() else None
            
            joints_info[joint_name] = {
                'path': path,
                'body0': str(body0) if body0 else None,
                'body1': str(body1) if body1 else None,
            }
            
            # Check if it's a revolute joint
            if prim.IsA(UsdPhysics.RevoluteJoint):
                rev_joint = UsdPhysics.RevoluteJoint(prim)
                axis_attr = rev_joint.GetAxisAttr()
                if axis_attr:
                    axis = axis_attr.Get()
                    joints_info[joint_name]['axis'] = axis
    
    # Print results
    print("="*80)
    print("LINKS INFORMATION")
    print("="*80)
    
    for link_name in sorted(links.keys()):
        link = links[link_name]
        print(f"\n{link_name}:")
        print(f"  Local position:  ({link['local_pos'][0]:8.4f}, {link['local_pos'][1]:8.4f}, {link['local_pos'][2]:8.4f})")
        print(f"  World position:  ({link['world_pos'][0]:8.4f}, {link['world_pos'][1]:8.4f}, {link['world_pos'][2]:8.4f})")
        print(f"  Mass:            {link['mass']:.4f} kg")
        print(f"  Inertia:         ({link['inertia'][0]:.8f}, {link['inertia'][1]:.8f}, {link['inertia'][2]:.8f})")
        
        if link['collisions']:
            print(f"  Collisions:")
            for col in link['collisions']:
                print(f"    - {col['type']}: {col['name']}")
                if 'radius' in col and 'height' in col:
                    print(f"      Radius: {col['radius']:.4f}m, Height: {col['height']:.4f}m")
                elif 'radius' in col:
                    print(f"      Radius: {col['radius']:.4f}m")
                elif 'size' in col:
                    print(f"      Size: {col['size']:.4f}m")
    
    print("\n" + "="*80)
    print("JOINTS INFORMATION")
    print("="*80)
    
    for joint_name in sorted(joints_info.keys()):
        joint = joints_info[joint_name]
        print(f"\n{joint_name}:")
        print(f"  Body 0: {joint['body0']}")
        print(f"  Body 1: {joint['body1']}")
        if 'axis' in joint:
            print(f"  Axis: {joint['axis']}")
    
    # Calculate standing height
    print("\n" + "="*80)
    print("STANDING HEIGHT CALCULATION")
    print("="*80)
    
    if 'base_link' in links:
        base_z = links['base_link']['world_pos'][2]
        print(f"\nBase link z-position: {base_z:.4f}m")
    else:
        print("\n⚠️  base_link not found")
        base_z = 0.0
    
    # Find wheel radius
    wheel_radius = 0.0
    for link_name, link in links.items():
        if 'wheel' in link_name.lower():
            for col in link['collisions']:
                if col['type'] == 'Cylinder' and 'radius' in col:
                    wheel_radius = col['radius']
                    print(f"Wheel radius (from {link_name}): {wheel_radius:.4f}m")
                    break
            if wheel_radius > 0:
                break
    
    if wheel_radius > 0:
        standing_height = base_z + wheel_radius
        print(f"\n{'='*80}")
        print(f"STANDING HEIGHT: {standing_height:.4f}m")
        print(f"{'='*80}")
        
        expected_height = 0.5562
        error = abs(standing_height - expected_height)
        error_pct = (error / expected_height) * 100
        
        print(f"\nExpected (from training): {expected_height:.4f}m")
        print(f"Difference: {error:.4f}m ({error_pct:.1f}%)")
        
        if error_pct < 5.0:
            print("✅ Geometry matches training environment!")
        else:
            print("⚠️  Geometry differs from training environment")
    else:
        print("\n⚠️  Could not determine wheel radius")


if __name__ == "__main__":
    extract_geometry()

#!/usr/bin/env python3
"""
Deep dive into USD geometry - find all collision shapes.
"""

from pxr import Usd, UsdGeom, UsdPhysics, Gf
from pathlib import Path


def deep_geometry_extraction():
    """Extract all geometry details from USD file."""
    
    # Get paths
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    usd_path = str(project_root / "lab/flamingo/assets/data/Robots/Flamingo/flamingo_rev03_1_1/flamingo_rev03_1_1_merge_joints.usd")
    
    print("="*80)
    print("DEEP USD GEOMETRY ANALYSIS")
    print("="*80)
    
    # Open the USD stage
    stage = Usd.Stage.Open(usd_path)
    
    if not stage:
        print("❌ Failed to open USD stage")
        return
    
    print(f"\nAnalyzing: {usd_path}\n")
    
    # Find all geometry prims
    print("="*80)
    print("ALL GEOMETRY PRIMITIVES")
    print("="*80)
    
    all_geoms = []
    
    for prim in stage.Traverse():
        # Check for all geometry types
        if prim.IsA(UsdGeom.Gprim):
            geom_type = prim.GetTypeName()
            path = str(prim.GetPath())
            name = prim.GetName()
            
            geom_info = {
                'name': name,
                'path': path,
                'type': geom_type,
            }
            
            # Get transform
            if UsdGeom.Xformable.Get(stage, prim.GetPath()):
                xformable = UsdGeom.Xformable(prim)
                world_xform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                trans = world_xform.ExtractTranslation()
                geom_info['position'] = (trans[0], trans[1], trans[2])
            
            # Extract type-specific parameters
            if prim.IsA(UsdGeom.Cylinder):
                cylinder = UsdGeom.Cylinder(prim)
                radius_attr = cylinder.GetRadiusAttr()
                height_attr = cylinder.GetHeightAttr()
                axis_attr = cylinder.GetAxisAttr()
                
                if radius_attr:
                    geom_info['radius'] = radius_attr.Get()
                if height_attr:
                    geom_info['height'] = height_attr.Get()
                if axis_attr:
                    geom_info['axis'] = axis_attr.Get()
                    
            elif prim.IsA(UsdGeom.Sphere):
                sphere = UsdGeom.Sphere(prim)
                radius_attr = sphere.GetRadiusAttr()
                if radius_attr:
                    geom_info['radius'] = radius_attr.Get()
                    
            elif prim.IsA(UsdGeom.Cube):
                cube = UsdGeom.Cube(prim)
                size_attr = cube.GetSizeAttr()
                if size_attr:
                    geom_info['size'] = size_attr.Get()
                    
            elif prim.IsA(UsdGeom.Capsule):
                capsule = UsdGeom.Capsule(prim)
                radius_attr = capsule.GetRadiusAttr()
                height_attr = capsule.GetHeightAttr()
                axis_attr = capsule.GetAxisAttr()
                
                if radius_attr:
                    geom_info['radius'] = radius_attr.Get()
                if height_attr:
                    geom_info['height'] = height_attr.Get()
                if axis_attr:
                    geom_info['axis'] = axis_attr.Get()
            
            all_geoms.append(geom_info)
    
    # Print all geometries
    for geom in all_geoms:
        print(f"\n{geom['type']}: {geom['name']}")
        print(f"  Path: {geom['path']}")
        if 'position' in geom:
            print(f"  Position: ({geom['position'][0]:8.4f}, {geom['position'][1]:8.4f}, {geom['position'][2]:8.4f})")
        if 'radius' in geom:
            print(f"  Radius: {geom['radius']:.4f}m")
        if 'height' in geom:
            print(f"  Height: {geom['height']:.4f}m")
        if 'size' in geom:
            print(f"  Size: {geom['size']:.4f}m")
        if 'axis' in geom:
            print(f"  Axis: {geom['axis']}")
    
    # Calculate standing height
    print("\n" + "="*80)
    print("STANDING HEIGHT CALCULATION")
    print("="*80)
    
    # Find wheel geometries
    wheel_geoms = [g for g in all_geoms if 'wheel' in g['path'].lower() and 'collision' in g['path'].lower()]
    
    if wheel_geoms:
        print(f"\nFound {len(wheel_geoms)} wheel collision geometries:")
        for wg in wheel_geoms:
            print(f"\n  {wg['name']} ({wg['type']})")
            if 'position' in wg:
                print(f"    Position: ({wg['position'][0]:8.4f}, {wg['position'][1]:8.4f}, {wg['position'][2]:8.4f})")
            if 'radius' in wg:
                print(f"    Radius: {wg['radius']:.4f}m")
                
                # Calculate standing height
                # The wheel center is at wg['position'][2]
                # Standing height = wheel_center_z + wheel_radius
                if 'position' in wg:
                    wheel_center_z = wg['position'][2]
                    wheel_radius = wg['radius']
                    standing_height = wheel_center_z + wheel_radius
                    
                    print(f"\n    Wheel center Z: {wheel_center_z:.4f}m")
                    print(f"    Wheel radius:   {wheel_radius:.4f}m")
                    print(f"    Ground level:   {wheel_center_z - wheel_radius:.4f}m")
                    print(f"    Standing height (base at Z=0): {abs(wheel_center_z - wheel_radius):.4f}m")
                    
                    # The actual standing height is the distance from ground to base_link
                    # If wheels are at -0.4537 and radius is R, ground is at -0.4537-R
                    # Standing height = 0 - (-0.4537-R) = 0.4537+R
                    actual_standing_height = abs(wheel_center_z - wheel_radius)
                    
                    print(f"\n    {'='*60}")
                    print(f"    CALCULATED STANDING HEIGHT: {actual_standing_height:.4f}m")
                    print(f"    {'='*60}")
                    
                    expected = 0.5562
                    error = abs(actual_standing_height - expected)
                    error_pct = (error / expected) * 100
                    
                    print(f"\n    Expected (training): {expected:.4f}m")
                    print(f"    Difference: {error:.4f}m ({error_pct:.1f}%)")
                    
                    if error_pct < 5.0:
                        print(f"    ✅ GEOMETRY MATCHES!")
                    else:
                        print(f"    ⚠️  Geometry mismatch")
                    
                    break  # Only need one wheel
    else:
        print("\n⚠️  No wheel collision geometries found")
        
        # Try to infer from wheel link positions
        print("\nTrying to infer from wheel link positions...")
        for geom in all_geoms:
            if 'wheel' in geom['path'].lower():
                print(f"\n  {geom['name']}: {geom['type']}")
                if 'position' in geom:
                    print(f"    Position: {geom['position']}")
                if 'radius' in geom:
                    print(f"    Radius: {geom['radius']}")


if __name__ == "__main__":
    deep_geometry_extraction()

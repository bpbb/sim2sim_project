#!/usr/bin/env python3
"""
Alternative USD to MJCF converter using pxr (USD Python bindings).

This script reads the USD file directly and extracts geometry information
to create a corrected MuJoCo XML file.
"""

import argparse
from pathlib import Path


def try_pxr_conversion():
    """Try to use USD Python bindings (pxr) for conversion."""
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
        
        print("✅ USD Python bindings (pxr) available!")
        
        # Get paths
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent
        usd_path = str(project_root / "lab/flamingo/assets/data/Robots/Flamingo/flamingo_rev03_1_1/flamingo_rev03_1_1_merge_joints.usd")
        
        print(f"\nLoading USD file: {usd_path}")
        
        # Open the USD stage
        stage = Usd.Stage.Open(usd_path)
        
        if not stage:
            print("❌ Failed to open USD stage")
            return False
        
        print("✅ USD stage loaded successfully")
        
        # Traverse the stage to find geometry
        print("\n" + "="*80)
        print("USD STAGE CONTENTS")
        print("="*80)
        
        def print_prim_info(prim, indent=0):
            """Recursively print prim information."""
            prefix = "  " * indent
            print(f"{prefix}{prim.GetName()} ({prim.GetTypeName()})")
            
            # Check for transform
            if UsdGeom.Xformable(prim):
                xform = UsdGeom.Xformable(prim)
                # Get local transform
                ops = xform.GetOrderedXformOps()
                if ops:
                    print(f"{prefix}  Transform ops: {len(ops)}")
            
            # Check for mesh geometry
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                print(f"{prefix}  Mesh with vertices")
            
            # Check for physics
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                print(f"{prefix}  Has RigidBody physics")
            
            # Recurse to children
            for child in prim.GetChildren():
                print_prim_info(child, indent + 1)
        
        # Start from root
        root = stage.GetPseudoRoot()
        for prim in root.GetChildren():
            print_prim_info(prim)
        
        print("\n" + "="*80)
        print("GEOMETRY EXTRACTION")
        print("="*80)
        
        # Try to find base_link and wheels
        base_link = None
        wheels = []
        
        for prim in stage.Traverse():
            name = prim.GetName().lower()
            
            if 'base' in name and 'link' in name:
                base_link = prim
                print(f"\n✅ Found base_link: {prim.GetPath()}")
                
                # Try to get position
                if UsdGeom.Xformable(prim):
                    xform = UsdGeom.Xformable(prim)
                    # Get world transform at default time
                    world_xform = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    translation = world_xform.ExtractTranslation()
                    print(f"   Position: ({translation[0]:.4f}, {translation[1]:.4f}, {translation[2]:.4f})")
            
            if 'wheel' in name:
                wheels.append(prim)
                print(f"\n✅ Found wheel: {prim.GetPath()}")
                
                # Try to get cylinder geometry
                if prim.IsA(UsdGeom.Cylinder):
                    cylinder = UsdGeom.Cylinder(prim)
                    radius = cylinder.GetRadiusAttr().Get()
                    height = cylinder.GetHeightAttr().Get()
                    print(f"   Radius: {radius:.4f}m, Height: {height:.4f}m")
        
        if base_link and wheels:
            print("\n✅ Successfully extracted geometry information!")
            print("\nYou can now manually create the MuJoCo XML with correct dimensions.")
            return True
        else:
            print("\n⚠️  Could not find all required components")
            return False
        
    except ImportError as e:
        print(f"❌ USD Python bindings not available: {e}")
        print("\nTo install USD Python bindings:")
        print("  pip install usd-core")
        return False
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_installation_guide():
    """Print guide for installing necessary tools."""
    print("\n" + "="*80)
    print("INSTALLATION OPTIONS")
    print("="*80)
    print("\nOption A: Install USD Python bindings (Recommended)")
    print("-" * 80)
    print("pip install usd-core")
    print("\nThis allows reading USD files directly in Python.")
    
    print("\n\nOption B: Use Isaac Sim (Most Accurate)")
    print("-" * 80)
    print("1. Install Isaac Sim from: https://developer.nvidia.com/isaac-sim")
    print("2. Run the export script inside Isaac Sim's Python environment")
    print("3. Or use the GUI method (see manual instructions)")
    
    print("\n\nOption C: Manual XML Creation")
    print("-" * 80)
    print("If you have the geometry specifications, you can manually edit")
    print("the existing XML file to match the correct dimensions.")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Convert Flamingo USD to MuJoCo MJCF")
    args = parser.parse_args()
    
    print("="*80)
    print("USD to MJCF Converter (Alternative Method)")
    print("="*80)
    
    success = try_pxr_conversion()
    
    if not success:
        print_installation_guide()
        
        print("\n" + "="*80)
        print("RECOMMENDED NEXT STEPS")
        print("="*80)
        print("\n1. Install USD Python bindings:")
        print("   pip install usd-core")
        print("\n2. Re-run this script to extract geometry")
        print("\n3. Or follow the manual Isaac Sim instructions from:")
        print("   python scripts/export_flamingo_mjcf.py")
        print("="*80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Export Flamingo USD model to MuJoCo MJCF format using Isaac Sim.

This script uses Isaac Sim's MJCF exporter to convert the USD model
to a MuJoCo XML file with correct geometry matching the training environment.

Expected standing height: 0.5562m (base_z + wheel_radius)
"""

import os
import argparse
from pathlib import Path


def export_to_mjcf_isaaclab():
    """
    Export using Isaac Lab's built-in conversion utilities.
    This is the recommended approach if Isaac Lab is available.
    """
    try:
        import omni.isaac.lab.sim as sim_utils
        from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
        
        print("Using Isaac Lab conversion utilities...")
        
        # Get paths
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent
        usd_path = project_root / "lab/flamingo/assets/data/Robots/Flamingo/flamingo_rev03_1_1/flamingo_rev03_1_1_merge_joints.usd"
        output_path = project_root / "assets/flamingo_correct.xml"
        
        print(f"USD file: {usd_path}")
        print(f"Output file: {output_path}")
        
        if not usd_path.exists():
            print(f"ERROR: USD file not found at {usd_path}")
            return False
            
        # Note: Isaac Lab doesn't have direct MJCF export
        # We need to use Isaac Sim's omni.kit.commands
        print("\nIsaac Lab detected, but MJCF export requires Isaac Sim.")
        print("Please use the Isaac Sim script below.")
        return False
        
    except ImportError:
        print("Isaac Lab not available, trying Isaac Sim...")
        return False


def export_to_mjcf_isaacsim():
    """
    Export using Isaac Sim's MJCF exporter.
    This requires running inside Isaac Sim's Python environment.
    """
    try:
        import omni.kit.commands
        from pxr import Usd, UsdGeom
        import omni.usd
        
        print("Using Isaac Sim MJCF exporter...")
        
        # Get paths
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent
        usd_path = str(project_root / "lab/flamingo/assets/data/Robots/Flamingo/flamingo_rev03_1_1/flamingo_rev03_1_1_merge_joints.usd")
        output_path = str(project_root / "assets/flamingo_correct.xml")
        
        print(f"USD file: {usd_path}")
        print(f"Output file: {output_path}")
        
        if not os.path.exists(usd_path):
            print(f"ERROR: USD file not found at {usd_path}")
            return False
        
        # Load the USD stage
        print("\nLoading USD stage...")
        stage = Usd.Stage.Open(usd_path)
        
        if not stage:
            print("ERROR: Failed to load USD stage")
            return False
            
        print(f"Stage loaded successfully")
        
        # Find the robot prim (usually at root or /World/Robot)
        root_prim = stage.GetDefaultPrim()
        if not root_prim:
            root_prim = stage.GetPrimAtPath("/")
        
        prim_path = str(root_prim.GetPath())
        print(f"Root prim path: {prim_path}")
        
        # Export to MJCF
        print("\nExporting to MJCF...")
        try:
            omni.kit.commands.execute(
                "ExportMJCF",
                file_path=output_path,
                prim_path=prim_path,
                export_physics=True,
                export_visuals=True
            )
            print(f"✅ Export successful: {output_path}")
            return True
            
        except Exception as e:
            print(f"ERROR during export: {e}")
            print("\nTrying alternative export method...")
            
            # Alternative: Use omni.isaac.mjcf if available
            try:
                import omni.isaac.mjcf as mjcf
                mjcf.export_mjcf(
                    output_path=output_path,
                    prim_path=prim_path,
                    export_physics=True
                )
                print(f"✅ Export successful (alternative method): {output_path}")
                return True
            except Exception as e2:
                print(f"ERROR with alternative method: {e2}")
                return False
        
    except ImportError as e:
        print(f"Isaac Sim not available: {e}")
        print("\nThis script must be run inside Isaac Sim's Python environment.")
        return False


def print_manual_instructions():
    """Print manual instructions for exporting in Isaac Sim GUI."""
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    usd_path = project_root / "lab/flamingo/assets/data/Robots/Flamingo/flamingo_rev03_1_1/flamingo_rev03_1_1_merge_joints.usd"
    output_path = project_root / "assets/flamingo_correct.xml"
    
    print("\n" + "="*80)
    print("MANUAL EXPORT INSTRUCTIONS")
    print("="*80)
    print("\nIf automatic export fails, follow these steps in Isaac Sim:\n")
    print("1. Launch Isaac Sim")
    print(f"\n2. Open the USD file:")
    print(f"   File → Open → {usd_path}")
    print("\n3. Open the Python console (Window → Script Editor)")
    print("\n4. Run this code in the console:")
    print("-" * 80)
    print(f"""
import omni.kit.commands
from pxr import Usd

# Get the stage
stage = omni.usd.get_context().get_stage()
root_prim = stage.GetDefaultPrim()

# Export to MJCF
omni.kit.commands.execute(
    "ExportMJCF",
    file_path="{output_path}",
    prim_path=str(root_prim.GetPath()),
    export_physics=True,
    export_visuals=True
)

print("Export complete!")
""")
    print("-" * 80)
    print(f"\n5. The exported file will be at: {output_path}")
    print("\n6. Verify the geometry:")
    print("   - Check base_link z-position")
    print("   - Check wheel radius")
    print("   - Standing height should be ≈ 0.5562m")
    print("\n" + "="*80)


def verify_exported_xml(xml_path):
    """Verify the exported XML has correct geometry."""
    import xml.etree.ElementTree as ET
    
    if not os.path.exists(xml_path):
        print(f"XML file not found: {xml_path}")
        return False
    
    print(f"\nVerifying exported XML: {xml_path}")
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Find base_link position
        base_link = root.find(".//body[@name='base_link']")
        if base_link is not None:
            pos = base_link.get('pos', '0 0 0')
            base_z = float(pos.split()[2])
            print(f"  Base link z-position: {base_z:.4f}m")
        else:
            print("  WARNING: base_link not found")
            base_z = 0.0
        
        # Find wheel radius
        wheel_geoms = root.findall(".//geom[@type='cylinder']")
        wheel_radius = 0.0
        for geom in wheel_geoms:
            if 'wheel' in geom.get('name', '').lower():
                size = geom.get('size', '0 0')
                wheel_radius = float(size.split()[0])
                print(f"  Wheel radius: {wheel_radius:.4f}m")
                break
        
        # Calculate standing height
        standing_height = base_z + wheel_radius
        print(f"  Standing height: {standing_height:.4f}m")
        
        # Compare with expected
        expected_height = 0.5562
        error = abs(standing_height - expected_height)
        error_pct = (error / expected_height) * 100
        
        print(f"\n  Expected height: {expected_height:.4f}m")
        print(f"  Error: {error:.4f}m ({error_pct:.1f}%)")
        
        if error_pct < 5.0:
            print("  ✅ Geometry looks correct!")
            return True
        else:
            print("  ⚠️  Geometry may be incorrect")
            return False
            
    except Exception as e:
        print(f"  ERROR parsing XML: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export Flamingo USD to MuJoCo MJCF")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify an existing XML file")
    parser.add_argument("--xml", type=str,
                       help="Path to XML file to verify")
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent
    default_output = project_root / "assets/flamingo_correct.xml"
    
    if args.verify_only:
        xml_path = args.xml if args.xml else default_output
        verify_exported_xml(xml_path)
        return
    
    print("="*80)
    print("Flamingo USD to MuJoCo MJCF Exporter")
    print("="*80)
    
    # Try automatic export
    success = export_to_mjcf_isaaclab()
    
    if not success:
        success = export_to_mjcf_isaacsim()
    
    if not success:
        print("\n⚠️  Automatic export not available.")
        print_manual_instructions()
    else:
        # Verify the exported file
        verify_exported_xml(str(default_output))
        
        print("\n" + "="*80)
        print("NEXT STEPS")
        print("="*80)
        print("\n1. Verify the geometry is correct (see above)")
        print("\n2. Test with the existing policy:")
        print(f"""
cd {project_root}
python scripts/transfer_flamingo_sim2sim.py \\
    --policy logs/co_rl/Flamingo_Flat_Stand_Drive/ppo/2026-02-07_18-03-28/model_4999.pt \\
    --xml assets/flamingo_correct.xml \\
    --init_height 0.5562 \\
    --pd_scale 1.0 \\
    --cmd_vx 0.0 \\
    --duration 20.0
""")
        print("="*80)


if __name__ == "__main__":
    main()

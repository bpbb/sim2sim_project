#!/usr/bin/env python3
"""
Create a corrected MuJoCo XML by scaling the existing one to match training height.

Training height: 0.5562m
Current XML height: 0.2991m (base_z=0.2461 + wheel_radius=0.053)
Scale factor: 0.5562 / 0.2991 ≈ 1.8598
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np


def scale_position_string(pos_str, scale):
    """Scale a position string like '0.1 0.2 0.3'"""
    values = [float(x) * scale for x in pos_str.split()]
    return ' '.join(f'{v:.6f}' for v in values)


def scale_size_string(size_str, scale):
    """Scale a size string"""
    values = [float(x) * scale for x in size_str.split()]
    return ' '.join(f'{v:.6f}' for v in values)


def scale_inertia(inertia_str, scale):
    """Scale inertia (scales with mass * length^2, so scale^3 for uniform density)"""
    # For uniform scaling, inertia scales as scale^5 (mass * length^2)
    # But we're keeping mass the same and only scaling geometry, so scale^2
    inertia_scale = scale ** 2
    values = [float(x) * inertia_scale for x in inertia_str.split()]
    return ' '.join(f'{v:.8f}' for v in values)


def create_corrected_xml():
    """Create corrected XML file with proper scaling."""
    
    # Paths
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent  # Go up two levels from scripts/ to project root
    input_xml = project_root / "sim2sim_onnx/assets/flamingo_torque.xml"
    output_xml = project_root / "main/assets/flamingo_correct.xml"
    
    # Make sure output directory exists
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate scale factor
    target_height = 0.5562  # Training height
    current_base_z = 0.2461
    current_wheel_r = 0.053
    current_height = current_base_z + current_wheel_r  # 0.2991
    
    scale = target_height / current_height
    
    print("="*80)
    print("CREATING CORRECTED MUJOCO XML")
    print("="*80)
    print(f"\nInput XML:  {input_xml}")
    print(f"Output XML: {output_xml}")
    print(f"\nCurrent standing height: {current_height:.4f}m")
    print(f"Target standing height:  {target_height:.4f}m")
    print(f"Scale factor: {scale:.6f}")
    
    # Parse XML
    tree = ET.parse(input_xml)
    root = tree.getroot()
    
    # Track what we're scaling
    scaled_elements = {
        'body_positions': 0,
        'geom_sizes': 0,
        'geom_positions': 0,
        'inertial_positions': 0,
        'inertial_inertias': 0,
        'joint_positions': 0,
    }
    
    # Scale all body positions
    for body in root.iter('body'):
        if 'pos' in body.attrib:
            old_pos = body.attrib['pos']
            body.attrib['pos'] = scale_position_string(old_pos, scale)
            scaled_elements['body_positions'] += 1
    
    # Scale all geometry
    for geom in root.iter('geom'):
        # Scale position
        if 'pos' in geom.attrib:
            old_pos = geom.attrib['pos']
            geom.attrib['pos'] = scale_position_string(old_pos, scale)
            scaled_elements['geom_positions'] += 1
        
        # Scale size
        if 'size' in geom.attrib:
            old_size = geom.attrib['size']
            geom.attrib['size'] = scale_size_string(old_size, scale)
            scaled_elements['geom_sizes'] += 1
    
    # Scale all inertial properties
    for inertial in root.iter('inertial'):
        # Scale position
        if 'pos' in inertial.attrib:
            old_pos = inertial.attrib['pos']
            inertial.attrib['pos'] = scale_position_string(old_pos, scale)
            scaled_elements['inertial_positions'] += 1
        
        # Scale inertia tensor
        if 'diaginertia' in inertial.attrib:
            old_inertia = inertial.attrib['diaginertia']
            inertial.attrib['diaginertia'] = scale_inertia(old_inertia, scale)
            scaled_elements['inertial_inertias'] += 1
    
    # Scale joint positions (if any)
    for joint in root.iter('joint'):
        if 'pos' in joint.attrib:
            old_pos = joint.attrib['pos']
            joint.attrib['pos'] = scale_position_string(old_pos, scale)
            scaled_elements['joint_positions'] += 1
    
    # Scale site positions
    for site in root.iter('site'):
        if 'pos' in site.attrib:
            old_pos = site.attrib['pos']
            site.attrib['pos'] = scale_position_string(old_pos, scale)
        if 'size' in site.attrib:
            old_size = site.attrib['size']
            site.attrib['size'] = scale_size_string(old_size, scale)
    
    # Write output
    tree.write(str(output_xml), encoding='utf-8', xml_declaration=True)
    
    print(f"\n✅ Corrected XML created successfully!")
    print(f"\nScaled elements:")
    for key, count in scaled_elements.items():
        print(f"  {key}: {count}")
    
    # Verify the output
    print(f"\n{'='*80}")
    print("VERIFICATION")
    print("="*80)
    
    verify_tree = ET.parse(output_xml)
    verify_root = verify_tree.getroot()
    
    # Find base_link
    base_link = verify_root.find(".//body[@name='base_link']")
    if base_link is not None:
        base_pos = base_link.get('pos', '0 0 0')
        base_z = float(base_pos.split()[2])
        print(f"\nBase link z-position: {base_z:.4f}m")
    else:
        print("\n⚠️  base_link not found")
        base_z = 0.0
    
    # Find wheel - look for cylinder geom in wheel_link body
    wheel_radius = 0.0
    for body in verify_root.iter('body'):
        if 'wheel' in body.get('name', '').lower():
            for geom in body.iter('geom'):
                if geom.get('type') == 'cylinder':
                    size = geom.get('size', '0 0')
                    wheel_radius = float(size.split()[0])
                    print(f"Wheel radius: {wheel_radius:.4f}m")
                    break
            if wheel_radius > 0:
                break
    
    standing_height = base_z + wheel_radius
    print(f"\n{'='*80}")
    print(f"CALCULATED STANDING HEIGHT: {standing_height:.4f}m")
    print(f"{'='*80}")
    
    error = abs(standing_height - target_height)
    error_pct = (error / target_height) * 100
    
    print(f"\nTarget height: {target_height:.4f}m")
    print(f"Achieved height: {standing_height:.4f}m")
    print(f"Error: {error:.4f}m ({error_pct:.2f}%)")
    
    if error_pct < 1.0:
        print("\n✅ GEOMETRY CORRECT! Ready to use.")
    elif error_pct < 5.0:
        print("\n✅ Geometry close enough (< 5% error)")
    else:
        print("\n⚠️  Geometry may still have issues")
    
    return output_xml


if __name__ == "__main__":
    output_path = create_corrected_xml()
    
    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print("="*80)
    print(f"\n1. The corrected XML is ready at:")
    print(f"   {output_path}")
    print(f"\n2. Test with the existing policy:")
    print(f"""
cd /home/drl-68/sim2sim_project/Two-wheel-Legged-Bot/main

python scripts/transfer_flamingo_sim2sim.py \\
    --policy logs/co_rl/Flamingo_Flat_Stand_Drive/ppo/2026-02-07_18-03-28/model_4999.pt \\
    --xml ../assets/flamingo_correct.xml \\
    --init_height 0.5562 \\
    --pd_scale 1.0 \\
    --cmd_vx 0.0 \\
    --duration 20.0
""")
    print("="*80)

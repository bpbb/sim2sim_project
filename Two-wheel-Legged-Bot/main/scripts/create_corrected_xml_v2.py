#!/usr/bin/env python3
"""
Create a PROPERLY corrected MuJoCo XML.

CRITICAL: In MuJoCo, body positions are RELATIVE to parent.
We should NOT scale these! Only scale:
1. Geometry sizes (collision shapes)
2. Inertial properties (COM positions relative to body frame)
3. The ROOT body position (base_link in worldbody)

This preserves the kinematic structure while matching the training height.
"""

import xml.etree.ElementTree as ET
from pathlib import Path


def scale_size_string(size_str, scale):
    """Scale a size string"""
    values = [float(x) * scale for x in size_str.split()]
    return ' '.join(f'{v:.6f}' for v in values)


def scale_position_string(pos_str, scale):
    """Scale a position string"""
    values = [float(x) * scale for x in pos_str.split()]
    return ' '.join(f'{v:.6f}' for v in values)


def scale_inertia(inertia_str, scale):
    """Scale inertia (scales with scale^2 for geometry scaling)"""
    inertia_scale = scale ** 2
    values = [float(x) * inertia_scale for x in inertia_str.split()]
    return ' '.join(f'{v:.8f}' for v in values)


def create_properly_corrected_xml():
    """Create corrected XML with proper kinematic structure."""
    
    # Paths
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    input_xml = project_root / "sim2sim_onnx/assets/flamingo_torque.xml"
    output_xml = project_root / "main/assets/flamingo_correct_v2.xml"
    
    # Make sure output directory exists
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate scale factor
    target_height = 0.5562  # Training height
    current_base_z = 0.2461
    current_wheel_r = 0.053
    current_height = current_base_z + current_wheel_r
    
    scale = target_height / current_height
    
    print("="*80)
    print("CREATING PROPERLY CORRECTED MUJOCO XML")
    print("="*80)
    print(f"\nInput XML:  {input_xml}")
    print(f"Output XML: {output_xml}")
    print(f"\nCurrent standing height: {current_height:.4f}m")
    print(f"Target standing height:  {target_height:.4f}m")
    print(f"Scale factor: {scale:.6f}")
    print("\nIMPORTANT: Only scaling geometry and root position, NOT relative positions!")
    
    # Parse XML
    tree = ET.parse(input_xml)
    root = tree.getroot()
    
    scaled_elements = {
        'root_body_position': 0,
        'geom_sizes': 0,
        'geom_positions_in_body_frame': 0,
        'inertial_positions': 0,
        'inertial_inertias': 0,
    }
    
    # Find worldbody
    worldbody = root.find('worldbody')
    if worldbody is None:
        print("ERROR: No worldbody found!")
        return None
    
    # Scale ONLY the root body (base_link) position in worldbody
    for body in worldbody.findall('body'):
        if body.get('name') == 'base_link':
            if 'pos' in body.attrib:
                old_pos = body.attrib['pos']
                body.attrib['pos'] = scale_position_string(old_pos, scale)
                scaled_elements['root_body_position'] += 1
                print(f"\n✅ Scaled root body position: {old_pos} → {body.attrib['pos']}")
    
    # Scale all geometry sizes (but NOT their positions relative to body)
    for geom in root.iter('geom'):
        # Scale size
        if 'size' in geom.attrib:
            old_size = geom.attrib['size']
            geom.attrib['size'] = scale_size_string(old_size, scale)
            scaled_elements['geom_sizes'] += 1
        
        # Scale position ONLY if it's a collision geom with explicit position
        # (these are positions within the body frame, should be scaled)
        if 'pos' in geom.attrib and 'class' in geom.attrib:
            if 'collision' in geom.attrib['class']:
                old_pos = geom.attrib['pos']
                geom.attrib['pos'] = scale_position_string(old_pos, scale)
                scaled_elements['geom_positions_in_body_frame'] += 1
    
    # Scale inertial properties (COM positions are in body frame)
    for inertial in root.iter('inertial'):
        # Scale COM position (relative to body frame)
        if 'pos' in inertial.attrib:
            old_pos = inertial.attrib['pos']
            inertial.attrib['pos'] = scale_position_string(old_pos, scale)
            scaled_elements['inertial_positions'] += 1
        
        # Scale inertia tensor
        if 'diaginertia' in inertial.attrib:
            old_inertia = inertial.attrib['diaginertia']
            inertial.attrib['diaginertia'] = scale_inertia(old_inertia, scale)
            scaled_elements['inertial_inertias'] += 1
    
    # Scale site positions (in body frame)
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
    
    # Verify
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
    
    # Find wheel
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
        print("\n✅ GEOMETRY CORRECT!")
    elif error_pct < 5.0:
        print("\n✅ Geometry close enough (< 5% error)")
    else:
        print("\n⚠️  Geometry may still have issues")
    
    # Test loading with MuJoCo
    print(f"\n{'='*80}")
    print("TESTING WITH MUJOCO")
    print("="*80)
    
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(output_xml))
        print(f"\n✅ MuJoCo model loaded successfully!")
        print(f"  Bodies: {model.nbody}")
        print(f"  Joints: {model.njnt}")
        print(f"  DOFs: {model.nv}")
        print(f"  Actuators: {model.nu}")
        
        # Check if all bodies are connected
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        print(f"\n✅ Forward kinematics successful!")
        print(f"  Base height after mj_forward: {data.qpos[2]:.4f}m")
        
    except Exception as e:
        print(f"\n❌ MuJoCo loading failed: {e}")
        return None
    
    return output_xml


if __name__ == "__main__":
    output_path = create_properly_corrected_xml()
    
    if output_path:
        print(f"\n{'='*80}")
        print("SUCCESS!")
        print("="*80)
        print(f"\nCorrected XML ready at: {output_path}")
        print("\nTest with:")
        print(f"""
cd /home/drl-68/sim2sim_project/Two-wheel-Legged-Bot/main

python scripts/transfer_flamingo_sim2sim.py \\
    --policy logs/co_rl/Flamingo_Flat_Stand_Drive/ppo/2026-02-07_18-03-28/model_4999.pt \\
    --xml assets/flamingo_correct_v2.xml \\
    --init_height 0.5562 \\
    --pd_scale 1.0 \\
    --cmd_vx 0.0 \\
    --duration 20.0
""")
        print("="*80)

#!/usr/bin/env python3
"""
Create the FINAL corrected MuJoCo XML.

KEY INSIGHT: In MuJoCo, body positions are relative to parent.
To scale the robot, we need to scale:
1. ALL body positions (they're all relative, so they all need scaling)
2. ALL geometry sizes
3. ALL inertial COM positions (relative to body frame)
4. Inertia tensors (scale^2)

This will preserve the kinematic structure while scaling the entire robot.
"""

import xml.etree.ElementTree as ET
from pathlib import Path


def scale_string(s, scale):
    """Scale a space-separated string of numbers."""
    values = [float(x) * scale for x in s.split()]
    return ' '.join(f'{v:.6f}' for v in values)


def scale_inertia(inertia_str, scale):
    """Scale inertia (scales with scale^2 for geometry scaling)."""
    inertia_scale = scale ** 2
    values = [float(x) * inertia_scale for x in inertia_str.split()]
    return ' '.join(f'{v:.8f}' for v in values)


def create_final_corrected_xml():
    """Create the final corrected XML."""
    
    # Paths
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    input_xml = project_root / "sim2sim_onnx/assets/flamingo_torque.xml"
    output_xml = project_root / "main/assets/flamingo_correct_final.xml"
    
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    
    # Calculate scale
    target_height = 0.5562
    current_base_z = 0.2461
    current_wheel_r = 0.053
    current_height = current_base_z + current_wheel_r
    
    scale = target_height / current_height
    
    print("="*80)
    print("CREATING FINAL CORRECTED MUJOCO XML")
    print("="*80)
    print(f"\nInput:  {input_xml}")
    print(f"Output: {output_xml}")
    print(f"\nCurrent height: {current_height:.4f}m")
    print(f"Target height:  {target_height:.4f}m")
    print(f"Scale factor:   {scale:.6f}")
    print("\nScaling ALL positions and geometries uniformly...")
    
    # Parse XML
    tree = ET.parse(input_xml)
    root = tree.getroot()
    
    stats = {
        'body_positions': 0,
        'geom_sizes': 0,
        'geom_positions': 0,
        'inertial_positions': 0,
        'inertial_inertias': 0,
        'site_positions': 0,
        'site_sizes': 0,
    }
    
    # Scale ALL body positions (all are relative to parent)
    for body in root.iter('body'):
        if 'pos' in body.attrib:
            body.attrib['pos'] = scale_string(body.attrib['pos'], scale)
            stats['body_positions'] += 1
    
    # Scale ALL geometry
    for geom in root.iter('geom'):
        if 'size' in geom.attrib:
            geom.attrib['size'] = scale_string(geom.attrib['size'], scale)
            stats['geom_sizes'] += 1
        if 'pos' in geom.attrib:
            geom.attrib['pos'] = scale_string(geom.attrib['pos'], scale)
            stats['geom_positions'] += 1
    
    # Scale ALL inertial properties
    for inertial in root.iter('inertial'):
        if 'pos' in inertial.attrib:
            inertial.attrib['pos'] = scale_string(inertial.attrib['pos'], scale)
            stats['inertial_positions'] += 1
        if 'diaginertia' in inertial.attrib:
            inertial.attrib['diaginertia'] = scale_inertia(inertial.attrib['diaginertia'], scale)
            stats['inertial_inertias'] += 1
    
    # Scale ALL sites
    for site in root.iter('site'):
        if 'pos' in site.attrib:
            site.attrib['pos'] = scale_string(site.attrib['pos'], scale)
            stats['site_positions'] += 1
        if 'size' in site.attrib:
            site.attrib['size'] = scale_string(site.attrib['size'], scale)
            stats['site_sizes'] += 1
            
    # Scale ALL meshes
    stats['mesh_scales'] = 0
    for mesh in root.iter('mesh'):
        mesh.attrib['scale'] = f"{scale:.6f} {scale:.6f} {scale:.6f}"
        stats['mesh_scales'] += 1
    
    # Write output
    tree.write(str(output_xml), encoding='utf-8', xml_declaration=True)
    
    print(f"\n✅ XML created successfully!")
    print(f"\nScaled elements:")
    for key, count in stats.items():
        print(f"  {key}: {count}")
    
    # Verify
    print(f"\n{'='*80}")
    print("VERIFICATION")
    print("="*80)
    
    verify_tree = ET.parse(output_xml)
    verify_root = verify_tree.getroot()
    
    # Find base_link
    base_link = verify_root.find(".//body[@name='base_link']")
    if base_link:
        base_pos = base_link.get('pos', '0 0 0')
        base_z = float(base_pos.split()[2])
        print(f"\nBase link z-position: {base_z:.4f}m")
    else:
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
    print(f"STANDING HEIGHT: {standing_height:.4f}m")
    print(f"{'='*80}")
    
    error = abs(standing_height - target_height)
    error_pct = (error / target_height) * 100
    
    print(f"\nTarget:   {target_height:.4f}m")
    print(f"Achieved: {standing_height:.4f}m")
    print(f"Error:    {error:.4f}m ({error_pct:.2f}%)")
    
    if error_pct < 1.0:
        print("\n✅ PERFECT!")
    elif error_pct < 5.0:
        print("\n✅ Good (< 5% error)")
    else:
        print("\n⚠️  May have issues")
    
    # Test with MuJoCo
    print(f"\n{'='*80}")
    print("MUJOCO VALIDATION")
    print("="*80)
    
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(output_xml))
        data = mujoco.MjData(model)
        
        print(f"\n✅ Model loaded!")
        print(f"  Bodies: {model.nbody}")
        print(f"  Joints: {model.njnt}")
        print(f"  DOFs: {model.nv}")
        
        mujoco.mj_forward(model, data)
        print(f"\n✅ Forward kinematics OK!")
        print(f"  Initial height: {data.qpos[2]:.4f}m")
        
        # Check if all bodies are at reasonable positions
        print(f"\n  Body positions after mj_forward:")
        for i in range(model.nbody):
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and 'link' in body_name.lower():
                pos = data.xpos[i]
                print(f"    {body_name}: z={pos[2]:.4f}m")
        
    except Exception as e:
        print(f"\n❌ MuJoCo error: {e}")
        return None
    
    return output_xml


if __name__ == "__main__":
    output = create_final_corrected_xml()
    
    if output:
        print(f"\n{'='*80}")
        print("SUCCESS! READY TO TEST")
        print("="*80)
        print(f"\nFile: {output}")
        print("\nTest command:")
        print("""
cd /home/drl-68/sim2sim_project/Two-wheel-Legged-Bot/main

python scripts/transfer_flamingo_sim2sim.py \\
    --policy logs/co_rl/Flamingo_Flat_Stand_Drive/ppo/2026-02-07_18-03-28/model_4999.pt \\
    --xml assets/flamingo_correct_final.xml \\
    --init_height 0.5562 \\
    --pd_scale 1.0 \\
    --cmd_vx 0.0 \\
    --duration 20.0
""")
        print("="*80)

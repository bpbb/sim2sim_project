#!/usr/bin/env python3
import xml.etree.ElementTree as ET
from pxr import Usd, UsdGeom, UsdPhysics
from pathlib import Path
import numpy as np

def get_usd_data(usd_path):
    stage = Usd.Stage.Open(usd_path)
    data = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            name = prim.GetName()
            xformable = UsdGeom.Xformable(prim)
            local_xform = xformable.GetLocalTransformation()
            local_trans = local_xform.ExtractTranslation()
            
            mass = 0.0
            inertia = [0.0, 0.0, 0.0]
            if prim.HasAPI(UsdPhysics.MassAPI):
                mass_api = UsdPhysics.MassAPI(prim)
                mass = mass_api.GetMassAttr().Get()
                diagonal_inertia = mass_api.GetDiagonalInertiaAttr().Get()
                if diagonal_inertia:
                    inertia = [diagonal_inertia[0], diagonal_inertia[1], diagonal_inertia[2]]
            
            data[name] = {
                'pos': [local_trans[0], local_trans[1], local_trans[2]],
                'mass': mass,
                'inertia': inertia
            }
    return data

def main():
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    input_xml = project_root / "sim2sim_onnx/assets/flamingo_torque.xml"
    output_xml = project_root / "main/assets/flamingo_correct_usd.xml"
    usd_path = str(project_root / "main/lab/flamingo/assets/data/Robots/Flamingo/flamingo_rev03_1_1/flamingo_rev03_1_1_merge_joints.usd")
    
    usd_data = get_usd_data(usd_path)
    tree = ET.parse(input_xml)
    root = tree.getroot()
    
    # Scale factor for everything else (visuals, etc.)
    target_height = 0.5562
    current_height = 0.2461 + 0.053
    scale = target_height / current_height
    
    for body in root.iter('body'):
        name = body.get('name')
        if name in usd_data:
            d = usd_data[name]
            # Update position (MuJoCo body pos is relative to parent)
            # EXCEPT for base_link which we set to the target height
            if name == 'base_link':
                body.set('pos', f"0 0 {target_height - scale*0.053}") # Approximately correct
            else:
                body.set('pos', f"{d['pos'][0]:.6f} {d['pos'][1]:.6f} {d['pos'][2]:.6f}")
            
            # Update inertial
            inertial = body.find('inertial')
            if inertial is not None:
                inertial.set('mass', f"{d['mass']:.6f}")
                inertial.set('diaginertia', f"{d['inertia'][0]:.8f} {d['inertia'][1]:.8f} {d['inertia'][2]:.8f}")
                # We should probably reset inertal pos to 0 since USD mass properties are usually at prim origin
                # or we keep them if we can extract COM. Let's keep them original but scaled for now.
                if 'pos' in inertial.attrib:
                    old_pos = [float(x) for x in inertial.attrib['pos'].split()]
                    new_pos = [x * scale for x in old_pos]
                    inertial.set('pos', f"{new_pos[0]:.6f} {new_pos[1]:.6f} {new_pos[2]:.6f}")

    # Scale ALL geom sizes and site sizes
    for geom in root.iter('geom'):
        if 'size' in geom.attrib:
            sizes = [float(x) * scale for x in geom.attrib['size'].split()]
            geom.set('size', ' '.join(f"{s:.6f}" for s in sizes))
        if 'pos' in geom.attrib:
            pos = [float(x) * scale for x in geom.attrib['pos'].split()]
            geom.set('pos', ' '.join(f"{p:.6f}" for p in pos))
            
    for mesh in root.iter('mesh'):
        mesh.set('scale', f"{scale:.6f} {scale:.6f} {scale:.6f}")

    for site in root.iter('site'):
        if 'pos' in site.attrib:
            pos = [float(x) * scale for x in site.attrib['pos'].split()]
            site.set('pos', ' '.join(f"{p:.6f}" for p in pos))
        if 'size' in site.attrib:
            size = [float(x) * scale for x in site.attrib['size'].split()]
            site.set('size', ' '.join(f"{s:.6f}" for s in size))

    tree.write(str(output_xml), encoding='utf-8', xml_declaration=True)
    print(f"Created USD-informed XML: {output_xml}")

if __name__ == "__main__":
    main()

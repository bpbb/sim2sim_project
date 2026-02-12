import numpy as np
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as xml_et

def euler_to_quat(roll, pitch, yaw):
    cx = np.cos(roll / 2)
    sx = np.sin(roll / 2)
    cy = np.cos(pitch / 2)
    sy = np.sin(pitch / 2)
    cz = np.cos(yaw / 2)
    sz = np.sin(yaw / 2)
    return np.array([
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz,
    ])

class IsaacLabTerrainGenerator:
    def __init__(self, num_rows=1, num_cols=4, terrain_width=1.0, terrain_length=1.0, 
                 horizontal_scale=0.1, vertical_scale=0.001):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.terrain_width = terrain_width # meters
        self.terrain_length = terrain_length # meters
        self.horizontal_scale = horizontal_scale # meters per pixel
        self.vertical_scale = vertical_scale # meters per unit in image (0-255)
        
        self.cell_width_px = int(terrain_width / horizontal_scale)
        self.cell_length_px = int(terrain_length / horizontal_scale)
        
        self.total_width_px = self.num_cols * self.cell_width_px
        self.total_length_px = self.num_rows * self.cell_length_px
        
        # Heightfield in meters
        self.height_field = np.zeros((self.total_length_px, self.total_width_px), dtype=np.float32)

    def add_rough_terrain(self, row, col, noise_range=1.0):
        r_start, r_end = row * self.cell_length_px, (row + 1) * self.cell_length_px
        c_start, c_end = col * self.cell_width_px, (col + 1) * self.cell_width_px
        
        noise = np.random.uniform(-noise_range, noise_range, (self.cell_length_px, self.cell_width_px))
        self.height_field[r_start:r_end, c_start:c_end] = noise

    def add_sloped_terrain(self, row, col, slope=0.5):
        r_start, r_end = row * self.cell_length_px, (row + 1) * self.cell_length_px
        c_start, c_end = col * self.cell_width_px, (col + 1) * self.cell_width_px
        
        x = np.linspace(0, self.terrain_width * slope, self.cell_width_px)
        slope_grid = np.tile(x, (self.cell_length_px, 1))
        self.height_field[r_start:r_end, c_start:c_end] = slope_grid

    def add_stairs_terrain(self, row, col, step_width=0.4, step_height=0.1):
        r_start, r_end = row * self.cell_length_px, (row + 1) * self.cell_length_px
        c_start, c_end = col * self.cell_width_px, (col + 1) * self.cell_width_px
        
        num_steps = int(self.terrain_width / step_width)
        step_width_px = int(step_width / self.horizontal_scale)
        
        stairs = np.zeros((self.cell_length_px, self.cell_width_px))
        for i in range(num_steps):
            stairs[:, i*step_width_px:(i+1)*step_width_px] = i * step_height
            
        self.height_field[r_start:r_end, c_start:c_end] = stairs

    def add_pyramid_stairs_terrain(self, row, col, step_width=0.4, step_height=0.1):
        r_start, r_end = row * self.cell_length_px, (row + 1) * self.cell_length_px
        c_start, c_end = col * self.cell_width_px, (col + 1) * self.cell_width_px
        
        # Pyramid stairs: distance to edge determines height
        stairs = np.zeros((self.cell_length_px, self.cell_width_px))
        
        for r in range(self.cell_length_px):
            for c in range(self.cell_width_px):
                # distance to closest edge in pixels
                dr = min(r, self.cell_length_px - r)
                dc = min(c, self.cell_width_px - c)
                dist_px = min(dr, dc)
                dist_m = dist_px * self.horizontal_scale
                
                num_step = int(dist_m / step_width)
                stairs[r, c] = num_step * step_height
                
        self.height_field[r_start:r_end, c_start:c_end] = stairs

    def add_discrete_obstacles(self, row, col, num_obstacles=20, obstacle_height=0.1, obstacle_size=0.4):
        r_start, r_end = row * self.cell_length_px, (row + 1) * self.cell_length_px
        c_start, c_end = col * self.cell_width_px, (col + 1) * self.cell_width_px
        
        obs_size_px = int(obstacle_size / self.horizontal_scale)
        
        grid = np.zeros((self.cell_length_px, self.cell_width_px))
        for _ in range(num_obstacles):
            r = np.random.randint(0, self.cell_length_px - obs_size_px)
            c = np.random.randint(0, self.cell_width_px - obs_size_px)
            grid[r:r+obs_size_px, c:c+obs_size_px] = obstacle_height
            
        self.height_field[r_start:r_end, c_start:c_end] = grid

    def export_height_field(self, filename="isaaclab_hfield.png"):
        # Normalize to 0-1 for matplotlib imsave
        # MuJoCo uses the image intensity to scale the hfield height.
        # Height = image_val * height_scale
        
        # Let's say we want 1.0 meters to be the max possible height in the image.
        height_scale = 1.0 
        img = (self.height_field / height_scale).clip(0, 1)
        plt.imsave(filename, img, cmap='gray')
        return height_scale

    def create_scene_xml(self, hfield_img_path, output_xml_path, robot_xml_path):
        root = xml_et.Element("mujoco", model="isaaclab_terrain")
        
        # Include robot
        xml_et.SubElement(root, "include", file=robot_xml_path)
        
        # Visual settings
        visual = xml_et.SubElement(root, "visual")
        xml_et.SubElement(visual, "headlight", diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3", specular="0 0 0")
        xml_et.SubElement(visual, "rgba", haze="0.15 0.25 0.35 1")
        
        # Asset
        asset = xml_et.SubElement(root, "asset")
        xml_et.SubElement(asset, "texture", type="skybox", builtin="gradient", rgb1="0.3 0.5 0.7", rgb2="0 0 0", width="512", height="3072")
        xml_et.SubElement(asset, "texture", type="2d", name="groundplane", builtin="checker", mark="edge", rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3", markrgb="0.8 0.8 0.8", width="300", height="300")
        xml_et.SubElement(asset, "material", name="groundplane", texture="groundplane", texuniform="true", texrepeat="5 5", reflectance="0.2")
        
        hfield_name = "isaaclab_hfield"
        size_x = self.total_width_px * self.horizontal_scale
        size_y = self.total_length_px * self.horizontal_scale
        # MuJoCo hfield size: [x_half_size, y_half_size, z_max, z_base]
        # INCREASED z_max to 0.5 to make roughness very visible
        hfield = xml_et.SubElement(asset, "hfield", name=hfield_name, file=hfield_img_path)
        hfield.attrib["size"] = f"{size_x/2} {size_y/2} 0.1 0.1"
        
        # Worldbody
        worldbody = xml_et.SubElement(root, "worldbody")
        xml_et.SubElement(worldbody, "light", pos="0 0 10", dir="0 0 -1", directional="true")
        
        # Hfield geom
        xml_et.SubElement(worldbody, "geom", type="hfield", hfield=hfield_name, pos="0 0 0", name="terrain")
        
        tree = xml_et.ElementTree(root)
        tree.write(output_xml_path)

if __name__ == "__main__":
    # Create a 1x10 grid (a long path)
    num_rows = 1
    num_cols = 4
    cell_size = 1.5
    generator = IsaacLabTerrainGenerator(num_rows=num_rows, num_cols=num_cols, 
                                           terrain_width=cell_size, terrain_length=cell_size,
                                           horizontal_scale=0.05)
    
    # Fill grid: Start flat, then get very rough
    for c in range(num_cols):
        r = 0
        if c == 0:
            # First cell is flat for initialization
            continue
        
        # Subsequent cells are very rough
        generator.add_rough_terrain(r, c, noise_range=0.4) # 0.3m noise on a 0.5m scale is very rough
                
    hfield_path = "/home/drl-68/sim2sim_project/unitree_mujoco/unitree_robots/go2/assets/isaaclab_hfield.png"
    generator.export_height_field(hfield_path)
    
    xml_path = "/home/drl-68/sim2sim_project/unitree_mujoco/unitree_robots/go2/scene_isaaclab_terrain.xml"
    robot_xml = "go2.xml" 
    # Note: MuJoCo will look in 'assets/' because of go2.xml's compiler meshdir setting
    generator.create_scene_xml("isaaclab_hfield.png", xml_path, robot_xml)
    
    print(f"Generated {hfield_path} and {xml_path}")
    print(f"Total path length: {num_cols * cell_size}m")
    # Centered at 0, path from -20 to +20. Start middle of cell 0 (-18m)
    start_x = -(num_cols * cell_size)/2 + cell_size/2
    print(f"Suggested start: x = {start_x}, y = 0")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

# Define Colors
COLORS = {
    "biomorphic_blue": "#0066A2",
    "biomorphic_blue_complimentary": "#FE8C00",
    "delft_blue": "#00A6D6",
    "color_x": "#F80031",
    "color_y": "#FFC700",
    "color_z": "#FF8100",
    "dark_grey": "#2e2e2e",
    "color_contact": "red",
    "color_trunk": "#8B4513"
}

import matplotlib as mpl
mpl.rcParams['text.usetex'] = False
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['axes.unicode_minus'] = False


# --- 1. Parse MTL ---
def parse_mtl(mtl_path):
    colors = {}
    current = None
    with open(mtl_path) as f:
        for line in f:
            if line.startswith("newmtl"):
                current = line.split()[1]
            elif line.startswith("Kd") and current is not None:
                _, r, g, b = line.split()
                colors[current] = (float(r), float(g), float(b), 1.0)
    return colors

def add_mesh(ax,
             obj_path, mtl_path,
             location, rotation):

    mtl_colors = parse_mtl(mtl_path)

    # --- 2. Parse OBJ ---
    vertices = []
    faces = []
    face_colors = []
    current_material = None

    with open(obj_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("usemtl"):
                current_material = line.split()[1]
            elif line.startswith("f "):
                idx = [int(part.split("/")[0])-1 for part in line.split()[1:4]]
                faces.append(idx)
                face_colors.append(mtl_colors.get(current_material, (0.7,0.7,0.7,1.0)))

    rot = R.from_euler(seq="xyz", angles=rotation)
    vertices = rot.apply(vertices)
    vertices = np.array(vertices) + location
    faces = np.array(faces)

    # --- 3. Build triangles for Poly3DCollection ---
    triangles = vertices[faces]

    poly = Poly3DCollection(triangles, facecolors=face_colors, linewidths=0.05)
    ax.add_collection3d(poly)  


def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    lim = 0.18

    # Store the location of the drone mesh
    loc = np.array([0.2, 0.1, 2])
    yaw = 45.0  # degrees
    
    # Plot target aka add cylinder in 3d
    cylinder_height = 4.0
    cylinder_radius = 0.025
    cylinder_rot = R.from_euler('x', 90, degrees=True).as_matrix()
    cylinder_pos = [0.0, cylinder_height / 2, 2.5 - cylinder_radius/2]
    theta = np.linspace(0, 2 * np.pi, 100)
    z = np.linspace(0, cylinder_height, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = cylinder_radius * np.cos(theta_grid)
    y_grid = cylinder_radius * np.sin(theta_grid)
    xyz = np.vstack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()])
    rotated_xyz = cylinder_rot @ xyz
    x_grid = rotated_xyz[0, :].reshape(x_grid.shape) + cylinder_pos[0]
    y_grid = rotated_xyz[1, :].reshape(y_grid.shape) + cylinder_pos[1]
    z_grid = rotated_xyz[2, :].reshape(z_grid.shape) + cylinder_pos[2]
    ax.plot_surface(x_grid, y_grid, z_grid, color=COLORS["color_trunk"], alpha=0.6) 

    add_mesh(ax=ax,
        obj_path="FeelyDroneFullyOpen/FeelyDroneFullyOpen.obj",
            mtl_path="FeelyDroneFullyOpen/FeelyDroneFullyOpen.mtl",
            location=loc,
            rotation=np.deg2rad([0, 0, yaw]))

    ax.set_xlim([-0.5 * lim, 1.5 * lim])
    ax.set_ylim([-1 * lim, 1.0 * lim])
    ax.set_zlim([2-lim, 2+lim])
    ax.set_axis_off()

    ax.view_init(elev=90, azim=0)

    fig.savefig("offset.png", bbox_inches='tight', pad_inches=0.1,
                transparent=True)
    
if __name__ == "__main__":
    main()

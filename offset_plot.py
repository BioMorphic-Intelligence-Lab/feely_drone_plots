import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

mpl.rcParams['text.usetex'] = False
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['axes.unicode_minus'] = False

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

PRESETS = {
    "none": {
        "loc": [0.0, 0.0, 2.2],
        "yaw": 0.0,
        "cylinder_euler": [90.0, 0.0, 0.0],
        "view_elev": 90,
        "view_azim":  0,
        "zlim_offset": 0.0,
        "xlim_offset": 0.0,
    },
    "positional": {
        "loc": [0.20, 0.0, 2.2],
        "yaw": 0.0,
        "cylinder_euler": [90.0, 0.0, 0.0],
        "view_elev": 90,
        "view_azim":  0,
        "zlim_offset": 0.0,
        "xlim_offset": 0.2,
    },
    "rotational": {
        "loc": [0.0, 0.0, 2.2],
        "yaw": 30.0,
        "cylinder_euler": [90.0, 0.0, 0.0],
        "view_elev": 90,
        "view_azim":  0,
        "zlim_offset": 0.0,
        "xlim_offset": 0.0,
    },
    "combined": {
        "loc": [0.20, 0.0, 2.2],
        "yaw": 30.0,
        "cylinder_euler": [90.0, 0.0, 0.0],
        "view_elev": 90,
        "view_azim":  0,
        "zlim_offset": 0.0,
        "xlim_offset": 0.2,
    },
    "inclinational": {
        "loc": [0.0, 0.0, 2.5],
        "yaw": 0.0,
        "cylinder_euler": [80.0, 0.0, 0.0],
        "view_elev": 20,
        "view_azim": 0,
        "zlim_offset": 0.30,
        "xlim_offset": 0.0,
    },
}


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


def add_cylinder(ax, preset):
    cylinder_height = 4.0
    cylinder_radius = 0.025
    cylinder_rot = R.from_euler(
        'xyz', preset["cylinder_euler"], degrees=True).as_matrix()
    cylinder_pos = [0.0, cylinder_height / 2, 2.5 - cylinder_radius / 2]
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


def _draw_box(ax, vertices, face_color):
    """Draw a box from its 8 vertices using Poly3DCollection."""
    idx = [
        [0, 1, 2, 3], [4, 5, 6, 7],
        [0, 1, 5, 4], [2, 3, 7, 6],
        [0, 3, 7, 4], [1, 2, 6, 5],
    ]
    faces = [[vertices[i] for i in f] for f in idx]
    ax.add_collection3d(Poly3DCollection(
        faces, facecolors=face_color,
        linewidths=0.1, edgecolors=(0.0, 0.0, 0.0, 1.0), alpha=1.0))


def add_hbar(ax, preset):
    """Draw an H-bar matching the geometry in hbar_0.100.urdf."""
    length = 5.0
    thickness = 0.01
    span = 0.100

    rot = R.from_euler(
        'xyz', preset["cylinder_euler"], degrees=True).as_matrix()
    pos = np.array([0.0, length / 2, 2.5])

    ht = thickness / 2
    hs = span / 2

    color = (0.75, 0.75, 0.78, 1.0)
    
    boxes = [
        (np.array([0, 0, length / 2]),  np.array([ht, hs, length / 2]), color),
        (np.array([0, hs, length / 2]), np.array([hs, ht, length / 2]), color),
        (np.array([0, -hs, length / 2]), np.array([hs, ht, length / 2]), color),
    ]

    signs = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ])

    for center, half, color in boxes:
        verts = center + signs * half
        verts = (rot @ verts.T).T + pos
        _draw_box(ax, verts, color)


def main():
    parser = argparse.ArgumentParser(
        description="Generate offset visualisation images for the FeelyDrone.")
    parser.add_argument(
        "mode",
        choices=PRESETS.keys(),
        help="Type of offset to visualise.")
    parser.add_argument(
        "-t", "--target",
        choices=["cylinder", "hbar"],
        default="cylinder",
        help="Target object to display (default: cylinder).")
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output filename (default: offset_<mode>.png).")
    args = parser.parse_args()

    preset = PRESETS[args.mode]
    loc = np.array(preset["loc"])
    yaw = preset["yaw"]
    output = args.output or f"offset_{args.mode}.png"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if args.target == "hbar":
        add_hbar(ax=ax, preset=preset)
    else:
        add_cylinder(ax=ax, preset=preset)

    lim = 0.18

    add_mesh(ax=ax,
             obj_path="FeelyDroneFullyOpen/FeelyDroneFullyOpen.obj",
             mtl_path="FeelyDroneFullyOpen/FeelyDroneFullyOpen.mtl",
             location=loc,
             rotation=np.deg2rad([0, 0, yaw]))

    zlim_off = preset["zlim_offset"]
    xlim_off = preset["xlim_offset"]
    ax.set_xlim([-lim + xlim_off, lim + xlim_off])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim + loc[2] + zlim_off, lim + loc[2] + zlim_off])

    
    #plt.show()
    ax.set_axis_off()
    ax.view_init(elev=preset["view_elev"], azim=preset["view_azim"])

    fig.savefig(output, bbox_inches='tight', pad_inches=0.1,
                transparent=True)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

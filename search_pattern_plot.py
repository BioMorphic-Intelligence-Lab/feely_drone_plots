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
    "color_contact": "red"
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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
tau = np.linspace(0, 1, 100, endpoint=True)
search_pattern_fun = lambda t: np.array([0.5 * np.sin(2 * 2 * np.pi * t),
                                        0.5 * np.sin(    2 * np.pi * t),
                                        1.75 * np.ones_like(t)])
search_pattern = search_pattern_fun(tau)

lim = np.max(np.abs(search_pattern[:2, :])) + 0.1

# Plot the target pose estimate
target_pos_estimate = np.array([0, 0, 2])
# Plot connection between target pos estimate and search pattern
ax.plot(np.zeros(2), np.zeros(2),
        [search_pattern_fun(0.5)[2], target_pos_estimate[2]],
        color=COLORS["dark_grey"], linestyle="--")
ax.plot(*target_pos_estimate, marker='o', markersize=10,
        color=COLORS["biomorphic_blue"],zorder=100)

# Plot the search pattern
ax.plot(search_pattern[0, :],
        search_pattern[1, :],
        search_pattern[2, :], 
        linestyle="--",
        color=COLORS["delft_blue"])

# Store the locations of the drone mesh
loc = np.stack((
    search_pattern_fun(0.125),
    search_pattern_fun(0.375),
    search_pattern_fun(0.625),
    search_pattern_fun(0.875),
    search_pattern_fun(0.5),
))

for i in range(len(loc)-1):
    add_mesh(ax=ax,
           obj_path="FeelyDroneFullyOpen/FeelyDroneFullyOpen.obj",
            mtl_path="FeelyDroneFullyOpen/FeelyDroneFullyOpen.mtl",
            location=loc[i],
            rotation=np.deg2rad([0, 0, 0]))

add_mesh(ax=ax,
         obj_path="FeelyDroneFullyUp/FeelyDroneFullyUp.obj",
         mtl_path="FeelyDroneFullyUp/FeelyDroneFullyUp.mtl",
         location=loc[-1],
         rotation=np.deg2rad([0, 0, 0]))


# Plot the maximally reachable points
ax.plot(np.ones(2) * (max(search_pattern[1, :]) + 0.05),
        [-lim, lim],
        min(search_pattern[2, :]), color=COLORS["dark_grey"], linestyle="--",
        zorder=110)
ax.plot(np.ones(2) * (min(search_pattern[1, :]) - 0.25),
        [-lim, lim],
        min(search_pattern[2, :]), color=COLORS["dark_grey"], linestyle="--",
        zorder=110)

ax.set_xlim([-lim, lim])
ax.set_ylim([-lim, lim])
ax.set_zlim([2-lim, 2+lim])

ax.set_xticks(np.linspace(-lim, lim, 5))
ax.set_yticks(np.linspace(-lim, lim, 5))
ax.set_zticks(np.linspace(2-lim, 2+lim, 3))

#ax.set_xlabel(r"x [\$m\$]")
#ax.set_ylabel(r"y [\$m\$]")
#ax.set_zlabel(r"z [\$m\$]")

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
ax.xaxis.labelpad = -5
ax.yaxis.labelpad = -5
ax.zaxis.labelpad = -12

fig.savefig("search_pattern.svg", bbox_inches='tight', pad_inches=0.0,
            transparent=True)

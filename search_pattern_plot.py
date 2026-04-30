import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

from feely_drone_common.search_pattern import (
    SinusoidalSearchPattern,
    LinearSearchPattern,
    SquareSearchPattern,
    SpiralSearchPattern,
)

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

PATTERN_CHOICES = ["sinusoidal", "linear", "square", "spiral"]


def create_pattern(name):
    """Instantiate a SearchPattern with sensible defaults for visualisation."""
    if name == "sinusoidal":
        params = np.array([
            [0.5, 0.5, 0.0],   # amplitude
            [2.0, 1.0, 1.0],   # frequency
            [0.0, 0.0, 0.0],   # phase
            [0.0, 0.0, 1.75],  # offset
        ])
        return SinusoidalSearchPattern(params)
    elif name == "linear":
        params = np.array([
            [1.0, 0.0, 0.0],   # slope
            [-0.5, 0.0, 1.75], # offset
        ])
        return LinearSearchPattern(params)
    elif name == "square":
        params = np.array([
            [1.0, 0.0, 0.0],  # side length (first element)
            [0.0, 0.0, 1.75], # center
        ])
        return SquareSearchPattern(params)
    elif name == "spiral":
        params = np.array([
            [0.5, 3.0, 0.0],  # max_radius, rotation_speed
            [0.0, 0.0, 1.75], # center
        ])
        return SpiralSearchPattern(params)
    else:
        raise ValueError(f"Unknown pattern: {name}")


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


def add_mesh(ax, obj_path, mtl_path, location, rotation):
    mtl_colors = parse_mtl(mtl_path)

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
                face_colors.append(mtl_colors.get(current_material, (0.7, 0.7, 0.7, 1.0)))

    rot = R.from_euler(seq="xyz", angles=rotation)
    vertices = rot.apply(vertices)
    vertices = np.array(vertices) + location
    faces = np.array(faces)

    triangles = vertices[faces]
    poly = Poly3DCollection(triangles, facecolors=face_colors, linewidths=0.05)
    ax.add_collection3d(poly)


def main():
    parser = argparse.ArgumentParser(
        description="Plot a search pattern from feely_drone_common.")
    parser.add_argument(
        "-p", "--pattern",
        choices=PATTERN_CHOICES,
        default="sinusoidal",
        help="Search pattern type to plot (default: sinusoidal)")
    parser.add_argument(
        "--no-mesh",
        action="store_true",
        help="Skip rendering drone meshes")
    args = parser.parse_args()

    pattern = create_pattern(args.pattern)

    tau = np.linspace(0, 1, 100, endpoint=True)
    search_pattern = np.array([pattern.f(t) for t in tau]).T  # shape (3, N)

    lim = np.max(np.abs(search_pattern[:2, :])) + 0.1

    z_center = np.mean(search_pattern[2, :])
    target_pos_estimate = np.array([0, 0, z_center + 0.25])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(np.zeros(2), np.zeros(2),
            [pattern.f(0.5)[2], target_pos_estimate[2]],
            color=COLORS["dark_grey"], linestyle="--")
    ax.plot(*target_pos_estimate, marker='o', markersize=10,
            color=COLORS["biomorphic_blue"], zorder=100)

    ax.plot(search_pattern[0, :],
            search_pattern[1, :],
            search_pattern[2, :],
            linestyle="--",
            color=COLORS["delft_blue"])

    if not args.no_mesh:
        if args.pattern == "sinusoidal":
            loc_fully_open = np.stack((
                pattern.f(0.125),
                pattern.f(0.375),
                pattern.f(0.625),
                pattern.f(0.875)
            ))

            loc_fully_up = np.stack((
                pattern.f(0.5),
            ))
        elif args.pattern == "square":
            loc_fully_open = np.stack((
                pattern.f(0.375),
                pattern.f(0.875),
            ))
            loc_fully_up = np.stack((
                pattern.f(0.125),
                pattern.f(0.625),
            ))
        elif args.pattern == "spiral":
            loc_fully_open = np.stack((
                pattern.f(0.575),
                pattern.f(0.8)
            ))
            loc_fully_up = np.stack((
                pattern.f(0.0),
            ))

        for i in range(len(loc_fully_open)):
            add_mesh(ax=ax,
                     obj_path="FeelyDroneFullyOpen/FeelyDroneFullyOpen.obj",
                     mtl_path="FeelyDroneFullyOpen/FeelyDroneFullyOpen.mtl",
                     location=loc_fully_open[i],
                     rotation=np.deg2rad([0, 0, 0]))
        for i in range(len(loc_fully_up)):
            add_mesh(ax=ax,
                 obj_path="FeelyDroneFullyUp/FeelyDroneFullyUp.obj",
                 mtl_path="FeelyDroneFullyUp/FeelyDroneFullyUp.mtl",
                 location=loc_fully_up[i],
                 rotation=np.deg2rad([0, 0, 0]))



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
    ax.set_zlim([z_center - lim, z_center + lim])

    ax.set_xticks(np.linspace(-lim, lim, 5))
    ax.set_yticks(np.linspace(-lim, lim, 5))
    ax.set_zticks(np.linspace(z_center - lim, z_center + lim, 3))

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.xaxis.labelpad = -5
    ax.yaxis.labelpad = -5
    ax.zaxis.labelpad = -12

    ax.set_xlabel(r"\$x\$ [m]")
    ax.set_ylabel(r"\$y\$ [m]")
    ax.set_zlabel(r"\$z\$ [m]")

    fig.savefig(f"search_pattern_{args.pattern}.svg",
                bbox_inches='tight', pad_inches=0.0, transparent=True)


if __name__ == "__main__":
    main()

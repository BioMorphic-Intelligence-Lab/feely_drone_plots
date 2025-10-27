import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from matplotlib.patches import FancyBboxPatch

from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from rosbags.rosbag2 import Reader

# Define Colors
COLORS = {
    "biomorphic_blue": "#0066A2",
    "biomorphic_blue_complimentary": "#FE8C00",
    "delft_blue": "#00A6D6",
    "color_x": "#008E2B",
    "color_yaw": "#001A83",
    "dark_grey": "#2e2e2e",
    "contact": "#ED5349",
    "color_trunk": "#8B4513"
}

STATE_COLORS = {
    'initcolor': '#FFFFFF',
    'searchcolor': '#00A7D6', 
    'graspcolor': '#0C2340',
    'finalcolor': '#6CC24A',
    'errorcolor': '#E03C31'
}

import matplotlib as mpl
mpl.rcParams['text.usetex'] = False
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['axes.unicode_minus'] = False

def guess_msgtype(path: Path) -> str:
    """Guess message type name from path."""
    name = path.relative_to(path.parents[2]).with_suffix('')
    if 'msg' not in name.parts:
        name = name.parent / 'msg' / name.name
    return str(name)

def rosbag2data(path: str):

    ############## Register non-standard msg types ##############
    typestore = get_typestore(Stores.ROS2_JAZZY)
    add_types = {}

    for pathstr in [
        "/home/antbre/projects/feely_drone/feely_drone_ros2/src/custom_msgs/msg/StateMachineState.msg",
        "/home/antbre/projects/feely_drone/feely_drone_ros2/src/custom_msgs/msg/TouchData.msg"
        ]:
        msgpath = Path(pathstr)
        msgdef = msgpath.read_text(encoding='utf-8')
        add_types.update(get_types_from_msg(msgdef, guess_msgtype(msgpath)))

    typestore.register(add_types)

    ##############################################################
    ############## Load all the data #############################
    ##############################################################

    t_ref = []
    t_pose = []
    t_contact = []
    t_touch = []
    t_target = []
    t_state_machine = []

    ref_position = []
    ref_yaw = []
    position = []
    yaw = []
    contact = []
    touch_data = []
    target = []
    target_yaw = []
    state_machine = []

    # Create reader instance and open for reading.
    with Reader(path) as reader:
        # Iterate over messages.
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == '/feely_drone/in/ref_pose':
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                t_ref += [float(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec)]
                ref_position +=[[msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]]
                rot = R.from_quat([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
                ref_yaw += [rot.as_euler('xyz', degrees=False)[2]]
            if connection.topic == '/feely_drone/out/pose':
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                t_pose += [float(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec)]
                position +=[[msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]]
                rot = R.from_quat([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
                yaw += [rot.as_euler('xyz', degrees=False)[2]]
            if connection.topic == '/target/out/pose':
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                t_target += [float(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec)]
                target +=[[msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]]
                target_yaw += [2 * np.arctan2(msg.pose.orientation.z, msg.pose.orientation.w)]
            if connection.topic == '/feely_drone/out/bin_touch_state':
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                t_contact += [float(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec)]
                contact += [msg.position]
            if connection.topic == '/feely_drone/out/touch_data':
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                t_touch += [float(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec)]
                touch_data += [msg.raw_data]
            if connection.topic == '/feely_drone/out/state_machine_state':
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                t_state_machine += [float(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec)]
                state_machine += [msg.state]
  
    
    t_start = min(np.concatenate((t_ref, t_pose, t_contact, t_touch,
                                  t_target, t_state_machine)))
    t_ref = np.array(t_ref) - t_start
    t_pose = np.array(t_pose) - t_start
    t_target = np.array(t_target) - t_start
    t_contact = np.array(t_contact) - t_start
    t_touch = np.array(t_touch) - t_start
    t_state_machine = np.array(t_state_machine) - t_start - 80 + 61

    ref_position = np.array(ref_position)
    ref_yaw = np.array(ref_yaw)
    position = np.array(position)
    yaw = np.array(yaw)
    contact = np.array(contact)
    touch_data = np.array(touch_data)
    target = np.array(target)
    target_yaw = np.array(target_yaw)
    state_machine = np.array(state_machine)

    return {"t_ref": t_ref, "ref_position": ref_position, "ref_yaw": ref_yaw,
            "t_pose": t_pose, "position": position, "yaw": yaw,
            "t_contact": t_contact, "contact": contact, 
            "t_touch": t_touch, "touch_data": touch_data,
            "t_target": t_target, "target": target, "target_yaw": target_yaw,
            "t_state_machine": t_state_machine, "state_machine": state_machine} 

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def process_data(data, cutoff=0):

    # Find the first index where dx is > 0.5 m/s indicating start of movement
    dx_abs = np.abs(np.diff(data["position"][:, 0], prepend=np.zeros(1))) / 0.01
    movement_start_idx = np.argmax((dx_abs > 0.2) & (data["position"][:, 2] > 1.25)) - 500
    if movement_start_idx < 0:
        movement_start_idx = 0
    flight_start_idx = 0 #np.argmax(data["position"][:,2] > 1.5) - 10

    target_pos = np.zeros(3)
    target_pos[0] = np.mean(data["target"][:, 0])
    target_pos[1] = np.mean(data["target"][:, 1])

    data["t_ref"] = data["t_ref"] - data["t_pose"][movement_start_idx]
    data["t_target"] = data["t_target"] - data["t_pose"][movement_start_idx]
    data["t_pose"] = data["t_pose"][flight_start_idx:] - data["t_pose"][movement_start_idx]
    data["position"] = data["position"][flight_start_idx:, :] - target_pos
    data["yaw"] = (data["yaw"][flight_start_idx:])
    data["target"] = data["target"] - target_pos
    data["target_yaw"] = normalize_angle(np.mean(data["target_yaw"], axis=0)) * np.ones_like(data["target_yaw"])
    
    return data

def make_time_series_plot(data, end_times):
    fig = plt.figure(figsize=6 * np.array([3,1]))
    gs = gridspec.GridSpec(2, 1, hspace=0.2, wspace=0.15,
                           left=0.08, right=0.98, top=0.90, bottom=0.2) 
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    #ax3 = fig.add_subplot(gs[2], sharex=ax1)
    axs = [ax1, ax2]#, ax3]

    t_end = max(end_times) + 1

    for i, d in enumerate(data):
        # Find end index
        end_idx = np.argmax(d["t_pose"] > end_times[i])
        # Find first contact index
        contacts = (d["state_machine"] == 2)
        contact_idx = np.argmax(contacts)
        # Find index of position closest to contact
        contact_time = d["t_state_machine"][contact_idx]
        pose_idx = np.argmin(np.abs(d["t_pose"] - contact_time))

        axs[0].plot(d["t_pose"][:end_idx], d["position"][:end_idx,0], label=r"\$x\$", color=COLORS["color_x"], alpha=0.4)
        axs[0].plot(d["t_pose"][end_idx], d["position"][end_idx, 0], marker="o", color=COLORS["dark_grey"],
                    markersize=8, label="Perched", zorder=10, alpha=0.8)
        #axs[0].plot(d["t_pose"][pose_idx], d["position"][pose_idx, 0], marker="o", color=COLORS["contact"],
        #            markersize=8, label="Contact", zorder=10, alpha=0.8)
        axs[0].set_ylabel(r"\$x\$ [m]")
        axs[0].set_yticks(np.linspace(-2.0, 1.0, 4, endpoint=True))
        
        axs[1].plot(d["t_pose"][:end_idx], d["yaw"][:end_idx] * 180/np.pi, label="yaw", color=COLORS["color_yaw"], alpha=0.8)
        axs[1].plot(d["t_pose"][end_idx], d["yaw"][end_idx] * 180/np.pi, marker="o", color=COLORS["dark_grey"],
                    markersize=8, label="Perched", zorder=10, alpha=0.8)
        #axs[1].plot(d["t_pose"][pose_idx], d["yaw"][pose_idx] * 180/np.pi, marker="o", color=COLORS["contact"],
        #        markersize=8, label="Contact", zorder=10, alpha=0.8)
        axs[1].set_ylim([-32, 32])
        axs[1].set_yticks([-30, -15, 0, 15, 30])
        axs[1].set_xticks(np.linspace(0, 100, 11, endpoint=True))
        axs[1].set_ylabel(r"Yaw [\$^\circ\$]")
        axs[1].set_xlabel(r"Time [s]")

        #axs[2].step(d["t_state_machine"], d["state_machine"], where='post', color=COLORS["dark_grey"])

    t_target = np.array([0, t_end])
    axs[1].plot(t_target, np.zeros_like(t_target), linestyle="--", label=r"target yaw", color="black")
    axs[0].plot(t_target, np.zeros_like(t_target), linestyle="--", label=r"target \$x\$", color="black")
    axs[0].set_xlim([0, t_end])
    plt.setp(axs[0].get_xticklabels(), visible=False)

    xlabelpad = 20
    ylabelpad = 45
    tickpad = 20

    axs[0].tick_params(axis='both', pad=tickpad)
    axs[1].tick_params(axis='both', pad=tickpad)
   
    axs[0].yaxis.labelpad = ylabelpad
    axs[0].xaxis.labelpad = xlabelpad
    axs[1].yaxis.labelpad = ylabelpad
    axs[1].xaxis.labelpad = xlabelpad

    #axs[2].set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
    #axs[2].set_ylim([-0.5, 7.5])

    return fig

def make_3d_plot(data, end_times, trial_names):
    fig = plt.figure(figsize=10 * np.array([1.5,1]))
    gs = gridspec.GridSpec(2, 3, hspace=0.1, wspace=0.25,
                           width_ratios=[1, 1, 0.05]) 
    axs = [
        fig.add_subplot(gs[0, 0], projection='3d'),
        fig.add_subplot(gs[1, 0], projection='3d'),
        fig.add_subplot(gs[0, 1], projection='3d'),
        fig.add_subplot(gs[1, 1], projection='3d'),
    ]

    # Add legend axis spanning the bottom row
    legend_ax = fig.add_subplot(gs[:, 2])

    for i, d in enumerate(data):

        end_idx = np.argmax(d["t_pose"] > end_times[i])
        # Create points for LineCollection
        points = np.array([d["position"][:end_idx, 0],
                        d["position"][:end_idx, 1],
                        d["position"][:end_idx, 2]]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        alphas_linear = np.linspace(0.0, 1.0, len(segments))
        lc1 = Line3DCollection(segments, alpha=alphas_linear, colors=COLORS["delft_blue"], linewidths=5)
        axs[i].add_collection3d(lc1)
            
        
        # Plot target aka add cylinder in 3d
        cylinder_height = 4.0
        cylinder_radius = 0.1
        cylinder_rot = R.from_euler('x', 90, degrees=True).as_matrix()
        cylinder_pos = [0.0, cylinder_height / 2, d["target"][0,2] - cylinder_radius/2]
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
        axs[i].plot_surface(x_grid, y_grid, z_grid, color=COLORS["color_trunk"], alpha=0.6)   
        
        # Set limits and labels
        axs[i].set_xlim([-2.0, 1.0])
        axs[i].set_ylim([-1.5, 1.5])
        axs[i].set_zlim([0.0, 3.0])
        axs[i].set_xlabel(r"\$x\$ [m]")
        axs[i].set_ylabel(r"\$y\$ [m]")
        axs[i].set_zlabel(r"\$z\$ [m]")
        axs[i].view_init(elev=20., azim=-45)

        # Set ticks
        axs[i].set_xticks(np.linspace(-2.0, 1.0, 4, endpoint=True))
        axs[i].set_yticks(np.linspace(-1.5, 1.5, 4, endpoint=True))
        axs[i].set_zticks(np.linspace(0.0, 3.0, 4, endpoint=True))

        xlabelpad = 20
        ylabelpad = 20
        zlabelpad = 20
        tickpad = 10

        axs[i].tick_params(axis='both', pad=tickpad)
        axs[i].tick_params(axis='both', pad=tickpad)
    
        axs[i].yaxis.labelpad = ylabelpad
        axs[i].xaxis.labelpad = xlabelpad
        axs[i].zaxis.labelpad = zlabelpad

        # Create a 2D overlay axis for the text box (to get independent width/height control)
        overlay_ax = fig.add_axes(axs[i].get_position(), frameon=False)
        overlay_ax.set_xlim(0, 1)
        overlay_ax.set_ylim(0, 1)
        overlay_ax.set_xticks([])
        overlay_ax.set_yticks([])
        
        # Define box dimensions independently
        box_width = 0.5   # width as fraction of axes width
        box_height = 0.15  # height as fraction of axes height
        box_x = 0.05       # x position
        box_y = 0.86       # y position (top-left style positioning)
        
        # Add the rounded rectangle with independent width/height control
        rounded_box = FancyBboxPatch(
            (box_x, box_y - box_height), box_width, box_height,  # subtract height for top-left positioning
            boxstyle="round,pad=0.01",
            facecolor='grey',
            alpha=0.3,
            edgecolor='none',
            transform=overlay_ax.transAxes
        )
        overlay_ax.add_patch(rounded_box)

        # Add text in the top-left corner
        axs[i].text2D(0.1, 0.8, rf"Trial {trial_names[i]}", 
                    transform=axs[i].transAxes, 
                    fontsize=12,  
                    ha="left", va="top")

    legend_ax.clear()  # Clear any existing content

    # Create custom alpha representation
    for j in range(100):
        alpha_val = (j + 1) / 100.0  # From 0.01 to 1.0
        legend_ax.axhspan(j, j+1, color=COLORS["delft_blue"], alpha=alpha_val)

    # Customize the legend axis
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 100)
    legend_ax.set_ylabel(r'Trial Progression [%]', fontsize=12)

    legend_ax.yaxis.tick_right()
    legend_ax.yaxis.set_label_position("right")

    # Set ticks
    legend_ax.set_yticks(np.linspace(0, 100, 5, endpoint=True))
    legend_ax.set_xticks([])

    # Remove unnecessary spines
    legend_ax.spines['bottom'].set_visible(False)
    legend_ax.spines['top'].set_visible(False)
    legend_ax.spines['left'].set_visible(False)
    legend_ax.spines['right'].set_visible(True)

    legend_ax.tick_params(axis='y', pad=ylabelpad)
    legend_ax.yaxis.labelpad = 1.5 * ylabelpad
    
    return fig

def make_contact_plot(data, end_time, index):
    fig = plt.figure(figsize=7 * np.array([2.5,1]))
    gs = gridspec.GridSpec(2, 1, hspace=0.2, wspace=0.15,
                           left=0.08, right=0.98, top=0.95, bottom=0.15,
                           height_ratios=[2, 1]) 
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    #ax3 = fig.add_subplot(gs[2], sharex=ax1)
    axs = [ax1, ax2]#, ax3]

    d = data[index]

    t_start = min(d["t_contact"][0], d["t_touch"][0])
    t_end = end_time - t_start
       
    # Find end index
    contact_end_idx = np.argmax(d["t_contact"] > end_time)
    touch_end_idx = np.argmax(d["t_touch"] > end_time)
    
    # Mask for before 10s and after 30s
    t = d["t_contact"][:contact_end_idx] - t_start
    mask_before = t < 2.5
    mask_after = t > 30

    contacts_scaled =  np.linspace(1, 12, num=12, endpoint=True) * d["contact"][:contact_end_idx, :]
    axs[0].plot(t[mask_before],
                contacts_scaled[mask_before],
                linewidth=0, marker="o",
                label=r"Binary Contact Signal", alpha=0.4, color=COLORS["contact"])
    axs[0].plot(t[mask_after] - 30 + 10,
                contacts_scaled[mask_after],
                linewidth=0, marker="o",
            label=r"Binary Contact Signal", alpha=0.4, color=COLORS["contact"])
    
    # Add a break indicator ("...") at the jump
    if np.any(mask_before) and np.any(mask_after):
        # Plot a short vertical dotted line and text
        axs[0].text(5, 4.5, "...", ha="center", va="center", fontsize=16, color="black")

    # Plot each touch_data channel with progressive shades of grey, skipping 10-30s
    num_channels = d["touch_data"].shape[1]
    drawn = False
    for i in range(num_channels):
        grey_val = 0.3 + 0.6 * (i / (num_channels - 1))  # 0.3 (dark) to 0.9 (light)
        color = (grey_val, grey_val, grey_val)

        # Mask for before 10s and after 30s
        t = d["t_touch"][:touch_end_idx] - t_start
        mask_before = t < 2.5
        mask_after = t > 30

        # Plot before 10s
        axs[1].plot(
            t[mask_before],
            d["touch_data"][:touch_end_idx, i][mask_before],
            label=f"Raw {i+1}" if i == 0 else None,
            alpha=0.8,
            color=color
        )
        # Plot after 30s
        axs[1].plot(
            t[mask_after] - 30 + 10,
            d["touch_data"][:touch_end_idx, i][mask_after],
            alpha=0.8,
            color=color
        )

        # Add a break indicator ("...") at the jump
        if np.any(mask_before) and np.any(mask_after) and not drawn:
            drawn = True
            # Find y at the end of before and start of after
            y_before = d["touch_data"][:touch_end_idx, i][mask_before][-1]
            y_after = d["touch_data"][:touch_end_idx, i][mask_after][0]
            # Plot a short vertical dotted line and text
            axs[1].text(5, (y_before + y_after)/2, "...", ha="center", va="center", fontsize=16, color="black")

    axs[1].set_xticks([0, 5, 10, 20, 30, 40, 60, 70, 80, 90, 100])
    axs[1].set_xticklabels([0, "...", 30, 40, 60, 70, 80, 90, 100, 110, 120])
    axs[1].set_ylabel(r"Value [-]")
    axs[1].set_xlabel(r"Time [s]")

    #axs[2].step(d["t_state_machine"], d["state_machine"], where='post', color=COLORS["dark_grey"])

    axs[0].set_ylabel(r"Contact \$\in\mathcal{B}\$")
    axs[0].set_ylim([0.5, 9.5])
    axs[0].set_yticks(np.linspace(1, 9, num=9, endpoint=True))
    axs[0].set_yticklabels(
        [rf"\$\mathcal{{C}}_{{{i}}}\$"  for i in range(1, 10)]
    )
    axs[0].set_xlim([0, t_end - 20])
    plt.setp(axs[0].get_xticklabels(), visible=False)

    xlabelpad = 20
    ylabelpad = 45
    tickpad = 20

    axs[0].tick_params(axis='both', pad=tickpad)
    axs[1].tick_params(axis='both', pad=tickpad)
   
    axs[0].yaxis.labelpad = - 0.8 * ylabelpad
    axs[0].xaxis.labelpad = xlabelpad
    axs[1].yaxis.labelpad = ylabelpad
    axs[1].xaxis.labelpad = xlabelpad

    #axs[2].set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
    #axs[2].set_ylim([-0.5, 7.5])

    return fig


def main():
    paths = [
            "/home/antbre/Desktop/ZeroOffset/rosbags_2025_08_18/rosbag2-11_50_16_success_0_offset",
            "/home/antbre/Desktop/ZeroOffset/rosbags_2025_08_18/rosbag2-18_48_51_success_0_offset",
            "/home/antbre/Desktop/ZeroOffset/rosbags_2025_08_19/rosbag2-12_04_02_success_0_offset",
            "/home/antbre/Desktop/ZeroOffset/rosbags_2025_08_19/rosbag2-12_14_55_success_0_offset",
            #
            "/home/antbre/Desktop/YawOffset/rosbags_2025_09_01/rosbag2-11_50_21_success_20deg_yaw_offset",
            "/home/antbre/Desktop/YawOffset/rosbags_2025_09_03/rosbag2-11_39_49_success-20degYaw",
            "/home/antbre/Desktop/YawOffset/rosbags_2025_09_03/rosbag2-09_26_05_success_45deg_yaw",
            "/home/antbre/Desktop/YawOffset/rosbags_2025_09_03/rosbag2-11_43_31_success-45degYaw",
            #
            "/home/antbre/Desktop/PosOffset/rosbags_2025_09_03/rosbag2-11_51_58_success_-0.25posOffset",
            "/home/antbre/Desktop/PosOffset/rosbags_2025_09_03/rosbag2-11_55_05_success0.25posOffset",
            "/home/antbre/Desktop/PosOffset/rosbags_2025_09_04/rosbag2-19_58_29_success-0.6posOffset",
            "/home/antbre/Desktop/PosOffset/rosbags_2025_09_04/rosbag2-20_01_01_success0.6posOffset",
            #
            "/home/antbre/Desktop/PosOffset/rosbags_2025_09_04/rosbag2-20_07_54successDoublePosAndYawOffset-0.25-0.25-25",
            "/home/antbre/Desktop/PosOffset/rosbags_2025_09_04/rosbag2-20_10_01successDoublePosAndYawOffset-0.25-0.25-25"
            ]


    data = np.array([process_data(rosbag2data(p)) for p in paths])
    end_times = np.array([43.0, 46.0, 35.0, 33.0,
                          33.5, 43.4, 52.0, 48.0,
                          32.0, 38.0, 37.0, 73.0,
                          41.0, 42.0
                          ])
    
    # Create and save time series plots
    #fig = make_time_series_plot(data, end_times)
    #fig.savefig("time_series_plot.svg")

    # Create and save contact threshold plot
    #fig = make_contact_plot(data, 54, -1)
    #fig.savefig("contact_plot.svg")
    
    fig = make_3d_plot(data[[3, 7, 11, 13]],
                       end_times[[3, 7, 11, 13]],
                       trial_names=["III", "VII", "XI", "XIII"])
    fig.savefig(f"3d_plot.svg", bbox_inches='tight', pad_inches=0.5,
        transparent=False)

if __name__=="__main__":
    main()



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from rosbags.rosbag2 import Reader

# Define Colors
COLORS = {
    "biomorphic_blue": "#0066A2",
    "biomorphic_blue_complimentary": "#FE8C00",
    "delft_blue": "#00A6D6",
    "color_x": "#008E2B",
    "color_y": "#FF5100",
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

STATE_INT_TO_COLOR = {
    0: 'initcolor',    # UNDEFINED
    9: 'initcolor',    # TAKEOFF
    1: 'searchcolor',  # SEARCHING
    2: 'graspcolor',   # TOUCHED
    3: 'graspcolor',   # APPROACH
    4: 'graspcolor',   # POSITION
    5: 'graspcolor',   # ROTATION
    6: 'finalcolor',   # FINALIZE
    7: 'finalcolor',   # PERCH
    8: 'errorcolor',   # ABORT
}

STATE_INT_TO_NAME = {
    0: r'UNDEFINED', 9: r'TAKEOFF', 1: r'SEARCH',
    2: r'TOUCH', 3: r'APPROACH', 4: r'POSITION',
    5: r'ROTATION', 6: r'FINALIZE', 7: r'PERCH', 8: r'ABORT',
}

import matplotlib as mpl
mpl.rcParams['text.usetex'] = False
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['axes.unicode_minus'] = False

paths = [
        # Zero Offset
        "/home/antbre/Desktop/ZeroOffset/rosbags_2025_08_18/rosbag2-11_50_16_success_0_offset",
        "/home/antbre/Desktop/ZeroOffset/rosbags_2025_08_18/rosbag2-18_48_51_success_0_offset",
        "/home/antbre/Desktop/ZeroOffset/rosbags_2025_08_19/rosbag2-12_04_02_success_0_offset",
        "/home/antbre/Desktop/ZeroOffset/rosbags_2025_08_19/rosbag2-12_14_55_success_0_offset",
        # Yaw Offset
        "/home/antbre/Desktop/YawOffset/rosbags_2025_09_01/rosbag2-11_50_21_success_20deg_yaw_offset",
        "/home/antbre/Desktop/YawOffset/rosbags_2025_09_03/rosbag2-11_39_49_success-20degYaw",
        "/home/antbre/Desktop/YawOffset/rosbags_2025_09_03/rosbag2-09_26_05_success_45deg_yaw",
        "/home/antbre/Desktop/YawOffset/rosbags_2025_09_03/rosbag2-11_43_31_success-45degYaw",
        # Positional Offset
        "/home/antbre/Desktop/PosOffset/rosbags_2025_09_03/rosbag2-11_51_58_success_-0.25posOffset",
        "/home/antbre/Desktop/PosOffset/rosbags_2025_09_03/rosbag2-11_55_05_success0.25posOffset",
        "/home/antbre/Desktop/PosOffset/rosbags_2025_09_04/rosbag2-19_58_29_success-0.6posOffset",
        "/home/antbre/Desktop/PosOffset/rosbags_2025_09_04/rosbag2-20_01_01_success0.6posOffset",
        # Inclines
        "/home/antbre/Desktop/InclOffset/rosbags_2026_05_07/rosbag2-16_31_49_success10deg",
        "/home/antbre/Desktop/InclOffset/rosbags_2026_05_07/rosbag2-17_48_07_success-10deg",
        "/home/antbre/Desktop/InclOffset/rosbags_2026_05_08/rosbag2-09_32_53-success30deg_crashPostPerch",
        "/home/antbre/Desktop/InclOffset/rosbags_2026_05_08/rosbag2-16_44_04-success-30deg",
        # Combined Offset
        "/home/antbre/Desktop/PosOffset/rosbags_2025_09_04/rosbag2-20_07_54successDoublePosAndYawOffset-0.25-0.25-25",
        "/home/antbre/Desktop/PosOffset/rosbags_2025_09_04/rosbag2-20_10_01successDoublePosAndYawOffset-0.25-0.25-25",
        # H-Bar
        # No Offset
        "/home/antbre/Desktop/HBar/rosbags_2026_05_08/noOffset/rosbag2-17_39_44_success_no_offset",
        "/home/antbre/Desktop/HBar/rosbags_2026_05_08/noOffset/rosbag2-17_43_41_success_no_offset",
        "/home/antbre/Desktop/HBar/rosbags_2026_05_09/noOffset/rosbag2-10_30_38_successNoOffsetLongHoverBelow",
        "/home/antbre/Desktop/HBar/rosbags_2026_05_09/noOffset/rosbag2-10_42_57_successNoOffset",
        # Pos Offset
        "/home/antbre/Desktop/HBar/rosbags_2026_05_09/posOffset/rosbag2-10_54_19success-0.25m",
        "/home/antbre/Desktop/HBar/rosbags_2026_05_09/posOffset/rosbag2-12_26_04_success0.25m",
        "/home/antbre/Desktop/HBar/rosbags_2026_05_09/posOffset/rosbag2-14_19_15_success-0.6m",
        "/home/antbre/Desktop/HBar/rosbags_2026_05_09/posOffset/rosbag2-14_33_03success0.6m",
]


start_times = [2, 0, 5, 6, # Cylinder no offset
               3, 3, 3, 3, # Cylinder yaw offset
               5, 5, 7, 5, # Cylinder positional offset
               15, 0, 25, 20, # Cylinder incline offset
               8, 8,       # Cylinder combined offset
               5, 52.7, 5, 20, # H-Bar no offset
               5, 5, 10, 22.1  # H-Bar pos offset
]

roman_numerals = ['(I)', '(II)', '(III)', '(IV)',
                  '(V)', '(VI)', '(VII)', '(VIII)',
                  '(IX)', '(X)', '(XI)', '(XII)',
                  '(XIII)', '(XIV)', '(XV)', '(XVI)',
                  '(XVII)', '(XVIII)', '(XIX)', '(XX)']

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

    TOUCH_DATA_OLD = "std_msgs/Header header\nint64[] raw_data\nint64[] filtered_data\nint64[] baseline_data"
    add_types.update(get_types_from_msg(
        TOUCH_DATA_OLD, 'custom_msgs/msg/TouchDataOld'))

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
    t_servo = []

    ref_position = []
    ref_yaw = []
    position = []
    yaw = []
    contact = []
    touch_data = []
    target = []
    target_yaw = []
    state_machine = []
    servo_states = []

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
                try:
                    msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                except Exception:
                    msg = typestore.deserialize_cdr(
                        rawdata, 'custom_msgs/msg/TouchDataOld')
                t_touch += [float(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec)]
                touch_data += [msg.raw_data]
            if connection.topic == '/feely_drone/in/servo_states':
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                t_servo += [float(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec)]
                servo_states += [list(msg.position)]
            if connection.topic == '/feely_drone/out/state_machine_state':
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                t_state_machine += [float(msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec)]
                state_machine += [msg.state]
  
    
    all_times = [t_ref, t_pose, t_contact, t_touch, t_target, t_state_machine]
    if t_servo:
        all_times.append(t_servo)
    t_start = min(np.concatenate(all_times))
    t_ref = np.array(t_ref) - t_start
    t_pose = np.array(t_pose) - t_start
    t_target = np.array(t_target) - t_start
    t_contact = np.array(t_contact) - t_start
    t_touch = np.array(t_touch) - t_start
    t_state_machine = np.array(t_state_machine) - t_start 
    t_servo = np.array(t_servo) - t_start if t_servo else np.array([])

    ref_position = np.array(ref_position)
    ref_yaw = np.array(ref_yaw)
    position = np.array(position)
    yaw = np.array(yaw)
    contact = np.array(contact)
    touch_data = np.array(touch_data)
    target = np.array(target)
    target_yaw = np.array(target_yaw)
    state_machine = np.array(state_machine)
    servo_states = np.array(servo_states) if servo_states else np.array([]).reshape(0, 3)

    return {"t_ref": t_ref, "ref_position": ref_position, "ref_yaw": ref_yaw,
            "t_pose": t_pose, "position": position, "yaw": yaw,
            "t_contact": t_contact, "contact": contact, 
            "t_touch": t_touch, "touch_data": touch_data,
            "t_target": t_target, "target": target, "target_yaw": target_yaw,
            "t_state_machine": t_state_machine, "state_machine": state_machine,
            "t_servo": t_servo, "servo_states": servo_states} 

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def process_data(data, cutoff=0, target_x_offset=0):

    # Find the first index where dx is > 0.5 m/s indicating start of movement
    dx_abs = np.abs(np.diff(data["position"][:, 0], prepend=np.zeros(1))) / 0.01
    movement_start_idx = np.argmax((dx_abs > 0.2) & (data["position"][:, 2] > 1.25)) - 500
    if movement_start_idx < 0:
        movement_start_idx = 0
    flight_start_idx = 0 #np.argmax(data["position"][:,2] > 1.5) - 10

    data["target"][:, 0] += target_x_offset

    target_pos = np.zeros(3)
    target_pos[0] = np.mean(data["target"][:, 0])
    target_pos[1] = np.mean(data["target"][:, 1])



    t_offset = data["t_pose"][movement_start_idx]
    data["t_ref"] = data["t_ref"] - t_offset
    data["t_target"] = data["t_target"] - t_offset
    data["t_contact"] = data["t_contact"] - t_offset
    data["t_touch"] = data["t_touch"] - t_offset
    data["t_state_machine"] = data["t_state_machine"] - t_offset
    if len(data["t_servo"]) > 0:
        data["t_servo"] = data["t_servo"] - t_offset
    data["t_pose"] = data["t_pose"][flight_start_idx:] - t_offset
    data["position"] = data["position"][flight_start_idx:, :] - target_pos
    data["ref_position"] = data["ref_position"] - target_pos
    data["yaw"] = (data["yaw"][flight_start_idx:])
    data["target"] = data["target"] - target_pos
    data["target_yaw"] = normalize_angle(np.mean(data["target_yaw"], axis=0)) * np.ones_like(data["target_yaw"])
    
    return data

def draw_state_backgrounds(ax, t_state_machine, state_machine, t_min=None, t_max=None, label=True):
    """Draw colored background spans and vertical transition lines for state-machine states."""
    if len(state_machine) == 0:
        return

    transitions = np.where(np.diff(state_machine) != 0)[0]
    seg_starts = np.concatenate(([0], transitions + 1))
    seg_ends = np.concatenate((transitions + 1, [len(state_machine)]))

    for s_start, s_end in zip(seg_starts, seg_ends):
        state_val = int(state_machine[s_start])
        color_key = STATE_INT_TO_COLOR.get(state_val, 'initcolor')
        color = STATE_COLORS[color_key]
        t0 = t_state_machine[s_start]
        t1 = t_state_machine[min(s_end, len(t_state_machine) - 1)]

        if t_min is not None:
            t0 = max(t0, t_min)
        if t_max is not None:
            t1 = min(t1, t_max)
        if t0 >= t1:
            continue

        text_color = 'white' if color_key == 'graspcolor' else COLORS['dark_grey']
        ax.axvspan(t0, t1, color=color, alpha=0.15, zorder=0)

        if label:
            name = STATE_INT_TO_NAME.get(state_val, '')
            label_x = (t0 + t1) / 2
            
            # Find all other spans already to be labeled (store in ax if not exists)
            if not hasattr(ax, '_label_positions'):
                ax._label_positions = []
            
            min_separation = 0.10 * (t_max - t_min if (t_max is not None and t_min is not None) else 1.0)
            base_y = 1.02
            max_offset = 0.20
            n_attempts = 5
            for n in range(n_attempts):
                offset = (n // 2) * 0.16  # stagger every other, 0, +, -, +, - etc.
                sign = (-1)**n
                y = base_y + sign * offset
                too_close = False
                for other_x, other_y in ax._label_positions:
                    if abs(other_x - label_x) < min_separation and abs(other_y - y) < 0.06:
                        too_close = True
                        break
                if not too_close:
                    break
            else:
                y = base_y  # fallback if all else fails

            ax.text(label_x, y, name,
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=7,
                    color=COLORS['dark_grey'])
            ax._label_positions.append((label_x, y))
             

    for idx in transitions:
        t_line = t_state_machine[idx + 1]
        if t_min is not None and t_line < t_min:
            continue
        if t_max is not None and t_line > t_max:
            continue
        ax.axvline(t_line, color=COLORS['dark_grey'], linewidth=0.5,
                   linestyle='--', alpha=0.5, zorder=1)

def make_time_series_plot(data, end_times):
    fig = plt.figure(figsize=7 * np.array([1.2, 1.3]))
    gs = gridspec.GridSpec(2, 1, hspace=0.1, wspace=0.01,
                           left=0.15, right=0.99, top=0.975, bottom=0.15) 
    ax1 = fig.add_subplot(gs[0])
    #ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[1], sharex=ax1)
    axs = [ax1,
           #ax2,
           ax3]

    t_end = max(end_times) + 1

    axs[0].set_ylim([-2.2, 1.2])

    offset1 = -0.15
    offset2 = -0.15
    for i, d in enumerate(data):

        # Find end index
        end_idx = np.argmax(d["t_pose"] > end_times[i])
        start_idx = np.argmax(d["t_pose"] > start_times[i])

        # Extract position and yaw
        position =  d["position"][start_idx:end_idx, :]
        yaw = d["yaw"][start_idx:end_idx]

        # Sliding window average
        window_size = 150  # Adjust the window size as needed
        yaw = np.convolve(yaw, np.ones(window_size)/window_size, mode='valid')
        #position[:, 0] = np.convolve(position[:, 0], np.ones(window_size)/window_size, mode='valid')

        # Find first contact index
        contacts = (d["state_machine"] == 2)
        contact_idx = np.argmax(contacts)
        # Find index of position closest to contact
        contact_time = d["t_state_machine"][contact_idx]
        pose_idx = np.argmin(np.abs(d["t_pose"] - contact_time))

        if "HBar" in paths[i]:
            axs[0].plot(d["t_pose"][start_idx:end_idx]-d["t_pose"][start_idx],
            position[:, 0], label=r"\$x\$", color=COLORS["color_x"], linestyle="--", alpha=0.4)
        else:
            axs[0].plot(d["t_pose"][start_idx:end_idx]-d["t_pose"][start_idx],
                        position[:, 0], label=r"\$x\$", color=COLORS["color_x"], alpha=0.4)
        
        axs[0].plot(d["t_pose"][end_idx]-d["t_pose"][start_idx],
                    position[-1, 0], marker="o", color=COLORS["dark_grey"],
                    markersize=8, label="Perched", zorder=10, alpha=0.8)
    
         # up/down alternating offset
        """if position[-1, 0] < 0:
            offset1 *= -1
            ymin = -1.8 + offset1
            ymax = 0

            axs[0].text(d["t_pose"][end_idx]-d["t_pose"][start_idx],
                        ymin - 0.05,
                        f"{roman_numerals[i]}",
                        color=COLORS["dark_grey"], fontsize=10,
                        va="top",
                        ha="center")

            # Normalize to [0,1] range inside the axes
            y0, y1 = axs[0].get_ylim()
            ymin_norm = (ymin - y0) / (y1 - y0)
            ymax_norm = (ymax - y0) / (y1 - y0)
            axs[0].axvline(d["t_pose"][end_idx]-d["t_pose"][start_idx],
                           ymin=ymin_norm,
                           ymax=ymax_norm,
                           color=COLORS["dark_grey"],
                           linestyle="--", alpha=0.5)
        else:
            offset2 *= -1
            ymin = 0
            ymax = 0.8 + offset2

            axs[0].text(d["t_pose"][end_idx]-d["t_pose"][start_idx],
                        ymax + 0.05,
                        f"{roman_numerals[i]}",
                        color=COLORS["dark_grey"], fontsize=10,
                        va="bottom",
                        ha="center")
            # Normalize to [0,1] range inside the axes
            y0, y1 = axs[0].get_ylim()
            ymin_norm = (ymin - y0) / (y1 - y0)
            ymax_norm = (ymax - y0) / (y1 - y0)
            axs[0].axvline(d["t_pose"][end_idx]-d["t_pose"][start_idx],
                           ymin=ymin_norm,
                           ymax=ymax_norm,
                           color=COLORS["dark_grey"],
                           linestyle="--", alpha=0.5)
        """
        #axs[1].plot(d["t_pose"][start_idx:end_idx]-d["t_pose"][start_idx],
        #            d["position"][start_idx:end_idx,1], label=r"\$y\$", color=COLORS["color_y"], alpha=0.4)
        #axs[1].plot(d["t_pose"][end_idx]-d["t_pose"][start_idx],
        #            d["position"][end_idx, 1], marker="o", color=COLORS["dark_grey"],
        #            markersize=8, label="Perched", zorder=10, alpha=0.8)
        
        if "HBar" in paths[i]:
            axs[1].plot(d["t_pose"][start_idx+window_size//2:end_idx-window_size//2 + 1]-d["t_pose"][start_idx+window_size//2],
                        yaw * 180/np.pi, label="yaw", color=COLORS["color_yaw"], alpha=0.8, linestyle="--")
        else:
            axs[1].plot(d["t_pose"][start_idx+window_size//2:end_idx-window_size//2 + 1]-d["t_pose"][start_idx+window_size//2],
                        yaw * 180/np.pi, label="yaw", color=COLORS["color_yaw"], alpha=0.8)
        axs[1].plot(d["t_pose"][end_idx-window_size//2 + 1]-d["t_pose"][start_idx+window_size//2],
                    yaw[-1] * 180/np.pi, marker="o", color=COLORS["dark_grey"],
                    markersize=8, label="Perched", zorder=10, alpha=0.8)    

    axs[0].set_ylabel(r"\$x\$ [m]")
    axs[0].set_yticks(np.linspace(-2.0, 1.0, 4, endpoint=True))

    #axs[1].set_ylabel(r"\$y\$ [m]")
    #axs[1].set_yticks(np.linspace(-1.0, 1.0, 3, endpoint=True))
    axs[1].set_ylim([-32, 32])
    axs[1].set_yticks([-30, -15, 0, 15, 30])
    axs[1].set_xticks(np.linspace(0, 100, 11, endpoint=True))
    axs[1].set_ylabel(r"Yaw [\$^\circ\$]")
    axs[1].set_xlabel(r"Time [s]")

        
    t_target = np.array([0, t_end])
    axs[1].plot(t_target, np.zeros_like(t_target), linestyle="--", label=r"target yaw", color="black")
    axs[0].plot(t_target, np.zeros_like(t_target), linestyle="--", label=r"target \$x\$", color="black")
    legend1_handles = [
        Line2D([0], [0], color=COLORS["color_x"], linestyle="-", label="Cylinder"),
        Line2D([0], [0], color=COLORS["color_x"], linestyle="--", label="T-Bar"),
    ]
    legend2_handles = [
        Line2D([0], [0], color=COLORS["color_yaw"], linestyle="-", label="Cylinder"),
        Line2D([0], [0], color=COLORS["color_yaw"], linestyle="--", label="T-Bar"),
    ]
    axs[0].legend(handles=legend1_handles, loc="lower right", fontsize=20, framealpha=0.8)
    axs[1].legend(handles=legend2_handles, loc="upper right", fontsize=20, framealpha=0.8)

    axs[0].set_xlim([0, 72])#t_end - min(start_times)])
    plt.setp(axs[0].get_xticklabels(), visible=False)
    #plt.setp(axs[1].get_xticklabels(), visible=False)

    xlabelpad = 20
    ylabelpad = 25
    tickpad = 20

    axs[0].tick_params(axis='both', pad=tickpad)
    #axs[1].tick_params(axis='both', pad=tickpad)
    axs[1].tick_params(axis='both', pad=tickpad)
   
    axs[0].yaxis.labelpad = ylabelpad
    axs[0].xaxis.labelpad = xlabelpad
    #axs[1].yaxis.labelpad = ylabelpad
    #axs[1].xaxis.labelpad = xlabelpad
    axs[1].yaxis.labelpad = ylabelpad
    axs[1].xaxis.labelpad = xlabelpad

    return fig

def make_top_view_plot(data, end_times):
    fig = plt.figure(figsize=6 * np.array([1, 2.6]))
    gs = gridspec.GridSpec(4, 1, hspace=0.05, wspace=0.25,
                           left=0.25, right=0.98, top=1.0, bottom=0.075) 

    ax_top_view1 = fig.add_subplot(gs[0])
    ax_top_view2 = fig.add_subplot(gs[1], sharey=ax_top_view1)
    ax_top_view3 = fig.add_subplot(gs[2], sharey=ax_top_view1)
    ax_top_view4 = fig.add_subplot(gs[3], sharey=ax_top_view1)
    axs_top_view = [ax_top_view1, ax_top_view2, ax_top_view3, ax_top_view4]

    t_end = max(end_times) + 1

    for i, d in enumerate(data):
        # Find end index
        end_idx = np.argmax(d["t_pose"] > end_times[i])
        start_idx = np.argmax(d["t_pose"] > start_times[i])
        # Find first contact index
        contacts = (d["state_machine"] == 2)
        contact_idx = np.argmax(contacts)
        # Find index of position closest to contact
        contact_time = d["t_state_machine"][contact_idx]
        pose_idx = np.argmin(np.abs(d["t_pose"] - contact_time))

        if i < 4:
            axs_top_view[0].plot(d["position"][start_idx:end_idx,0], d["position"][start_idx:end_idx,1],
                            color=COLORS["delft_blue"], alpha=0.8)
            axs_top_view[0].plot(d["position"][end_idx,0], d["position"][end_idx,1], marker="o", color=COLORS["dark_grey"],
                        markersize=8, label="Perched", zorder=101, alpha=0.8)
        elif i < 8:
            axs_top_view[1].plot(d["position"][start_idx:end_idx,0], d["position"][start_idx:end_idx,1],
                color=COLORS["delft_blue"], alpha=0.8)
            axs_top_view[1].plot(d["position"][end_idx,0], d["position"][end_idx,1], marker="o", color=COLORS["dark_grey"],
                        markersize=8, label="Perched", zorder=101, alpha=0.8)
        elif i < 12:
            axs_top_view[2].plot(d["position"][start_idx:end_idx,0], d["position"][start_idx:end_idx,1],
                color=COLORS["delft_blue"], alpha=0.8)
            axs_top_view[2].plot(d["position"][end_idx,0], d["position"][end_idx,1], marker="o", color=COLORS["dark_grey"],
                        markersize=8, label="Perched", zorder=101, alpha=0.8)
        else:
            axs_top_view[3].plot(d["position"][start_idx:end_idx,0], d["position"][start_idx:end_idx,1],
                color=COLORS["delft_blue"], alpha=0.8)
            axs_top_view[3].plot(d["position"][end_idx,0], d["position"][end_idx,1], marker="o", color=COLORS["dark_grey"],
                        markersize=8, label="Perched", zorder=101, alpha=0.8)
        
    xlabelpad = 25
    ylabelpad = 45
    tickpad = 25

    for j in range(4):
        rect = patches.Rectangle(
            (-0.25, -10),      # bottom-left corner (x, y)
            0.5, 20,        # width, height
            linewidth=0,
            facecolor=COLORS["color_trunk"],
            alpha=0.5,
            zorder=100
        )
        axs_top_view[j].add_patch(rect)
        axs_top_view[j].yaxis.labelpad = ylabelpad
        axs_top_view[j].xaxis.labelpad = xlabelpad
        axs_top_view[j].tick_params(axis='both', pad=tickpad)
        axs_top_view[j].set_yticks(np.linspace(-1.0, 1.0, 3, endpoint=True))
        axs_top_view[j].set_ylabel(r"\$y\$ [m]")
        axs_top_view[j].set_ylim([-1.3, 1.3])
        axs_top_view[j].set_xlim([-2.0, 1.25])
        axs_top_view[j].set_xticks(np.linspace(-2.0, 1.0, 4, endpoint=True))
        axs_top_view[j].set_xticklabels([])
        axs_top_view[j].set_aspect('equal')

        box_x = 0.035      # x position
        box_y = 0.24       # y position (top-left style positioning)
        box_width = 0.2    # width as fraction of axes width
        box_height = 0.2   # height as fraction of axes height
        # Add the rounded rectangle with independent width/height control
        rounded_box = FancyBboxPatch(
            (box_x, box_y - box_height), box_width, box_height,  # subtract height for top-left positioning
            boxstyle="round,pad=0.01",
            facecolor='grey',
            alpha=0.3,
            edgecolor='none',
            transform=axs_top_view[j].transAxes
        )
        axs_top_view[j].add_patch(rounded_box)

        axs_top_view[j].text(-1.8, -1.05, rf"{roman_numerals[j]}")
    
    axs_top_view[-1].set_xlabel(r"\$x\$ [m]")
    axs_top_view[-1].set_xticklabels(np.linspace(-2.0, 1.0, 4, endpoint=True))

    return fig

def make_3d_plot(data, end_times, trial_names):
    fig = plt.figure(figsize=10 * np.array([1.1, 1.0]))
    gs = gridspec.GridSpec(3, 2, hspace=0.01, wspace=0.2,
                           height_ratios=[1, 1, 0.025]) 
    axs = [
        fig.add_subplot(gs[0, 0], projection='3d'),
        fig.add_subplot(gs[1, 0], projection='3d'),
        fig.add_subplot(gs[0, 1], projection='3d'),
        fig.add_subplot(gs[1, 1], projection='3d'),
    ]

    # Add legend axis spanning the bottom row
    legend_ax = fig.add_subplot(gs[2, :])

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

        xlabelpad = 10
        ylabelpad = 10
        zlabelpad = 2
        tickpad = 1

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
        box_width = 0.62   # width as fraction of axes width
        box_height = 0.15  # height as fraction of axes height
        box_x = 0.25       # x position
        box_y = 0.95       # y position (top-left style positioning)
        
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
        txt = axs[i].text2D(0.275, 0.94, trial_names[i], 
                    transform=axs[i].transAxes, 
                    fontsize=12,  
                    ha="left", va="top")
        txt.set_in_layout(False)


    legend_ax.clear()  # Clear any existing content

    # Create custom alpha representation
    for j in range(100):
        alpha_val = (j + 1) / 100.0  # From 0.01 to 1.0
        legend_ax.axvspan(j, j+1, color=COLORS["delft_blue"], alpha=alpha_val)

    # Customize the legend axis
    legend_ax.set_ylim(0, 1)
    legend_ax.set_xlim(0, 100)
    legend_ax.set_xlabel(r'Trial Progression [%]', fontsize=12)

    #legend_ax.yaxis.tick_right()
    #legend_ax.yaxis.set_label_position("right")

    # Set ticks
    legend_ax.set_xticks(np.linspace(0, 100, 5, endpoint=True))
    legend_ax.set_yticks([])

    # Remove unnecessary spines
    legend_ax.spines['bottom'].set_visible(True)
    legend_ax.spines['top'].set_visible(False)
    legend_ax.spines['left'].set_visible(False)
    legend_ax.spines['right'].set_visible(False)

    legend_ax.tick_params(axis='x', pad=xlabelpad)
    legend_ax.xaxis.labelpad = xlabelpad
    
    return fig

def make_contact_plot(data, end_time, index):
    fig = plt.figure(figsize=7 * np.array([1.75, 0.75]))
    gs = gridspec.GridSpec(2, 1, hspace=0.2, wspace=0.15,
                           left=0.08, right=0.98, top=0.95, bottom=0.15,
                           height_ratios=[3, 1]) 
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
    axs[1].set_yticks([0, 200, 400])
    axs[1].set_yticklabels([])
    axs[1].set_ylabel(r"Value [-]")
    axs[1].set_xlabel(r"Time [s]")

    #axs[2].step(d["t_state_machine"], d["state_machine"], where='post', color=COLORS["dark_grey"])

    axs[0].set_ylabel(r"Contact \$\in\mathcal{B}\$")
    axs[0].set_ylim([0.5, 9.5])
    axs[0].set_yticks(np.linspace(1, 9, num=9, endpoint=True))
    axs[0].set_yticklabels(
        [rf"\$\mathcal{{C}}_{{{i}}}\$" for i in range(1, 10)],
    )
    axs[0].set_xlim([0, t_end - 20])
    plt.setp(axs[0].get_xticklabels(), visible=False)

    xlabelpad = 20
    ylabelpad = 55
    xtickpad = 20
    ytickpad = 0

    axs[0].tick_params(axis='x', pad=xtickpad)
    axs[0].tick_params(axis='y', pad=ytickpad)
    axs[1].tick_params(axis='x', pad=xtickpad)
    axs[1].tick_params(axis='y', pad=ytickpad)
   
    axs[0].yaxis.labelpad = -0.7 * ylabelpad
    axs[0].xaxis.labelpad = xlabelpad
    axs[1].yaxis.labelpad = ylabelpad
    axs[1].xaxis.labelpad = xlabelpad

    #axs[2].set_yticks([0, 1, 2, 3, 4, 5, 6, 7])
    #axs[2].set_ylim([-0.5, 7.5])

    return fig

def make_full_trial_overview(data, end_time, index):
    fig = plt.figure(figsize=6 * np.array([1.5, 1.0]))
    gs = gridspec.GridSpec(3, 1, hspace=0.15, wspace=0.15,
                           left=0.10, right=0.98, top=0.93, bottom=0.10,
                           height_ratios=[1, 1, 1])
    ax_pos = fig.add_subplot(gs[0])
    ax_fingers = fig.add_subplot(gs[1], sharex=ax_pos)
    ax_contact = fig.add_subplot(gs[2], sharex=ax_pos)

    start_times[index] -= 2.0

    d = data[index]

    start_idx = np.argmax(d["t_pose"] > start_times[index])
    t_start = d["t_pose"][start_idx]

    t_pose = d["t_pose"][:]
    t_ref = d["t_ref"][:]
    t_contact = d["t_contact"][:]

    # --- Top: x, y, z position + reference ---
    color_z = '#6C3FC5'
    ax_pos.plot(t_pose, d["position"][:, 0],
                color=COLORS["color_x"], label=r"\$x\$")
    ax_pos.plot(t_ref, d["ref_position"][:, 0],
                color=COLORS["color_x"], linestyle="--", alpha=0.5, label=r"\$x_\mathrm{ref}\$")
    ax_pos.plot(t_pose, d["position"][:, 1],
                color=COLORS["color_y"], label=r"\$y\$")
    ax_pos.plot(t_ref, d["ref_position"][:, 1],
                color=COLORS["color_y"], linestyle="--", alpha=0.5, label=r"\$y_\mathrm{ref}\$")
    ax_pos.plot(t_pose, d["position"][:, 2],
                color=color_z, label=r"\$z\$")
    ax_pos.plot(t_ref, d["ref_position"][:, 2],
                color=color_z, linestyle="--", alpha=0.5, label=r"\$z_\mathrm{ref}\$")
    ax_pos.set_ylabel(r"Position [\$m\$]")
    ax_pos.legend(loc="lower right", ncol=3, fontsize=7, framealpha=0.8)    

    # --- Middle: finger opening values ---
    finger_colors = [COLORS["color_x"], COLORS["color_y"], COLORS["color_yaw"]]
    if len(d["t_servo"]) > 0:
        t_servo = d["t_servo"][:]
        for f in range(d["servo_states"].shape[1]):
            ax_fingers.plot(t_servo, d["servo_states"][:, f],
                            color=finger_colors[f % len(finger_colors)],
                            label=f"Finger {f+1}")
    ax_fingers.set_ylim([-0.05, 1.05])
    ax_fingers.set_yticks([0.0, 0.5, 1.0])
    ax_fingers.set_ylabel(r"Opening \$\alpha\$ [-]")
    ax_fingers.legend(
        loc="lower left",
        ncol=3,
        fontsize=12,
        framealpha=0.8,
        borderaxespad=0.75,
        #handletextpad=1.5,           # more spacing between marker and label
        columnspacing=3.0            # more spacing between columns
    )

    # --- Bottom: binary contact for 9 sensors ---
    num_contact_channels = min(d["contact"].shape[1], 9)
    contacts_scaled = (np.linspace(1, num_contact_channels, num=num_contact_channels, endpoint=True)
                       * d["contact"][:, :num_contact_channels])
    ax_contact.plot(t_contact, contacts_scaled,
                    linewidth=0, marker="o", markersize=4,
                    alpha=0.6, color=COLORS["contact"])
    ax_contact.set_ylabel(r"Contact \$\in\mathcal{B}\$")
    ax_contact.set_ylim([0.5, num_contact_channels + 0.5])
    ax_contact.set_yticks(np.linspace(1, num_contact_channels, num=num_contact_channels, endpoint=True))
    ax_contact.set_yticklabels(
        [rf"\$\mathcal{{C}}_{{{i}}}\$" for i in range(1, num_contact_channels + 1)],
    )
    ax_contact.set_xlabel(r"Time [s]")


    # --- State-machine background shading on all three axes ---
    t_sm = d["t_state_machine"][:]
    sm = d["state_machine"][:]
    for i_ax, ax in enumerate([ax_pos, ax_fingers, ax_contact]):
        draw_state_backgrounds(ax, t_sm, sm,
                               t_min=t_start, t_max=end_time,
                               label=(i_ax == 0))
    # Set x tick label pad and x tick pad for all axes
    xtickpad = 10  # or any fixed value you prefer
    ytickpad = 15       # or any fixed value you prefer
    for ax in [ax_pos, ax_fingers, ax_contact]:
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.tick_params(axis='y', pad=ytickpad)
        ax.tick_params(axis='x', pad=xtickpad)

    ax_contact.yaxis.labelpad = -40

    ax_pos.set_xlim([t_start, end_time])
    plt.setp(ax_pos.get_xticklabels(), visible=False)
    plt.setp(ax_fingers.get_xticklabels(), visible=False)

    return fig

def main():

    target_x_offsets = [0, 0, 0, 0, # Cylinder no offset
                        0, 0, 0, 0, # Cylinder yaw offset
                        0, 0, 0, 0, # Cylinder positional offset
                        0, 0, 0, 0, # Cylinder incline offset
                        0, 0,             # Cylinder combined offset
                        0.0, 0.0, 0.0, 0.0, # H-Bar no offset
                        0.0, 0.15, 0.0, 0.0, # H-Bar pos offset
                        ]
    data = np.array([process_data(rosbag2data(p), target_x_offset=target_x_offsets[i]) for i, p in enumerate(paths)])
    end_times = np.array([43.0, 46.0, 35.0, 33.0, # Cylinder no offset
                          33.5, 43.4, 52.0, 46.5, # Cylinder yaw offset
                          32.0, 38.0, 37.0, 73.0, # Cylinder positional offset
                          66.0, 60.0, 65.0, 65.0, # Cylinder incline offset
                          41.0, 42.0,             # Cylinder combined offset
                          75.0, 110.0, 73.2, 76.1, # H-Bar no offset
                          65.5, 49.2, 40.0, 55.0, # H-Bar pos offset
                          ])
    
    # Create and save time series plots
    fig = make_time_series_plot(data, end_times)
    fig.savefig("time_series_plot.svg")

    # Create and save top view plots
    #fig = make_top_view_plot(data, end_times)
    #fig.savefig("top_view_plot.svg")

    # Create and save contact threshold plot
    #fig = make_contact_plot(data, 54, -1)
    #fig.savefig("contact_plot.svg")
    
    
    #fig = make_full_trial_overview(data, 34.0, 0)
    #fig.savefig("full_trial_overview.svg")

    # Create and save 3D plots
    #fig = make_3d_plot(data[[3, 7, 11, 13]],
    #                   end_times[[3, 7, 11, 13]],
    #                   trial_names=[r"No Offset (III):\\  (\$\SI{0.0}{\meter}\$, \$\SI{0.0}{\meter}\$, \$\SI{0.0}{\degree}\$)",
    #                                r"Rotational Offset (VII):\\ (\$\SI{0.0}{\meter}\$, \$\SI{0.0}{\meter}\$, \$\SI{20}{\degree}\$)",
    #                                r"Positional Offset (XI):\\ (\$\SI{-0.6}{\meter}\$, \$\SI{0.0}{\meter}\$, \$\SI{0.0}{\degree}\$)",
    #                                r"Combined Offset (XIII):\\ (\$\SI{-0.25}{\meter}\$, \$\SI{-0.25}{\meter}\$, \$\SI{-15}{\degree}\$)"])
    #fig.savefig(f"3d_plot.svg", bbox_inches='tight', pad_inches=0.35,
    #    transparent=False)

if __name__=="__main__":
    main()



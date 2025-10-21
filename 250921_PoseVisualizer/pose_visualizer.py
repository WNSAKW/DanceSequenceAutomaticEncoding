import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
import ast
import os
import glob
from PIL import Image, ImageTk

# --- Constants ---
# The connection between toes (31, 32) has been removed.
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19),
    (15, 21), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29),
    (27, 31), (24, 26), (26, 28), (28, 30), (28, 32)
]
FOOT_INDICES = [27, 28, 29, 30, 31, 32]
HIP_INDICES = [23, 24]
IMAGE_DISPLAY_SIZE = (400, 400)

# --- Data Processing Functions ---
def load_pose_data(filepath):
    try:
        df = pd.read_csv(filepath)
        all_frames = []
        for _, row in df.iterrows():
            frame_points = [ast.literal_eval(row[f'p{i}']) for i in range(33)]
            all_frames.append(np.array(frame_points, dtype=float))
        return np.array(all_frames)
    except Exception as e:
        messagebox.showerror("Read Error", f"Could not read or parse CSV file: {e}")
        return None

def normalize_by_first_frame(all_frames):
    """Normalizes all frames based on the floor of the first frame, ensuring no negative height."""
    if len(all_frames) == 0:
        return all_frames

    first_frame = all_frames[0]
    
    # Determine offsets from the first frame
    foot_y_coords = first_frame[FOOT_INDICES, 1]
    floor_y = np.max(foot_y_coords)
    hip_center = np.mean(first_frame[HIP_INDICES], axis=0)
    center_x, _, center_z = hip_center
    
    # Create a single translation vector to apply to all frames
    translation = np.array([-center_x, -floor_y, -center_z])
    
    # Apply the same transformation and invert Y-axis for all frames
    temp_frames = []
    for frame in all_frames:
        new_frame = frame + translation
        new_frame[:, 1] *= -1 # Invert Y-axis to make it upright
        temp_frames.append(new_frame)
    
    processed_frames = np.array(temp_frames)

    # Final check: find the global minimum height and shift everything up
    # This ensures the lowest point of the entire animation is at y=0.
    min_height = processed_frames[:, :, 1].min()
    processed_frames[:, :, 1] -= min_height

    return processed_frames

# --- Main Application Class ---
class PoseVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("2D/3D Pose Visualizer (Fixed Floor)")
        self.root.geometry("1100x700")
        self.ani = None

        # --- Layout ---
        top_frame = tk.Frame(root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        main_container = tk.Frame(root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.plot_frame = tk.Frame(main_container, relief=tk.SUNKEN, borderwidth=1)
        self.plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_frame = tk.Frame(main_container, relief=tk.SUNKEN, borderwidth=1, width=IMAGE_DISPLAY_SIZE[0])
        self.image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        self.image_frame.pack_propagate(False)

        # --- Widgets ---
        self.select_button = tk.Button(top_frame, text="Select CSV File and Start", command=self.on_select_file)
        self.select_button.pack(side=tk.LEFT)
        self.info_label = tk.Label(top_frame, text="Please select a file to begin.")
        self.info_label.pack(side=tk.LEFT, padx=10)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.image_label = tk.Label(self.image_frame, text="Image Display Area", bg='gray')
        self.image_label.pack(fill=tk.BOTH, expand=True)

    def on_select_file(self):
        filepath = filedialog.askopenfilename(title="Select a CSV file", initialdir=".", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
        if not filepath: return

        csv_filename = os.path.basename(filepath)
        track_name = os.path.splitext(csv_filename)[0]
        image_dir = os.path.join(os.path.dirname(filepath), track_name)

        if not os.path.isdir(image_dir):
            messagebox.showwarning("Path Error", f"Could not find image folder:\n{image_dir}")
            return

        raw_frames = load_pose_data(filepath)
        if raw_frames is None: return
        
        self.all_frames = normalize_by_first_frame(raw_frames)
        self.image_files = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))

        if len(self.all_frames) != len(self.image_files):
            messagebox.showwarning("Count Mismatch", f"CSV frames ({len(self.all_frames)}) vs. images ({len(self.image_files)}).")

        self.info_label.config(text=f"Loaded: {csv_filename} ({len(self.all_frames)} frames)")
        self.start_animation()

    def start_animation(self):
        if self.ani: self.ani.event_source.stop()

        self.ax.clear()
        
        # --- Axis limits setup ---
        all_points = self.all_frames.reshape(-1, 3)
        x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
        y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max() # y_min is now 0
        z_min, z_max = all_points[:, 2].min(), all_points[:, 2].max()

        # Center the horizontal plane (X and Z axes)
        x_range = x_max - x_min
        z_range = z_max - z_min
        horizontal_max_range = max(x_range, z_range) / 2.0
        mid_x = (x_max + x_min) / 2
        mid_z = (z_max + z_min) / 2
        self.ax.set_xlim(mid_x - horizontal_max_range, mid_x + horizontal_max_range)
        self.ax.set_ylim(mid_z - horizontal_max_range, mid_z + horizontal_max_range)

        # Set the height axis (Y-data on plot's Z-axis) to start from 0
        self.ax.set_zlim(0, y_max * 1.1) # Add 10% padding at the top

        self.ax.set_xlabel("X-axis"); self.ax.set_ylabel("Z-axis (Depth)"); self.ax.set_zlabel("Y-axis (Height)")
        self.ax.view_init(elev=15, azim=-75)

        self.scatter = self.ax.scatter([], [], [], c='red', marker='o', s=20)
        self.lines = [self.ax.plot([], [], [], 'b-')[0] for _ in POSE_CONNECTIONS]
        self.frame_text = self.ax.text2D(0.05, 0.95, '', transform=self.ax.transAxes)

        self.ani = animation.FuncAnimation(self.fig, self._update_animation, frames=len(self.all_frames), interval=33, blit=False, repeat=True)
        self.canvas.draw()

    def _update_animation(self, frame_num):
        points = self.all_frames[frame_num]
        self.scatter._offsets3d = (points[:, 0], points[:, 2], points[:, 1])
        for i, (p1_idx, p2_idx) in enumerate(POSE_CONNECTIONS):
            p1, p2 = points[p1_idx], points[p2_idx]
            self.lines[i].set_data([p1[0], p2[0]], [p1[2], p2[2]])
            self.lines[i].set_3d_properties([p1[1], p2[1]])
        self.frame_text.set_text(f'Frame: {frame_num + 1}/{len(self.all_frames)}')

        if frame_num < len(self.image_files):
            img_path = self.image_files[frame_num]
            try:
                img = Image.open(img_path)
                img.thumbnail(IMAGE_DISPLAY_SIZE, Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image=img)
                self.image_label.config(image=photo)
                self.image_label.image = photo
            except Exception as e:
                self.image_label.config(text=f"Could not load image\n{os.path.basename(img_path)}")
        
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = PoseVisualizerApp(root)
    root.mainloop()
#!/usr/bin/env python3
"""
Standalone script to create 3D drone trajectory animation from Dymos database files.
Requires: dymos_solution.db and dymos_simulation.db files

Usage:
    python animate_trajectory.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import openmdao.api as om
from pathlib import Path

# ============================================================================
# CONFIGURATION - Modify these parameters as needed
# ============================================================================

# Database file paths
SOLUTION_DB = 'dymos_solution_fifth.db'  # Change to your solution DB filename
SIMULATION_DB = 'dymos_simulation_fifth.db'  # Change to your simulation DB filename

# Mission parameters
X_PAYLOAD = 500.0  # m - Payload X coordinate
Y_FINAL = 500.0    # m - Final Y coordinate
Z_FINAL = 100.0    # m - Cruise altitude

# Phase names (must match your Dymos trajectory)
PHASE_NAMES = ['climb1', 'cruise1', 'descent1', 'climb2', 'cruise2', 'descent2']

# Animation settings
SKIP_FRAMES = 2      # Skip frames for speed (lower = smoother but slower)
INTERVAL_MS = 30     # Milliseconds between frames (lower = faster)
SAVE_ANIMATION = False  # Set to True to save animation
OUTPUT_FORMAT = 'mp4'   # 'mp4' or 'gif'
OUTPUT_FILENAME = 'drone_trajectory_animation'

# ============================================================================
# LOAD DATA FROM DATABASE FILES
# ============================================================================

def load_trajectory_data(solution_file, simulation_file, phase_names):
    """
    Load trajectory data from Dymos database files.
    
    Parameters
    ----------
    solution_file : str
        Path to dymos_solution.db file
    simulation_file : str
        Path to dymos_simulation.db file
    phase_names : list
        List of phase names in trajectory
        
    Returns
    -------
    dict
        Dictionary containing trajectory data
    """
    print(f"\n{'='*70}")
    print(f"LOADING TRAJECTORY DATA")
    print(f"{'='*70}")
    
    # Check if files exist
    if not Path(solution_file).exists():
        raise FileNotFoundError(f"Solution file not found: {solution_file}")
    if not Path(simulation_file).exists():
        raise FileNotFoundError(f"Simulation file not found: {simulation_file}")
    
    # Load solution data
    print(f"Loading solution from: {solution_file}")
    sol_reader = om.CaseReader(solution_file)
    sol = sol_reader.get_case('final')
    
    # Load simulation data
    print(f"Loading simulation from: {simulation_file}")
    sim_reader = om.CaseReader(simulation_file)
    sim = sim_reader.get_case('final')
    
    # Extract data for each phase
    data = {
        'x_sol': {},
        'y_sol': {},
        'z_sol': {},
        't_sol': {},
        'x_sim': {},
        'y_sim': {},
        'z_sim': {},
        't_sim': {},
    }
    
    print(f"\nExtracting data for {len(phase_names)} phases:")
    for phase in phase_names:
        print(f"  - {phase}")
        # Solution data
        data['x_sol'][phase] = sol.get_val(f'traj.{phase}.timeseries.x')
        data['y_sol'][phase] = sol.get_val(f'traj.{phase}.timeseries.y')
        data['z_sol'][phase] = sol.get_val(f'traj.{phase}.timeseries.z')
        data['t_sol'][phase] = sol.get_val(f'traj.{phase}.timeseries.time')
        
        # Simulation data
        data['x_sim'][phase] = sim.get_val(f'traj.{phase}.timeseries.x')
        data['y_sim'][phase] = sim.get_val(f'traj.{phase}.timeseries.y')
        data['z_sim'][phase] = sim.get_val(f'traj.{phase}.timeseries.z')
        data['t_sim'][phase] = sim.get_val(f'traj.{phase}.timeseries.time')
    
    # Concatenate all phases into continuous trajectory
    data['x_traj'] = np.concatenate([data['x_sim'][ph].flatten() for ph in phase_names])
    data['y_traj'] = np.concatenate([data['y_sim'][ph].flatten() for ph in phase_names])
    data['z_traj'] = np.concatenate([data['z_sim'][ph].flatten() for ph in phase_names])
    data['t_traj'] = np.concatenate([data['t_sim'][ph].flatten() for ph in phase_names])
    
    # Calculate payload pickup index (end of descent1)
    descent1_idx = 0
    for i, phase in enumerate(phase_names):
        if phase == 'descent1':
            for j in range(i + 1):
                descent1_idx += len(data['x_sim'][phase_names[j]])
            break
    
    data['descent1_end_idx'] = descent1_idx
    data['phase_names'] = phase_names
    
    print(f"\n{'='*70}")
    print(f"DATA LOADED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"Total trajectory points: {len(data['x_traj'])}")
    print(f"Total mission time: {data['t_traj'][-1]:.2f} seconds")
    print(f"Payload pickup at index: {descent1_idx}")
    print(f"Payload pickup time: {data['t_traj'][descent1_idx-1]:.2f} seconds")
    print(f"{'='*70}\n")
    
    return data

# ============================================================================
# CREATE ANIMATION
# ============================================================================

def create_animation(data, x_payload, y_final, z_final, skip_frames=2, interval_ms=30):
    """
    Create 3D trajectory animation with drone and payload visualization.
    
    Parameters
    ----------
    data : dict
        Trajectory data dictionary
    x_payload : float
        Payload X coordinate
    y_final : float
        Final Y coordinate
    z_final : float
        Cruise altitude
    skip_frames : int
        Number of frames to skip for performance
    interval_ms : int
        Milliseconds between animation frames
        
    Returns
    -------
    fig, anim
        Matplotlib figure and animation objects
    """
    print("Creating animation...")
    
    # Extract trajectory arrays
    x_traj = data['x_traj']
    y_traj = data['y_traj']
    z_traj = data['z_traj']
    t_traj = data['t_traj']
    descent1_end_idx = data['descent1_end_idx']
    phase_names = data['phase_names']
    
    # Calculate phase boundaries for status display
    phase_boundaries = [0]
    for phase in phase_names:
        phase_boundaries.append(phase_boundaries[-1] + len(data['x_sim'][phase]))
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the full trajectory as a faint reference line
    ax.plot(x_traj, y_traj, z_traj, 'gray', alpha=0.25, linewidth=1.5, 
            linestyle='--', label='Planned Path')
    
    # Plot waypoints
    ax.plot([0], [0], [0], 's', color='green', markersize=14, 
            markeredgecolor='black', markeredgewidth=2.5, label='Start', zorder=10)
    payload_pickup_marker = ax.plot([x_payload], [0], [0], '*', color='gold', 
            markersize=28, markeredgecolor='black', markeredgewidth=2.5, 
            label='Payload', zorder=10)[0]
    ax.plot([x_payload], [y_final], [0], 's', color='red', markersize=14, 
            markeredgecolor='black', markeredgewidth=2.5, label='Destination', zorder=10)
    
    # Initialize animated elements
    trail_line, = ax.plot([], [], [], 'b-', linewidth=2.5, alpha=0.7, label='Drone Trail')
    
    # Drone representation (quadcopter)
    drone_body, = ax.plot([], [], [], 'o', color='#1f77b4', markersize=18, 
                          markeredgecolor='black', markeredgewidth=2.5, zorder=20, label='Drone')
    drone_rotor1, = ax.plot([], [], [], 'o', color='red', markersize=6, 
                            markeredgecolor='darkred', markeredgewidth=1, zorder=19)
    drone_rotor2, = ax.plot([], [], [], 'o', color='red', markersize=6, 
                            markeredgecolor='darkred', markeredgewidth=1, zorder=19)
    drone_rotor3, = ax.plot([], [], [], 'o', color='red', markersize=6, 
                            markeredgecolor='darkred', markeredgewidth=1, zorder=19)
    drone_rotor4, = ax.plot([], [], [], 'o', color='red', markersize=6, 
                            markeredgecolor='darkred', markeredgewidth=1, zorder=19)
    
    # Payload when carried
    payload_carried, = ax.plot([], [], [], '*', color='gold', markersize=22, 
                               markeredgecolor='orange', markeredgewidth=2, zorder=18)
    
    # Text annotations
    time_text = ax.text2D(0.02, 0.97, '', transform=ax.transAxes, fontsize=13, 
                         fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', 
                                  alpha=0.9, edgecolor='black', linewidth=2))
    phase_text = ax.text2D(0.02, 0.90, '', transform=ax.transAxes, fontsize=12,
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                                    alpha=0.9, edgecolor='black', linewidth=2))
    status_text = ax.text2D(0.02, 0.83, '', transform=ax.transAxes, fontsize=11,
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', 
                                    alpha=0.85, edgecolor='black', linewidth=1.5))
    
    # Set labels and title
    ax.set_xlabel('X Position (m)', fontsize=13, labelpad=12, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=13, labelpad=12, fontweight='bold')
    ax.set_zlabel('Z Position (m)', fontsize=13, labelpad=12, fontweight='bold')
    ax.set_title('Drone Payload Delivery Mission - Time-Optimal Trajectory', 
                 fontsize=15, fontweight='bold', pad=25)
    
    # Set axis limits with padding
    ax.set_xlim(-50, x_payload + 100)
    ax.set_ylim(-50, y_final + 100)
    ax.set_zlim(-10, z_final + 30)
    
    # Set viewing angle
    ax.view_init(elev=22, azim=50)
    ax.grid(True, alpha=0.4, linewidth=0.8)
    
    # Legend
    ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
             edgecolor='black', fancybox=True, shadow=True)
    
    # Animation parameters
    total_frames = len(x_traj) // skip_frames
    rotor_offset = 5  # meters
    
    def init():
        """Initialize animation"""
        trail_line.set_data([], [])
        trail_line.set_3d_properties([])
        drone_body.set_data([], [])
        drone_body.set_3d_properties([])
        for rotor in [drone_rotor1, drone_rotor2, drone_rotor3, drone_rotor4]:
            rotor.set_data([], [])
            rotor.set_3d_properties([])
        payload_carried.set_data([], [])
        payload_carried.set_3d_properties([])
        time_text.set_text('')
        phase_text.set_text('')
        status_text.set_text('')
        return (trail_line, drone_body, drone_rotor1, drone_rotor2, 
                drone_rotor3, drone_rotor4, payload_carried, 
                time_text, phase_text, status_text)
    
    def animate(frame):
        """Animation function"""
        idx = frame * skip_frames
        if idx >= len(x_traj):
            idx = len(x_traj) - 1
        
        # Update trail
        trail_line.set_data(x_traj[:idx+1], y_traj[:idx+1])
        trail_line.set_3d_properties(z_traj[:idx+1])
        
        # Current position
        x_curr, y_curr, z_curr = x_traj[idx], y_traj[idx], z_traj[idx]
        
        # Update drone body
        drone_body.set_data([x_curr], [y_curr])
        drone_body.set_3d_properties([z_curr])
        
        # Update rotors (quadcopter configuration)
        drone_rotor1.set_data([x_curr + rotor_offset], [y_curr + rotor_offset])
        drone_rotor1.set_3d_properties([z_curr + 1])
        
        drone_rotor2.set_data([x_curr - rotor_offset], [y_curr + rotor_offset])
        drone_rotor2.set_3d_properties([z_curr + 1])
        
        drone_rotor3.set_data([x_curr + rotor_offset], [y_curr - rotor_offset])
        drone_rotor3.set_3d_properties([z_curr + 1])
        
        drone_rotor4.set_data([x_curr - rotor_offset], [y_curr - rotor_offset])
        drone_rotor4.set_3d_properties([z_curr + 1])
        
        # Update time display
        time_text.set_text(f'Time: {t_traj[idx]:.1f} / {t_traj[-1]:.1f} s')
        
        # Determine current phase
        current_phase_idx = 0
        for i in range(len(phase_boundaries) - 1):
            if phase_boundaries[i] <= idx < phase_boundaries[i + 1]:
                current_phase_idx = i
                break
        
        phase_name = phase_names[current_phase_idx]
        
        # Update phase and status based on current phase
        phase_info = {
            'climb1': ('CLIMB 1', 'Ascending to cruise altitude', 'lightblue'),
            'cruise1': ('CRUISE 1', 'En route to payload location', 'lightyellow'),
            'descent1': ('DESCENT 1', 'Landing for payload pickup', 'lightcoral'),
            'climb2': ('CLIMB 2 ✓', 'Ascending with payload', 'lightgreen'),
            'cruise2': ('CRUISE 2 ✓', 'En route to destination', 'lightyellow'),
            'descent2': ('DESCENT 2 ✓', 'Landing at destination', 'lightcoral'),
        }
        
        phase, status, color = phase_info.get(phase_name, ('UNKNOWN', 'Unknown phase', 'white'))
        
        phase_text.set_text(f'Phase: {phase}')
        phase_text.set_bbox(dict(boxstyle='round,pad=0.5', facecolor=color, 
                                 alpha=0.9, edgecolor='black', linewidth=2))
        status_text.set_text(f'Status: {status}')
        
        # Handle payload pickup
        if idx >= descent1_end_idx:
            # Payload attached to drone
            payload_carried.set_data([x_curr], [y_curr])
            payload_carried.set_3d_properties([z_curr - 3])
            # Hide ground marker
            if idx == descent1_end_idx:
                payload_pickup_marker.set_visible(False)
        else:
            # No payload carried yet
            payload_carried.set_data([], [])
            payload_carried.set_3d_properties([])
        
        return (trail_line, drone_body, drone_rotor1, drone_rotor2, 
                drone_rotor3, drone_rotor4, payload_carried, 
                time_text, phase_text, status_text)
    
    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=total_frames,
                        interval=interval_ms, blit=True, repeat=True)
    
    print(f"Animation created successfully!")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Frame rate: ~{1000/interval_ms:.1f} FPS")
    
    return fig, anim

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("DRONE TRAJECTORY ANIMATION FROM DATABASE FILES")
    print("="*70)
    
    try:
        # Load data
        data = load_trajectory_data(SOLUTION_DB, SIMULATION_DB, PHASE_NAMES)
        
        # Create animation
        fig, anim = create_animation(data, X_PAYLOAD, Y_FINAL, Z_FINAL, 
                                     SKIP_FRAMES, INTERVAL_MS)
        
        # Save animation if requested
        if SAVE_ANIMATION:
            output_file = f"{OUTPUT_FILENAME}.{OUTPUT_FORMAT}"
            print(f"\nSaving animation as {OUTPUT_FORMAT.upper()}...")
            print(f"  Output file: {output_file}")
            
            if OUTPUT_FORMAT == 'mp4':
                anim.save(output_file, writer='ffmpeg', fps=30, dpi=150, 
                         bitrate=2000, extra_args=['-vcodec', 'libx264'])
            elif OUTPUT_FORMAT == 'gif':
                anim.save(output_file, writer='pillow', fps=20, dpi=100)
            else:
                print(f"  Warning: Unknown format '{OUTPUT_FORMAT}', skipping save")
            
            print(f"  Animation saved successfully!")
        
        # Display animation
        plt.tight_layout()
        plt.show()
        
        print("\n" + "="*70)
        print("ANIMATION COMPLETE")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
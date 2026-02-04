import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import openmdao.api as om
import dymos as dm
from dymos.models.atmosphere.atmos_1976 import USatm1976Comp

import sys 
import os

sys.path.append("/home/omdao/Aviary-1/")

# Import your components
from aviary.mission.sixdof.six_dof_EOM import SixDOF_EOM
from aviary.mission.sixdof.force_component_calc import ForceComponentResolver
from aviary.mission.sixdof.AeroSphereComp import AeroSphereComp
from aviary.mission.sixdof.SimplifiedForceComponentResolver import SimplifiedForceResolver
from aviary.mission.sixdof.force_component_calc_copy import ForceComponentResolver_Copy

from openmdao.utils.general_utils import set_pyoptsparse_opt
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')
if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver


class vtolODE(om.Group):
    """
    Fixed ODE that provides all necessary inputs
    """
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        
    def setup(self):
        nn = self.options['num_nodes']

        # Convert z (NED, down positive) to h (altitude, up positive)
        self.add_subsystem('altitude_calc',
                          om.ExecComp('h = -z',
                                     h={'units': 'm', 'shape': (nn,)},
                                     z={'units': 'm', 'shape': (nn,)}),
                          promotes_inputs=['z'],
                          promotes_outputs=['h'])

        # Atmosphere component
        self.add_subsystem('atm', 
                          USatm1976Comp(num_nodes=nn),
                          promotes_inputs=['*'],
                          promotes_outputs=['rho', 'sos', 'temp'])
        
        # Compute wind angles
        self.add_subsystem('wind_angles',
                          om.ExecComp(['V = (u**2 + v**2 + w**2 + 1e-8)**0.5',
                                      'alpha = arctan2(w, u + 1e-8)',
                                      'beta = arcsin(fmin(fmax(v / ((u**2 + v**2 + w**2 + 1e-8)**0.5 + 1e-8), -0.99), 0.99))'],
                                     V={'units': 'm/s', 'shape': (nn,)},
                                     alpha={'units': 'rad', 'shape': (nn,)},
                                     beta={'units': 'rad', 'shape': (nn,)},
                                     u={'units': 'm/s', 'shape': (nn,)},
                                     v={'units': 'm/s', 'shape': (nn,)},
                                     w={'units': 'm/s', 'shape': (nn,)}),
                          promotes_inputs=['u', 'v', 'w'],
                          promotes_outputs=['alpha', 'beta'])

        # Aerodynamic parameters
        self.add_subsystem('aero_params',
                          om.ExecComp(['A = pi * radius**2',
                                      'Cd_val = Cd'],
                                     A={'units': 'm**2', 'shape': (nn,)},
                                     Cd_val={'shape': (nn,)},
                                     radius={'units': 'm', 'shape': (1,), 'val': 0.12},
                                     Cd={'shape': (1,), 'val': 0.5}),
                          promotes_inputs=[('radius', 'sphere_radius'), 
                                         ('Cd', 'sphere_Cd')])

        # Aerodynamics
        self.add_subsystem('aero', 
                          AeroSphereComp(num_nodes=nn),
                          promotes_inputs=['u', 'v', 'w'])
        
        # Connect aero parameters
        self.connect('aero_params.Cd_val', 'aero.Cd')
        self.connect('rho', 'aero.rho')

        self.add_subsystem('total_thrust',
                           om.ExecComp('T = (T_x**2 + T_y**2 + T_z**2)**0.5',
                           T={'units': 'N', 'shape': (nn,)},
                           T_x={'units': 'N', 'shape': (nn,)},
                           T_y={'units': 'N', 'shape': (nn,)},
                           T_z={'units': 'N', 'shape': (nn,)}),
                        promotes_inputs=['T_x', 'T_y', 'T_z'])

        # Force resolution
        self.add_subsystem('forces', 
                          ForceComponentResolver_Copy(num_nodes=nn),
                          promotes_inputs=['u', 'v', 'w', 'T_x', 'T_y', 'T_z',
                                           'roll', 'pitch', 'yaw'])
        
        # Connect aero forces
        self.connect('aero.drag', 'forces.drag')
        self.connect('aero.lift', 'forces.lift')
        self.connect('aero.side', 'forces.side')
        self.connect('total_thrust.T', 'forces.thrust')

        # Equations of motion
        self.add_subsystem('eom', 
                          SixDOF_EOM(num_nodes=nn),
                          promotes_inputs=['mass', 'u', 'v', 'w', 
                                           'roll_angle_vel', 'pitch_angle_vel', 'yaw_ang_vel',
                                           'roll', 'pitch', 'yaw', 
                                           'x', 'y', 'z', 'g', 'lx', 
                                           'ly', 'lz', 'J_xz', 'J_xy', 'J_yz', 'J_xx', 'J_yy',
                                           'J_zz'],
                          promotes_outputs=['*'])
        
        self.connect('forces.Fx', 'eom.Fx')
        self.connect('forces.Fy', 'eom.Fy')
        self.connect('forces.Fz', 'eom.Fz')


def load_waypoints(filename):
    """
    Load waypoints from a text file.
    
    Parameters:
    -----------
    filename : str
        Path to text file with 3 columns: x, y, z coordinates
        
    Returns:
    --------
    waypoints : numpy array
        Array of shape (n, 3) containing [x, y, z] coordinates
    """
    waypoints = np.loadtxt(filename)
    if waypoints.ndim == 1:
        waypoints = waypoints.reshape(1, -1)
    
    # Ensure we have exactly 3 columns
    if waypoints.shape[1] != 3:
        raise ValueError(f"Expected 3 columns (x, y, z), got {waypoints.shape[1]}")
    
    return waypoints


def create_phase_sequence(waypoints, start_position=np.array([0, 0, 0])):
    """
    Create a sequence of phase names and waypoint targets.
    
    Parameters:
    -----------
    waypoints : numpy array
        Array of shape (n, 3) containing [x, y, z] coordinates for pickups/dropoffs
    start_position : numpy array
        Starting position [x, y, z], default is [0, 0, 0]
        
    Returns:
    --------
    phase_info : list of dicts
        Each dict contains: {
            'name': phase name,
            'type': 'climb', 'cruise', or 'descent',
            'start': [x, y, z] start position,
            'end': [x, y, z] end position,
            'z_cruise': cruise altitude
        }
    """
    phase_info = []
    current_pos = start_position.copy()
    
    for idx, waypoint in enumerate(waypoints):
        waypoint_num = idx + 1
        
        # Determine cruise altitude (use max of current z or waypoint z, plus buffer)
        z_cruise = max(abs(current_pos[2]), abs(waypoint[2])) + 100.0
        
        # Phase 1: Climb from current position to cruise altitude
        climb_end = current_pos.copy()
        climb_end[2] = -z_cruise  # Negative because NED coordinates
        
        phase_info.append({
            'name': f'climb{waypoint_num}',
            'type': 'climb',
            'start': current_pos.copy(),
            'end': climb_end.copy(),
            'z_cruise': z_cruise
        })
        
        # Phase 2: Cruise at altitude to waypoint x,y position
        cruise_end = waypoint.copy()
        cruise_end[2] = -z_cruise  # Maintain cruise altitude
        
        phase_info.append({
            'name': f'cruise{waypoint_num}',
            'type': 'cruise',
            'start': climb_end.copy(),
            'end': cruise_end.copy(),
            'z_cruise': z_cruise
        })
        
        # Phase 3: Descend to waypoint
        phase_info.append({
            'name': f'descent{waypoint_num}',
            'type': 'descent',
            'start': cruise_end.copy(),
            'end': waypoint.copy(),
            'z_cruise': z_cruise
        })
        
        # Update current position for next iteration
        current_pos = waypoint.copy()
    
    return phase_info


def setup_trajectory(waypoints_file, vehicle_params=None):
    """
    Set up the trajectory optimization problem based on waypoints file.
    
    Parameters:
    -----------
    waypoints_file : str
        Path to waypoints text file
    vehicle_params : dict, optional
        Dictionary of vehicle parameters. Defaults provided if None.
        
    Returns:
    --------
    p : OpenMDAO Problem
        Configured problem ready to run
    phase_sequence : list
        List of phase names in order
    phase_info : list
        Detailed phase information
    waypoints : numpy array
        The loaded waypoints
    """
    # Default vehicle parameters
    if vehicle_params is None:
        vehicle_params = {
            'mass_empty': 2.0,  # kg
            'mass_payload': 0.5,  # kg
            'J_xx': 0.20,  # kg*m^2
            'J_yy': 0.20,  # kg*m^2
            'J_zz': 0.35,  # kg*m^2
            'J_xz': 0.0,  # kg*m^2
            'J_xy': 0.0, # kg*m^2
            'J_yz': 0.0, # kg*m^2
            'sphere_radius': 0.5,  # m
            'sphere_Cd': 0.47,
            'g': 9.81,  # m/s^2
        }
    
    # Load waypoints
    waypoints = load_waypoints(waypoints_file)
    print(f"Loaded {len(waypoints)} waypoint(s) from {waypoints_file}")
    for i, wp in enumerate(waypoints):
        print(f"  Waypoint {i+1}: x={wp[0]:.1f}, y={wp[1]:.1f}, z={wp[2]:.1f}")
    
    # Create phase sequence
    phase_info = create_phase_sequence(waypoints)
    phase_sequence = [phase['name'] for phase in phase_info]
    
    print(f"\nCreated {len(phase_sequence)} phases:")
    for phase in phase_info:
        print(f"  {phase['name']}: {phase['type']}")
    
    # Create problem
    p = om.Problem()
    
    p.driver = om.pyOptSparseDriver()
    p.driver.options["optimizer"] = "SNOPT"
    p.driver.opt_settings['Major iteration limit'] = 1000
    p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-4
    p.driver.opt_settings['Major optimality tolerance'] = 1.0E-3
    p.driver.opt_settings['iSumm'] = 6
    p.driver.opt_settings['Verify level'] = 0
    p.driver.declare_coloring()
    
    # Create trajectory
    traj = dm.Trajectory()
    
    # Add trajectory-level parameters
    param_targets = {phase['name']: ['g'] for phase in phase_info}
    traj.add_parameter('g', units='m / s**2', targets=param_targets,
                       opt=False, static_target=True, val=vehicle_params['g'])
    
    param_targets = {phase['name']: ['sphere_radius'] for phase in phase_info}
    traj.add_parameter('sphere_radius', units='m', targets=param_targets,
                       opt=False, static_target=True, val=vehicle_params['sphere_radius'])
    
    param_targets = {phase['name']: ['sphere_Cd'] for phase in phase_info}
    traj.add_parameter('sphere_Cd', targets=param_targets,
                       opt=False, static_target=True, val=vehicle_params['sphere_Cd'])
    
    # Create and add phases
    for i, phase in enumerate(phase_info):
        phase_name = phase['name']
        
        # Determine mass (with or without payload)
        # Payload is picked up at waypoint, so after descent phases
        waypoint_idx = (i // 3)  # Which waypoint we're heading to/from
        if i % 3 == 2:  # descent phase just completed
            # After pickup
            mass = vehicle_params['mass_empty'] + vehicle_params['mass_payload']
        else:
            # Before pickup or between waypoints
            if waypoint_idx == 0:
                mass = vehicle_params['mass_empty']
            else:
                mass = vehicle_params['mass_empty'] + vehicle_params['mass_payload']
        
        # Create phase
        ph = dm.Phase(ode_class=vtolODE,
                      transcription=dm.Radau(num_segments=10, order=3))
        
        ph = traj.add_phase(phase_name, ph)
        
        # Time options
        if i == 0:
            ph.set_time_options(fix_initial=True, duration_bounds=(5, 100), 
                               duration_ref=30, units='s')
        else:
            ph.set_time_options(fix_initial=False, duration_bounds=(5, 100), 
                               duration_ref=30, units='s')
        
        # Add parameters
        ph.add_parameter('mass', val=mass, static_target=True, targets=['mass'], units='kg')
        ph.add_parameter('J_xx', val=vehicle_params['J_xx'], static_target=True, 
                        targets=['J_xx'], units='kg*m**2')
        ph.add_parameter('J_yy', val=vehicle_params['J_yy'], static_target=True, 
                        targets=['J_yy'], units='kg*m**2')
        ph.add_parameter('J_zz', val=vehicle_params['J_zz'], static_target=True, 
                        targets=['J_zz'], units='kg*m**2')
        ph.add_parameter('J_xz', val=vehicle_params['J_xz'], static_target=True, 
                        targets=['J_xz'], units='kg*m**2')
        ph.add_parameter('J_xy', val=vehicle_params['J_xy'], static_target=True,
                         targets=['J_xy'], units='kg*m**2')
        ph.add_parameter('J_yz', val=vehicle_params['J_yz'], static_target=True,
                         targets=['J_yz'], units='kg*m**2')
        ph.add_parameter('lx', val=0.0, static_target=True, targets=['lx'], units='N*m')
        ph.add_parameter('ly', val=0.0, static_target=True, targets=['ly'], units='N*m')
        ph.add_parameter('lz', val=0.0, static_target=True, targets=['lz'], units='N*m')
        
        # Add states
        ph.add_state('x', rate_source='dx_dt', units='m', ref=100, defect_ref=10,
                    fix_initial=(i==0), fix_final=False)
        ph.add_state('y', rate_source='dy_dt', units='m', ref=100, defect_ref=10,
                    fix_initial=(i==0), fix_final=False)
        ph.add_state('z', rate_source='dz_dt', units='m', ref=100, defect_ref=10,
                    fix_initial=(i==0), fix_final=False)
        ph.add_state('u', rate_source='dx_accel', units='m/s', ref=10, defect_ref=1,
                    fix_initial=(i==0), fix_final=False)
        ph.add_state('v', rate_source='dy_accel', units='m/s', ref=10, defect_ref=1,
                    fix_initial=(i==0), fix_final=False)
        ph.add_state('w', rate_source='dz_accel', units='m/s', ref=10, defect_ref=1,
                    fix_initial=(i==0), fix_final=False)
        ph.add_state('roll', rate_source='roll_angle_rate_eq', units='rad', ref=1, defect_ref=0.1,
                    fix_initial=(i==0), fix_final=False)
        ph.add_state('pitch', rate_source='pitch_angle_rate_eq', units='rad', ref=1, defect_ref=0.1,
                    fix_initial=(i==0), fix_final=False)
        ph.add_state('yaw', rate_source='yaw_angle_rate_eq', units='rad', ref=1, defect_ref=0.1,
                    fix_initial=(i==0), fix_final=False)
        ph.add_state('roll_angle_vel', rate_source='roll_accel', units='rad/s', 
                    ref=1, defect_ref=0.1, fix_initial=(i==0), fix_final=False)
        ph.add_state('pitch_angle_vel', rate_source='pitch_accel', units='rad/s',
                    ref=1, defect_ref=0.1, fix_initial=(i==0), fix_final=False)
        ph.add_state('yaw_ang_vel', rate_source='yaw_accel', units='rad/s',
                    ref=1, defect_ref=0.1, fix_initial=(i==0), fix_final=False)
        
        # Add controls
        ph.add_control('T_x', units='N', opt=True, lower=-50, upper=50, ref=10,
                      rate_continuity=False, rate2_continuity=False)
        ph.add_control('T_y', units='N', opt=True, lower=-50, upper=50, ref=10,
                      rate_continuity=False, rate2_continuity=False)
        ph.add_control('T_z', units='N', opt=True, lower=-100, upper=0, ref=10,
                      rate_continuity=False, rate2_continuity=False)
        #ph.add_control('lx', targets=['lx'], opt=False, units='N*m', val=0.0)
        #ph.add_control('ly', targets=['ly'], opt=False, units='N*m', val=0.0)
        #ph.add_control('lz', targets=['lz'], opt=False, units='N*m', val=0.0)
        
        # Set boundary constraints
        if i == 0:
            # First phase starts at origin
            ph.add_boundary_constraint('x', loc='initial', equals=0.0)
            ph.add_boundary_constraint('y', loc='initial', equals=0.0)
            ph.add_boundary_constraint('z', loc='initial', equals=0.0)
            ph.add_boundary_constraint('u', loc='initial', equals=0.0)
            ph.add_boundary_constraint('v', loc='initial', equals=0.0)
            ph.add_boundary_constraint('w', loc='initial', equals=0.0)
            ph.add_boundary_constraint('roll', loc='initial', equals=0.0)
            ph.add_boundary_constraint('pitch', loc='initial', equals=0.0)
            ph.add_boundary_constraint('yaw', loc='initial', equals=0.0)
            ph.add_boundary_constraint('roll_angle_vel', loc='initial', equals=0.0)
            ph.add_boundary_constraint('pitch_angle_vel', loc='initial', equals=0.0)
            ph.add_boundary_constraint('yaw_ang_vel', loc='initial', equals=0.0)
        
        # Final boundary constraints
        ph.add_boundary_constraint('x', loc='final', equals=phase['end'][0])
        ph.add_boundary_constraint('y', loc='final', equals=phase['end'][1])
        ph.add_boundary_constraint('z', loc='final', equals=phase['end'][2])
        
        # Final velocity constraints (hover at waypoints)
        if phase['type'] == 'descent':
            ph.add_boundary_constraint('u', loc='final', equals=0.0)
            ph.add_boundary_constraint('v', loc='final', equals=0.0)
            ph.add_boundary_constraint('w', loc='final', equals=0.0)
            ph.add_boundary_constraint('roll', loc='final', equals=0.0)
            ph.add_boundary_constraint('pitch', loc='final', equals=0.0)
            ph.add_boundary_constraint('roll_angle_vel', loc='final', equals=0.0)
            ph.add_boundary_constraint('pitch_angle_vel', loc='final', equals=0.0)
            ph.add_boundary_constraint('yaw_ang_vel', loc='final', equals=0.0)
            
    
    # Link phases
    for i in range(len(phase_info) - 1):
        traj.link_phases([phase_info[i]['name'], phase_info[i+1]['name']], 
                        ['time', 'x', 'y', 'z', 'u', 'v', 'w',
                         'roll', 'pitch', 'yaw', 
                         'roll_angle_vel', 'pitch_angle_vel', 'yaw_ang_vel'])
    
    last_phase_name = phase_info[-1]['name']
    traj.add_objective(f'{last_phase_name}.timeseries.time', index=-1, ref=100)
    
    p.model.add_subsystem('traj', traj)
    
    p.setup()
    
    # Set initial guesses
    for i, phase in enumerate(phase_info):
        phase_name = phase['name']
        
        p.set_val(f'traj.{phase_name}.t_initial', 0.0 if i == 0 else 30.0*i)
        p.set_val(f'traj.{phase_name}.t_duration', 30.0)
        
        # Position initial guesses (linear interpolation)
        p.set_val(f'traj.{phase_name}.states:x', 
                 [phase['start'][0], phase['end'][0]], units='m')
        p.set_val(f'traj.{phase_name}.states:y',
                 [phase['start'][1], phase['end'][1]], units='m')
        p.set_val(f'traj.{phase_name}.states:z',
                 [phase['start'][2], phase['end'][2]], units='m')
        
        # Velocity guesses
        p.set_val(f'traj.{phase_name}.states:u', [0, 0], units='m/s')
        p.set_val(f'traj.{phase_name}.states:v', [0, 0], units='m/s')
        p.set_val(f'traj.{phase_name}.states:w', [0, 0], units='m/s')
        
        # Attitude guesses
        p.set_val(f'traj.{phase_name}.states:roll', [0, 0], units='rad')
        p.set_val(f'traj.{phase_name}.states:pitch', [0, 0], units='rad')
        p.set_val(f'traj.{phase_name}.states:yaw', [0, 0], units='rad')
        p.set_val(f'traj.{phase_name}.states:roll_angle_vel', [0, 0], units='rad/s')
        p.set_val(f'traj.{phase_name}.states:pitch_angle_vel', [0, 0], units='rad/s')
        p.set_val(f'traj.{phase_name}.states:yaw_ang_vel', [0, 0], units='rad/s')
        
        # Control guesses
        p.set_val(f'traj.{phase_name}.controls:T_x', [0, 0], units='N')
        p.set_val(f'traj.{phase_name}.controls:T_y', [0, 0], units='N')
        p.set_val(f'traj.{phase_name}.controls:T_z', [0, 0], units='N')

        # Determine mass for thrust guess
        waypoint_idx = (i // 3)
        if i % 3 == 2:
            mass_val = vehicle_params['mass_empty'] + vehicle_params['mass_payload']
        else:
            if waypoint_idx == 0:
                mass_val = vehicle_params['mass_empty']
            else:
                mass_val = vehicle_params['mass_empty'] + vehicle_params['mass_payload']
        
        p.set_val(f'traj.{phase_name}.controls:T_z', 
                 [-mass_val * vehicle_params['g'], -mass_val * vehicle_params['g']], 
                 units='N')
    
    return p, phase_sequence, phase_info, waypoints


def plot_trajectory(p, phase_sequence, waypoints):
    """
    Plot the optimized trajectory in 3D.
    
    Parameters:
    -----------
    p : OpenMDAO Problem
        Solved problem
    phase_sequence : list
        List of phase names
    waypoints : numpy array
        Waypoint coordinates
    """
    # Get solution
    sol = om.CaseReader('dymos_solution.db').get_case('final')
    
    # Extract trajectory data
    x_sim = {}
    y_sim = {}
    z_sim = {}
    x_sol = {}
    y_sol = {}
    z_sol = {}
    
    for phase_name in phase_sequence:
        x_sim[phase_name] = p.get_val(f'traj.{phase_name}.timeseries.x')
        y_sim[phase_name] = p.get_val(f'traj.{phase_name}.timeseries.y')
        z_sim[phase_name] = p.get_val(f'traj.{phase_name}.timeseries.z')
        
        x_sol[phase_name] = sol.get_val(f'traj.{phase_name}.states:x')
        y_sol[phase_name] = sol.get_val(f'traj.{phase_name}.states:y')
        z_sol[phase_name] = sol.get_val(f'traj.{phase_name}.states:z')
    
    # Create 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectory
    for phase_name in phase_sequence:
        ax.plot(x_sim[phase_name], y_sim[phase_name], z_sim[phase_name], 
               '-', color='C0', linewidth=2)
        ax.plot(x_sol[phase_name], y_sol[phase_name], z_sol[phase_name], 
               'o', color='C1', markersize=4)
    
    # Mark waypoints
    for i, wp in enumerate(waypoints):
        ax.plot([wp[0]], [wp[1]], [wp[2]], marker='*', color='gold', 
               markersize=20, markeredgecolor='black', markeredgewidth=1.5,
               label=f'Waypoint {i+1}' if i == 0 else '')
    
    # Mark start
    ax.plot([0], [0], [0], marker='s', color='green', 
           markersize=10, markeredgecolor='black', label='Start')
    
    # Mark end
    final_wp = waypoints[-1]
    ax.plot([final_wp[0]], [final_wp[1]], [final_wp[2]], marker='s', color='red', 
           markersize=10, markeredgecolor='black', label='End')
    
    # Labels
    ax.set_xlabel('X Position (m)', fontsize=12, labelpad=10)
    ax.set_ylabel('Y Position (m)', fontsize=12, labelpad=10)
    ax.set_zlabel('Z Position (m)', fontsize=12, labelpad=10)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='C0', linewidth=2, label='Simulation'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='C1', 
               markeredgecolor='C1', markersize=6, label='Solution Points'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
               markeredgecolor='black', markeredgewidth=1.5, markersize=12, 
               label='Waypoints'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
               markeredgecolor='black', markersize=8, label='Start'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
               markeredgecolor='black', markersize=8, label='End')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    ax.view_init(elev=20, azim=45)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Create multi-view plot
    fig2 = plt.figure(figsize=(15, 5))
    
    # Top view (X-Y)
    ax1 = fig2.add_subplot(131)
    for phase_name in phase_sequence:
        ax1.plot(x_sim[phase_name], y_sim[phase_name], '-', color='C0', linewidth=2)
        ax1.plot(x_sol[phase_name], y_sol[phase_name], 'o', color='C1', markersize=3)
    for wp in waypoints:
        ax1.plot(wp[0], wp[1], marker='*', color='gold', markersize=15,
                markeredgecolor='black', markeredgewidth=1.5)
    ax1.plot(0, 0, 's', color='green', markersize=8, markeredgecolor='black')
    ax1.plot(final_wp[0], final_wp[1], 's', color='red', markersize=8, 
            markeredgecolor='black')
    ax1.set_xlabel('X Position (m)')
    ax1.set_ylabel('Y Position (m)')
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    ax1.text(0.5, -0.18, '(a) Top View', transform=ax1.transAxes,
            ha='center', va='top', fontsize=12, fontweight='bold')
    
    # Side view (X-Z)
    ax2 = fig2.add_subplot(132)
    for phase_name in phase_sequence:
        ax2.plot(x_sim[phase_name], z_sim[phase_name], '-', color='C0', linewidth=2)
        ax2.plot(x_sol[phase_name], z_sol[phase_name], 'o', color='C1', markersize=3)
    for wp in waypoints:
        ax2.plot(wp[0], wp[2], marker='*', color='gold', markersize=15,
                markeredgecolor='black', markeredgewidth=1.5)
    ax2.plot(0, 0, 's', color='green', markersize=8, markeredgecolor='black')
    ax2.plot(final_wp[0], final_wp[2], 's', color='red', markersize=8,
            markeredgecolor='black')
    ax2.set_xlabel('X Position (m)')
    ax2.set_ylabel('Z Position (m)')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.5, -0.18, '(b) Side View', transform=ax2.transAxes,
            ha='center', va='top', fontsize=12, fontweight='bold')
    
    # Front view (Y-Z)
    ax3 = fig2.add_subplot(133)
    for phase_name in phase_sequence:
        ax3.plot(y_sim[phase_name], z_sim[phase_name], '-', color='C0', linewidth=2)
        ax3.plot(y_sol[phase_name], z_sol[phase_name], 'o', color='C1', markersize=3)
    for wp in waypoints:
        ax3.plot(wp[1], wp[2], marker='*', color='gold', markersize=15,
                markeredgecolor='black', markeredgewidth=1.5)
    ax3.plot(0, 0, 's', color='green', markersize=8, markeredgecolor='black')
    ax3.plot(final_wp[1], final_wp[2], 's', color='red', markersize=8,
            markeredgecolor='black')
    ax3.set_xlabel('Y Position (m)')
    ax3.set_ylabel('Z Position (m)')
    ax3.grid(True, alpha=0.3)
    ax3.text(0.5, -0.18, '(c) Front View', transform=ax3.transAxes,
            ha='center', va='top', fontsize=12, fontweight='bold')
    
    fig2.legend(handles=legend_elements, loc='upper center',
               bbox_to_anchor=(0.5, 0.98), ncol=5, fontsize=10, frameon=True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    # Example usage
    waypoints_file = "aviary/mission/sixdof/waypoints.txt"  # Change this to your file
    
    # Optional: Customize vehicle parameters
    custom_params = {
        'mass_empty': 2.0,
        'mass_payload': 0.5,
        'J_xx': 0.20,
        'J_yy': 0.20,
        'J_zz': 0.35,
        'J_xz': 0.0,
        'J_xy': 0.0,
        'J_yz': 0.0,
        'sphere_radius': 0.5,
        'sphere_Cd': 0.47,
        'g': 9.81,
    }
    
    # Setup and solve
    p, phase_sequence, phase_info, waypoints = setup_trajectory(
        waypoints_file, vehicle_params=custom_params
    )
    
    print("\nRunning optimization...")
    dm.run_problem(p, simulate=True, make_plots=False)
    
    print("\nOptimization complete!")
    print(f"Total mission time: {p.get_val('traj.descent{}.time'.format(len(waypoints)), units='s')[-1]:.2f} s")
    
    # Plot results
    plot_trajectory(p, phase_sequence, waypoints)
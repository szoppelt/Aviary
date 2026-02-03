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
# for a simple force component -- excludes lift and side
from aviary.mission.sixdof.SimplifiedForceComponentResolver import SimplifiedForceResolver
# new force component
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

        # CRITICAL FIX 1: Convert z (NED, down positive) to h (altitude, up positive)
        self.add_subsystem('altitude_calc',
                          om.ExecComp('h = -z',
                                     h={'units': 'm', 'shape': (nn,)},
                                     z={'units': 'm', 'shape': (nn,)}),
                          promotes_inputs=['z'],
                          promotes_outputs=['h'])

        # Atmosphere component (needs h, not z)
        self.add_subsystem('atm', 
                          USatm1976Comp(num_nodes=nn),
                          promotes_inputs=['*'],
                          promotes_outputs=['rho', 'sos', 'temp'])
        
        # CRITICAL FIX 2: Compute angles needed by ForceComponentResolver
        # These relate body velocities to wind/NED frames
        self.add_subsystem('wind_angles',
                          om.ExecComp(['V = (u**2 + v**2 + w**2 + 1e-8)**0.5',
                                      'alpha = arctan2(w, u + 1e-8)',  # angle of attack
                                      'beta = arcsin(fmin(fmax(v / ((u**2 + v**2 + w**2 + 1e-8)**0.5 + 1e-8), -0.99), 0.99))'],
                                     V={'units': 'm/s', 'shape': (nn,)},
                                     alpha={'units': 'rad', 'shape': (nn,)},
                                     beta={'units': 'rad', 'shape': (nn,)},
                                     u={'units': 'm/s', 'shape': (nn,)},
                                     v={'units': 'm/s', 'shape': (nn,)},
                                     w={'units': 'm/s', 'shape': (nn,)}),
                          promotes_inputs=['u', 'v', 'w'],
                          promotes_outputs=['alpha', 'beta'])

        # CRITICAL FIX 3: Add aerodynamic parameters as inputs
        # These should be set as parameters in your phase
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
                                           'roll_ang_vel', 'pitch_ang_vel', 'yaw_ang_vel',
                                           'roll', 'pitch', 'yaw', 
                                           'x', 'y', 'z', 'g', 'lx', 
                                           'ly', 'lz', 'J_xz', 'J_xx', 'J_yy',
                                           'J_zz'],
                          promotes_outputs=['*'])
        
        self.connect('forces.Fx', 'eom.Fx')
        self.connect('forces.Fy', 'eom.Fy')
        self.connect('forces.Fz', 'eom.Fz')
        


# Adding other stuff to potentially ignore everything below
        
p = om.Problem()

p.driver = om.pyOptSparseDriver()
p.driver.options["optimizer"] = "SNOPT"
p.driver.opt_settings['Major iteration limit'] = 1000
p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-4 # Relaxed
p.driver.opt_settings['Major optimality tolerance'] = 1.0E-3 # Relaxed
p.driver.opt_settings['iSumm'] = 6
p.driver.opt_settings['Verify level'] = 0
#p.driver.opt_settings['Linesearch tolerance'] = 0.9 # Conservative
p.driver.declare_coloring()

traj = dm.Trajectory()
traj_names = ['climb1', 'cruise1', 'descent1', 'climb2', 'cruise2', 'descent2']

# Parameters (before payload)
traj.add_parameter('g', units='m / s**2', 
                   targets={'climb1': ['g'], 'cruise1': ['g'], 'descent1': ['g'],
                            'climb2': ['g'], 'cruise2': ['g'], 'descent2': ['g']},
                    opt=False, static_target=True, val=9.81)
traj.add_parameter('sphere_radius', units='m', 
                   targets={'climb1': ['sphere_radius'], 'cruise1': ['sphere_radius'], 'descent1': ['sphere_radius'],
                            'climb2': ['sphere_radius'], 'cruise2': ['sphere_radius'], 'descent2': ['sphere_radius']},
                    opt=False, static_target=True, val=0.5)
traj.add_parameter('sphere_Cd', targets={'climb1': ['sphere_Cd'], 'cruise1': ['sphere_Cd'], 'descent1': ['sphere_Cd'], 
                                         'climb2': ['sphere_Cd'], 'cruise2': ['sphere_Cd'], 'descent2': ['sphere_Cd']},
                    opt=False, static_target=True, val=0.47)
# traj.add_parameter('lx', units='N*m', 
#                    targets={'climb1': ['lx'], 'cruise1': ['lx'], 'descent1': ['lx']},
#                    opt=False, static_target=True)
# traj.add_parameter('ly', units='N*m', 
#                    targets={'climb1': ['ly'], 'cruise1': ['ly'], 'descent1': ['ly']},
#                    opt=False, static_target=True)
# traj.add_parameter('lz', units='N*m', 
#                    targets={'climb1': ['lz'], 'cruise1': ['lz'], 'descent1': ['lz']},
#                    opt=False, static_target=True)


x_payload = 500.0 # m; payload coordinate (500, 0, 0)
z_final = 100.0 # m; z altitude to fly to for both climbs/cruise
y_final = 500.0 # m; final coordinate for y-direction (500, 500, 0)
# Overall 
# First phase (climb1)

climb1 = dm.Phase(ode_class=vtolODE,
                 transcription=dm.Radau(num_segments=10, order=3))

climb1 = traj.add_phase('climb1', climb1)

climb1.set_time_options(fix_initial=True, duration_bounds=(5, 100), duration_ref=30, units='s')
climb1.add_parameter('mass', val=2, static_target=True, targets=['mass'], units='kg')
climb1.add_parameter('J_xx', val=0.20, static_target=True, targets=['J_xx'], units='kg*m**2')
climb1.add_parameter('J_yy', val=0.20, static_target=True, targets=['J_yy'], units='kg*m**2')
climb1.add_parameter('J_zz', val=0.20, static_target=True, targets=['J_zz'], units='kg*m**2')
climb1.add_parameter('J_xz', val=0.0, static_target=True, targets=['J_xz'], units='kg*m**2')
climb1.add_state('u', fix_initial=True, fix_final=False, rate_source='dx_accel', 
                targets=['u'], units='m/s', ref=1, defect_ref=1)
climb1.add_state('v', fix_initial=True, fix_final=False, rate_source='dy_accel', 
                targets=['v'], units='m/s', ref=1, defect_ref=1)
climb1.add_state('w', fix_initial=True, fix_final=False, rate_source='dz_accel', 
                targets=['w'], units='m/s', ref=1, defect_ref=1)
climb1.add_state('roll_ang_vel', fix_initial=True, fix_final=False, rate_source='roll_accel',
                targets=['roll_ang_vel'], units='rad/s', ref=1, defect_ref=1)
climb1.add_state('pitch_ang_vel', fix_initial=True, fix_final=False, rate_source='pitch_accel',
                targets=['pitch_ang_vel'], units='rad/s', ref=1, defect_ref=1)
climb1.add_state('yaw_ang_vel', fix_initial=True, fix_final=False, rate_source='yaw_accel',
                targets=['yaw_ang_vel'], units='rad/s', ref=1, defect_ref=1)
climb1.add_state('roll', fix_initial=True, fix_final=False, rate_source='roll_angle_rate_eq', 
                targets=['roll'], units='rad', ref=1, defect_ref=1)
climb1.add_state('pitch', fix_initial=True, fix_final=False, rate_source='pitch_angle_rate_eq', 
                targets=['pitch'], units='rad', ref=1, defect_ref=1)
climb1.add_state('yaw', fix_initial=True, fix_final=False, rate_source='yaw_angle_rate_eq', 
                targets=['yaw'], units='rad', ref=1, defect_ref=1)
climb1.add_state('x', fix_initial=True, fix_final=False, rate_source='dx_dt',
                targets=['x'], units='m', ref=10, defect_ref=0.5)
climb1.add_state('y', fix_initial=True, fix_final=False, rate_source='dy_dt',
                targets=['y'], units='m', ref=10, defect_ref=0.5)
climb1.add_state('z', fix_initial=True, fix_final=False, rate_source='dz_dt',
                targets=['z'], units='m', ref=z_final, defect_ref=2.0)

# Controls (without explicit scaling for now)
climb1.add_control('T_x', targets=['T_x'], opt=True,units='N', lower=-100.0, upper=100.0)
climb1.add_control('T_y', targets=['T_y'], opt=True,units='N', lower=-100.0, upper=100.0)
climb1.add_control('T_z', targets=['T_z'], opt=True,units='N', lower=-100.0, upper=100.0)
climb1.add_control('lx', targets=['lx'], opt=False, units='N*m', val=0.0)
climb1.add_control('ly', targets=['ly'], opt=False, units='N*m', val=0.0)
climb1.add_control('lz', targets=['lz'], opt=False, units='N*m', val=0.0)
climb1.add_boundary_constraint('z', loc='final', equals=z_final, units='m', scaler=0.01)
#climb1.add_path_constraint('T_climb1=(T_x**2+T_y**2+T_z**2)**0.5', lower=0.0, upper=25) # upper = mg
climb1.add_path_constraint('x', lower=0.0, upper=50.0, units='m')
#climb1.add_path_constraint('y', lower=0.0, upper=10, units='m')


# Second phase (cruise1)
cruise1 = dm.Phase(ode_class=vtolODE,
                 transcription=dm.Radau(num_segments=10, order=3))

cruise1 = traj.add_phase('cruise1', cruise1)

cruise1.set_time_options(fix_initial=False, initial_bounds=(10, 100), duration_bounds=(30, 200), duration_ref=40, units='s')
cruise1.add_parameter('mass', val=2, static_target=True, targets=['mass'], units='kg')
cruise1.add_parameter('J_xx', val=0.20, static_target=True, targets=['J_xx'], units='kg*m**2')
cruise1.add_parameter('J_yy', val=0.20, static_target=True, targets=['J_yy'], units='kg*m**2')
cruise1.add_parameter('J_zz', val=0.20, static_target=True, targets=['J_zz'], units='kg*m**2')
cruise1.add_parameter('J_xz', val=0.0, static_target=True, targets=['J_xz'], units='kg*m**2')
cruise1.add_state('u', fix_initial=False, fix_final=False, rate_source='dx_accel', targets=['u'], units='m/s', ref=10, defect_ref=1)
cruise1.add_state('v', fix_initial=False, fix_final=False, rate_source='dy_accel', targets=['v'], units='m/s', ref=1, defect_ref=0.1)
cruise1.add_state('w', fix_initial=False, fix_final=False, rate_source='dz_accel', targets=['w'], units='m/s', ref=1, defect_ref=0.1)
cruise1.add_state('roll_ang_vel', fix_initial=False, fix_final=False, rate_source='roll_accel', targets=['roll_ang_vel'], units='rad/s', ref=1, defect_ref=0.1)
cruise1.add_state('pitch_ang_vel', fix_initial=False, fix_final=False, rate_source='pitch_accel', targets=['pitch_ang_vel'], units='rad/s', ref=1, defect_ref=0.1)
cruise1.add_state('yaw_ang_vel', fix_initial=False, fix_final=False, rate_source='yaw_accel', targets=['yaw_ang_vel'], units='rad/s', ref=1, defect_ref=0.1)
cruise1.add_state('roll', fix_initial=False, fix_final=False, rate_source='roll_angle_rate_eq', targets=['roll'], units='rad', ref=1, defect_ref=0.1)
cruise1.add_state('pitch', fix_initial=False, fix_final=False, rate_source='pitch_angle_rate_eq', targets=['pitch'], units='rad', ref=1, defect_ref=0.1)
cruise1.add_state('yaw', fix_initial=False, fix_final=False, rate_source='yaw_angle_rate_eq', targets=['yaw'], units='rad', ref=1, defect_ref=0.1)
cruise1.add_state('x', fix_initial=False, fix_final=False, rate_source='dx_dt', targets=['x'], units='m', ref=x_payload, defect_ref=5.0)
cruise1.add_state('y', fix_initial=False, fix_final=False, rate_source='dy_dt', targets=['y'], units='m', ref=10, defect_ref=0.5)
cruise1.add_state('z', fix_initial=False, fix_final=False, rate_source='dz_dt', targets=['z'], units='m', ref=z_final, defect_ref=2.0)
# Controls (without explicit scaling for now)
cruise1.add_control('T_x', targets=['T_x'], opt=True,units='N', lower=-100.0, upper=100.0)
cruise1.add_control('T_y', targets=['T_y'], opt=True,units='N', lower=-100.0, upper=100.0)
cruise1.add_control('T_z', targets=['T_z'], opt=True,units='N', lower=-100.0, upper=100.0)
cruise1.add_control('lx', targets=['lx'], opt=False, units='N*m', val=0.0)
cruise1.add_control('ly', targets=['ly'], opt=False, units='N*m', val=0.0)
cruise1.add_control('lz', targets=['lz'], opt=False, units='N*m', val=0.0)
#cruise1.add_path_constraint('z', lower=z_final - 50.0, upper=z_final + 50.0, units='m')
#cruise1.add_boundary_constraint('z', loc='initial', equals=z_final, units='m')
cruise1.add_boundary_constraint('z', loc='final', lower=z_final - 20, upper=z_final + 20, units='m', scaler=0.01)
#cruise1.add_path_constraint('T_cruise1=(T_x**2+T_y**2+T_z**2)**0.5', lower=0.0, upper=25) # upper = mg
cruise1.add_boundary_constraint('x', loc='final', equals=x_payload, units='m', scaler=0.002)
#cruise1.add_path_constraint('y', loc='final', lower=-5.0, upper=5.0, units='m')
#cruise1.add_path_constraint('y', lower=0.0, upper=10.0, units='m')
#cruise1.add_path_constraint('z', lower=60.0, upper=100.0, units='m')

descent1 = dm.Phase(ode_class=vtolODE,
                 transcription=dm.Radau(num_segments=10, order=3))

descent1 = traj.add_phase('descent1', descent1)
descent1.set_time_options(fix_initial=False, initial_bounds=(50, 200), duration_bounds=(10, 200), duration_ref=30, units='s')
descent1.add_parameter('mass', val=2, static_target=True, targets=['mass'], units='kg')
descent1.add_parameter('J_xx', val=0.20, static_target=True, targets=['J_xx'], units='kg*m**2')
descent1.add_parameter('J_yy', val=0.20, static_target=True, targets=['J_yy'], units='kg*m**2')
descent1.add_parameter('J_zz', val=0.20, static_target=True, targets=['J_zz'], units='kg*m**2')
descent1.add_parameter('J_xz', val=0.0, static_target=True, targets=['J_xz'], units='kg*m**2')
descent1.add_state('u', fix_initial=False, fix_final=True, rate_source='dx_accel', targets=['u'], units='m/s', ref=1, defect_ref=0.1)
descent1.add_state('v', fix_initial=False, fix_final=True, rate_source='dy_accel', targets=['v'], units='m/s', ref=1, defect_ref=0.1)
descent1.add_state('w', fix_initial=False, fix_final=True, rate_source='dz_accel', targets=['w'], units='m/s', ref=1, defect_ref=0.1)
descent1.add_state('roll_ang_vel', fix_initial=False, fix_final=False, rate_source='roll_accel', targets=['roll_ang_vel'], units='rad/s', ref=1, defect_ref=0.1)
descent1.add_state('pitch_ang_vel', fix_initial=False, fix_final=False, rate_source='pitch_accel', targets=['pitch_ang_vel'], units='rad/s', ref=1, defect_ref=0.1)
descent1.add_state('yaw_ang_vel', fix_initial=False, fix_final=False, rate_source='yaw_accel', targets=['yaw_ang_vel'], units='rad/s', ref=1, defect_ref=0.1)
descent1.add_state('roll', fix_initial=False, fix_final=False, rate_source='roll_angle_rate_eq', targets=['roll'], units='rad', ref=1, defect_ref=0.1)
descent1.add_state('pitch', fix_initial=False, fix_final=False, rate_source='pitch_angle_rate_eq', targets=['pitch'], units='rad', ref=1, defect_ref=0.1)
descent1.add_state('yaw', fix_initial=False, fix_final=False, rate_source='yaw_angle_rate_eq', targets=['yaw'], units='rad', ref=1, defect_ref=0.1)
descent1.add_state('x', fix_initial=False, fix_final=False, rate_source='dx_dt', targets=['x'], units='m', ref=x_payload, defect_ref=2.0)
descent1.add_state('y', fix_initial=False, fix_final=False, rate_source='dy_dt', targets=['y'], units='m', ref=10, defect_ref=0.5)
descent1.add_state('z', fix_initial=False, fix_final=True, rate_source='dz_dt', targets=['z'], units='m', ref=z_final, defect_ref=2.0)
# Controls (without explicit scaling for now)
descent1.add_control('T_x', targets=['T_x'], opt=True, units='N', lower=-100.0, upper=100.0)
descent1.add_control('T_y', targets=['T_y'], opt=True, units='N', lower=-100.0, upper=100.0)
descent1.add_control('T_z', targets=['T_z'], opt=True, units='N', lower=-100.0, upper=100.0)
descent1.add_control('lx', targets=['lx'], opt=False, units='N*m', val=0.0)
descent1.add_control('ly', targets=['ly'], opt=False, units='N*m', val=0.0)
descent1.add_control('lz', targets=['lz'], opt=False, units='N*m', val=0.0)
descent1.add_boundary_constraint('x', loc='final', equals=x_payload, units='m', scaler=0.002)
descent1.add_boundary_constraint('y', loc='final', equals=0.0, units='m', scaler=0.01)
#descent1.add_boundary_constraint('z', loc='final', equals=0.0, units='m', scaler=0.01)
#descent1.add_path_constraint('T_descent1=(T_x**2+T_y**2+T_z**2)**0.5', lower=0.0, upper=25) # upper = mg
descent1.add_path_constraint('x', lower=0.0, upper=x_payload + 20, units='m')
#descent1.add_path_constraint('y', lower=-5.0, upper=5.0, units='m')

climb2 = dm.Phase(ode_class=vtolODE,
                 transcription=dm.Radau(num_segments=10, order=3))

climb2 = traj.add_phase('climb2', climb2)
climb2.set_time_options(fix_initial=False, initial_bounds=(50, 200), duration_bounds=(10, 200), duration_ref=30, units='s')
climb2.add_parameter('mass', val=2.2, static_target=True, targets=['mass'], units='kg')
climb2.add_parameter('J_xx', val=0.22, static_target=True, targets=['J_xx'], units='kg*m**2')
climb2.add_parameter('J_yy', val=0.22, static_target=True, targets=['J_yy'], units='kg*m**2')
climb2.add_parameter('J_zz', val=0.22, static_target=True, targets=['J_zz'], units='kg*m**2')
climb2.add_parameter('J_xz', val=0.0, static_target=True, targets=['J_xz'], units='kg*m**2')
climb2.add_state('u', fix_initial=True, fix_final=False, rate_source='dx_accel', targets=['u'], units='m/s', ref=1, defect_ref=0.1)
climb2.add_state('v', fix_initial=True, fix_final=False, rate_source='dy_accel', targets=['v'], units='m/s', ref=1, defect_ref=0.1)
climb2.add_state('w', fix_initial=True, fix_final=False, rate_source='dz_accel', targets=['w'], units='m/s', ref=1, defect_ref=0.1)
climb2.add_state('roll_ang_vel', fix_initial=False, fix_final=False, rate_source='roll_accel', targets=['roll_ang_vel'], units='rad/s', ref=1, defect_ref=0.1)
climb2.add_state('pitch_ang_vel', fix_initial=False, fix_final=False, rate_source='pitch_accel', targets=['pitch_ang_vel'], units='rad/s', ref=1, defect_ref=0.1)
climb2.add_state('yaw_ang_vel', fix_initial=False, fix_final=False, rate_source='yaw_accel', targets=['yaw_ang_vel'], units='rad/s', ref=1, defect_ref=0.1)
climb2.add_state('roll', fix_initial=False, fix_final=False, rate_source='roll_angle_rate_eq', targets=['roll'], units='rad', ref=1, defect_ref=0.1)
climb2.add_state('pitch', fix_initial=False, fix_final=False, rate_source='pitch_angle_rate_eq', targets=['pitch'], units='rad', ref=1, defect_ref=0.1)
climb2.add_state('yaw', fix_initial=False, fix_final=False, rate_source='yaw_angle_rate_eq', targets=['yaw'], units='rad', ref=1, defect_ref=0.1)
climb2.add_state('x', fix_initial=False, fix_final=False, rate_source='dx_dt', targets=['x'], units='m', ref=x_payload, defect_ref=2.0)
climb2.add_state('y', fix_initial=False, fix_final=False, rate_source='dy_dt', targets=['y'], units='m', ref=10, defect_ref=0.5)
climb2.add_state('z', fix_initial=True, fix_final=False, rate_source='dz_dt', targets=['z'], units='m', ref=z_final, defect_ref=2.0)
climb2.add_control('T_x', targets=['T_x'], opt=True, units='N', lower=-100.0, upper=100.0)
climb2.add_control('T_y', targets=['T_y'], opt=True, units='N', lower=-100.0, upper=100.0)
climb2.add_control('T_z', targets=['T_z'], opt=True, units='N', lower=-100.0, upper=100.0)
climb2.add_control('lx', targets=['lx'], opt=False, units='N*m', val=0.0)
climb2.add_control('ly', targets=['ly'], opt=False, units='N*m', val=0.0)
climb2.add_control('lz', targets=['lz'], opt=False, units='N*m', val=0.0)

climb2.add_boundary_constraint('z', loc='final', equals=z_final, units='m', scaler=0.01)
#climb1.add_path_constraint('T_climb1=(T_x**2+T_y**2+T_z**2)**0.5', lower=0.0, upper=25) # upper = mg
climb2.add_path_constraint('x', lower=x_payload-20, upper=x_payload+20, units='m')


cruise2 = dm.Phase(ode_class=vtolODE,
                 transcription=dm.Radau(num_segments=10, order=3))

cruise2 = traj.add_phase('cruise2', cruise2)
cruise2.set_time_options(fix_initial=False, initial_bounds=(50, 200), duration_bounds=(10, 200), duration_ref=30, units='s')
cruise2.add_parameter('mass', val=2.2, static_target=True, targets=['mass'], units='kg')
cruise2.add_parameter('J_xx', val=0.22, static_target=True, targets=['J_xx'], units='kg*m**2')
cruise2.add_parameter('J_yy', val=0.22, static_target=True, targets=['J_yy'], units='kg*m**2')
cruise2.add_parameter('J_zz', val=0.22, static_target=True, targets=['J_zz'], units='kg*m**2')
cruise2.add_parameter('J_xz', val=0.0, static_target=True, targets=['J_xz'], units='kg*m**2')
cruise2.add_state('u', fix_initial=False, fix_final=False, rate_source='dx_accel', targets=['u'], units='m/s', ref=1, defect_ref=0.1)
cruise2.add_state('v', fix_initial=False, fix_final=False, rate_source='dy_accel', targets=['v'], units='m/s', ref=1, defect_ref=0.1)
cruise2.add_state('w', fix_initial=False, fix_final=False, rate_source='dz_accel', targets=['w'], units='m/s', ref=1, defect_ref=0.1)
cruise2.add_state('roll_ang_vel', fix_initial=False, fix_final=False, rate_source='roll_accel', targets=['roll_ang_vel'], units='rad/s', ref=1, defect_ref=0.1)
cruise2.add_state('pitch_ang_vel', fix_initial=False, fix_final=False, rate_source='pitch_accel', targets=['pitch_ang_vel'], units='rad/s', ref=1, defect_ref=0.1)
cruise2.add_state('yaw_ang_vel', fix_initial=False, fix_final=False, rate_source='yaw_accel', targets=['yaw_ang_vel'], units='rad/s', ref=1, defect_ref=0.1)
cruise2.add_state('roll', fix_initial=False, fix_final=False, rate_source='roll_angle_rate_eq', targets=['roll'], units='rad', ref=1, defect_ref=0.1)
cruise2.add_state('pitch', fix_initial=False, fix_final=False, rate_source='pitch_angle_rate_eq', targets=['pitch'], units='rad', ref=1, defect_ref=0.1)
cruise2.add_state('yaw', fix_initial=False, fix_final=False, rate_source='yaw_angle_rate_eq', targets=['yaw'], units='rad', ref=1, defect_ref=0.1)
cruise2.add_state('x', fix_initial=False, fix_final=False, rate_source='dx_dt', targets=['x'], units='m', ref=x_payload, defect_ref=2.0)
cruise2.add_state('y', fix_initial=False, fix_final=False, rate_source='dy_dt', targets=['y'], units='m', ref=y_final, defect_ref=2.0)
cruise2.add_state('z', fix_initial=False, fix_final=False, rate_source='dz_dt', targets=['z'], units='m', ref=z_final, defect_ref=2.0)
cruise2.add_control('T_x', targets=['T_x'], opt=True, units='N', lower=-100.0, upper=100.0)
cruise2.add_control('T_y', targets=['T_y'], opt=True, units='N', lower=-100.0, upper=100.0)
cruise2.add_control('T_z', targets=['T_z'], opt=True, units='N', lower=-100.0, upper=100.0)
cruise2.add_control('lx', targets=['lx'], opt=False, units='N*m', val=0.0)
cruise2.add_control('ly', targets=['ly'], opt=False, units='N*m', val=0.0)
cruise2.add_control('lz', targets=['lz'], opt=False, units='N*m', val=0.0)

cruise2.add_boundary_constraint('z', loc='final', lower=z_final - 20, upper=z_final + 20, units='m')
#cruise1.add_path_constraint('T_cruise1=(T_x**2+T_y**2+T_z**2)**0.5', lower=0.0, upper=25) # upper = mg
cruise2.add_boundary_constraint('x', loc='final', lower=x_payload - 20, upper=x_payload + 20, units='m')
cruise2.add_boundary_constraint('y', loc='final', lower=y_final - 20, upper=y_final + 20, units='m')

descent2 = dm.Phase(ode_class=vtolODE,
                 transcription=dm.Radau(num_segments=10, order=3))

descent2 = traj.add_phase('descent2', descent2)
descent2.set_time_options(fix_initial=False, initial_bounds=(50, 200), duration_bounds=(10, 200), duration_ref=30, units='s')
descent2.add_parameter('mass', val=2.2, static_target=True, targets=['mass'], units='kg')
descent2.add_parameter('J_xx', val=0.22, static_target=True, targets=['J_xx'], units='kg*m**2')
descent2.add_parameter('J_yy', val=0.22, static_target=True, targets=['J_yy'], units='kg*m**2')
descent2.add_parameter('J_zz', val=0.22, static_target=True, targets=['J_zz'], units='kg*m**2')
descent2.add_parameter('J_xz', val=0.0, static_target=True, targets=['J_xz'], units='kg*m**2')
descent2.add_state('u', fix_initial=False, fix_final=True, rate_source='dx_accel', targets=['u'], units='m/s', ref=1, defect_ref=0.1)
descent2.add_state('v', fix_initial=False, fix_final=True, rate_source='dy_accel', targets=['v'], units='m/s', ref=1, defect_ref=0.1)
descent2.add_state('w', fix_initial=False, fix_final=True, rate_source='dz_accel', targets=['w'], units='m/s', ref=1, defect_ref=0.1)
descent2.add_state('roll_ang_vel', fix_initial=False, fix_final=False, rate_source='roll_accel', targets=['roll_ang_vel'], units='rad/s', ref=1, defect_ref=0.1)
descent2.add_state('pitch_ang_vel', fix_initial=False, fix_final=False, rate_source='pitch_accel', targets=['pitch_ang_vel'], units='rad/s', ref=1, defect_ref=0.1)
descent2.add_state('yaw_ang_vel', fix_initial=False, fix_final=False, rate_source='yaw_accel', targets=['yaw_ang_vel'], units='rad/s', ref=1, defect_ref=0.1)
descent2.add_state('roll', fix_initial=False, fix_final=False, rate_source='roll_angle_rate_eq', targets=['roll'], units='rad', ref=1, defect_ref=0.1)
descent2.add_state('pitch', fix_initial=False, fix_final=False, rate_source='pitch_angle_rate_eq', targets=['pitch'], units='rad', ref=1, defect_ref=0.1)
descent2.add_state('yaw', fix_initial=False, fix_final=False, rate_source='yaw_angle_rate_eq', targets=['yaw'], units='rad', ref=1, defect_ref=0.1)
descent2.add_state('x', fix_initial=False, fix_final=False, rate_source='dx_dt', targets=['x'], units='m', ref=x_payload, defect_ref=2.0)
descent2.add_state('y', fix_initial=False, fix_final=False, rate_source='dy_dt', targets=['y'], units='m', ref=y_final, defect_ref=2.0)
descent2.add_state('z', fix_initial=False, fix_final=True, rate_source='dz_dt', targets=['z'], units='m', ref=z_final, defect_ref=2.0)
descent2.add_control('T_x', targets=['T_x'], opt=True, units='N', lower=-100.0, upper=100.0)
descent2.add_control('T_y', targets=['T_y'], opt=True, units='N', lower=-100.0, upper=100.0)
descent2.add_control('T_z', targets=['T_z'], opt=True, units='N', lower=-100.0, upper=100.0)
descent2.add_control('lx', targets=['lx'], opt=False, units='N*m', val=0.0)
descent2.add_control('ly', targets=['ly'], opt=False, units='N*m', val=0.0)
descent2.add_control('lz', targets=['lz'], opt=False, units='N*m', val=0.0)
descent2.add_boundary_constraint('y', loc='final', equals=y_final, units='m', scaler=0.01)
descent2.add_boundary_constraint('x', loc='final', equals=x_payload, units='m', scaler=0.01)
descent2.add_objective('time', loc='final', ref=240)



traj.link_phases(['climb1', 'cruise1', 'descent1', 'climb2', 'cruise2', 'descent2'],
                    vars=['time', 'u', 'v', 'w',
                          'roll_ang_vel', 'pitch_ang_vel', 'yaw_ang_vel',
                          'roll', 'pitch', 'yaw', 'x', 'y', 'z'],
                    connected=True)

p.model.add_subsystem('traj', subsys=traj)
p.setup(check=True)

#print("\n=== Verifying Linkage Constraints ===")
#print("Number of linkage constraints:", len([c for c in p.model.traj._linkage_constraints]))

# Initial guesses

p.set_val('traj.parameters:sphere_radius', val=0.5, units='m')
p.set_val('traj.parameters:sphere_Cd', val=0.47)
p.set_val('traj.parameters:g', val=9.81, units='m/s**2')

# Hover thrust (T=W=mg)
weight1 = 2 * 9.81 # before payload
weight2 = 2.2 * 9.81 # after payload


climb1 = p.model.traj.phases.climb1
cruise1 = p.model.traj.phases.cruise1
descent1 = p.model.traj.phases.descent1
climb2 = p.model.traj.phases.climb2
cruise2 = p.model.traj.phases.cruise2
descent2 = p.model.traj.phases.descent2

climb1.set_time_options(fix_initial=True)
climb1.set_time_val(initial=0.0, duration=30, units='s')

#climb1.add_boundary_constraint('z', loc='initial', equals=0.0, units='m')
#climb1.add_boundary_constraint('u', loc='initial', equals=0.0, units='m/s')
#climb1.add_boundary_constraint('v', loc='initial', equals=0.0, units='m/s')
#climb1.add_boundary_constraint('w', loc='initial', equals=0.0, units='m/s')

climb1.set_state_val('u', vals=[0, 0], units='m/s')
climb1.set_state_val('v', vals=[0, 0], units='m/s')
climb1.set_state_val('w', vals=[0, 2.5], units='m/s')
climb1.set_state_val('roll_ang_vel', vals=[0, 0], units='rad/s')
climb1.set_state_val('pitch_ang_vel', vals=[0, 0], units='rad/s')
climb1.set_state_val('yaw_ang_vel', vals=[0, 0], units='rad/s')
climb1.set_state_val('roll', vals=[0, 0], units='rad')
climb1.set_state_val('pitch', vals=[0, 0], units='rad')
climb1.set_state_val('yaw', vals=[0, 0], units='rad')
climb1.set_state_val('x', vals=[0, 0], units='m')
climb1.set_state_val('y', vals=[0, 0], units='m')
climb1.set_state_val('z', vals=[0, z_final], units='m')
climb1.set_control_val('T_x', vals=[0, 0], units='N')
climb1.set_control_val('T_y', vals=[0, 0], units='N')
climb1.set_control_val('T_z', vals=[0, -weight1*1.5], units='N')
climb1.set_control_val('lx', vals=[0.0, 0.0], units='N*m')
climb1.set_control_val('ly', vals=[0.0, 0.0], units='N*m')
climb1.set_control_val('lz', vals=[0.0, 0.0], units='N*m')

cruise1.set_time_val(initial=30.0, duration=60.0, units='s')
cruise1.set_state_val('u', vals=[10, 10], units='m/s')
cruise1.set_state_val('v', vals=[0, 0], units='m/s')
cruise1.set_state_val('w', vals=[0, 0], units='m/s')
cruise1.set_state_val('roll_ang_vel', vals=[0, 0], units='rad/s')
cruise1.set_state_val('pitch_ang_vel', vals=[0, 0], units='rad/s')
cruise1.set_state_val('yaw_ang_vel', vals=[0, 0], units='rad/s')
cruise1.set_state_val('roll', vals=[0, 0], units='rad')
cruise1.set_state_val('pitch', vals=[0, 0], units='rad')
cruise1.set_state_val('yaw', vals=[0, 0], units='rad')
cruise1.set_state_val('x', vals=[0, x_payload], units='m')
cruise1.set_state_val('y', vals=[0, 0], units='m')
cruise1.set_state_val('z', vals=[z_final, z_final], units='m')
cruise1.set_control_val('T_x', vals=[1.33, 1.33], units='N')
cruise1.set_control_val('T_y', vals=[0, 0], units='N')
cruise1.set_control_val('T_z', vals=[-weight1, -weight1], units='N')
cruise1.set_control_val('lx', vals=[0.0, 0.0], units='N*m')
cruise1.set_control_val('ly', vals=[0.0, 0.0], units='N*m')
cruise1.set_control_val('lz', vals=[0.0, 0.0], units='N*m')

descent1.set_time_val(initial=90, duration=30, units='s')
descent1.set_state_val('u', vals=[0, 0], units='m/s')
descent1.set_state_val('v', vals=[0, 0], units='m/s')
descent1.set_state_val('w', vals=[2.5, 0], units='m/s')
descent1.set_state_val('roll_ang_vel', vals=[0, 0], units='rad/s')
descent1.set_state_val('pitch_ang_vel', vals=[0, 0], units='rad/s')
descent1.set_state_val('yaw_ang_vel', vals=[0, 0], units='rad/s')
descent1.set_state_val('roll', vals=[0, 0], units='rad')
descent1.set_state_val('pitch', vals=[0, 0], units='rad')
descent1.set_state_val('yaw', vals=[0, 0], units='rad')
descent1.set_state_val('x', vals=[x_payload, x_payload], units='m')
descent1.set_state_val('y', vals=[0, 0], units='m')
descent1.set_state_val('z', vals=[z_final, 0], units='m')
descent1.set_control_val('T_x', vals=[0, 0], units='N')
descent1.set_control_val('T_y', vals=[0, 0], units='N')
descent1.set_control_val('T_z', vals=[-weight1*0.5, 0], units='N')
descent1.set_control_val('lx', vals=[0.0, 0.0], units='N*m')
descent1.set_control_val('ly', vals=[0.0, 0.0], units='N*m')
descent1.set_control_val('lz', vals=[0.0, 0.0], units='N*m')

climb2.set_time_val(initial=120, duration=30, units='s')
climb2.set_state_val('u', vals=[0, 0], units='m/s')
climb2.set_state_val('v', vals=[0, 0], units='m/s')
climb2.set_state_val('w', vals=[0, 2.5], units='m/s')
climb2.set_state_val('roll_ang_vel', vals=[0, 0], units='rad/s')
climb2.set_state_val('pitch_ang_vel', vals=[0, 0], units='rad/s')
climb2.set_state_val('yaw_ang_vel', vals=[0, 0], units='rad/s')
climb2.set_state_val('roll', vals=[0, 0], units='rad')
climb2.set_state_val('pitch', vals=[0, 0], units='rad')
climb2.set_state_val('yaw', vals=[0, 0], units='rad')
climb2.set_state_val('x', vals=[x_payload, x_payload], units='m')
climb2.set_state_val('y', vals=[0, 0], units='m')
climb2.set_state_val('z', vals=[0, z_final], units='m')
climb2.set_control_val('T_x', vals=[0, 0], units='N')
climb2.set_control_val('T_y', vals=[0, 0], units='N')
climb2.set_control_val('T_z', vals=[0, -weight2*1.5], units='N')
climb2.set_control_val('lx', vals=[0.0, 0.0], units='N*m')
climb2.set_control_val('ly', vals=[0.0, 0.0], units='N*m')
climb2.set_control_val('lz', vals=[0.0, 0.0], units='N*m')

cruise2.set_time_val(initial=150, duration=60, units='s')
cruise2.set_state_val('u', vals=[0, 0], units='m/s')
cruise2.set_state_val('v', vals=[10, 10], units='m/s')
cruise2.set_state_val('w', vals=[0, 2.5], units='m/s')
cruise2.set_state_val('roll_ang_vel', vals=[0, 0], units='rad/s')
cruise2.set_state_val('pitch_ang_vel', vals=[0, 0], units='rad/s')
cruise2.set_state_val('yaw_ang_vel', vals=[0, 0], units='rad/s')
cruise2.set_state_val('roll', vals=[0, 0], units='rad')
cruise2.set_state_val('pitch', vals=[0, 0], units='rad')
cruise2.set_state_val('yaw', vals=[0, 0], units='rad')
cruise2.set_state_val('x', vals=[x_payload, x_payload], units='m')
cruise2.set_state_val('y', vals=[0, y_final], units='m')
cruise2.set_state_val('z', vals=[z_final, z_final], units='m')
cruise2.set_control_val('T_x', vals=[0, 0], units='N')
cruise2.set_control_val('T_y', vals=[1.35, 1.35], units='N')
cruise2.set_control_val('T_z', vals=[-weight2, -weight2], units='N')
cruise2.set_control_val('lx', vals=[0.0, 0.0], units='N*m')
cruise2.set_control_val('ly', vals=[0.0, 0.0], units='N*m')
cruise2.set_control_val('lz', vals=[0.0, 0.0], units='N*m')

descent2.set_time_val(initial=210, duration=30, units='s')
descent2.set_state_val('u', vals=[0, 0], units='m/s')
descent2.set_state_val('v', vals=[0, 0], units='m/s')
descent2.set_state_val('w', vals=[2.5, 0], units='m/s')
descent2.set_state_val('roll_ang_vel', vals=[0, 0], units='rad/s')
descent2.set_state_val('pitch_ang_vel', vals=[0, 0], units='rad/s')
descent2.set_state_val('yaw_ang_vel', vals=[0, 0], units='rad/s')
descent2.set_state_val('roll', vals=[0, 0], units='rad')
descent2.set_state_val('pitch', vals=[0, 0], units='rad')
descent2.set_state_val('yaw', vals=[0, 0], units='rad')
descent2.set_state_val('x', vals=[x_payload, x_payload], units='m')
descent2.set_state_val('y', vals=[y_final, y_final], units='m')
descent2.set_state_val('z', vals=[z_final, 0], units='m')
descent2.set_control_val('T_x', vals=[0, 0], units='N')
descent2.set_control_val('T_y', vals=[0, 0], units='N')
descent2.set_control_val('T_z', vals=[-weight2*0.5, 0], units='N')
descent2.set_control_val('lx', vals=[0.0, 0.0], units='N*m')
descent2.set_control_val('ly', vals=[0.0, 0.0], units='N*m')
descent2.set_control_val('lz', vals=[0.0, 0.0], units='N*m')

p.run_model()
print("\n=== Checking Phase Continuity ===")
for state in ['u', 'v', 'w',
                          'roll_ang_vel', 'pitch_ang_vel', 'yaw_ang_vel',
                          'roll', 'pitch', 'yaw', 'x', 'y', 'z']:
    climb1_end = p.get_val(f'traj.phases.climb1.timeseries.{state}')[-1]
    cruise1_start = p.get_val(f'traj.phases.cruise1.timeseries.{state}')[0]
    cruise1_end = p.get_val(f'traj.phases.cruise1.timeseries.{state}')[-1]
    descent1_start = p.get_val(f'traj.phases.descent1.timeseries.{state}')[0]
    descent1_end = p.get_val(f'traj.phases.descent1.timeseries.{state}')[-1]
    climb2_start = p.get_val(f'traj.phases.climb2.timeseries.{state}')[0]
    climb2_end = p.get_val(f'traj.phases.climb2.timeseries.{state}')[-1]
    cruise2_start = p.get_val(f'traj.phases.cruise2.timeseries.{state}')[0]
    cruise2_end = p.get_val(f'traj.phases.cruise2.timeseries.{state}')[-1]
    descent2_start = p.get_val(f'traj.phases.descent2.timeseries.{state}')[0]

    print(f"\n{state}:")
    print(f"   climb1 end:    {climb1_end}")
    print(f"   cruise1 start:   {cruise1_start}  gap:  {abs(climb1_end-cruise1_start)}")
    print(f"   cruise1 end:   {cruise1_end}")
    print(f"   descent1 start:   {descent1_start}  (gap:  {abs(cruise1_end-descent1_start)})")
    print(f"   descent1_end:   {descent1_end}")
    print(f"   climb2_start:   {climb2_start}   (gap:   {abs(descent1_end - climb2_start)})")
    print(f"   climb2_end:   {climb2_end}")
    print(f"   cruise2_start:   {cruise2_start}   (gap:   {abs(climb2_end - cruise2_start)})")
    print(f"   cruise2_end:   {cruise2_end}")
    print(f"   descent2_start:   {descent2_start} (gap:   {abs(cruise2_end - descent2_start)})")

print("\n=== Checking Dynamics Magnitudes ===")
for phase_name in traj_names:
    for accel in ['dx_accel', 'dy_accel', 'dz_accel',
                  'roll_accel', 'pitch_accel', 'yaw_accel']:
        val = p.get_val(f'traj.phases.{phase_name}.rhs_all.{accel}')
        print(f"{phase_name}.{accel}: min={val.min():.2e}, max={val.max():.2e}, mean={np.abs(val).mean():.2e}")

        # Flag is accelerations are huge
        if np.abs(val).max() > 100:
            print(f"   WARNING: Very large accelerations!")

print("\n=== Checking Forces ===")
for phase_name in traj_names: 
    Fx = p.get_val(f'traj.phases.{phase_name}.rhs_all.forces.Fx')
    Fy = p.get_val(f'traj.phases.{phase_name}.rhs_all.forces.Fy')
    Fz = p.get_val(f'traj.phases.{phase_name}.rhs_all.forces.Fz')
    T_x = p.get_val(f'traj.phases.{phase_name}.rhs_all.T_x')
    T_y = p.get_val(f'traj.phases.{phase_name}.rhs_all.T_y')
    T_z = p.get_val(f'traj.phases.{phase_name}.rhs_all.T_z')
    drag = p.get_val(f'traj.phases.{phase_name}.rhs_all.aero.drag')
    T_mag = np.sqrt(T_x**2 + T_y**2 + T_z**2)

    print(f"\n{phase_name}:")
    print(f"   T_x:  {T_x.mean():.2f} N")
    print(f"   T_y:  {T_y.mean():.2f} N")
    print(f"   T_z:  {T_z.mean():.2f} N")
    print(f"   T_mag:  {T_mag.mean():.2f} N")
    print(f"   Drag:  {drag.mean():.2f} N")
    print(f"   Fx: min={Fx.min():.2f}, max={Fx.max():.2f}, mean={Fx.mean():.2f}")
    print(f"   Fy: min={Fy.min():.2f}, max={Fy.max():.2f}, mean={Fy.mean():.2f}")
    print(f"   Fz: min={Fz.min():.2f}, max={Fz.max():.2f}, mean={Fz.mean():.2f}")

    if np.abs(Fx).max() > 500 or np.abs(Fz).max() > 500:
        print(f"   ERROR: Forces are way too large!")

print("\n=== Checking Angles ===")
for phase_name in traj_names:
    alpha = p.get_val(f'traj.phases.{phase_name}.rhs_all.wind_angles.alpha')
    beta = p.get_val(f'traj.phases.{phase_name}.rhs_all.wind_angles.beta')

    print(f"\n{phase_name}:")
    for angle_name, angle_val in [('alpha', alpha), ('beta', beta)]:
        print(f"   {angle_name}: min={np.rad2deg(angle_val.min()):.2f} deg, "
              f"max={np.rad2deg(angle_val.max()):.2f} deg")
        if np.any(np.isnan(angle_val)):
            print(f"   ERROR: {angle_name} contains NaN!")

baseline_time = p.get_val(f'traj.descent2.timeseries.time', units='s')[-1]

dm.run_problem(p, 
               run_driver=True,
               simulate=True,
               solution_record_file='dymos_solution_fifth.db', 
               simulation_record_file='dymos_simulation_fifth.db')

obj_time = p.get_val('traj.descent2.timeseries.time', units='s')[-1]
print(f'Objective value: {obj_time} s')
print(f'Time difference (baseline - objective): {baseline_time - obj_time} s')
print(f'Percent difference: {obj_time / abs(baseline_time - obj_time) * 100}%')

# Post processing
sol = om.CaseReader(p.get_outputs_dir() / 'dymos_solution_fifth.db').get_case('final')
sim = om.CaseReader(traj.sim_prob.get_outputs_dir() / 'dymos_simulation_fifth.db').get_case('final')

t_sol = dict((phs, sol.get_val(f'traj.{phs}.timeseries.time'.format(phs)))
             for phs in traj_names)
z_sol = dict((phs, sol.get_val(f'traj.{phs}.timeseries.z'.format(phs)))
             for phs in traj_names)
x_sol = dict((phs, sol.get_val(f'traj.{phs}.timeseries.x'.format(phs)))
             for phs in traj_names)
y_sol = dict((phs, sol.get_val(f'traj.{phs}.timeseries.y'.format(phs)))
             for phs in traj_names)

t_sim = dict((phs, sim.get_val(f'traj.{phs}.timeseries.time'.format(phs)))
            for phs in traj_names)
z_sim = dict((phs, sim.get_val(f'traj.{phs}.timeseries.z'.format(phs)))
            for phs in traj_names)
x_sim = dict((phs, sim.get_val(f'traj.{phs}.timeseries.x'.format(phs)))
            for phs in traj_names)
y_sim = dict((phs, sim.get_val(f'traj.{phs}.timeseries.y'.format(phs)))
             for phs in traj_names)

#font
plt.rcParams.update({# Use mathtext, not LaTeX
                      'text.usetex': False,
                      # Use the Computer modern font
                      'font.family': 'serif',
                      'font.serif': 'cmr10',
                      'mathtext.fontset': 'cm',
                      'figure.autolayout': True,
                      # Use ASCII minus
                      'axes.unicode_minus': False,})

# fig = plt.figure(figsize=(12, 9))
# ax = fig.add_subplot(111, projection='3d')

# # Plot each phase
# for ph in traj_names:
#     # Simulation (line)
#     ax.plot(x_sim[ph], y_sim[ph], z_sim[ph], '-', color='C0', linewidth=2)
#     # Solution (dots)
#     ax.plot(x_sol[ph], y_sol[ph], z_sol[ph], 'o', 
#             color='C1', markersize=4, markeredgecolor='C1', markerfacecolor='C1')

# # Mark the payload pickup location
# ax.plot([x_payload], [0], [0], marker='*', color='gold', 
#         markersize=20, markeredgecolor='black', markeredgewidth=1.5,
#         label='Payload Pickup')

# # Mark start and end points
# ax.plot([0], [0], [0], marker='s', color='green', 
#         markersize=10, markeredgecolor='black', label='Start')
# ax.plot([x_payload], [y_final], [0], marker='s', color='red', 
#         markersize=10, markeredgecolor='black', label='End')

# # Labels and formatting
# ax.set_xlabel('X Position (m)', fontsize=12, labelpad=10)
# ax.set_ylabel('Y Position (m)', fontsize=12, labelpad=10)
# ax.set_zlabel('Z Position (m)', fontsize=12, labelpad=10)

# # Legend
# ax.legend(['Simulation', 'Solution Points', 'Payload Pickup', 'Start', 'End'], 
#           loc='upper left', fontsize=10)

# # Set viewing angle for better perspective
# ax.view_init(elev=20, azim=45)

# # Add grid
# ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

# # Optional: Create multiple views
# fig2 = plt.figure(figsize=(15, 5))

# # Top view (X-Y plane)
# ax1 = fig2.add_subplot(131)
# for ph in traj_names:
#     ax1.plot(x_sim[ph], y_sim[ph], '-', color='C0', linewidth=2)
#     ax1.plot(x_sol[ph], y_sol[ph], 'o', color='C1', markersize=3)
# ax1.plot(x_payload, 0, marker='*', color='gold', markersize=15, 
#          markeredgecolor='black', markeredgewidth=1.5)
# ax1.plot(0, 0, 's', color='green', markersize=8, markeredgecolor='black')
# ax1.plot(x_payload, y_final, 's', color='red', markersize=8, markeredgecolor='black')
# ax1.set_xlabel('X Position (m)')
# ax1.set_ylabel('Y Position (m)')
# ax1.grid(True, alpha=0.3)
# ax1.axis('equal')

# # Side view (X-Z plane)
# ax2 = fig2.add_subplot(132)
# for ph in traj_names:
#     ax2.plot(x_sim[ph], z_sim[ph], '-', color='C0', linewidth=2)
#     ax2.plot(x_sol[ph], z_sol[ph], 'o', color='C1', markersize=3)
# ax2.plot(x_payload, 0, marker='*', color='gold', markersize=15,
#          markeredgecolor='black', markeredgewidth=1.5)
# ax2.plot(0, 0, 's', color='green', markersize=8, markeredgecolor='black')
# ax2.plot(x_payload, 0, 's', color='red', markersize=8, markeredgecolor='black')
# ax2.set_xlabel('X Position (m)')
# ax2.set_ylabel('Z Position (m)')
# ax2.grid(True, alpha=0.3)

# # Front view (Y-Z plane)
# ax3 = fig2.add_subplot(133)
# for ph in traj_names:
#     ax3.plot(y_sim[ph], z_sim[ph], '-', color='C0', linewidth=2)
#     ax3.plot(y_sol[ph], z_sol[ph], 'o', color='C1', markersize=3)
# ax3.plot(0, 0, marker='*', color='gold', markersize=15,
#          markeredgecolor='black', markeredgewidth=1.5)
# ax3.plot(0, 0, 's', color='green', markersize=8, markeredgecolor='black')
# ax3.plot(y_final, 0, 's', color='red', markersize=8, markeredgecolor='black')
# ax3.set_xlabel('Y Position (m)')
# ax3.set_ylabel('Z Position (m)')
# ax3.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.show()

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot each phase
for ph in traj_names:
    # Simulation (line)
    sim_line = ax.plot(x_sim[ph], y_sim[ph], z_sim[ph], '-', color='C0', linewidth=2)
    # Solution (dots)
    sol_points = ax.plot(x_sol[ph], y_sol[ph], z_sol[ph], 'o', 
                         color='C1', markersize=4, markeredgecolor='C1', markerfacecolor='C1')

# Mark the payload pickup location
payload_marker = ax.plot([x_payload], [0], [0], marker='*', color='gold', 
                        markersize=20, markeredgecolor='black', markeredgewidth=1.5,
                        label='Payload Pickup')

# Mark start and end points
start_marker = ax.plot([0], [0], [0], marker='s', color='green', 
                       markersize=10, markeredgecolor='black', label='Start')
end_marker = ax.plot([x_payload], [y_final], [0], marker='s', color='red', 
                     markersize=10, markeredgecolor='black', label='End')

# Labels and formatting
ax.set_xlabel('X Position (m)', fontsize=12, labelpad=10)
ax.set_ylabel('Y Position (m)', fontsize=12, labelpad=10)
ax.set_zlabel('Z Position (m)', fontsize=12, labelpad=10)

# Create custom legend entries with correct colors/markers
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='C0', linewidth=2, label='Simulation'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='C1', 
           markeredgecolor='C1', markersize=6, label='Solution Points'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
           markeredgecolor='black', markeredgewidth=1.5, markersize=12, label='Payload Pickup'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
           markeredgecolor='black', markersize=8, label='Start'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
           markeredgecolor='black', markersize=8, label='End')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

# Set viewing angle for better perspective
ax.view_init(elev=20, azim=45)

# Add grid
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Optional: Create multiple views with master legend
fig2 = plt.figure(figsize=(15, 5))

# Top view (X-Y plane)
ax1 = fig2.add_subplot(131)
for ph in traj_names:
    ax1.plot(x_sim[ph], y_sim[ph], '-', color='C0', linewidth=2)
    ax1.plot(x_sol[ph], y_sol[ph], 'o', color='C1', markersize=3)
ax1.plot(x_payload, 0, marker='*', color='gold', markersize=15, 
        markeredgecolor='black', markeredgewidth=1.5)
ax1.plot(0, 0, 's', color='green', markersize=8, markeredgecolor='black')
ax1.plot(x_payload, y_final, 's', color='red', markersize=8, markeredgecolor='black')
ax1.set_xlabel('X Position (m)')
ax1.set_ylabel('Y Position (m)')
ax1.grid(True, alpha=0.3)
ax1.axis('equal')
ax1.text(0.5, -0.18, '(a)', transform=ax1.transAxes, 
         ha='center', va='top', fontsize=12, fontweight='bold')

# Side view (X-Z plane)
ax2 = fig2.add_subplot(132)
for ph in traj_names:
    ax2.plot(x_sim[ph], z_sim[ph], '-', color='C0', linewidth=2)
    ax2.plot(x_sol[ph], z_sol[ph], 'o', color='C1', markersize=3)
ax2.plot(x_payload, 0, marker='*', color='gold', markersize=15,
        markeredgecolor='black', markeredgewidth=1.5)
ax2.plot(0, 0, 's', color='green', markersize=8, markeredgecolor='black')
ax2.plot(x_payload, 0, 's', color='red', markersize=8, markeredgecolor='black')
ax2.set_xlabel('X Position (m)')
ax2.set_ylabel('Z Position (m)')
ax2.grid(True, alpha=0.3)
ax2.text(0.5, -0.18, '(b)', transform=ax2.transAxes, 
         ha='center', va='top', fontsize=12, fontweight='bold')

# Front view (Y-Z plane)
ax3 = fig2.add_subplot(133)
for ph in traj_names:
    ax3.plot(y_sim[ph], z_sim[ph], '-', color='C0', linewidth=2)
    ax3.plot(y_sol[ph], z_sol[ph], 'o', color='C1', markersize=3)
ax3.plot(0, 0, marker='*', color='gold', markersize=15,
        markeredgecolor='black', markeredgewidth=1.5)
ax3.plot(0, 0, 's', color='green', markersize=8, markeredgecolor='black')
ax3.plot(y_final, 0, 's', color='red', markersize=8, markeredgecolor='black')
ax3.set_xlabel('Y Position (m)')
ax3.set_ylabel('Z Position (m)')
ax3.grid(True, alpha=0.3)
ax3.text(0.5, -0.18, '(c)', transform=ax3.transAxes, 
         ha='center', va='top', fontsize=12, fontweight='bold')

# Create master legend for the multi-view figure
fig2.legend(handles=legend_elements, loc='upper center', 
           bbox_to_anchor=(0.5, 0.98), ncol=5, fontsize=10, frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave room for legend at top
plt.show()



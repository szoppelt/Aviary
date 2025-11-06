import numpy as np
import matplotlib.pyplot as plt
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

# Parameters (static)
traj.add_parameter('mass', units='kg', 
                   targets={'climb': ['mass'], 'cruise': ['mass'], 'descent': ['mass']},
                   opt=False, static_target=True)
traj.add_parameter('J_xx', units='kg * m**2', 
                   targets={'climb': ['J_xx'], 'cruise': ['J_xx'], 'descent': ['J_xx']},
                   opt=False, static_target=True)
traj.add_parameter('J_yy', units='kg * m**2', 
                   targets={'climb': ['J_yy'], 'cruise': ['J_yy'], 'descent': ['J_yy']},
                   opt=False, static_target=True)
traj.add_parameter('J_zz', units='kg * m**2', 
                   targets={'climb': ['J_zz'], 'cruise': ['J_zz'], 'descent': ['J_zz']},
                   opt=False, static_target=True)
traj.add_parameter('J_xz', units='kg * m**2', 
                   targets={'climb': ['J_xz'], 'cruise': ['J_xz'], 'descent': ['J_xz']},
                   opt=False, static_target=True)
traj.add_parameter('g', units='m / s**2', 
                   targets={'climb': ['g'], 'cruise': ['g'], 'descent': ['g']},
                    opt=False, static_target=True)
traj.add_parameter('sphere_radius', units='m', 
                   targets={'climb': ['sphere_radius'], 'cruise': ['sphere_radius'], 'descent': ['sphere_radius']},
                    opt=False, static_target=True, val=0.12)
traj.add_parameter('sphere_Cd', targets={'climb': ['sphere_Cd'], 'cruise': ['sphere_Cd'], 'descent': ['sphere_Cd']},
                    opt=False, static_target=True, val=0.47)
# traj.add_parameter('lx', units='N*m', 
#                    targets={'climb': ['lx'], 'cruise': ['lx'], 'descent': ['lx']},
#                    opt=False, static_target=True)
# traj.add_parameter('ly', units='N*m', 
#                    targets={'climb': ['ly'], 'cruise': ['ly'], 'descent': ['ly']},
#                    opt=False, static_target=True)
# traj.add_parameter('lz', units='N*m', 
#                    targets={'climb': ['lz'], 'cruise': ['lz'], 'descent': ['lz']},
#                    opt=False, static_target=True)


z_final = 100.0 # m

# First phase (climb)

climb = dm.Phase(ode_class=vtolODE,
                 transcription=dm.Radau(num_segments=10, order=3))

climb = traj.add_phase('climb', climb)

climb.set_time_options(fix_initial=True, duration_bounds=(5, 100), duration_ref=40, units='s')
climb.add_state('u', fix_initial=True, fix_final=False, rate_source='dx_accel', 
                targets=['u'], units='m/s', ref=1, defect_ref=1)
climb.add_state('v', fix_initial=True, fix_final=False, rate_source='dy_accel', 
                targets=['v'], units='m/s', ref=1, defect_ref=1)
climb.add_state('w', fix_initial=True, fix_final=False, rate_source='dz_accel', 
                targets=['w'], units='m/s', ref=1, defect_ref=1)
climb.add_state('roll_ang_vel', fix_initial=True, fix_final=False, rate_source='roll_accel',
                targets=['roll_ang_vel'], units='rad/s', ref=1, defect_ref=1)
climb.add_state('pitch_ang_vel', fix_initial=True, fix_final=False, rate_source='pitch_accel',
                targets=['pitch_ang_vel'], units='rad/s', ref=1, defect_ref=1)
climb.add_state('yaw_ang_vel', fix_initial=True, fix_final=False, rate_source='yaw_accel',
                targets=['yaw_ang_vel'], units='rad/s', ref=1, defect_ref=1)
climb.add_state('roll', fix_initial=True, fix_final=False, rate_source='roll_angle_rate_eq', 
                targets=['roll'], units='rad', ref=1, defect_ref=1)
climb.add_state('pitch', fix_initial=True, fix_final=False, rate_source='pitch_angle_rate_eq', 
                targets=['pitch'], units='rad', ref=1, defect_ref=1)
climb.add_state('yaw', fix_initial=True, fix_final=False, rate_source='yaw_angle_rate_eq', 
                targets=['yaw'], units='rad', ref=1, defect_ref=1)
climb.add_state('x', fix_initial=True, fix_final=False, rate_source='dx_dt',
                targets=['x'], units='m', ref=10, defect_ref=0.5)
climb.add_state('y', fix_initial=True, fix_final=False, rate_source='dy_dt',
                targets=['y'], units='m', ref=10, defect_ref=0.5)
climb.add_state('z', fix_initial=True, fix_final=False, rate_source='dz_dt',
                targets=['z'], units='m', ref=z_final, defect_ref=2.0)

# Controls (without explicit scaling for now)
climb.add_control('T_x', targets=['T_x'], opt=True,units='N', lower=-100.0, upper=100.0)
climb.add_control('T_y', targets=['T_y'], opt=True,units='N', lower=-100.0, upper=100.0)
climb.add_control('T_z', targets=['T_z'], opt=True,units='N', lower=-100.0, upper=100.0)
climb.add_control('lx', targets=['lx'], opt=False, units='N*m', val=0.0)
climb.add_control('ly', targets=['ly'], opt=False, units='N*m', val=0.0)
climb.add_control('lz', targets=['lz'], opt=False, units='N*m', val=0.0)
climb.add_boundary_constraint('z', loc='final', equals=z_final, units='m', scaler=0.01)
#climb.add_path_constraint('T_climb=(T_x**2+T_y**2+T_z**2)**0.5', lower=0.0, upper=25) # upper = mg
climb.add_path_constraint('x', lower=0.0, upper=100, units='m')
#climb.add_path_constraint('y', lower=0.0, upper=10, units='m')


# Second phase (cruise)
cruise = dm.Phase(ode_class=vtolODE,
                 transcription=dm.Radau(num_segments=10, order=3))

cruise = traj.add_phase('cruise', cruise)

cruise.set_time_options(fix_initial=False, initial_bounds=(10, 100), duration_bounds=(30, 200), duration_ref=80, units='s')
cruise.add_state('u', fix_initial=False, fix_final=False, rate_source='dx_accel', targets=['u'], units='m/s', ref=10, defect_ref=1)
cruise.add_state('v', fix_initial=False, fix_final=False, rate_source='dy_accel', targets=['v'], units='m/s', ref=1, defect_ref=0.1)
cruise.add_state('w', fix_initial=False, fix_final=False, rate_source='dz_accel', targets=['w'], units='m/s', ref=1, defect_ref=0.1)
cruise.add_state('roll_ang_vel', fix_initial=False, fix_final=False, rate_source='roll_accel', targets=['roll_ang_vel'], units='rad/s', ref=1, defect_ref=0.1)
cruise.add_state('pitch_ang_vel', fix_initial=False, fix_final=False, rate_source='pitch_accel', targets=['pitch_ang_vel'], units='rad/s', ref=1, defect_ref=0.1)
cruise.add_state('yaw_ang_vel', fix_initial=False, fix_final=False, rate_source='yaw_accel', targets=['yaw_ang_vel'], units='rad/s', ref=1, defect_ref=0.1)
cruise.add_state('roll', fix_initial=False, fix_final=False, rate_source='roll_angle_rate_eq', targets=['roll'], units='rad', ref=1, defect_ref=0.1)
cruise.add_state('pitch', fix_initial=False, fix_final=False, rate_source='pitch_angle_rate_eq', targets=['pitch'], units='rad', ref=1, defect_ref=0.1)
cruise.add_state('yaw', fix_initial=False, fix_final=False, rate_source='yaw_angle_rate_eq', targets=['yaw'], units='rad', ref=1, defect_ref=0.1)
cruise.add_state('x', fix_initial=False, fix_final=False, rate_source='dx_dt', targets=['x'], units='m', ref=150, defect_ref=5.0)
cruise.add_state('y', fix_initial=False, fix_final=False, rate_source='dy_dt', targets=['y'], units='m', ref=10, defect_ref=0.5)
cruise.add_state('z', fix_initial=False, fix_final=False, rate_source='dz_dt', targets=['z'], units='m', ref=z_final, defect_ref=2.0)
# Controls (without explicit scaling for now)
cruise.add_control('T_x', targets=['T_x'], opt=True,units='N', lower=-100.0, upper=100.0)
cruise.add_control('T_y', targets=['T_y'], opt=True,units='N', lower=-100.0, upper=100.0)
cruise.add_control('T_z', targets=['T_z'], opt=True,units='N', lower=-100.0, upper=100.0)
cruise.add_control('lx', targets=['lx'], opt=False, units='N*m', val=0.0)
cruise.add_control('ly', targets=['ly'], opt=False, units='N*m', val=0.0)
cruise.add_control('lz', targets=['lz'], opt=False, units='N*m', val=0.0)
#cruise.add_path_constraint('z', lower=z_final - 50.0, upper=z_final + 50.0, units='m')
#cruise.add_boundary_constraint('z', loc='initial', equals=z_final, units='m')
cruise.add_boundary_constraint('z', loc='final', lower=z_final - 20, upper=z_final + 20, units='m')
#cruise.add_path_constraint('T_cruise=(T_x**2+T_y**2+T_z**2)**0.5', lower=0.0, upper=25) # upper = mg
cruise.add_path_constraint('x', lower=0.0, upper=1500.0, units='m')
#cruise.add_path_constraint('y', lower=0.0, upper=10.0, units='m')
#cruise.add_path_constraint('z', lower=60.0, upper=100.0, units='m')

descent = dm.Phase(ode_class=vtolODE,
                 transcription=dm.Radau(num_segments=10, order=3))

descent = traj.add_phase('descent', descent)
descent.set_time_options(fix_initial=False, initial_bounds=(50, 200), duration_bounds=(10, 200), duration_ref=40, units='s')
descent.add_state('u', fix_initial=False, fix_final=True, rate_source='dx_accel', targets=['u'], units='m/s', ref=1, defect_ref=0.1)
descent.add_state('v', fix_initial=False, fix_final=True, rate_source='dy_accel', targets=['v'], units='m/s', ref=1, defect_ref=0.1)
descent.add_state('w', fix_initial=False, fix_final=True, rate_source='dz_accel', targets=['w'], units='m/s', ref=1, defect_ref=0.1)
descent.add_state('roll_ang_vel', fix_initial=False, fix_final=False, rate_source='roll_accel', targets=['roll_ang_vel'], units='rad/s', ref=1, defect_ref=0.1)
descent.add_state('pitch_ang_vel', fix_initial=False, fix_final=False, rate_source='pitch_accel', targets=['pitch_ang_vel'], units='rad/s', ref=1, defect_ref=0.1)
descent.add_state('yaw_ang_vel', fix_initial=False, fix_final=False, rate_source='yaw_accel', targets=['yaw_ang_vel'], units='rad/s', ref=1, defect_ref=0.1)
descent.add_state('roll', fix_initial=False, fix_final=False, rate_source='roll_angle_rate_eq', targets=['roll'], units='rad', ref=1, defect_ref=0.1)
descent.add_state('pitch', fix_initial=False, fix_final=False, rate_source='pitch_angle_rate_eq', targets=['pitch'], units='rad', ref=1, defect_ref=0.1)
descent.add_state('yaw', fix_initial=False, fix_final=False, rate_source='yaw_angle_rate_eq', targets=['yaw'], units='rad', ref=1, defect_ref=0.1)
descent.add_state('x', fix_initial=False, fix_final=False, rate_source='dx_dt', targets=['x'], units='m', ref=10, defect_ref=1.0)
descent.add_state('y', fix_initial=False, fix_final=False, rate_source='dy_dt', targets=['y'], units='m', ref=10, defect_ref=0.5)
descent.add_state('z', fix_initial=False, fix_final=True, rate_source='dz_dt', targets=['z'], units='m', ref=z_final, defect_ref=2.0)
descent.add_objective('time', loc='final', ref=120.0) # minimize time
# Controls (without explicit scaling for now)
descent.add_control('T_x', targets=['T_x'], opt=True, units='N', lower=-100.0, upper=100.0)
descent.add_control('T_y', targets=['T_y'], opt=True, units='N', lower=-100.0, upper=100.0)
descent.add_control('T_z', targets=['T_z'], opt=True, units='N', lower=-100.0, upper=100.0)
descent.add_control('lx', targets=['lx'], opt=False, units='N*m', val=0.0)
descent.add_control('ly', targets=['ly'], opt=False, units='N*m', val=0.0)
descent.add_control('lz', targets=['lz'], opt=False, units='N*m', val=0.0)
#descent.add_boundary_constraint('z', loc='final', equals=0.0, units='m', scaler=0.01)
#descent.add_path_constraint('T_descent=(T_x**2+T_y**2+T_z**2)**0.5', lower=0.0, upper=25) # upper = mg
#descent.add_path_constraint('x', lower=0.0, upper=1000.0, units='m')
#descent.add_path_constraint('y', lower=0.0, upper=10.0, units='m')



traj.link_phases(['climb', 'cruise', 'descent'],
                    vars=['time', 'u', 'v', 'w',
                          'roll_ang_vel', 'pitch_ang_vel', 'yaw_ang_vel',
                          'roll', 'pitch', 'yaw', 'x', 'y', 'z'],
                    connected=True)

p.model.add_subsystem('traj', subsys=traj)
p.setup(check=True)

#print("\n=== Verifying Linkage Constraints ===")
#print("Number of linkage constraints:", len([c for c in p.model.traj._linkage_constraints]))

# Initial guesses

p.set_val('traj.parameters:mass', val=2.0, units='kg')
p.set_val('traj.parameters:J_xx', val=0.2, units='kg*m**2')
p.set_val('traj.parameters:J_yy', val=0.2, units='kg*m**2')
p.set_val('traj.parameters:J_zz', val=0.2, units='kg*m**2')
p.set_val('traj.parameters:J_xz', val=0.0, units='kg*m**2')
p.set_val('traj.parameters:sphere_radius', val=0.5, units='m')
p.set_val('traj.parameters:sphere_Cd', val=0.47)
p.set_val('traj.parameters:g', val=9.81, units='m/s**2')


climb = p.model.traj.phases.climb
cruise = p.model.traj.phases.cruise
descent = p.model.traj.phases.descent

climb.set_time_options(fix_initial=True)
climb.set_time_val(initial=0.0, duration=40, units='s')

# Hover thrust (T=W=mg)
weight = 2 * 9.81

#climb.add_boundary_constraint('z', loc='initial', equals=0.0, units='m')
#climb.add_boundary_constraint('u', loc='initial', equals=0.0, units='m/s')
#climb.add_boundary_constraint('v', loc='initial', equals=0.0, units='m/s')
#climb.add_boundary_constraint('w', loc='initial', equals=0.0, units='m/s')
climb.set_state_val('u', vals=[0, 0], units='m/s')
climb.set_state_val('v', vals=[0, 0], units='m/s')
climb.set_state_val('w', vals=[0, 2.5], units='m/s')
climb.set_state_val('roll_ang_vel', vals=[0, 0], units='rad/s')
climb.set_state_val('pitch_ang_vel', vals=[0, 0], units='rad/s')
climb.set_state_val('yaw_ang_vel', vals=[0, 0], units='rad/s')
climb.set_state_val('roll', vals=[0, 0], units='rad')
climb.set_state_val('pitch', vals=[0, 0], units='rad')
climb.set_state_val('yaw', vals=[0, 0], units='rad')
climb.set_state_val('x', vals=[0, 10], units='m')
climb.set_state_val('y', vals=[0, 0], units='m')
climb.set_state_val('z', vals=[0, z_final], units='m')
climb.set_control_val('T_x', vals=[0, 0], units='N')
climb.set_control_val('T_y', vals=[0, 0], units='N')
climb.set_control_val('T_z', vals=[0, -weight*1.5], units='N')
climb.set_control_val('lx', vals=[0.0, 0.0], units='N*m')
climb.set_control_val('ly', vals=[0.0, 0.0], units='N*m')
climb.set_control_val('lz', vals=[0.0, 0.0], units='N*m')

cruise.set_time_val(initial=40.0, duration=80.0, units='s')
cruise.set_state_val('u', vals=[10, 10], units='m/s')
cruise.set_state_val('v', vals=[0, 0], units='m/s')
cruise.set_state_val('w', vals=[0, 0], units='m/s')
cruise.set_state_val('roll_ang_vel', vals=[0, 0], units='rad/s')
cruise.set_state_val('pitch_ang_vel', vals=[0, 0], units='rad/s')
cruise.set_state_val('yaw_ang_vel', vals=[0, 0], units='rad/s')
cruise.set_state_val('roll', vals=[0, 0], units='rad')
cruise.set_state_val('pitch', vals=[0, 0], units='rad')
cruise.set_state_val('yaw', vals=[0, 0], units='rad')
cruise.set_state_val('x', vals=[10, 810], units='m')
cruise.set_state_val('y', vals=[0, 0], units='m')
cruise.set_state_val('z', vals=[z_final, z_final], units='m')
cruise.set_control_val('T_x', vals=[1.28, 1.28], units='N')
cruise.set_control_val('T_y', vals=[0, 0], units='N')
cruise.set_control_val('T_z', vals=[-weight, -weight], units='N')
cruise.set_control_val('lx', vals=[0.0, 0.0], units='N*m')
cruise.set_control_val('ly', vals=[0.0, 0.0], units='N*m')
cruise.set_control_val('lz', vals=[0.0, 0.0], units='N*m')

descent.set_time_val(initial=120, duration=40, units='s')
descent.set_state_val('u', vals=[0, 0], units='m/s')
descent.set_state_val('v', vals=[0, 0], units='m/s')
descent.set_state_val('w', vals=[2.5, 0], units='m/s')
descent.set_state_val('roll_ang_vel', vals=[0, 0], units='rad/s')
descent.set_state_val('pitch_ang_vel', vals=[0, 0], units='rad/s')
descent.set_state_val('yaw_ang_vel', vals=[0, 0], units='rad/s')
descent.set_state_val('roll', vals=[0, 0], units='rad')
descent.set_state_val('pitch', vals=[0, 0], units='rad')
descent.set_state_val('yaw', vals=[0, 0], units='rad')
descent.set_state_val('x', vals=[810, 820], units='m')
descent.set_state_val('y', vals=[0, 0], units='m')
descent.set_state_val('z', vals=[z_final, 0], units='m')
descent.set_control_val('T_x', vals=[0, 0], units='N')
descent.set_control_val('T_y', vals=[0, 0], units='N')
descent.set_control_val('T_z', vals=[-weight*0.5, 0], units='N')
descent.set_control_val('lx', vals=[0.0, 0.0], units='N*m')
descent.set_control_val('ly', vals=[0.0, 0.0], units='N*m')
descent.set_control_val('lz', vals=[0.0, 0.0], units='N*m')

p.run_model()
print("\n=== Checking Phase Continuity ===")
for state in ['u', 'v', 'w',
                          'roll_ang_vel', 'pitch_ang_vel', 'yaw_ang_vel',
                          'roll', 'pitch', 'yaw', 'x', 'y', 'z']:
    climb_end = p.get_val(f'traj.phases.climb.timeseries.{state}')[-1]
    cruise_start = p.get_val(f'traj.phases.cruise.timeseries.{state}')[0]
    cruise_end = p.get_val(f'traj.phases.cruise.timeseries.{state}')[-1]
    descent_start = p.get_val(f'traj.phases.descent.timeseries.{state}')[0]

    print(f"\n{state}:")
    print(f"   Climb end:    {climb_end}")
    print(f"   Cruise start:   {cruise_start}  gap:  {abs(climb_end-cruise_start)}")
    print(f"   Cruise end:   {cruise_end}")
    print(f"   Descent start:   {descent_start}  (gap:  {abs(cruise_end-descent_start)})")

z_cruise = p.get_val('traj.phases.cruise.timeseries.z')
print(f"\n=== Cruise Altitude Check ===")
print(f"   z range: [{z_cruise.min():.2f}, {z_cruise.max():.2f}]")
print(f"   Constraint: [150, 50]")
if z_cruise.min() < 150 or z_cruise.max() > 50:
    print("  WARNING: Initial guess violates cruise path constraint!")

print("\n=== Checking Dynamics Magnitudes ===")
for phase_name in ['climb', 'cruise', 'descent']:
    for accel in ['dx_accel', 'dy_accel', 'dz_accel',
                  'roll_accel', 'pitch_accel', 'yaw_accel']:
        val = p.get_val(f'traj.phases.{phase_name}.rhs_all.{accel}')
        print(f"{phase_name}.{accel}: min={val.min():.2e}, max={val.max():.2e}, mean={np.abs(val).mean():.2e}")

        # Flag is accelerations are huge
        if np.abs(val).max() > 100:
            print(f"   WARNING: Very large accelerations!")

print("\n=== Checking Forces ===")
for phase_name in ['climb', 'cruise', 'descent']: 
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
for phase_name in ['climb', 'cruise', 'descent']:
    alpha = p.get_val(f'traj.phases.{phase_name}.rhs_all.wind_angles.alpha')
    beta = p.get_val(f'traj.phases.{phase_name}.rhs_all.wind_angles.beta')

    print(f"\n{phase_name}:")
    for angle_name, angle_val in [('alpha', alpha), ('beta', beta)]:
        print(f"   {angle_name}: min={np.rad2deg(angle_val.min()):.2f} deg, "
              f"max={np.rad2deg(angle_val.max()):.2f} deg")
        if np.any(np.isnan(angle_val)):
            print(f"   ERROR: {angle_name} contains NaN!")

dm.run_problem(p, 
               run_driver=True,
               simulate=True,
               solution_record_file='dymos_solution_4.db', 
               simulation_record_file='dymos_simulation_4.db')

#exp_out = traj.simulate()

# Post processing
sol = om.CaseReader(p.get_outputs_dir() / 'dymos_solution_4.db').get_case('final')
sim = om.CaseReader(traj.sim_prob.get_outputs_dir() / 'dymos_simulation_4.db').get_case('final')

t_sol = dict((phs, sol.get_val(f'traj.{phs}.timeseries.time'.format(phs)))
             for phs in ['climb', 'cruise', 'descent'])
z_sol = dict((phs, sol.get_val(f'traj.{phs}.timeseries.z'.format(phs)))
             for phs in ['climb', 'cruise','descent'])
x_sol = dict((phs, sol.get_val(f'traj.{phs}.timeseries.x'.format(phs)))
             for phs in ['climb', 'cruise', 'descent'])

t_sim = dict((phs, sim.get_val(f'traj.{phs}.timeseries.time'.format(phs)))
            for phs in ['climb', 'cruise', 'descent'])
z_sim = dict((phs, sim.get_val(f'traj.{phs}.timeseries.z'.format(phs)))
            for phs in ['climb', 'cruise','descent'])
x_sim = dict((phs, sim.get_val(f'traj.{phs}.timeseries.x'.format(phs)))
            for phs in ['climb', 'cruise', 'descent'])

for ph in ['climb', 'cruise', 'descent']:
    plt.plot(t_sol[ph], z_sol[ph], 'o', mfc='C1', mec='C1', ms=3)
    plt.plot(t_sim[ph], z_sim[ph], '-', color='C0')
    plt.xlabel('time (s)')
    plt.ylabel('Altitude (m)')
    plt.title('Optimized Trajectory Plot - z v time')

plt.legend(labels=['Solution', 'Simulation'])
plt.show()

for ph in ['climb', 'cruise', 'descent']:
    plt.plot(x_sol[ph], z_sol[ph], 'o', mfc='C1', mec='C1', ms=3)
    plt.plot(x_sim[ph], z_sim[ph], '-', color='C0')
    plt.legend()
    plt.xlabel('Distance (m)')
    plt.ylabel('Altitude (m)')
    plt.title('Optimized Trajectory Plot - z v x')

plt.legend(labels=['Solution', 'Simulation'])
plt.show()


# # Plot altitude vs time across all phases
# for ph in ['climb', 'cruise', 'descent']:
#     t = exp_out.get_val(f'traj.{ph}.timeseries.time')
#     z = exp_out.get_val(f'traj.{ph}.timeseries.z')
#     plt.plot(t, z, label=ph)
#     plt.legend()
#     plt.xlabel('Time (s)')
#     plt.ylabel('Altitude z (m)')
#     plt.title('VTOL Trajectory with Climb, Cruise, Descent')
    
# plt.show()

# for ph in ['climb', 'cruise', 'descent']:
#     x = exp_out.get_val(f'traj.{ph}.timeseries.x')
#     plt.plot(x, z, label=ph)
#     plt.legend()
#     plt.xlabel('x (m)')
#     plt.ylabel('z (m)')
#     plt.title('altitude v distance trajectory')
# plt.show()

# Create plots with correct grid
# fig = plt.figure(figsize=(14, 10))
# ax_tz = plt.subplot2grid((2, 2), (0, 0))
# ax_xz = plt.subplot2grid((2, 2), (0, 1))
# ax_uvw = plt.subplot2grid((2, 2), (1, 0))
# ax_thrust = plt.subplot2grid((2, 2), (1, 1))

# fig.suptitle('UAV Mission Trajectory', fontsize=16)

# # Time vs altitude
# ax_tz.set_xlabel('Time (s)')
# ax_tz.set_ylabel('Altitude (m)')
# ax_tz.set_title('Altitude vs Time')
# ax_tz.grid(True)

# for phs in ['climb', 'cruise', 'descent']:
#     ax_tz.plot(t_sim[phs], -z_sim[phs], '-', color='C0', linewidth=2, label='simulation' if phs == 'climb' else '')
#     ax_tz.plot(t_sol[phs], -z_sol[phs], 'o', mfc='C1', mec='C1', ms=4, label='solution' if phs == 'climb' else '')

# ax_tz.legend()

# # X-Z trajectory
# ax_xz.set_xlabel('Downrange (m)')
# ax_xz.set_ylabel('Altitude (m)')
# ax_xz.set_title('Flight Path')
# ax_xz.grid(True)

# for phs in ['climb', 'cruise', 'descent']:
#     ax_xz.plot(x_sim[phs], -z_sim[phs], '-', color='C0', linewidth=2, label='simulation' if phs == 'climb' else '')
#     ax_xz.plot(x_sol[phs], -z_sol[phs], 'o', mfc='C1', mec='C1', ms=4, label='solution' if phs == 'climb' else '')

# ax_xz.legend()

# # Velocity components
# ax_uvw.set_xlabel('Time (s)')
# ax_uvw.set_ylabel('Velocity (m/s)')
# ax_uvw.set_title('Velocity Components')
# ax_uvw.grid(True)

# for phs in ['climb', 'cruise', 'descent']:
#     t = t_sim[phs]
#     u = sim.get_case('final').get_val(f'traj.{phs}.timeseries.u')
#     v = sim.get_case('final').get_val(f'traj.{phs}.timeseries.v')
#     w = sim.get_case('final').get_val(f'traj.{phs}.timeseries.w')
    
#     if phs == 'climb':
#         ax_uvw.plot(t, u, '-', color='C0', label='u (forward)')
#         ax_uvw.plot(t, v, '-', color='C1', label='v (side)')
#         ax_uvw.plot(t, w, '-', color='C2', label='w (down)')
#     else:
#         ax_uvw.plot(t, u, '-', color='C0')
#         ax_uvw.plot(t, v, '-', color='C1')
#         ax_uvw.plot(t, w, '-', color='C2')

# ax_uvw.legend()

# # Thrust components
# ax_thrust.set_xlabel('Time (s)')
# ax_thrust.set_ylabel('Thrust (N)')
# ax_thrust.set_title('Thrust Components')
# ax_thrust.grid(True)

# for phs in ['climb', 'cruise', 'descent']:
#     t = t_sim[phs]
#     T_x = sim.get_case('final').get_val(f'traj.{phs}.timeseries.T_x')
#     T_y = sim.get_case('final').get_val(f'traj.{phs}.timeseries.T_y')
#     T_z = sim.get_case('final').get_val(f'traj.{phs}.timeseries.T_z')
#     T_mag = np.sqrt(T_x**2 + T_y**2 + T_z**2)
    
#     if phs == 'climb':
#         ax_thrust.plot(t, T_x, '-', color='C0', label='T_x')
#         ax_thrust.plot(t, T_y, '-', color='C1', label='T_y')
#         ax_thrust.plot(t, T_z, '-', color='C2', label='T_z')
#         ax_thrust.plot(t, T_mag, '-', color='black', linewidth=2, label='|T| total')
#     else:
#         ax_thrust.plot(t, T_x, '-', color='C0')
#         ax_thrust.plot(t, T_y, '-', color='C1')
#         ax_thrust.plot(t, T_z, '-', color='C2')
#         ax_thrust.plot(t, T_mag, '-', color='black', linewidth=2)

# ax_thrust.axhline(y=hover_thrust, color='red', linestyle='--', label='Weight (mg)')
# ax_thrust.legend()

# plt.tight_layout()
# plt.savefig('vtol_trajectory_results.png', dpi=150)
# print("\nPlot saved as 'vtol_trajectory_results.png'")
# plt.show()

# # Print final statistics
# print("\n=== Mission Summary ===")
# final_time = t_sol['descent'][-1]
# final_x = x_sol['descent'][-1]
# print(f"Total mission time: {final_time:.2f} s")
# print(f"Total distance traveled: {final_x:.2f} m")
# print(f"Average speed: {final_x/final_time:.2f} m/s")


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
                          om.ExecComp(['V = (u**2 + v**2 + w**2)**0.5 + 1e-8',
                                      'alpha = arctan2(w, u)',  # angle of attack
                                      'beta = arcsin(v / ((u**2 + v**2 + w**2)**0.5 + 1e-8))',    # sideslip angle
                                      'gamma = -arcsin(w / ((u**2 + v**2 + w**2)**0.5 + 1e-8))',  # flight path angle (wind frame)
                                      'chi = arctan2(v, u)'],    # heading angle (wind frame)
                                     V={'units': 'm/s', 'shape': (nn,)},
                                     alpha={'units': 'rad', 'shape': (nn,)},
                                     beta={'units': 'rad', 'shape': (nn,)},
                                     gamma={'units': 'rad', 'shape': (nn,)},
                                     chi={'units': 'rad', 'shape': (nn,)},
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

        
        # For simplicity with a sphere, we can assume thrust is purely vertical
        # and wind angles equal body angles
        self.add_subsystem('thrust_angles',
                          om.ExecComp(['heading_angle = 1e-8',
                                      'flight_path_angle = 1e-8',
                                      'heading_angle_NED = 1e-8',
                                      'fpa_NED = -pi/2'],  # -90 deg = straight down
                                     heading_angle={'units': 'rad', 'shape': (nn,), 'val': np.zeros(nn)},
                                     flight_path_angle={'units': 'rad', 'shape': (nn,), 'val': np.zeros(nn)},
                                     heading_angle_NED={'units': 'rad', 'shape': (nn,), 'val': np.zeros(nn)},
                                     fpa_NED={'units': 'rad', 'shape': (nn,), 'val': np.ones(nn) * (-np.pi/2)}),
                          promotes_outputs=['heading_angle', 'flight_path_angle',
                                          'heading_angle_NED', 'fpa_NED'])

        # Force resolution
        self.add_subsystem('forces', 
                          ForceComponentResolver(num_nodes=nn),
                          promotes_inputs=['u', 'v', 'w', 'thrust'])
        
        # Connect aero forces
        self.connect('aero.drag', 'forces.drag')
        self.connect('aero.lift', 'forces.lift')
        self.connect('aero.side', 'forces.side')
        self.connect('heading_angle', 'forces.heading_angle')
        self.connect('flight_path_angle', 'forces.flight_path_angle')
        self.connect('heading_angle_NED', 'forces.heading_angle_NED')
        self.connect('fpa_NED', 'forces.fpa_NED')

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
p.driver.opt_settings['Major feasibility tolerance'] = 1.0E-6
p.driver.opt_settings['Major optimality tolerance'] = 1.0E-5
p.driver.opt_settings['iSumm'] = 6
p.driver.opt_settings['Verify level'] = 3
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


z_final = -100.0 # m

# First phase (climb)

climb = dm.Phase(ode_class=vtolODE,
                 transcription=dm.Radau(num_segments=5, order=3))

climb = traj.add_phase('climb', climb)

climb.set_time_options(fix_initial=True, duration_bounds=(.5, 40), units='s')
climb.add_state('u', fix_initial=True, fix_final=False, rate_source='dx_accel', 
                targets=['u'], units='m/s')
climb.add_state('v', fix_initial=True, fix_final=False, rate_source='dy_accel', 
                targets=['v'], units='m/s')
climb.add_state('w', fix_initial=True, fix_final=False, rate_source='dz_accel', 
                targets=['w'], units='m/s')
climb.add_state('roll_ang_vel', fix_initial=True, fix_final=False, rate_source='roll_accel',
                targets=['roll_ang_vel'], units='rad/s')
climb.add_state('pitch_ang_vel', fix_initial=True, fix_final=False, rate_source='pitch_accel',
                targets=['pitch_ang_vel'], units='rad/s')
climb.add_state('yaw_ang_vel', fix_initial=True, fix_final=False, rate_source='yaw_accel',
                targets=['yaw_ang_vel'], units='rad/s')
climb.add_state('roll', fix_initial=True, fix_final=False, rate_source='roll_angle_rate_eq', 
                targets=['roll'], units='rad')
climb.add_state('pitch', fix_initial=True, fix_final=False, rate_source='pitch_angle_rate_eq', 
                targets=['pitch'], units='rad')
climb.add_state('yaw', fix_initial=True, fix_final=False, rate_source='yaw_angle_rate_eq', 
                targets=['yaw'], units='rad')
climb.add_state('x', fix_initial=True, fix_final=False, rate_source='dx_dt',
                targets=['x'], units='m')
climb.add_state('y', fix_initial=True, fix_final=False, rate_source='dy_dt',
                targets=['y'], units='m')
climb.add_state('z', fix_initial=True, fix_final=False, rate_source='dz_dt',
                targets=['z'], units='m')

# Controls (without explicit scaling for now)
climb.add_control('thrust', targets=['thrust'], opt=True,
                 units='N', lower=0.0, upper=200.0)
climb.add_control('lx', targets=['lx'], opt=True,
                 units='N*m', lower=-5.0, upper=5.0)
climb.add_control('ly', targets=['ly'], opt=True,
                 units='N*m', lower=-5.0, upper=5.0)
climb.add_control('lz', targets=['lz'], opt=True,
                 units='N*m', lower=-5.0, upper=5.0)
climb.add_boundary_constraint('z', loc='final', equals=z_final, units='m')


# Second phase (cruise)
cruise = dm.Phase(ode_class=vtolODE,
                 transcription=dm.Radau(num_segments=5, order=3))

cruise = traj.add_phase('cruise', cruise)

cruise.set_time_options(fix_initial=False, initial_bounds=(0.5, 50), duration_bounds=(0.5, 80), duration_ref=80, units='s')
cruise.add_state('u', fix_initial=False, fix_final=False, rate_source='dx_accel', targets=['u'], units='m/s')
cruise.add_state('v', fix_initial=False, fix_final=False, rate_source='dy_accel', targets=['v'], units='m/s')
cruise.add_state('w', fix_initial=False, fix_final=False, rate_source='dz_accel', targets=['w'], units='m/s')
cruise.add_state('roll_ang_vel', fix_initial=False, fix_final=False, rate_source='roll_accel', targets=['roll_ang_vel'], units='rad/s')
cruise.add_state('pitch_ang_vel', fix_initial=False, fix_final=False, rate_source='pitch_accel', targets=['pitch_ang_vel'], units='rad/s')
cruise.add_state('yaw_ang_vel', fix_initial=False, fix_final=False, rate_source='yaw_accel', targets=['yaw_ang_vel'], units='rad/s')
cruise.add_state('roll', fix_initial=False, fix_final=False, rate_source='roll_angle_rate_eq', targets=['roll'], units='rad')
cruise.add_state('pitch', fix_initial=False, fix_final=False, rate_source='pitch_angle_rate_eq', targets=['pitch'], units='rad')
cruise.add_state('yaw', fix_initial=False, fix_final=False, rate_source='yaw_angle_rate_eq', targets=['yaw'], units='rad')
cruise.add_state('x', fix_initial=False, fix_final=False, rate_source='dx_dt', targets=['x'], units='m')
cruise.add_state('y', fix_initial=False, fix_final=False, rate_source='dy_dt', targets=['y'], units='m')
cruise.add_state('z', fix_initial=False, fix_final=False, rate_source='dz_dt', targets=['z'], units='m')
# Controls (without explicit scaling for now)
cruise.add_control('thrust', targets=['thrust'], opt=True, units='N', lower=0.0, upper=200.0)
cruise.add_control('lx', targets=['lx'], opt=True, units='N*m', lower=-5.0, upper=5.0)
cruise.add_control('ly', targets=['ly'], opt=True, units='N*m', lower=-5.0, upper=5.0)
cruise.add_control('lz', targets=['lz'], opt=True, units='N*m', lower=-5.0, upper=5.0)
cruise.add_path_constraint('z', lower=z_final - 1.0, upper=z_final + 1.0, units='m')

descent = dm.Phase(ode_class=vtolODE,
                 transcription=dm.Radau(num_segments=5, order=3))

descent = traj.add_phase('descent', descent)
descent.set_time_options(fix_initial=False, initial_bounds=(0.5, 50), duration_bounds=(0.5, 80), duration_ref=80, units='s')
descent.add_state('u', fix_initial=False, fix_final=False, rate_source='dx_accel', targets=['u'], units='m/s')
descent.add_state('v', fix_initial=False, fix_final=False, rate_source='dy_accel', targets=['v'], units='m/s')
descent.add_state('w', fix_initial=False, fix_final=False, rate_source='dz_accel', targets=['w'], units='m/s')
descent.add_state('roll_ang_vel', fix_initial=False, fix_final=False, rate_source='roll_accel', targets=['roll_ang_vel'], units='rad/s')
descent.add_state('pitch_ang_vel', fix_initial=False, fix_final=False, rate_source='pitch_accel', targets=['pitch_ang_vel'], units='rad/s')
descent.add_state('yaw_ang_vel', fix_initial=False, fix_final=False, rate_source='yaw_accel', targets=['yaw_ang_vel'], units='rad/s')
descent.add_state('roll', fix_initial=False, fix_final=False, rate_source='roll_angle_rate_eq', targets=['roll'], units='rad')
descent.add_state('pitch', fix_initial=False, fix_final=False, rate_source='pitch_angle_rate_eq', targets=['pitch'], units='rad')
descent.add_state('yaw', fix_initial=False, fix_final=False, rate_source='yaw_angle_rate_eq', targets=['yaw'], units='rad')
descent.add_state('x', fix_initial=False, fix_final=False, rate_source='dx_dt', targets=['x'], units='m')
descent.add_state('y', fix_initial=False, fix_final=False, rate_source='dy_dt', targets=['y'], units='m')
descent.add_state('z', fix_initial=False, fix_final=False, rate_source='dz_dt', targets=['z'], units='m')
descent.add_objective('time', loc='final') # minimize time
# Controls (without explicit scaling for now)
descent.add_control('thrust', targets=['thrust'], opt=True, units='N', lower=0.0, upper=200.0)
descent.add_control('lx', targets=['lx'], opt=True, units='N*m', lower=-5.0, upper=5.0)
descent.add_control('ly', targets=['ly'], opt=True, units='N*m', lower=-5.0, upper=5.0)
descent.add_control('lz', targets=['lz'], opt=True, units='N*m', lower=-5.0, upper=5.0)
descent.add_boundary_constraint('z', loc='final', equals=0.0, units='m')


traj.link_phases(['climb', 'cruise', 'descent'],
                    vars=['time', 'u', 'v', 'w',
                          'roll_ang_vel', 'pitch_ang_vel', 'yaw_ang_vel',
                          'roll', 'pitch', 'yaw', 'x', 'y', 'z'],
                    connected=True)

p.model.add_subsystem('traj', subsys=traj)
p.setup(check=True)

# Initial guesses

p.set_val('traj.parameters:mass', val=2.0, units='kg')
p.set_val('traj.parameters:J_xx', val=0.16, units='kg*m**2')
p.set_val('traj.parameters:J_yy', val=0.16, units='kg*m**2')
p.set_val('traj.parameters:J_zz', val=0.16, units='kg*m**2')
p.set_val('traj.parameters:J_xz', val=0.0, units='kg*m**2')
p.set_val('traj.parameters:sphere_radius', val=0.12, units='m')
p.set_val('traj.parameters:sphere_Cd', val=0.47)
p.set_val('traj.parameters:g', val=9.81, units='m/s**2')

climb = p.model.traj.phases.climb
cruise = p.model.traj.phases.cruise
descent = p.model.traj.phases.descent

climb.set_time_options(fix_initial=True)
climb.set_time_val(initial=0.0, duration=40, units='s')

# Hover thrust (m*g)
hover_thrust = 19.62 # N

climb.add_boundary_constraint('z', loc='initial', equals=0.0, units='m')
climb.add_boundary_constraint('u', loc='initial', equals=0.0, units='m/s')
climb.add_boundary_constraint('v', loc='initial', equals=0.0, units='m/s')
climb.add_boundary_constraint('w', loc='initial', equals=0.0, units='m/s')
climb.set_state_val('u', vals=[0, 0], units='m/s')
climb.set_state_val('v', vals=[0, 0], units='m/s')
climb.set_state_val('w', vals=[0, -5], units='m/s')
climb.set_state_val('roll_ang_vel', vals=[0, 0], units='rad/s')
climb.set_state_val('pitch_ang_vel', vals=[0, 0], units='rad/s')
climb.set_state_val('yaw_ang_vel', vals=[0, 0], units='rad/s')
climb.set_state_val('roll', vals=[0, 0], units='rad')
climb.set_state_val('pitch', vals=[0, 0], units='rad')
climb.set_state_val('yaw', vals=[0, 0], units='rad')
climb.set_state_val('x', vals=[0, 0.5], units='m')
climb.set_state_val('y', vals=[0, 0.5], units='m')
climb.set_state_val('z', vals=[0, -100], units='m')
climb.set_control_val('thrust', vals=[hover_thrust, hover_thrust*1.2], units='N')
climb.set_control_val('lx', vals=0.0, units='N*m')
climb.set_control_val('ly', vals=0.0, units='N*m')
climb.set_control_val('lz', vals=0.0, units='N*m')
# climb.set_parameter_val('mass', val=2.0, units='kg')
# climb.set_parameter_val('J_xx', val=0.16, units='kg*m**2')
# climb.set_parameter_val('J_yy', val=0.16, units='kg*m**2')
# climb.set_parameter_val('J_zz', val=0.16, units='kg*m**2')
# climb.set_parameter_val('J_xz', val=0.0, units='kg*m**2')
# climb.set_parameter_val('sphere_radius', val=0.12, units='m')
# climb.set_parameter_val('sphere_Cd', val=0.47)
# climb.set_parameter_val('g', val=9.81, units='m/s**2')

cruise.set_time_val(initial=40.0, duration=80.0, units='s')
cruise.set_state_val('u', vals=[0, 10], units='m/s')
cruise.set_state_val('v', vals=[0, 0], units='m/s')
cruise.set_state_val('w', vals=[-5, 0], units='m/s')
cruise.set_state_val('roll_ang_vel', vals=[0, 0], units='rad/s')
cruise.set_state_val('pitch_ang_vel', vals=[0, 0], units='rad/s')
cruise.set_state_val('yaw_ang_vel', vals=[0, 0], units='rad/s')
cruise.set_state_val('roll', vals=[0, 0], units='rad')
cruise.set_state_val('pitch', vals=[0, 0], units='rad')
cruise.set_state_val('yaw', vals=[0, 0], units='rad')
cruise.set_state_val('x', vals=[0, 200], units='m')
cruise.set_state_val('y', vals=[0, 0.5], units='m')
cruise.set_state_val('z', vals=[-100, -100], units='m')
cruise.set_control_val('thrust', vals=[hover_thrust*1.2, hover_thrust*0.8], units='N')
cruise.set_control_val('lx', vals=0.0, units='N*m')
cruise.set_control_val('ly', vals=0.0, units='N*m')
cruise.set_control_val('lz', vals=0.0, units='N*m')
# cruise.set_parameter_val('mass', val=2.0, units='kg')
# cruise.set_parameter_val('J_xx', val=0.16, units='kg*m**2')
# cruise.set_parameter_val('J_yy', val=0.16, units='kg*m**2')
# cruise.set_parameter_val('J_zz', val=0.16, units='kg*m**2')
# cruise.set_parameter_val('J_xz', val=0.0, units='kg*m**2')
# cruise.set_parameter_val('sphere_radius', val=0.12, units='m')
# cruise.set_parameter_val('sphere_Cd', val=0.47)
# cruise.set_parameter_val('g', val=9.81, units='m/s**2')

descent.set_time_val(initial=120, duration=40, units='s')
descent.set_state_val('u', vals=[10, 0], units='m/s')
descent.set_state_val('v', vals=[0, 0], units='m/s')
descent.set_state_val('w', vals=[0, 5], units='m/s')
descent.set_state_val('roll_ang_vel', vals=[0, 0], units='rad/s')
descent.set_state_val('pitch_ang_vel', vals=[0, 0], units='rad/s')
descent.set_state_val('yaw_ang_vel', vals=[0, 0], units='rad/s')
descent.set_state_val('roll', vals=[0, 0], units='rad')
descent.set_state_val('pitch', vals=[0, 0], units='rad')
descent.set_state_val('yaw', vals=[0, 0], units='rad')
descent.set_state_val('x', vals=[0, 0.5], units='m')
descent.set_state_val('y', vals=[0, 0.5], units='m')
descent.set_state_val('z', vals=[-100, 0], units='m')
descent.set_control_val('thrust', vals=[hover_thrust*0.8, 0], units='N')
descent.set_control_val('lx', vals=0.0, units='N*m')
descent.set_control_val('ly', vals=0.0, units='N*m')
descent.set_control_val('lz', vals=0.0, units='N*m')
# descent.set_parameter_val('mass', val=2.0, units='kg')
# descent.set_parameter_val('J_xx', val=0.16, units='kg*m**2')
# descent.set_parameter_val('J_yy', val=0.16, units='kg*m**2')
# descent.set_parameter_val('J_zz', val=0.16, units='kg*m**2')
# descent.set_parameter_val('J_xz', val=0.0, units='kg*m**2')
# descent.set_parameter_val('sphere_radius', val=0.12, units='m')
# descent.set_parameter_val('sphere_Cd', val=0.47)
# descent.set_parameter_val('g', val=9.81, units='m/s**2')

dm.run_problem(p, run_driver=True, simulate=True)

exp_out = traj.simulate()

# Post processing
# sol = om.CaseReader(p.get_outputs_dir() / 'dymos_solution.db').get_case('final')
# sim = om.CaseReader(traj.sim_prob.get_outputs_dir() / 'dymos_solution.db').get_case('final')

# t_sol = dict((phs, sol.get_val(f'traj.{phs}.timeseries.time'.format(phs)))
#              for phs in ['climb', 'cruise', 'descent'])
# z_sol = dict((phs, sol.get_val(f'traj.{phs}.timeseries.pos_z'.format(phs)))
#              for phs in ['climb', 'cruise','descent'])

# t_exp = dict((phs, sim.get_val(f'traj.{phs}.timeseries.time'.format(phs)))
#              for phs in ['climb', 'cruise', 'descent'])
# z_exp = dict((phs, sim.get_val(f'traj.{phs}.timeseries.pos_z'.format(phs)))
#              for phs in ['climb', 'cruise','descent'])

# fig, ax = plt.subplots(figsize=(12 , 6))

# Plot altitude vs time across all phases
for ph in ['climb', 'cruise', 'descent']:
    t = exp_out.get_val(f'traj.{ph}.timeseries.time')
    z = exp_out.get_val(f'traj.{ph}.timeseries.z')
    plt.plot(t, z, label=ph)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude z (m)')
    plt.title('VTOL Trajectory with Climb, Cruise, Descent')
    plt.show()
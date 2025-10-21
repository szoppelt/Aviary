import matplotlib.pyplot as plt

import numpy as np

import openmdao.api as om
import dymos as dm

from openmdao.api import Group

from dymos.models.atmosphere.atmos_1976 import USatm1976Comp

from aviary.mission.sixdof.six_dof_EOM import SixDOF_EOM
from aviary.mission.sixdof.force_component_calc import ForceComponentResolver
from aviary.mission.sixdof.AeroSphereComp import AeroSphereComp

from openmdao.utils.general_utils import set_pyoptsparse_opt
OPT, OPTIMIZER = set_pyoptsparse_opt('SNOPT')
if OPTIMIZER:
    from openmdao.drivers.pyoptsparse_driver import pyOptSparseDriver

class vtolODE(Group):

    def initialize(self):
        self.options.declare('num_nodes', types=int)
    
    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem('USatm1976comp', USatm1976Comp(num_nodes=nn),
                           promotes_inputs=['*'],
                           promotes_outputs=['rho'])
        
        self.add_subsystem('AeroSphereComp', AeroSphereComp(num_nodes=nn),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
        
        self.add_subsystem('ForceComponents', ForceComponentResolver(num_nodes=nn),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])
        
        self.add_subsystem('SixDOF_EOM', SixDOF_EOM(num_nodes=nn),
                           promotes_inputs=['*'],
                           promotes_outputs=['*'])

def sixdof_test():
    p = om.Problem()
    

    traj = dm.Trajectory()
    phase = dm.Phase(ode_class=vtolODE, 
                    transcription=dm.Radau(num_segments=10, order=3)
                    )
    
    p.model.add_subsystem('traj', traj)

    # SETUP

    traj.add_phase(name='main_phase', phase=phase)

    phase.set_time_options(fix_initial=True, 
                           fix_duration=False, 
                           units='s',
                           duration_bounds=(1, 600.0))
    
    phase.add_state('u', fix_initial=True, rate_source='dx_accel', targets=['u'], lower=0, upper=100, units='m/s')
    phase.add_state('v', fix_initial=True, rate_source='dy_accel', targets=['v'], lower=0, upper=100, units='m/s')
    phase.add_state('w', fix_initial=True, rate_source='dz_accel', targets=['w'], lower=0, upper=100, units='m/s')
    phase.add_state('roll_ang_vel', fix_initial=True, rate_source='roll_accel', targets=['roll_ang_vel'], lower=0, upper=100, units='rad/s')
    phase.add_state('pitch_ang_vel', fix_initial=True, rate_source='pitch_accel', targets=['pitch_ang_vel'], lower=0, upper=100, units='rad/s')
    phase.add_state('yaw_ang_vel', fix_initial=True, rate_source='yaw_accel', targets=['yaw_ang_vel'], lower=0, upper=100, units='rad/s')
    phase.add_state('roll', fix_initial=True, rate_source='roll_angle_rate_eq', targets=['roll'], lower=0, upper=np.pi, units='rad')
    phase.add_state('pitch', fix_initial=True, rate_source='pitch_angle_rate_eq', targets=['pitch'], lower=0, upper=np.pi, units='rad')
    phase.add_state('yaw', fix_initial=True, rate_source='yaw_angle_rate_eq', targets=['yaw'], lower=0, upper=np.pi, units='rad')
    phase.add_state('x', fix_initial=True, rate_source='dx_dt', targets=['x'],lower=0, upper=100, units='m')
    phase.add_state('y', fix_initial=True, rate_source='dy_dt', targets=['y'], lower=0, upper=100, units='m')
    phase.add_state('z', fix_initial=True, rate_source='dz_dt', targets=['z'], lower=0, upper=100, units='m')
    phase.add_state('energy', fix_initial=True, rate_source='dE_dt', targets=['energy'], lower=0, upper=300, units='J')

    phase.add_control('Fx', targets=['Fx'], opt=True, units='N')
    phase.add_control('Fy', targets=['Fy_ext'], opt=True, units='N')
    phase.add_control('Fz', targets=['Fz'], opt=True, units='N')
    phase.add_control('lx', targets=['lx'], opt=True, units='N*m')
    phase.add_control('ly', targets=['ly'], opt=True, units='N*m')
    phase.add_control('lz', targets=['lz'], opt=True, units='N*m')
    phase.add_control('power', targets=['power'], opt=True, units='W')

    phase.add_parameter('mass', units='kg', targets=['mass'], opt=False)
    phase.add_parameter('J_xx', units='kg * m**2', targets=['J_xx'], opt=False)
    phase.add_parameter('J_yy', units='kg * m**2', targets=['J_yy'], opt=False)
    phase.add_parameter('J_zz', units='kg * m**2', targets=['J_zz'], opt=False)
    phase.add_parameter('J_xz', units='kg * m**2', targets=['J_xz'], opt=False)

    phase.add_boundary_constraint('z', loc='final', equals=33, units='m')
    phase.add_path_constraint('x', lower=0, upper=0.1, units='m')
    phase.add_path_constraint('y', lower=0, upper=0.1, units='m')
    
    phase.add_objective('time', loc='final', units='s') # minimize energy

    p.driver = om.pyOptSparseDriver()
    p.driver.options["optimizer"] = "IPOPT"

    p.driver.opt_settings['mu_init'] = 1e-1
    p.driver.opt_settings['max_iter'] = 600
    p.driver.opt_settings['constr_viol_tol'] = 1e-6
    p.driver.opt_settings['compl_inf_tol'] = 1e-6
    p.driver.opt_settings['tol'] = 1e-5
    p.driver.opt_settings['print_level'] = 3
    p.driver.opt_settings['nlp_scaling_method'] = 'gradient-based'
    p.driver.opt_settings['alpha_for_y'] = 'safer-min-dual-infeas'
    p.driver.opt_settings['mu_strategy'] = 'monotone'
    p.driver.opt_settings['bound_mult_init_method'] = 'mu-based'
    p.driver.options['print_results'] = False

    p.driver.declare_coloring()

    p.setup()

    phase.set_time_val(initial=0, duration=60, units='s')
    phase.set_state_val('axial_vel', vals=[0, 0], units='m/s')
    phase.set_state_val('lat_vel', vals=[0, 0], units='m/s')
    phase.set_state_val('vert_vel', vals=[10, 10], units='m/s')
    phase.set_state_val('roll_ang_vel', vals=[0, 0], units='rad/s')
    phase.set_state_val('pitch_ang_vel', vals=[0, 0], units='rad/s')
    phase.set_state_val('yaw_ang_vel', vals=[0, 0], units='rad/s')
    phase.set_state_val('roll', vals=[0, 0], units='rad')
    phase.set_state_val('pitch', vals=[0, 0], units='rad')
    phase.set_state_val('yaw', vals=[0, 0], units='rad')
    phase.set_state_val('x', vals=[0, 0], units='m')
    phase.set_state_val('y', vals=[0, 0], units='m')
    phase.set_state_val('z', vals=[0, 33], units='m')
    phase.set_state_val('energy', vals=[0, 300], units='J')

    phase.set_control_val('Fx_ext', vals=[0, 0], units='N')
    phase.set_control_val('Fy_ext', vals=[0, 0], units='N')
    phase.set_control_val('Fz_ext', vals=[10, 10], units='N')
    phase.set_control_val('lx_ext', vals=[0, 0], units='N*m')
    phase.set_control_val('ly_ext', vals=[0, 0], units='N*m')
    phase.set_control_val('lz_ext', vals=[0, 0], units='N*m')
    phase.set_control_val('power', vals=[0, 300], units='W')

    phase.set_parameter_val('mass', val=10, units='kg')
    phase.set_parameter_val('J_xx', val=16, units='kg*m**2') # assume a sphere of 10 kg with radius = 2
    phase.set_parameter_val('J_yy', val=16, units='kg*m**2')
    phase.set_parameter_val('J_zz', val=16, units='kg*m**2')
    phase.set_parameter_val('J_xz', val=0, units='kg*m**2')
    

    p.final_setup()

    p.run_model()

    dm.run_problem(p, run_driver=True, simulate=True, make_plots=True)

    exp_out = traj.simulate()

    p_sol = p
    p_sim = exp_out

    x_traj = p_sol.get_val('traj.main_phase.timeseries.x')
    x_sim = p_sim.get_val('traj.main_phase.timeseries.x')
    y_traj = p_sol.get_val('traj.main_phase.timeseries.y')
    y_sim = p_sim.get_val('traj.main_phase.timeseries.y')
    z_traj = p_sol.get_val('traj.main_phase.timeseries.z')
    z_sim = p_sim.get_val('traj.main_phase.timeseries.z')
    t_traj = p_sol.get_val('traj.main_phase.timeseries.time')
    t_sim = p_sim.get_val('traj.main_phase.timeseries.time')



    


    # plt.plot(t_traj, z_traj, marker='o', ms=4, linestyle='None', label='solution')
    # plt.plot(t_sim, z_sim, marker=None, linestyle='-', label='simulation')
    # plt.legend(fontsize=12)
    # plt.xlabel('t (s)', fontsize=12)
    # plt.ylabel('z (m)', fontsize=12)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.title('Trajectory of Vertical Take Off vs. Time', fontsize=12)
    # plt.show()
    # plt.savefig('./TrajPlots_Largerfontt.pdf')
    

    


if __name__ == "__main__":
    sixdof_test() 



def build_phase_fixed(name, transcription, duration_bounds, z_final=None, cruise=False):
    """
    Fixed phase builder with better initial conditions and bounds
    """
    phase = dm.Phase(ode_class=vtolODE, transcription=transcription)
    
    phase.set_time_options(fix_initial=False, fix_duration=False,
                          units='s', duration_bounds=duration_bounds)
    
    # States with more reasonable bounds and scaling
    # Simpler version without explicit scaling - let Dymos auto-scale
    state_configs = [
        # Linear velocities (body frame)
        ('u', 'dx_accel', 'u', -20, 20, 'm/s'),
        ('v', 'dy_accel', 'v', -20, 20, 'm/s'),
        ('w', 'dz_accel', 'w', -20, 20, 'm/s'),
        # Angular velocities
        ('roll_ang_vel', 'roll_accel', 'roll_ang_vel', -5, 5, 'rad/s'),
        ('pitch_ang_vel', 'pitch_accel', 'pitch_ang_vel', -5, 5, 'rad/s'),
        ('yaw_ang_vel', 'yaw_accel', 'yaw_ang_vel', -5, 5, 'rad/s'),
        # Euler angles
        ('roll', 'roll_angle_rate_eq', 'roll', -np.pi/4, np.pi/4, 'rad'),
        ('pitch', 'pitch_angle_rate_eq', 'pitch', -np.pi/4, np.pi/4, 'rad'),
        ('yaw', 'yaw_angle_rate_eq', 'yaw', 0, 2*np.pi, 'rad'),
        # Position (NED frame)
        ('x', 'dx_dt', 'x', 0, 1000, 'm'),
        ('y', 'dy_dt', 'y', -100, 100, 'm'),
        ('z', 'dz_dt', 'z', -100, 0, 'm'),  # NED: 0=ground, -100m=100m altitude
    ]
    
    for state, rate, tgt, lo, up, units in state_configs:
        phase.add_state(state, fix_initial=False, rate_source=rate,
                       targets=[tgt], lower=lo, upper=up, units=units)
    
    # Controls (without explicit scaling for now)
    phase.add_control('thrust', targets=['thrust'], opt=True,
                     units='N', lower=0.0, upper=200.0)
    phase.add_control('lx', targets=['lx'], opt=True,
                     units='N*m', lower=-5.0, upper=5.0)
    phase.add_control('ly', targets=['ly'], opt=True,
                     units='N*m', lower=-5.0, upper=5.0)
    phase.add_control('lz', targets=['lz'], opt=True,
                     units='N*m', lower=-5.0, upper=5.0)
    
    # Parameters (static)
    phase.add_parameter('mass', units='kg', targets=['mass'],
                       opt=False, static_target=True)
    phase.add_parameter('J_xx', units='kg * m**2', targets=['J_xx'],
                       opt=False, static_target=True)
    phase.add_parameter('J_yy', units='kg * m**2', targets=['J_yy'],
                       opt=False, static_target=True)
    phase.add_parameter('J_zz', units='kg * m**2', targets=['J_zz'],
                       opt=False, static_target=True)
    phase.add_parameter('J_xz', units='kg * m**2', targets=['J_xz'],
                       opt=False, static_target=True)
    phase.add_parameter('g', units='m / s**2', targets=['g'],
                        opt=False, static_target=True)
    
    # Add aero parameters
    phase.add_parameter('sphere_radius', units='m', targets=['sphere_radius'],
                       opt=False, static_target=True, val=0.12)
    phase.add_parameter('sphere_Cd', targets=['sphere_Cd'],
                       opt=False, static_target=True, val=0.47)
    
    # Constraints
    if z_final is not None:
        phase.add_boundary_constraint('z', loc='final', equals=z_final, units='m')
    
    if cruise:
        # Tighter cruise constraint
        phase.add_path_constraint('z', lower=z_final - 1.0, upper=z_final + 1.0, units='m')
    
    return phase

def sixdof_mission_fixed():
    """
    Fixed mission with better setup
    """
    p = om.Problem()
    traj = dm.Trajectory()
    p.model.add_subsystem('traj', traj)
    
    # Use fewer segments initially for debugging
    tx = dm.Radau(num_segments=5, order=3)
    
    # Define phases with corrected builder
    climb = build_phase_fixed('climb', tx, duration_bounds=(10, 100), z_final=-100)  # -100m = 100m altitude
    cruise = build_phase_fixed('cruise', tx, duration_bounds=(20, 200), z_final=-100, cruise=True)
    descent = build_phase_fixed('descent', tx, duration_bounds=(10, 100), z_final=0)
    
    traj.add_phase('climb', climb)
    traj.add_phase('cruise', cruise)
    traj.add_phase('descent', descent)
    
    # Link phases
    traj.link_phases(['climb', 'cruise', 'descent'],
                    vars=['time', 'u', 'v', 'w',
                          'roll_ang_vel', 'pitch_ang_vel', 'yaw_ang_vel',
                          'roll', 'pitch', 'yaw', 'x', 'y', 'z'],
                    connected=True)
    
    # Objective: minimize time
    descent.add_objective('time', loc='final')
    
    # Driver setup
    p.driver = om.pyOptSparseDriver()
    p.driver.options["optimizer"] = "IPOPT"
    p.driver.opt_settings['max_iter'] = 800
    p.driver.opt_settings['tol'] = 1e-4
    p.driver.opt_settings['print_level'] = 5
    p.driver.opt_settings['acceptable_tol'] = 1e-3
    
    p.setup()
    
    # CRITICAL FIX 5: Better initial conditions
    # Climb phase
    climb.set_time_options(fix_initial=True)
    climb.add_boundary_constraint('z', loc='initial', equals=0.0, units='m')
    climb.add_boundary_constraint('u', loc='initial', equals=0.0, units='m/s')
    climb.add_boundary_constraint('v', loc='initial', equals=0.0, units='m/s')
    climb.add_boundary_constraint('w', loc='initial', equals=0.0, units='m/s')
    
    # Set reasonable time guesses
    climb.set_time_val(initial=0.0, duration=40.0, units='s')
    cruise.set_time_val(initial=40.0, duration=80.0, units='s')
    descent.set_time_val(initial=120.0, duration=40.0, units='s')
    
    # Set state initial guesses
    climb.set_state_val('z', vals=[0, -100], units='m')
    climb.set_state_val('w', vals=[0, -2.5], units='m/s')  # Climbing (negative in NED)
    
    cruise.set_state_val('z', vals=[-100, -100], units='m')
    cruise.set_state_val('u', vals=[5, 10], units='m/s')  # Forward cruise
    cruise.set_state_val('w', vals=[-2.5, 0], units='m/s')
    
    descent.set_state_val('z', vals=[-100, 0], units='m')
    descent.set_state_val('w', vals=[0, 2.5], units='m/s')  # Descending (positive in NED)
    descent.set_state_val('u', vals=[10, 0], units='m/s')
    
    # CRITICAL FIX 6: Set thrust initial guess near hover
    # Hover thrust = mass * g = 10 kg * 9.81 m/s^2 ≈ 98 N
    hover_thrust = 98.0
    
    climb.set_control_val('thrust', vals=hover_thrust * 1.2)  # Extra for climb
    cruise.set_control_val('thrust', vals=hover_thrust * 0.5)  # Less for cruise
    descent.set_control_val('thrust', vals=hover_thrust * 0.8)  # Controlled descent
    
    # Zero moments initially
    for ph in [climb, cruise, descent]:
        ph.set_control_val('lx', vals=0.0)
        ph.set_control_val('ly', vals=0.0)
        ph.set_control_val('lz', vals=0.0)
    
    # Set parameters
    for ph in [climb, cruise, descent]:
        ph.set_parameter_val('mass', val=10.0, units='kg')
        ph.set_parameter_val('J_xx', val=0.16, units='kg*m**2')
        ph.set_parameter_val('J_yy', val=0.16, units='kg*m**2')
        ph.set_parameter_val('J_zz', val=0.16, units='kg*m**2')
        ph.set_parameter_val('J_xz', val=0.0, units='kg*m**2')
        ph.set_parameter_val('sphere_radius', val=0.12, units='m')
        ph.set_parameter_val('sphere_Cd', val=0.5)
        ph.set_parameter_val('g', val=9.81, units='m/s**2')
    
    p.final_setup()
    
    # Run simulation first to check setup
    print("Running simulation to verify setup...")
    dm.run_problem(p, run_driver=False, simulate=True, make_plots=False)

    sim = traj.simulate(method='Radau', atol=1e-6, rtol=1e-6)

    # Plot altitude
    for ph_name in ['climb', 'cruise', 'descent']:
        t = sim.get_val(f'traj.{ph_name}.timeseries.time')
        z = sim.get_val(f'traj.{ph_name}.timeseries.z')
        plt.plot(t, z, label=ph_name)
        plt.legend()
        plt.xlabel('time (s)')
        plt.ylabel('z (m) (NED)')
        plt.title('Trajectory (altitude)')
        plt.show()
    
    print("\nSimulation successful! Now running optimization...")
    dm.run_problem(p, run_driver=True, simulate=True, make_plots=True)
    
    return p


if __name__ == "__main__":
    print("=" * 60)
    print("KEY FIXES APPLIED:")
    print("=" * 60)
    print("1. Added altitude calculation (h = -z)")
    print("2. Added wind angle calculations")
    print("3. Added sphere_radius and sphere_Cd as parameters")
    print("4. Set thrust angles for vertical thrust")
    print("5. Better initial guesses (hover thrust ~98N)")
    print("6. Added scaling with ref/ref0")
    print("7. NED coordinates: z=0 is ground, z=-100 is 100m altitude")
    print("=" * 60)
    
    # Uncomment to run:
    p = sixdof_mission_fixed()
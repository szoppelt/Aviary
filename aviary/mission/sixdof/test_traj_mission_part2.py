import matplotlib.pyplot as plt
import numpy as np
import openmdao.api as om
import dymos as dm

import sys
import os

sys.path.append("/home/omdao/Aviary-1/")


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

        self.add_subsystem('atm', USatm1976Comp(num_nodes=nn),
            promotes_inputs=['*'],
            promotes_outputs=['rho'])
        self.add_subsystem('aero', AeroSphereComp(num_nodes=nn),
            promotes_inputs=['*'],
            promotes_outputs=['*'])


        self.add_subsystem('forces', ForceComponentResolver(num_nodes=nn),
            promotes_inputs=['*'],
            promotes_outputs=['Fx', 'Fy', 'Fz'])
        self.add_subsystem('eom', SixDOF_EOM(num_nodes=nn),
            promotes_inputs=['*'],
            promotes_outputs=['*'])


def build_phase(name, transcription, duration_bounds, z_final=None, cruise=False):
    phase = dm.Phase(
            ode_class=vtolODE,
            transcription=transcription
            )


    phase.set_time_options(fix_initial=False, fix_duration=False,
    units='s', duration_bounds=duration_bounds)


    # States (same for all phases)
    for state, rate, tgt, lo, up, units in [
        ('u', 'dx_accel', 'u', 0, 100, 'm/s'),
        ('v', 'dy_accel', 'v', 0, 100, 'm/s'),
        ('w', 'dz_accel', 'w', 0, 100, 'm/s'),
        ('roll_ang_vel', 'roll_accel', 'roll_ang_vel', 0, 100, 'rad/s'),
        ('pitch_ang_vel', 'pitch_accel', 'pitch_ang_vel', 0, 100, 'rad/s'),
        ('yaw_ang_vel', 'yaw_accel', 'yaw_ang_vel', 0, 100, 'rad/s'),
        ('roll', 'roll_angle_rate_eq', 'roll', 0, np.pi, 'rad'),
        ('pitch', 'pitch_angle_rate_eq', 'pitch', 0, np.pi, 'rad'),
        ('yaw', 'yaw_angle_rate_eq', 'yaw', 0, np.pi, 'rad'),
        ('x', 'dx_dt', 'x', 0, 10000, 'm'),
        ('y', 'dy_dt', 'y', 0, 10000, 'm'),
        ('z', 'dz_dt', 'z', 0, 10000, 'm')]:
        phase.add_state(state, fix_initial=False, rate_source=rate, targets=[tgt], lower=lo, upper=up, units=units)


    # Controls
    #phase.add_control('Fx', targets=['Fx'], opt=True, units='N', lower=-200.0, upper=200.0)
    #phase.add_control('Fy', targets=['Fy'], opt=True, units='N', lower=-200.0, upper=200.0)
    #phase.add_control('Fz', targets=['Fz'], opt=True, units='N', lower=0.0, upper=400.0)
    phase.add_control('thrust', targets=['thrust'], opt=True, units='N', lower=-200.0, upper=200.0)
    phase.add_control('lx', targets=['lx'], opt=True, units='N*m', lower=-50.0, upper=50.0)
    phase.add_control('ly', targets=['ly'], opt=True, units='N*m', lower=-50.0, upper=50.0)
    phase.add_control('lz', targets=['lz'], opt=True, units='N*m', lower=-50.0, upper=50.0)


    # Parameters
    phase.add_parameter('mass', units='kg', targets=['mass'], opt=False)
    phase.add_parameter('J_xx', units='kg * m**2', targets=['J_xx'], opt=False)
    phase.add_parameter('J_yy', units='kg * m**2', targets=['J_yy'], opt=False)
    phase.add_parameter('J_zz', units='kg * m**2', targets=['J_zz'], opt=False)
    phase.add_parameter('J_xz', units='kg * m**2', targets=['J_xz'], opt=False)
    
    # Constraints per phase
    if z_final is not None:
        phase.add_boundary_constraint('z', loc='final', equals=z_final, units='m')
    if cruise:
        phase.add_path_constraint('z', lower=z_final - 0.1, upper=z_final + 0.1, units='m')


    return phase

def sixdof_mission():
    p = om.Problem()
    traj = dm.Trajectory()
    p.model.add_subsystem('traj', traj)


    tx = dm.Radau(num_segments=10, order=3)


    # Define phases
    climb = build_phase('climb', tx, duration_bounds=(1, 200), z_final=100)
    cruise = build_phase('cruise', tx, duration_bounds=(1, 500), z_final=100, cruise=True)
    descent = build_phase('descent', tx, duration_bounds=(1, 200), z_final=0)


    traj.add_phase('climb', climb)
    traj.add_phase('cruise', cruise)
    traj.add_phase('descent', descent)


    # Link phases (continuous states)
    traj.link_phases(['climb', 'cruise', 'descent'],
                    vars=['time', 
                          'u', 'v', 'w',
                            'roll_ang_vel', 'pitch_ang_vel', 'yaw_ang_vel',
                            'roll', 'pitch', 'yaw', 'x', 'y', 'z'],
                    connected=True)


    # Objective: minimize final time of descent
    descent.add_objective('time', loc='final')


    # Driver
    p.driver = om.pyOptSparseDriver()
    p.driver.options["optimizer"] = "IPOPT"
    p.driver.opt_settings['max_iter'] = 800
    p.driver.opt_settings['tol'] = 1e-5


    p.setup()


    # Initial guesses
    climb.set_time_options(fix_initial=True)
    climb.add_boundary_constraint('z', loc='initial', equals=0.0, units='m')
    climb.add_boundary_constraint('u', loc='initial', equals=1.0e-4, units='m/s')
    climb.add_boundary_constraint('v', loc='initial', equals=0.0, units='m/s')
    climb.add_boundary_constraint('w', loc='initial', equals=0.0, units='m/s')
    climb.set_time_val(initial=0, duration=50, units='s')
    climb.set_state_val('z', vals=[0, 100], units='m')
    cruise.set_time_val(initial=50, duration=200, units='s')
    cruise.set_state_val('z', vals=[100, 100], units='m')
    descent.set_time_val(initial=250, duration=50, units='s')
    descent.set_state_val('z', vals=[100, 0], units='m')
    


    # Parameters
    for ph in [climb, cruise, descent]:
        ph.set_parameter_val('mass', val=10, units='kg')
        ph.set_parameter_val('J_xx', val=16, units='kg*m**2')
        ph.set_parameter_val('J_yy', val=16, units='kg*m**2')
        ph.set_parameter_val('J_zz', val=16, units='kg*m**2')
        ph.set_parameter_val('J_xz', val=0, units='kg*m**2')

    #p.setup()
    p.final_setup()
    dm.run_problem(p, run_driver=False, simulate=True, make_plots=True)


    exp_out = traj.simulate()


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

if __name__ == "__main__":
    sixdof_mission()
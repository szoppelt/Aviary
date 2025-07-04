import matplotlib.pyplot as plt

import numpy as np

import openmdao.api as om
import dymos as dm

from openmdao.api import Group

from dymos.models.atmosphere.atmos_1976 import USatm1976Comp

from aviary.mission.sixdof.six_dof_EOM import SixDOF_EOM
from aviary.mission.sixdof.force_component_calc import ForceComponentResolver

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

    phase.set_time_options(fix_initial=True, fix_duration=False, units='s')
    
    phase.add_state('axial_vel', fix_initial=True, rate_source='dx_accel', targets=['axial_vel'], lower=0, upper=100, units='m/s')
    phase.add_state('lat_vel', fix_initial=True, rate_source='dy_accel', targets=['lat_vel'], lower=0, upper=100, units='m/s')
    phase.add_state('vert_vel', fix_initial=True, rate_source='dz_accel', targets=['vert_vel'], lower=0, upper=100, units='m/s')
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

    phase.add_control('Fx_ext', targets=['Fx_ext'], opt=True, units='N')
    phase.add_control('Fy_ext', targets=['Fy_ext'], opt=True, units='N')
    phase.add_control('Fz_ext', targets=['Fz_ext'], opt=True, units='N')
    phase.add_control('lx_ext', targets=['lx_ext'], opt=True, units='N*m')
    phase.add_control('ly_ext', targets=['ly_ext'], opt=True, units='N*m')
    phase.add_control('lz_ext', targets=['lz_ext'], opt=True, units='N*m')
    phase.add_control('power', targets=['power'], opt=True, units='W')

    phase.add_parameter('mass', units='kg', targets=['mass'], opt=False)
    phase.add_parameter('J_xx', units='kg * m**2', targets=['J_xx'], opt=False)
    phase.add_parameter('J_yy', units='kg * m**2', targets=['J_yy'], opt=False)
    phase.add_parameter('J_zz', units='kg * m**2', targets=['J_zz'], opt=False)
    phase.add_parameter('J_xz', units='kg * m**2', targets=['J_xz'], opt=False)

    phase.add_boundary_constraint('z', loc='final', equals=33, units='m')
    phase.add_path_constraint('x', lower=0, upper=0.1, units='m')
    phase.add_path_constraint('y', lower=0, upper=0.1, units='m')
    
    phase.add_objective('energy', loc='final', units='J') # minimize energy

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






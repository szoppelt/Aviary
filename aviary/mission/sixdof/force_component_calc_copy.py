import numpy as np
import openmdao.api as om
from aviary.variable_info.variables import Dynamic
from aviary.utils.functions import add_aviary_input


class ForceComponentResolver_Copy(om.ExplicitComponent):
    """
    This class will resolve forces (thrust, drag, lift, etc.) into their 
    respective x,y,z components for the 6 DOF equations of motion. 
    
    This class assumes that the total force is given and needs to be resolved 
    into the separate components.

    Assumptions:
        - Thrust is entirely in -z direction (T = (0,0,-T_z)^T) w.r.t. body CS
        - Assuming F_i is in body CS, and D, S, and L are in wind CS. Wind -> body rotation matrix
          was applied for coordinate transformations
        - Thrust is initially in NED CS. So, two rotations (NED -> wind and wind -> body) are applied

    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # inputs

        self.add_input(
            'u',
            val=np.zeros(nn),
            units='m/s',
            desc="axial velocity"
        )

        self.add_input(
            'v',
            val=np.zeros(nn),
            units='m/s',
            desc="lateral velocity"
        )

        self.add_input(
            'w',
            val=np.zeros(nn),
            units='m/s',
            desc="vertical velocity"
        )

        self.add_input(
            'drag',
            val=np.zeros(nn),
            units='N',
            desc="Drag vector (unresolved)"
        )

        self.add_input(
            'thrust',
            val=np.zeros(nn),
            units='N',
            desc="Thrust vector (unresolved)"
        )

        self.add_input(
            'lift',
            val=np.zeros(nn),
            units='N',
            desc="Lift vector (unresolved)"
        )

        self.add_input(
            'side',
            val=np.zeros(nn),
            units='N',
            desc="Side vector (unresolved)"
        )

        self.add_input(
            'T_x',
            val=np.zeros(nn),
            units='N',
            desc='x-direction thrust'
        )

        self.add_input(
            'T_y',
            val=np.zeros(nn),
            units='N',
            desc='y-direction thrust'
        )

        self.add_input(
            'T_z',
            val=np.zeros(nn),
            units='N',
            desc='z-direction thrust'
        )

        # self.add_input(
        #     'heading_angle',
        #     val=np.zeros(nn),
        #     units='rad',
        #     desc='Heading angle in body'
        # )

        # self.add_input(
        #     'flight_path_angle',
        #     val=np.zeros(nn),
        #     units='rad',
        #     desc="Flight path angle in body")
        
        # self.add_input(
        #     'heading_angle_NED',
        #     val=np.zeros(nn),
        #     units='rad',
        #     desc="Thrust heading angle in NED"
        # )

        # self.add_input(
        #     'fpa_NED',
        #     val=np.zeros(nn),
        #     units='rad',
        #     desc="Thrust flight path angle in NED"
        # )

        # self.add_input(
        #     'true_air_speed',
        #     val=np.zeros(nn),
        #     units='m/s',
        #     desc="True air speed"
        # ) # This is an aviary variable

        self.add_input(
            'roll',
            val=np.zeros(nn),
            units='rad',
            desc='roll angle'
        )

        self.add_input(
            'pitch',
            val=np.zeros(nn),
            units='rad',
            desc="pitch angle"
        )

        self.add_input(
            'yaw',
            val=np.zeros(nn),
            units='rad',
            desc="yaw angle"
        )

        # outputs

        self.add_output(
            'Fx',
            val=np.zeros(nn),
            units='N',
            desc="x-comp of final force"
        )

        self.add_output(
            'Fy',
            val=np.zeros(nn),
            units='N',
            desc="y-comp of final force"
        )

        self.add_output(
            'Fz',
            val=np.zeros(nn),
            units='N',
            desc="z-comp of final force"
        )

        ar = np.arange(nn)

        self.declare_partials(of='Fx', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='Fx', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='Fx', wrt='w', rows=ar, cols=ar)
        self.declare_partials(of='Fx', wrt='drag', rows=ar, cols=ar)
        self.declare_partials(of='Fx', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='Fx', wrt='side', rows=ar, cols=ar)
        self.declare_partials(of='Fx', wrt='T_x', rows=ar, cols=ar)
        self.declare_partials(of='Fx', wrt='T_y', rows=ar, cols=ar)
        self.declare_partials(of='Fx', wrt='T_z', rows=ar, cols=ar)
        self.declare_partials(of='Fx', wrt='roll', rows=ar, cols=ar)
        self.declare_partials(of='Fx', wrt='pitch', rows=ar, cols=ar)
        self.declare_partials(of='Fx', wrt='yaw', rows=ar, cols=ar)

        self.declare_partials(of='Fy', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='w', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='drag', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='side', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='T_x', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='T_y', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='T_z', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='roll', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='pitch', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='yaw', rows=ar, cols=ar)

        self.declare_partials(of='Fz', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='w', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='drag', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='side', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='T_x', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='T_y', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='T_z', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='roll', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='pitch', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='yaw', rows=ar, cols=ar)

    def compute(self, inputs, outputs):

        u = inputs['u']
        v = inputs['v']
        w = inputs['w']
        D = inputs['drag']
        T = inputs['thrust']
        T_x = inputs['T_x']
        T_y = inputs['T_y']
        T_z = inputs['T_z']
        phi = inputs['roll']
        theta = inputs['pitch']
        psi = inputs['yaw']
        L = inputs['lift']
        S = inputs['side'] # side force -- assume 0 for now
        # chi = inputs['heading_angle']
        # gamma = inputs['flight_path_angle']
        # chi_T = inputs['heading_angle_NED']
        # gamma_T = inputs['fpa_NED']

        nn = self.options['num_nodes']

        # true air speed

        V = np.sqrt(u**2 + v**2 + w**2 + 1e-8) # add epsilon value

        # angle of attack

        # divide by zero checks
        if np.any(u == 0):
            u[u == 0] = 1e-4
            alpha = np.arctan(w / u)
        else:
            alpha = np.arctan(w / u)

        # side slip angle

        # divide by zero checks
        if ((np.any(u != 0) or np.any(w != 0))) :
            beta = np.arctan(v / np.sqrt(u**2 + w**2))
        else:
            u[u == 0] = 1.0e-4
            beta = np.arctan(v / np.sqrt(u**2 + w**2))
        

        # some trig needed

        cos_a = np.cos(alpha)
        cos_b = np.cos(beta)
        sin_a = np.sin(alpha)
        sin_b = np.sin(beta)
        # cos_g = np.cos(gamma)
        # sin_g = np.sin(gamma)
        # cos_c = np.cos(chi)
        # sin_c = np.sin(chi)

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        cos_ph = np.cos(phi)
        sin_ph = np.sin(phi)
        cos_ps = np.cos(psi)
        sin_ps = np.sin(psi)

        # Thrust direction in NED -- as \hat{t}_n

        # t_hat_n = [
        #     np.cos(gamma_T) * np.cos(chi_T),
        #     np.cos(gamma_T) * np.sin(chi_T),
        #     -np.sin(gamma_T)
        # ]

        # t_hat_n = np.array(t_hat_n)
        # t_hat_n = t_hat_n.reshape((int(np.size(t_hat_n)), 1))

        # t_n_x = np.cos(gamma_T) * np.cos(chi_T)
        # t_n_y = np.cos(gamma_T) * np.sin(chi_T)
        # t_n_z = -np.sin(gamma_T)

        # C_{b-<n}

        # Cbn = [
        #     -sin_b * sin_c - cos_b * cos_c * np.cos(alpha - gamma),
        #      sin_b * cos_c - cos_b * sin_c * np.cos(alpha - gamma),
        #      cos_b * np.sin(alpha - gamma),
        #     cos_b * sin_c - sin_b * cos_c * np.cos(alpha - gamma),
        #      -cos_b * cos_c - sin_b * sin_c * np.cos(alpha - gamma),
        #      sin_b * np.sin(alpha - gamma),
        #     cos_c * np.sin(alpha - gamma),
        #      sin_c * np.sin(alpha - gamma),
        #      np.cos(alpha - gamma)
        # ]

        # Cbn = np.array(Cbn)
        # Cbn = Cbn.reshape((3, int(np.size(Cbn)/3)))

        # Cbn_11 = -sin_b * sin_c - cos_b * cos_c * np.cos(alpha - gamma)
        # Cbn_12 = sin_b * cos_c - cos_b * sin_c * np.cos(alpha - gamma)
        # Cbn_13 = cos_b * np.sin(alpha - gamma)

        # Cbn_21 = cos_b * sin_c - sin_b * cos_c * np.cos(alpha - gamma)
        # Cbn_22 = -cos_b * cos_c - sin_b * sin_c * np.cos(alpha - gamma)
        # Cbn_23 = sin_b * np.sin(alpha - gamma)

        # Cbn_31 = cos_c * np.sin(alpha - gamma)
        # Cbn_32 = sin_c * np.sin(alpha - gamma)
        # Cbn_33 = np.cos(alpha - gamma)

        # Thrust direction in body
        #t_hat_b = Cbn @ t_hat_n

        # Thrust force in body
        #Fb_thrust = T * t_hat_b

        # Thrust in (x,y,z)
        # F_T_x = Fb_thrust[0]
        # F_T_y = Fb_thrust[1]
        # F_T_z = Fb_thrust[2]

        # t_b_x = Cbn_11 * t_n_x + Cbn_12 * t_n_y + Cbn_13 * t_n_z
        # t_b_y = Cbn_21 * t_n_x + Cbn_22 * t_n_y + Cbn_23 * t_n_z
        # t_b_z = Cbn_31 * t_n_x + Cbn_32 * t_n_y + Cbn_33 * t_n_z

        # F_T_x = T * t_b_x
        # F_T_y = T * t_b_y
        # F_T_z = T * t_b_z

        F_T_x = T_x * cos_t * cos_ph + T_y * cos_t * sin_ph - T_z * sin_t
        F_T_y = T_x * (-cos_ps * sin_ph + sin_ps * sin_t * cos_ph) + T_y * (cos_ps * cos_ph + sin_ps * sin_t * sin_ph) + T_z * sin_ps * cos_t
        F_T_z = T_x * (sin_ps * sin_ph + cos_ps * sin_t * cos_ph) + T_y * (-sin_ps * cos_ph + cos_ps * sin_t * sin_ph) + T_z * cos_ps * cos_t
        

        outputs['Fx'] = -(cos_a * cos_b * D - cos_a * sin_b * S - sin_a * L) + F_T_x
        outputs['Fy'] = -(sin_b * D + cos_b * S) + F_T_y
        outputs['Fz'] = -(sin_a * cos_b * D + sin_a * sin_b * S + cos_a * L) + F_T_z

    def compute_partials(self, inputs, J):
        u = inputs['u']
        v = inputs['v']
        w = inputs['w']
        D = inputs['drag']
        T = inputs['thrust']
        L = inputs['lift']
        S = inputs['side'] # side force -- assume 0 for now
        # gamma = inputs['flight_path_angle']
        # chi = inputs['heading_angle']
        # chi_T = inputs['heading_angle_NED']
        # gamma_T = inputs['fpa_NED']
        T_x = inputs['T_x']
        T_y = inputs['T_y']
        T_z = inputs['T_z']
        phi = inputs['roll']
        theta = inputs['pitch']
        psi = inputs['yaw']

        V = np.sqrt(u**2 + v**2 + w**2 + 1e-8)

        # divide by zero checks
        if np.any(u == 0):
            u = 1e-4
            alpha = np.arctan(w / u)
        else:
            alpha = np.arctan(w / u)

        # side slip angle

        # divide by zero checks
        if (np.any(u != 0) or np.any(w != 0)) :
            beta = np.arctan(v / np.sqrt(u**2 + w**2))
        else:
            u = 1.0e-4
            beta = np.arctan(v / np.sqrt(u**2 + w**2))
        
        # t_hat_n = [
        #     np.cos(gamma_T) * np.cos(chi_T),
        #     np.cos(gamma_T) * np.sin(chi_T),
        #     -np.sin(gamma_T)
        # ]

        # t_hat_n = np.array(t_hat_n)
        # t_hat_n= t_hat_n.reshape((int(np.size(t_hat_n)), 1))


        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        cos_b, sin_b = np.cos(beta),  np.sin(beta)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        cos_ph, sin_ph = np.cos(phi), np.sin(phi)
        cos_ps, sin_ps = np.cos(psi), np.sin(psi)

        J['Fx', 'u'] = np.cos(alpha) * np.sin(beta) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * D + \
                         np.cos(beta) * np.sin(alpha) * (-w / (w**2 + u**2)) * D + \
                         (np.cos(alpha) * np.cos(beta) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * S - np.sin(beta) * np.sin(alpha) * (-w / (w**2 + u**2)) * S) + \
                         (np.cos(alpha) * (-w / (w**2 + u**2)) * L) 
        J['Fx', 'v'] = np.cos(alpha) * np.sin(beta) * (np.sqrt(w**2 + u**2) / V**2) * D + np.cos(alpha) * np.cos(beta) * (np.sqrt(w**2 + u**2) / V**2) * S 
        J['Fx', 'w'] = np.cos(alpha) * np.sin(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * D + np.sin(alpha) * np.cos(beta) * (u / (w**2 + u**2)) * D + \
                       np.cos(alpha) * np.cos(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * S - np.sin(alpha) * np.sin(beta) * (u / (w**2 + u**2)) * S + \
                       np.cos(alpha) * (u / (w**2 + u**2)) * L 
        J['Fx', 'drag'] = -np.cos(alpha) * np.cos(beta)
        J['Fx', 'lift'] = np.sin(alpha)
        J['Fx', 'side'] = np.cos(alpha) * np.sin(beta)
        J['Fx', 'T_x'] = cos_t * cos_ph
        J['Fx', 'T_y'] = cos_t * sin_ph
        J['Fx', 'T_z'] = -sin_t
        J['Fx', 'roll'] = -T_x * cos_t * sin_ph + T_y * cos_t * cos_ph 
        J['Fx', 'pitch'] = -T_x * sin_t * cos_ph - T_y * sin_t * sin_ph - T_z * cos_t
        J['Fx', 'yaw'] = 0.0

        J['Fy', 'u'] = -np.cos(beta) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * D + np.sin(beta) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * S 
        J['Fy', 'v'] = -np.cos(beta) * (np.sqrt(w**2 + u**2) / V**2) * D + np.sin(beta) * (np.sqrt(w**2 + u**2) / V**2) * S 
        J['Fy', 'w'] = -np.cos(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * D + np.sin(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * S 
        J['Fy', 'drag'] = -np.sin(beta)
        J['Fy', 'side'] = -np.cos(beta)
        J['Fy', 'T_x'] = -cos_ps * sin_ph + sin_ps * sin_t * cos_ph
        J['Fy', 'T_y'] = cos_ps * cos_ph + sin_ps * sin_t * sin_ph
        J['Fy', 'T_z'] = sin_ps * cos_t
        J['Fy', 'roll'] = T_x * (-cos_ps * cos_ph - sin_ps * sin_t * sin_ph) + T_y * (-cos_ps * sin_ph + sin_ps * sin_t * cos_ph) 
        J['Fy', 'pitch'] = T_x * sin_ps * cos_t * cos_ph + T_y * sin_ps * cos_t * sin_ph - T_z * sin_ps * sin_t
        J['Fy', 'yaw'] = T_x * (sin_ps * sin_ph + cos_ps * sin_t * cos_ph) + T_y * (-sin_ps * cos_ph + cos_ps * sin_t * sin_ph) + T_z * cos_ps * cos_t

        J['Fz', 'u'] = np.sin(alpha) * np.sin(beta) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * D - np.cos(alpha) * np.cos(beta) * (-w / (w**2 + u**2)) * D - \
                       np.sin(alpha) * np.cos(beta) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * S - np.cos(alpha) * np.sin(beta) * (-w / (w**2 + u**2)) * S + \
                       np.sin(alpha) * (-w / (w**2 + u**2)) * L 
        J['Fz', 'v'] = np.sin(alpha) * np.sin(beta) * (np.sqrt(w**2 + u**2) / V**2) * D - np.sin(alpha) * np.cos(beta) * (np.sqrt(w**2 + u**2) / V**2) * S 
        J['Fz', 'w'] = np.sin(alpha) * np.sin(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * D - np.cos(alpha) * np.cos(beta) * (u / (w**2 + u**2)) * D - \
                       np.sin(alpha) * np.cos(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * S - np.cos(alpha) * np.sin(beta) * (u / (w**2 + u**2)) * S + \
                       np.sin(alpha) * (u / (w**2 + u**2)) * L 
        J['Fz', 'drag'] = -np.sin(alpha) * np.cos(beta) 
        J['Fz', 'lift'] = -np.cos(alpha)
        J['Fz', 'side'] = -np.sin(alpha) * np.sin(beta)
        J['Fz', 'T_x'] = sin_ps * sin_ph + cos_ps * sin_t * cos_ph
        J['Fz', 'T_y'] = -sin_ps * cos_ph + cos_ps * sin_t * sin_ph
        J['Fz', 'T_z'] = cos_ps * cos_t
        J['Fz', 'roll'] = T_x * (sin_ps * cos_ph - cos_ps * sin_t * sin_ph) + T_y * (sin_ps * sin_ph + cos_ps * sin_t * cos_ph) 
        J['Fz', 'pitch'] = T_x * cos_ps * cos_t * cos_ph + T_y * cos_ps * cos_t * sin_ph - T_z * cos_ps * sin_t
        J['Fz', 'yaw'] = T_x * (cos_ps * sin_ph - sin_ps * sin_t * cos_ph) + T_y * (-cos_ps * cos_ph - sin_ps * sin_t * sin_ph) - T_z * sin_ps * cos_t

if __name__ == "__main__":
    p = om.Problem()
    p.model = om.Group()
    des_vars = p.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=['*'])

    des_vars.add_output('u', 0.5, units='m/s')
    des_vars.add_output('v', 0.6, units='m/s')
    des_vars.add_output('w', 0.7, units='m/s')
    des_vars.add_output('drag', 50, units='N')
    des_vars.add_output('thrust', 50, units='N')
    des_vars.add_output('lift', 60, units='N')
    des_vars.add_output('side', 70, units='N')
    des_vars.add_output('T_x', 10, units='N')
    des_vars.add_output('T_y', 10, units='N')
    des_vars.add_output('T_z', 10, units='N')
    des_vars.add_output('roll', np.pi/2, units='rad')
    des_vars.add_output('pitch', np.pi/2, units='rad')
    des_vars.add_output('yaw', np.pi/2, units='rad')
    

    p.model.add_subsystem('ForceComponentResolver_Copy', ForceComponentResolver_Copy(num_nodes=1), promotes=['*'])

    p.setup(check=False, force_alloc_complex=True)

    p.run_model()

    p.check_partials(compact_print=True, show_only_incorrect=True, method='cs')








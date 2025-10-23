import numpy as np
import openmdao.api as om
from aviary.variable_info.variables import Dynamic
from aviary.utils.functions import add_aviary_input


class ForceComponentResolver(om.ExplicitComponent):
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
            'heading_angle',
            val=np.zeros(nn),
            units='rad',
            desc='Heading angle in body'
        )

        self.add_input(
            'flight_path_angle',
            val=np.zeros(nn),
            units='rad',
            desc="Flight path angle in body")
        
        self.add_input(
            'heading_angle_NED',
            val=np.zeros(nn),
            units='rad',
            desc="Thrust heading angle in NED"
        )

        self.add_input(
            'fpa_NED',
            val=np.zeros(nn),
            units='rad',
            desc="Thrust flight path angle in NED"
        )

        # self.add_input(
        #     'true_air_speed',
        #     val=np.zeros(nn),
        #     units='m/s',
        #     desc="True air speed"
        # ) # This is an aviary variable

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
        self.declare_partials(of='Fx', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='Fx', wrt='heading_angle', rows=ar, cols=ar)
        self.declare_partials(of='Fx', wrt='flight_path_angle', rows=ar, cols=ar)
        self.declare_partials(of='Fx', wrt='heading_angle_NED', rows=ar, cols=ar)
        self.declare_partials(of='Fx', wrt='fpa_NED', rows=ar, cols=ar)

        self.declare_partials(of='Fy', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='w', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='drag', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='side', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='heading_angle', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='flight_path_angle', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='heading_angle_NED', rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt='fpa_NED', rows=ar, cols=ar)

        self.declare_partials(of='Fz', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='w', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='drag', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='lift', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='side', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='thrust', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='heading_angle', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='flight_path_angle', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='heading_angle_NED', rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt='fpa_NED', rows=ar, cols=ar)

    def compute(self, inputs, outputs):

        u = inputs['u']
        v = inputs['v']
        w = inputs['w']
        D = inputs['drag']
        T = inputs['thrust']
        L = inputs['lift']
        S = inputs['side'] # side force -- assume 0 for now
        chi = inputs['heading_angle']
        gamma = inputs['flight_path_angle']
        chi_T = inputs['heading_angle_NED']
        gamma_T = inputs['fpa_NED']

        nn = self.options['num_nodes']

        # true air speed

        V = np.sqrt(u**2 + v**2 + w**2 + 1e-8) # add epsilon value

        # angle of attack

        # divide by zero checks
        # if np.any(u == 0):
        #     u[u == 0] = 1e-4
        #     alpha = np.arctan(w / u)
        # else:
        #     alpha = np.arctan(w / u)

        alpha = np.arctan2(w, u)
        # side slip angle

        # divide by zero checks
        # if ((np.any(u != 0) or np.any(w != 0))) :
        #     beta = np.arctan(v / np.sqrt(u**2 + w**2))
        # else:
        #     u[u == 0] = 1.0e-4
        #     beta = np.arctan(v / np.sqrt(u**2 + w**2))

        beta = np.arctan2(v, np.sqrt(u**2 + w**2 + 1e-8))
        

        # some trig needed

        cos_a = np.cos(alpha)
        cos_b = np.cos(beta)
        sin_a = np.sin(alpha)
        sin_b = np.sin(beta)
        cos_g = np.cos(gamma)
        sin_g = np.sin(gamma)
        cos_c = np.cos(chi)
        sin_c = np.sin(chi)

        # Thrust direction in NED -- as \hat{t}_n

        # t_hat_n = [
        #     np.cos(gamma_T) * np.cos(chi_T),
        #     np.cos(gamma_T) * np.sin(chi_T),
        #     -np.sin(gamma_T)
        # ]

        # t_hat_n = np.array(t_hat_n)
        # t_hat_n = t_hat_n.reshape((int(np.size(t_hat_n)), 1))

        t_n_x = np.cos(gamma_T) * np.cos(chi_T)
        t_n_y = np.cos(gamma_T) * np.sin(chi_T)
        t_n_z = -np.sin(gamma_T)

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

        Cbn_11 = -sin_b * sin_c - cos_b * cos_c * np.cos(alpha - gamma)
        Cbn_12 = sin_b * cos_c - cos_b * sin_c * np.cos(alpha - gamma)
        Cbn_13 = cos_b * np.sin(alpha - gamma)

        Cbn_21 = cos_b * sin_c - sin_b * cos_c * np.cos(alpha - gamma)
        Cbn_22 = -cos_b * cos_c - sin_b * sin_c * np.cos(alpha - gamma)
        Cbn_23 = sin_b * np.sin(alpha - gamma)

        Cbn_31 = cos_c * np.sin(alpha - gamma)
        Cbn_32 = sin_c * np.sin(alpha - gamma)
        Cbn_33 = np.cos(alpha - gamma)

        # Thrust direction in body
        #t_hat_b = Cbn @ t_hat_n

        # Thrust force in body
        #Fb_thrust = T * t_hat_b

        # Thrust in (x,y,z)
        # F_T_x = Fb_thrust[0]
        # F_T_y = Fb_thrust[1]
        # F_T_z = Fb_thrust[2]

        t_b_x = Cbn_11 * t_n_x + Cbn_12 * t_n_y + Cbn_13 * t_n_z
        t_b_y = Cbn_21 * t_n_x + Cbn_22 * t_n_y + Cbn_23 * t_n_z
        t_b_z = Cbn_31 * t_n_x + Cbn_32 * t_n_y + Cbn_33 * t_n_z

        F_T_x = T * t_b_x
        F_T_y = T * t_b_y
        F_T_z = T * t_b_z
        

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
        gamma = inputs['flight_path_angle']
        chi = inputs['heading_angle']
        chi_T = inputs['heading_angle_NED']
        gamma_T = inputs['fpa_NED']

        V = np.sqrt(u**2 + v**2 + w**2 + 1e-8)

        # divide by zero checks
        # if np.any(u == 0):
        #     u = 1e-4
        #     alpha = np.arctan(w / u)
        # else:
        #     alpha = np.arctan(w / u)

        # side slip angle

        # divide by zero checks
        # if (np.any(u != 0) or np.any(w != 0)) :
        #     beta = np.arctan(v / np.sqrt(u**2 + w**2))
        # else:
        #     u = 1.0e-4
        #     beta = np.arctan(v / np.sqrt(u**2 + w**2))
        
        # t_hat_n = [
        #     np.cos(gamma_T) * np.cos(chi_T),
        #     np.cos(gamma_T) * np.sin(chi_T),
        #     -np.sin(gamma_T)
        # ]

        # t_hat_n = np.array(t_hat_n)
        # t_hat_n= t_hat_n.reshape((int(np.size(t_hat_n)), 1))

        alpha = np.arctan2(w, u)
        beta = np.arctan2(v, np.sqrt(u**2 + w**2 + 1e-8))

        cos_a, sin_a = np.cos(alpha), np.sin(alpha)
        cos_b, sin_b = np.cos(beta),  np.sin(beta)
        cos_g, sin_g = np.cos(gamma), np.sin(gamma)
        cos_c, sin_c = np.cos(chi),   np.sin(chi)

        t_n_x = np.cos(gamma_T) * np.cos(chi_T)
        t_n_y = np.cos(gamma_T) * np.sin(chi_T)
        t_n_z = -np.sin(gamma_T)

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

        Cbn_11 = -sin_b * sin_c - cos_b * cos_c * np.cos(alpha - gamma)
        Cbn_12 = sin_b * cos_c - cos_b * sin_c * np.cos(alpha - gamma)
        Cbn_13 = cos_b * np.sin(alpha - gamma)

        Cbn_21 = cos_b * sin_c - sin_b * cos_c * np.cos(alpha - gamma)
        Cbn_22 = -cos_b * cos_c - sin_b * sin_c * np.cos(alpha - gamma)
        Cbn_23 = sin_b * np.sin(alpha - gamma)

        Cbn_31 = cos_c * np.sin(alpha - gamma)
        Cbn_32 = sin_c * np.sin(alpha - gamma)
        Cbn_33 = np.cos(alpha - gamma)

        t_b_x = Cbn_11 * t_n_x + Cbn_12 * t_n_y + Cbn_13 * t_n_z
        t_b_y = Cbn_21 * t_n_x + Cbn_22 * t_n_y + Cbn_23 * t_n_z
        t_b_z = Cbn_31 * t_n_x + Cbn_32 * t_n_y + Cbn_33 * t_n_z

        # Derivatives of t_n
        # dt_n_dchi = [
        #     -np.cos(gamma_T) * np.sin(chi_T),
        #     np.cos(gamma_T) * np.cos(chi_T),
        #     np.zeros(int(np.size(gamma_T)))
        # ]

        # dt_n_dchi = np.array(dt_n_dchi, dtype="object")
        # dt_n_dchi = dt_n_dchi.reshape((int(np.size(dt_n_dchi)), 1))

        dt_n_x_dchi_T = -np.cos(gamma_T) * np.sin(chi_T)
        dt_n_y_dchi_T = np.cos(gamma_T) * np.cos(chi_T)
        dt_n_z_dchi_T = 0.0

        # dt_n_dgamma = [
        #     -np.sin(gamma_T) * np.cos(chi_T),
        #     -np.sin(gamma_T) * np.sin(chi_T),
        #     -np.cos(gamma_T)
        # ]

        # dt_n_dgamma = np.array(dt_n_dgamma, dtype="object")
        # dt_n_dgamma = dt_n_dgamma.reshape((int(np.size(dt_n_dgamma)), 1))

        dt_n_x_dgamma_T = -np.sin(gamma_T) * np.cos(chi_T)
        dt_n_y_dgamma_T = -np.sin(gamma_T) * np.sin(chi_T)
        dt_n_z_dgamma_T = -np.cos(gamma_T)

        # Rotation to body + derivatives
        # t_b = Cbn @ t_hat_n
        # dt_b_dchi = Cbn @ dt_n_dchi
        # dt_b_dgamma = Cbn @ dt_n_dgamma

        dt_b_x_dchi_T = Cbn_11 * dt_n_x_dchi_T + Cbn_12 * dt_n_y_dchi_T + Cbn_13 * dt_n_z_dchi_T
        dt_b_y_dchi_T = Cbn_21 * dt_n_x_dchi_T + Cbn_22 * dt_n_y_dchi_T + Cbn_23 * dt_n_z_dchi_T
        dt_b_z_dchi_T = Cbn_31 * dt_n_x_dchi_T + Cbn_32 * dt_n_y_dchi_T + Cbn_33 * dt_n_z_dchi_T

        dt_b_x_dgamma_T = Cbn_11 * dt_n_x_dgamma_T + Cbn_12 * dt_n_y_dgamma_T + Cbn_13 * dt_n_z_dgamma_T
        dt_b_y_dgamma_T = Cbn_21 * dt_n_x_dgamma_T + Cbn_22 * dt_n_y_dgamma_T + Cbn_23 * dt_n_z_dgamma_T
        dt_b_z_dgamma_T = Cbn_31 * dt_n_x_dgamma_T + Cbn_32 * dt_n_y_dgamma_T + Cbn_33 * dt_n_z_dgamma_T

        # note: d/dx arctan(a/x) = -a / (x^2 + a^2)
        # note: d/dx arctan(a / sqrt(b^2 + x^2)) = - ax / ((b^2 + x^2 + a^2) * sqrt(b^2 + x^2))
        # note: d/dx arctan(x / sqrt(a^2 + b^2)) = sqrt(a^2 + b^2) / (a^2 + b^2 + x^2)
        # note: d/dx arctan(x/a) = a / (a^2 + x^2)

        J['Fx', 'u'] = np.cos(alpha) * np.sin(beta) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * D + \
                         np.cos(beta) * np.sin(alpha) * (-w / (w**2 + u**2)) * D + \
                         (np.cos(alpha) * np.cos(beta) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * S - np.sin(beta) * np.sin(alpha) * (-w / (w**2 + u**2)) * S) + \
                         (np.cos(alpha) * (-w / (w**2 + u**2)) * L) + T * ((-np.cos(gamma_T) * np.cos(chi_T) * cos_b * sin_c * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) + 
                                                                           np.cos(gamma_T) * np.cos(chi_T) * sin_b * cos_c * np.cos(alpha - gamma) * 
                                                                           ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) + np.cos(gamma_T) * np.cos(chi_T) * cos_b * cos_c * 
                                                                           np.sin(alpha - gamma) * (-w / (w**2 + u**2))) + (
                                                                               np.cos(gamma_T) * np.sin(chi_T) * cos_b * cos_c * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) + 
                                                                                np.cos(gamma_T) * np.sin(chi_T) * sin_b * sin_c * np.cos(alpha - gamma) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) + 
                                                                           np.cos(gamma_T) * np.sin(chi_T) * cos_b * sin_c * np.sin(alpha -gamma) * (-w / (w**2 + u**2))) + 
                                                                           (np.sin(gamma_T) * sin_b * np.sin(alpha - gamma) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) - 
                                                                            np.sin(gamma_T) * cos_b * np.cos(alpha - gamma) * (-w / (w**2 + u**2))))
        J['Fx', 'v'] = np.cos(alpha) * np.sin(beta) * (np.sqrt(w**2 + u**2) / V**2) * D + np.cos(alpha) * np.cos(beta) * (np.sqrt(w**2 + u**2) / V**2) * S + T * (
            (-np.cos(gamma_T) * np.cos(chi_T) * cos_b * sin_c * (np.sqrt(w**2 + u**2) / V**2) + np.cos(gamma_T) * np.cos(chi_T) * sin_b * cos_c * np.cos(alpha - gamma) * (np.sqrt(w**2 + u**2) / V**2)) + 
             (np.cos(gamma_T) * np.sin(chi_T) * cos_b * cos_c * (np.sqrt(w**2 + u**2) / V**2) + np.cos(gamma_T) * np.sin(chi_T) * sin_b * sin_c * np.cos(alpha - gamma) * (np.sqrt(w**2 + u**2) / V**2)) +
             (np.sin(gamma_T) * sin_b * np.sin(alpha - gamma) * (np.sqrt(w**2 + u**2) / V**2))
        )
        J['Fx', 'w'] = np.cos(alpha) * np.sin(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * D + np.sin(alpha) * np.cos(beta) * (u / (w**2 + u**2)) * D + \
                       np.cos(alpha) * np.cos(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * S - np.sin(alpha) * np.sin(beta) * (u / (w**2 + u**2)) * S + \
                       np.cos(alpha) * (u / (w**2 + u**2)) * L + T * (
                           (-np.cos(gamma_T) * np.cos(chi_T) * cos_b * sin_c * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) + np.cos(gamma_T) * np.cos(chi_T) * cos_b * cos_c * np.sin(alpha - gamma) * (u / (w**2 + u**2)) +
                            np.cos(gamma_T) * np.cos(chi_T) * sin_b * cos_c * np.cos(alpha - gamma) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2))))) + 
                           (np.cos(gamma_T) * np.sin(chi_T) * cos_b * cos_c * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) + np.cos(gamma_T) * np.sin(chi_T) * cos_b * sin_c * np.sin(alpha - gamma) * (u / (w**2 + u**2)) + 
                            np.cos(gamma_T) * np.sin(chi_T) * sin_b * sin_c * np.cos(alpha - gamma) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2))))) + 
                           (-np.sin(gamma_T) * cos_b * np.cos(alpha - gamma) * (u / (w**2 + u**2)) + np.sin(gamma_T) * sin_b * np.sin(alpha - gamma) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))))
                       )
        J['Fx', 'drag'] = -np.cos(alpha) * np.cos(beta)
        J['Fx', 'lift'] = np.sin(alpha)
        J['Fx', 'side'] = np.cos(alpha) * np.sin(beta)
        J['Fx', 'thrust'] = t_b_x
        J['Fx', 'flight_path_angle'] = T * ((-cos_b * cos_c * np.sin(alpha - gamma)) * t_n_x + (-cos_b * sin_c * np.sin(alpha - gamma)) * t_n_y + 
                                                          (-cos_b * np.cos(alpha - gamma)) * t_n_z)
        J['Fx', 'heading_angle'] = T * ((-sin_b * cos_c + cos_b * sin_c * np.cos(alpha - gamma)) * t_n_x + 
                                        (-sin_b * sin_c - cos_b * cos_c * np.cos(alpha - gamma)) * t_n_y)
        J['Fx', 'heading_angle_NED'] = T * dt_b_x_dchi_T
        J['Fx', 'fpa_NED'] = T * dt_b_x_dgamma_T

        J['Fy', 'u'] = -np.cos(beta) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * D + np.sin(beta) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * S - \
                        T * np.cos(gamma_T) * np.cos(chi_T) * sin_b * sin_c * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) - \
                        np.cos(gamma_T) * np.cos(chi_T) * np.cos(beta) * cos_c * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * T * np.cos(alpha - gamma) + \
                        np.sin(alpha - gamma) * cos_c * (-w / (w**2 + u**2)) * T * np.sin(beta) * np.cos(gamma_T) * np.cos(chi_T) + \
                        sin_b * cos_c * np.cos(gamma_T) * np.sin(chi_T) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * T - \
                        np.cos(gamma_T) * np.sin(chi_T) * cos_b * sin_c * np.cos(alpha - gamma) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * T + \
                        np.cos(gamma_T) * np.sin(chi_T) * sin_b * sin_c * np.sin(alpha - gamma) * (-w / (w**2 + u**2)) * T - \
                        np.sin(gamma_T) * cos_b * np.sin(alpha - gamma) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * T - \
                        np.sin(gamma_T) * sin_b * np.cos(alpha - gamma) * (-w / (w**2 + u**2)) * T
        J['Fy', 'v'] = -np.cos(beta) * (np.sqrt(w**2 + u**2) / V**2) * D + np.sin(beta) * (np.sqrt(w**2 + u**2) / V**2) * S - \
                        np.cos(gamma_T) * np.cos(chi_T) * sin_b * sin_c * (np.sqrt(w**2 + u**2) / V**2) * T - \
                        np.cos(gamma_T) * np.cos(chi_T) * cos_b * cos_c * np.cos(alpha - gamma) * (np.sqrt(w**2 + u**2) / V**2) * T + \
                        np.cos(gamma_T) * np.sin(chi_T) * sin_b * cos_c * (np.sqrt(w**2 + u**2) / V**2) * T - \
                        np.cos(gamma_T) * np.sin(chi_T) * np.cos(beta) * sin_c * (np.sqrt(w**2 + u**2) / V**2) * T * np.cos(alpha - gamma) - \
                        np.sin(gamma_T) * cos_b * np.sin(alpha - gamma) * (np.sqrt(w**2 + u**2) / V**2) * T
        J['Fy', 'w'] = -np.cos(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * D + np.sin(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * S - \
                        np.cos(gamma_T) * np.cos(chi_T) * sin_b * sin_c * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * T - \
                        np.cos(gamma_T) * np.cos(chi_T) * cos_b * cos_c * np.cos(alpha - gamma) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * T + \
                        np.cos(gamma_T) * np.cos(chi_T) * sin_b * cos_c * np.sin(alpha - gamma) * (u / (w**2 + u**2)) * T  + \
                        np.cos(gamma_T) * np.sin(chi_T) * sin_b * cos_c * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * T - \
                        np.cos(gamma_T) * np.sin(chi_T) * cos_b * sin_c * np.cos(alpha - gamma) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * T + \
                        np.cos(gamma_T) * np.sin(chi_T) * sin_b * sin_c * np.sin(alpha - gamma) * (u / (w**2 + u**2)) * T - \
                        np.sin(gamma_T) * np.cos(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * T * np.sin(alpha - gamma) - \
                        np.sin(gamma_T) * np.cos(alpha - gamma) * (u / (w**2 + u**2)) * T * np.sin(beta)
        J['Fy', 'drag'] = -np.sin(beta)
        J['Fy', 'side'] = -np.cos(beta)
        J['Fy', 'thrust'] = t_b_y
        J['Fy', 'flight_path_angle'] = T * ((-sin_b * cos_c * np.sin(alpha - gamma)) * t_n_x + (-sin_b * sin_c * np.sin(alpha - gamma)) * t_n_y + 
                                                          (-sin_b * np.cos(alpha - gamma)) * t_n_z)
        J['Fy', 'heading_angle'] = T * ((cos_b * cos_c + sin_b * sin_c * np.cos(alpha - gamma)) * t_n_x + (cos_b * sin_c - sin_b * cos_c * np.cos(alpha - gamma)) * t_n_y)
        J['Fy', 'heading_angle_NED'] = T * dt_b_y_dchi_T
        J['Fy', 'fpa_NED'] = T * dt_b_y_dgamma_T

        J['Fz', 'u'] = np.sin(alpha) * np.sin(beta) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * D - np.cos(alpha) * np.cos(beta) * (-w / (w**2 + u**2)) * D - \
                       np.sin(alpha) * np.cos(beta) * ((-v * u) / ((V**2 * np.sqrt(w**2 + u**2)))) * S - np.cos(alpha) * np.sin(beta) * (-w / (w**2 + u**2)) * S + \
                       np.sin(alpha) * (-w / (w**2 + u**2)) * L + np.cos(gamma_T) * np.cos(chi_T) * cos_c * np.cos(alpha - gamma) * (-w / (w**2 + u**2)) * T + \
                       np.cos(gamma_T) * np.sin(chi_T) * sin_c * np.cos(alpha - gamma) * (-w / (w**2 + u**2)) * T + \
                       np.sin(gamma_T) * np.sin(alpha - gamma) * (-w / (w**2 + u**2)) * T
        J['Fz', 'v'] = np.sin(alpha) * np.sin(beta) * (np.sqrt(w**2 + u**2) / V**2) * D - np.sin(alpha) * np.cos(beta) * (np.sqrt(w**2 + u**2) / V**2) * S 
        J['Fz', 'w'] = np.sin(alpha) * np.sin(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * D - np.cos(alpha) * np.cos(beta) * (u / (w**2 + u**2)) * D - \
                       np.sin(alpha) * np.cos(beta) * ((-w * v) / ((V**2 * np.sqrt(w**2 + u**2)))) * S - np.cos(alpha) * np.sin(beta) * (u / (w**2 + u**2)) * S + \
                       np.sin(alpha) * (u / (w**2 + u**2)) * L + np.cos(gamma_T) * np.cos(chi_T) * cos_c * np.cos(alpha - gamma) * (u / (w**2 + u**2)) * T + \
                       np.cos(gamma_T) * np.sin(chi_T) * sin_c * np.cos(alpha - gamma) * (u / (w**2 + u**2)) * T + \
                       np.sin(gamma_T) * np.sin(alpha - gamma) * (u / (w**2 + u**2)) * T
        J['Fz', 'drag'] = -np.sin(alpha) * np.cos(beta) 
        J['Fz', 'lift'] = -np.cos(alpha)
        J['Fz', 'side'] = -np.sin(alpha) * np.sin(beta)
        J['Fz', 'thrust'] = t_b_z
        J['Fz', 'flight_path_angle'] = T * ((-cos_c * np.cos(alpha - gamma)) * t_n_x + (-sin_c * np.cos(alpha - gamma)) * t_n_y + 
                                                          np.sin(alpha - gamma) * t_n_z)
        J['Fz', 'heading_angle'] = T * ((-sin_c * np.sin(alpha - gamma)) * t_n_x + (cos_c * np.sin(alpha - gamma)) * t_n_y)
        J['Fz', 'heading_angle_NED'] = T * dt_b_z_dchi_T
        J['Fz', 'fpa_NED'] = T * dt_b_z_dgamma_T

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
    des_vars.add_output('heading_angle', units='rad')
    des_vars.add_output('flight_path_angle', units='rad')
    des_vars.add_output('heading_angle_NED', units='rad')
    des_vars.add_output('fpa_NED', units='rad')

    p.model.add_subsystem('ForceComponentResolver', ForceComponentResolver(num_nodes=1), promotes=['*'])

    p.setup(check=False, force_alloc_complex=True)

    p.run_model()

    p.check_partials(compact_print=True, show_only_incorrect=True, method='cs')








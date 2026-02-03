import numpy as np
import openmdao.api as om

class SixDOF_EOl(om.ExplicitComponent):
    """
    Six DOF EOl component, with particular emphasis for rotorcraft. 
    ASSUlPTJ_ONS:
        - Assume Flat Earth model (particularly for rotorcraft)
        - Earth is the internal f.o.r.
        - Gravity is constant and normal to the tangent plane (Earth's surface) -- g = (0 0 G)^T
        - (aircraft) mass is constant
        - aircraft is a rigid body
        - Symmetry in the xz plane (for moment of inertia matrix -- J_xy = J_yx = J_zy = J_yz = 0)

    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
    
    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(
            'mass',
            val=np.zeros(1),
            units='kg',
            desc="mass -- assume constant"
        )

        self.add_input(
            'u',
            val=np.zeros(nn),
            units='m/s', # meters per second
            desc="axial velocity of Cl wrt inertial CS resolved in aircraft body fixed CS"
        )

        self.add_input(
            'v',
            val=np.zeros(nn),
            units='m/s',
            desc="lateral velocity of Cl wrt inertial CS resolved in aircraft body fixed CS"
        )

        self.add_input(
            'w',
            val=np.zeros(nn),
            units='m/s',
            desc="vertical velocity of Cl wrt inertial CS resolved in aircraft body fixed CS"
        )

        self.add_input(
            'roll_angle_vel',
            val=np.zeros(nn),
            units='rad/s', # radians per second
            desc="roll angular velocity of body fixed CS wrt intertial CS"
        )

        self.add_input(
            'pitch_angle_vel',
            val=np.zeros(nn),
            units='rad/s',
            desc="pitch angular velocity of body fixed CS wrt intertial CS"
        )

        self.add_input(
            'yaw_ang_vel',
            val=np.zeros(nn),
            units='rad/s',
            desc="yaw angular velocity of body fixed CS wrt intertial CS"
        )

        self.add_input(
            'roll',
            val=np.zeros(nn),
            units='rad', # radians
            desc="roll angle"
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

        self.add_input(
            'x',
            val=np.zeros(nn),
            units='m',
            desc="x-axis position of aircraft resolved in North-East-Down (NED) CS"
        )

        self.add_input(
            'y',
            val=np.zeros(nn),
            units='m',
            desc="y-axis position of aircraft resolved in NED CS"
        )

        self.add_input(
            'z',
            val=np.zeros(nn),
            units='m',
            desc="z-axis position of aircraft resolved in NED CS"
        )

        self.add_input(
            'g',
            val=9.81,
            units='m/s**2',
            desc="acceleration due to gravity"
        )

        self.add_input(
            'Fx',
            val=np.zeros(nn),
            units='N',
            desc="external forces in the x direciton"
        )

        self.add_input(
            'Fy',
            val=np.zeros(nn),
            units='N',
            desc="external forces in the y direction"
        )

        self.add_input(
            'Fz',
            val=np.zeros(nn),
            units='N',
            desc="external forces in the z direction"
        )

        self.add_input(
            'lx',
            val=np.zeros(nn),
            units='kg*m**2/s**2', # kg times m^2 / s^2
            desc="external moments in the x direction"
        )

        self.add_input(
            'ly',
            val=np.zeros(nn),
            units='kg*m**2/s**2',
            desc="external moments in the y direction"
        )

        self.add_input(
            'lz',
            val=np.zeros(nn),
            units='kg*m**2/s**2',
            desc="external moments in the z direction"
        )

        # Below are the necessary components for the moment of inertia matrix (J)
        # Only xx, yy, zz, and xz are needed (xy and yz are 0 with assumptions)
        # For now, these are separated.
        # TODO: Rewrite J and EOl in matrix form

        self.add_input(
            'J_xz',
            val=np.zeros(1),
            units='kg*m**2',
            desc="x-z (top right and bottom left corner of 3x3 matrix, assuming symmetry) " \
            "component"
        )

        self.add_input(
            'J_xy',
            val=np.zeros(1),
            units='kg*m**2',
            desc="xy component of moment of inertia matrix"
        )

        self.add_input(
            'J_yz',
            val=np.zeros(1),
            units='kg*m**2',
            desc='yz component of inertia matrix'
        )

        self.add_input(
            'J_xx',
            val=np.zeros(1),
            units='kg*m**2',
            desc="first diag component"
        )

        self.add_input(
            'J_yy',
            val=np.zeros(1),
            units='kg*m**2',
            desc="second diag component"
        )

        self.add_input(
            'J_zz',
            val=np.zeros(1),
            units='kg*m**2',
            desc="third diag component"
        )

        # Outputs

        self.add_output(
            'dx_accel',
            val=np.zeros(nn),
            units='m/s**2', # meters per seconds squared
            desc="x-axis (roll-axis) velocity equation, " \
            "state: u",
            tags=['dymos.state_rate_source:u', 'dymos.state_units:m/s']
        )

        self.add_output(
            'dy_accel',
            val=np.zeros(nn),
            units='m/s**2',
            desc="y-axis (pitch axis) velocity equation, " \
            "state: v",
            tags=['dymos.state_rate_source:v', 'dymos.state_units:m/s']
        )

        self.add_output(
            'dz_accel',
            val=np.zeros(nn),
            units='m/s**2',
            desc="z-axis (yaw axis) velocity equation, " \
            "state: w",
            tags=['dymos.state_rate_source:w', 'dymos.state_units:m/s']
        )

        self.add_output(
            'roll_accel',
            val=np.zeros(nn),
            units='rad/s**2', # radians per second squared
            desc="roll equation, " \
            "state: p",
            tags=['dymos.state_rate_source:p', 'dymos.state_units:rad/s']
        )

        self.add_output(
            'pitch_accel',
            val=np.zeros(nn),
            units='rad/s**2',
            desc="pitch equation, " \
            "state: q",
            tags=['dymos.state_rate_source:q', 'dymos.state_units:rad/s']
        )

        self.add_output(
            'yaw_accel',
            val=np.zeros(nn),
            units='rad/s**2',
            desc="yaw equation, " \
            "state: r",
            tags=['dymos.state_rate_source:r', 'dymos.state_units:rad/s']
        )

        self.add_output(
            'roll_angle_rate_eq',
            val=np.zeros(nn),
            units='rad/s',
            desc="Euler angular roll rate",
            tags=['dymos.state_rate_source:roll', 'dymos.state_units:rad']
        )

        self.add_output(
            'pitch_angle_rate_eq',
            val=np.zeros(nn),
            units='rad/s',
            desc="Euler angular pitch rate",
            tags=['dymos.state_rate_source:pitch', 'dymos.state_units:rad']
        )

        self.add_output(
            'yaw_angle_rate_eq',
            val=np.zeros(nn),
            units='rad/s',
            desc="Euler angular yaw rate",
            tags=['dymos.state_rate_source:yaw', 'dymos.state_units:rad']
        )

        self.add_output(
            'dx_dt',
            val=np.zeros(nn),
            units='m/s',
            desc="x-position derivative of aircraft COl wrt point in NED CS",
            tags=['dymos.state_rate_source:x', 'dymos.state_units:m'] 
        )

        self.add_output(
            'dy_dt',
            val=np.zeros(nn),
            units='m/s',
            desc="y-position derivative of aircraft COl wrt point in NED CS",
            tags=['dymos.state_rate_source:y', 'dymos.state_units:m']
        )

        self.add_output(
            'dz_dt',
            val=np.zeros(nn),
            units='m/s',
            desc="z-position derivative of aircraft COl wrt point in NED CS",
            tags=['dymos.state_rate_source:z', 'dymos.state_units:m']
        )

        ar = np.arange(nn)
        self.declare_partials(of='dx_accel', wrt='mass')
        self.declare_partials(of='dx_accel', wrt='Fx', rows=ar, cols=ar)
        self.declare_partials(of='dx_accel', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='dx_accel', wrt='w', rows=ar, cols=ar)
        self.declare_partials(of='dx_accel', wrt='yaw_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='dx_accel', wrt='pitch_angle_vel', rows=ar, cols=ar)
        self.declare_partials(of='dx_accel', wrt='g')
        self.declare_partials(of='dx_accel', wrt='pitch', rows=ar, cols=ar)

        self.declare_partials(of='dy_accel', wrt='mass')
        self.declare_partials(of='dy_accel', wrt='Fy', rows=ar, cols=ar)
        self.declare_partials(of='dy_accel', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='dy_accel', wrt='w', rows=ar, cols=ar)
        self.declare_partials(of='dy_accel', wrt='yaw_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='dy_accel', wrt='roll_angle_vel', rows=ar, cols=ar)
        self.declare_partials(of='dy_accel', wrt='g')
        self.declare_partials(of='dy_accel', wrt='roll', rows=ar, cols=ar)
        self.declare_partials(of='dy_accel', wrt='pitch', rows=ar, cols=ar)

        self.declare_partials(of='dz_accel', wrt='mass')
        self.declare_partials(of='dz_accel', wrt='Fz', rows=ar, cols=ar)
        self.declare_partials(of='dz_accel', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='dz_accel', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='dz_accel', wrt='roll_angle_vel', rows=ar, cols=ar)
        self.declare_partials(of='dz_accel', wrt='pitch_angle_vel', rows=ar, cols=ar)
        self.declare_partials(of='dz_accel', wrt='g')
        self.declare_partials(of='dz_accel', wrt='roll', rows=ar, cols=ar)
        self.declare_partials(of='dz_accel', wrt='pitch', rows=ar, cols=ar)

        self.declare_partials(of='roll_accel', wrt='J_xz')
        self.declare_partials(of='roll_accel', wrt='J_xx')
        self.declare_partials(of='roll_accel', wrt='J_yy')
        self.declare_partials(of='roll_accel', wrt='J_zz')
        self.declare_partials(of='roll_accel', wrt='J_yz')
        self.declare_partials(of='roll_accel', wrt='J_xy')
        self.declare_partials(of='roll_accel', wrt='roll_angle_vel', rows=ar, cols=ar)
        self.declare_partials(of='roll_accel', wrt='pitch_angle_vel', rows=ar, cols=ar)
        self.declare_partials(of='roll_accel', wrt='yaw_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='roll_accel', wrt='lx', rows=ar, cols=ar)
        self.declare_partials(of='roll_accel', wrt='lz', rows=ar, cols=ar)

        self.declare_partials(of='pitch_accel', wrt='J_xz')
        self.declare_partials(of='pitch_accel', wrt='J_xx')
        self.declare_partials(of='pitch_accel', wrt='J_yy')
        self.declare_partials(of='pitch_accel', wrt='J_zz')
        self.declare_partials(of='pitch_accel', wrt='J_yz')
        self.declare_partials(of='pitch_accel', wrt='J_xy')
        self.declare_partials(of='pitch_accel', wrt='roll_angle_vel', rows=ar, cols=ar)
        self.declare_partials(of='pitch_accel', wrt='yaw_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='pitch_accel', wrt='ly', rows=ar, cols=ar)

        self.declare_partials(of='yaw_accel', wrt='J_xz')
        self.declare_partials(of='yaw_accel', wrt='J_xx')
        self.declare_partials(of='yaw_accel', wrt='J_yy')
        self.declare_partials(of='yaw_accel', wrt='J_zz')
        self.declare_partials(of='yaw_accel', wrt='J_yz')
        self.declare_partials(of='yaw_accel', wrt='J_xy')
        self.declare_partials(of='yaw_accel', wrt='roll_angle_vel', rows=ar, cols=ar)
        self.declare_partials(of='yaw_accel', wrt='pitch_angle_vel', rows=ar, cols=ar)
        self.declare_partials(of='yaw_accel', wrt='yaw_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='yaw_accel', wrt='lx', rows=ar, cols=ar)
        self.declare_partials(of='yaw_accel', wrt='lz', rows=ar, cols=ar)

        self.declare_partials(of='roll_angle_rate_eq', wrt='roll_angle_vel', rows=ar, cols=ar)
        self.declare_partials(of='roll_angle_rate_eq', wrt='pitch_angle_vel', rows=ar, cols=ar)
        self.declare_partials(of='roll_angle_rate_eq', wrt='yaw_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='roll_angle_rate_eq', wrt='roll', rows=ar, cols=ar)
        self.declare_partials(of='roll_angle_rate_eq', wrt='pitch', rows=ar, cols=ar)

        self.declare_partials(of='pitch_angle_rate_eq', wrt='pitch_angle_vel', rows=ar, cols=ar)
        self.declare_partials(of='pitch_angle_rate_eq', wrt='yaw_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='pitch_angle_rate_eq', wrt='roll', rows=ar, cols=ar)

        self.declare_partials(of='yaw_angle_rate_eq', wrt='pitch_angle_vel', rows=ar, cols=ar)
        self.declare_partials(of='yaw_angle_rate_eq', wrt='yaw_ang_vel', rows=ar, cols=ar)
        self.declare_partials(of='yaw_angle_rate_eq', wrt='roll', rows=ar, cols=ar)
        self.declare_partials(of='yaw_angle_rate_eq', wrt='pitch', rows=ar, cols=ar)

        self.declare_partials(of='dx_dt', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='dx_dt', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='dx_dt', wrt='w', rows=ar, cols=ar)
        self.declare_partials(of='dx_dt', wrt='roll', rows=ar, cols=ar)
        self.declare_partials(of='dx_dt', wrt='pitch', rows=ar, cols=ar)
        self.declare_partials(of='dx_dt', wrt='yaw', rows=ar, cols=ar)

        self.declare_partials(of='dy_dt', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='dy_dt', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='dy_dt', wrt='w', rows=ar, cols=ar)
        self.declare_partials(of='dy_dt', wrt='roll', rows=ar, cols=ar)
        self.declare_partials(of='dy_dt', wrt='pitch', rows=ar, cols=ar)
        self.declare_partials(of='dy_dt', wrt='yaw', rows=ar, cols=ar)

        self.declare_partials(of='dz_dt', wrt='u', rows=ar, cols=ar)
        self.declare_partials(of='dz_dt', wrt='v', rows=ar, cols=ar)
        self.declare_partials(of='dz_dt', wrt='w', rows=ar, cols=ar)
        self.declare_partials(of='dz_dt', wrt='roll', rows=ar, cols=ar)
        self.declare_partials(of='dz_dt', wrt='pitch', rows=ar, cols=ar)
        

    def compute(self, inputs, outputs):
        """
        Compute function for EOl. 
        TODO: Same as above, potentially rewrite equations for \
              matrix form, and add potential assymetry to moment \
              of inertia matrix.

        """

        # inputs

        mass = inputs['mass']
        u = inputs['u'] # u
        v = inputs['v'] # v
        w = inputs['w'] # w
        p = inputs['roll_angle_vel'] # p
        q = inputs['pitch_angle_vel'] # q
        r = inputs['yaw_ang_vel'] # r
        roll = inputs['roll'] # phi
        pitch = inputs['pitch'] # theta
        yaw = inputs['yaw'] # psi
        x = inputs['x'] # p1
        y = inputs['y'] # p2
        z = inputs['z'] # p3
        # time = inputs['time']
        g = inputs['g']
        Fx = inputs['Fx']
        Fy = inputs['Fy']
        Fz = inputs['Fz']
        lx = inputs['lx'] # l
        ly = inputs['ly'] # m
        lz = inputs['lz'] # n
        J_xz = inputs['J_xz']
        J_xx = inputs['J_xx']
        J_yy = inputs['J_yy']
        J_zz = inputs['J_zz']
        J_yz = inputs['J_yz']
        J_xy = inputs['J_xy']

        # Resolve gravity in body coordinate system -- denoted with subscript 'b'
        gx_b = -np.sin(pitch) * g
        gy_b = np.sin(roll) * np.cos(pitch) * g
        gz_b = np.cos(roll) * np.cos(pitch) * g

        # TODO: could add external forces and moments here if needed

        # Denominator for roll and yaw rate equations
        # Den = J_xx * J_zz - J_xz**2

        # roll-axis velocity equation

        dx_accel = 1 / mass * Fx + gx_b - w * q + v * r

        # pitch-axis velocity equation

        dy_accel = 1 / mass * Fy + gy_b - u * r + w * p

        # yaw-axis velocity equation

        dz_accel = 1 / mass * Fz + gz_b - v * p + u * q

        # Roll equation

        roll_accel = (
            (-1 * (J_xz)**(2) * J_yy + (
                2 * J_xy * J_xz * J_yz + ( 
                    -1 * J_xx * (J_yz)**(2) + (
                        -1 * (J_xy)**( 2 ) * J_zz + J_xx * J_yy * J_zz)))))**(-1) * (
                            (-1 * J_xz * J_yy + J_xy * J_yz) * (
                                lz + (J_xx * p * q + (
                                    -1 * J_yy * p * q + (J_xy * (-1 * (p)**(2) + (q)**(2)) + (
                                        -1 * J_yz * p * r + J_xz * q * r ))))) + (
                                            (-1 * (J_yz)**(2) + J_yy * J_zz) * (
                                                lx + (-1 * J_xz * p * q + (-1 * J_yz * (q)**(2) + (
                                                    J_xy * p * r + (J_yy * q * r + (
                                                        -1 * J_zz * q * r + J_yz * (r)**(2))))))) + (
                                                            J_xz * J_yz + -1 * J_xy * J_zz) * (
                                                                ly + (J_yz * p * q + (-1 * J_xx * p * r + (
                                                                    J_zz * p * r + (
                                                                        -1 * J_xy * q * r + J_xz * ((p)**(2) + -1 * (r)**(2)))))))))
        
        # Pitch equation

        pitch_accel = (
            (-1 * (J_xz)**(2) * J_yy + (
                2 * J_xy * J_xz * J_yz + (
                    -1 * J_xx * (J_yz)**(2) + (
                        -1 * (J_xy)**(2) * J_zz + J_xx * J_yy * J_zz)))))**(-1) * (
                            (J_xy * J_xz + -1 * J_xx * J_yz) * (
                                lz + (J_xx * p * q + (
                                    -1 * J_yy * p * q + (
                                        J_xy * (-1 * (p)**(2) + (q)**(2)) + (
                                            -1 * J_yz * p * r + J_xz * q * r))))) + ((
                                                J_xz * J_yz + -1 * J_xy * J_zz) * (lx + (
                                                    -1 * J_xz * p * q + (
                                                        -1 * J_yz * (q)**(2) + (
                                                            J_xy * p * r + (J_yy * q * r + (
                                                                -1 * J_zz * q * r + J_yz * (r)**(2))))))) + -1 * (
                                                                    (J_xz)**(2) + -1 * J_xx * J_zz) * (
                                                                        ly + (J_yz * p * q + (
                                                                            -1 * J_xx * p * r + (
                                                                                J_zz * p * r + (
                                                                                    -1 * J_xy * q * r + J_xz * (
                                                                                        (p)**(2) + -1 * (r)**(2)))))))))
        
        # Yaw equation

        yaw_accel = ((
            -1 * (J_xz)**(2) * J_yy + (
                2 * J_xy * J_xz * J_yz + (
                    -1 * J_xx * (J_yz)**(2) + (
                        -1 * (J_xy)**(2) * J_zz + J_xx * J_yy * J_zz )))))**(-1) * (
                            (-1 * (J_xy)**(2) + J_xx * J_yy) * (
                                lz + (J_xx * p * q + (
                                    -1 * J_yy * p * q + (
                                        J_xy * (-1 * (p)**(2) + (q)**(2)) + (
                                            -1 * J_yz * p * r + J_xz * q * r))))) + (
                                                (-1 * J_xz * J_yy + J_xy * J_yz) * (
                                                    lx + (-1 * J_xz * p * q + (
                                                        -1 * J_yz * (q)**(2) + (
                                                            J_xy * p * r + (J_yy * q * r + (
                                                                -1 * J_zz * q * r + J_yz * (r)**(2))))))) + (
                                                                    J_xy * J_xz + -1 * J_xx * J_yz) * (
                                                                        ly + (J_yz * p * q + (
                                                                            -1 * J_xx * p * r + (
                                                                                J_zz * p * r + (
                                                                                    -1 * J_xy * q * r + J_xz * (
                                                                                        (p)**(2) + -1 * (r)**(2)))))))))

        # Kinematic equations
        
        roll_angle_rate_eq = p + np.sin(roll) * np.tan(pitch) * q + \
                             np.cos(roll) * np.tan(pitch) * r
        
        pitch_angle_rate_eq = np.cos(roll) * q - np.sin(roll) * r

        yaw_angle_rate_eq = np.sin(roll) / np.cos(pitch) * q + \
                            np.cos(roll) / np.cos(pitch) * r

        # Position equations

        dx_dt = np.cos(pitch) * np.cos(yaw) * u + \
                (-np.cos(roll) * np.sin(yaw) + np.sin(roll) * np.sin(pitch) * np.cos(yaw)) * v + \
                (np.sin(roll) * np.sin(yaw) + np.cos(roll) * np.sin(pitch) * np.cos(yaw)) * w
        
        dy_dt = np.cos(pitch) * np.sin(yaw) * u + \
                (np.cos(roll) * np.cos(yaw) + np.sin(roll) * np.sin(pitch) * np.sin(yaw)) * v + \
                (-np.sin(roll) * np.cos(yaw) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) * w
        
        dz_dt = -np.sin(pitch) * u + \
                np.sin(roll) * np.cos(pitch) * v + \
                np.cos(roll) * np.cos(pitch) * w

        outputs['dx_accel'] = dx_accel
        outputs['dy_accel'] = dy_accel
        outputs['dz_accel'] = dz_accel
        outputs['roll_accel'] = roll_accel
        outputs['pitch_accel'] = pitch_accel
        outputs['yaw_accel'] = yaw_accel
        outputs['roll_angle_rate_eq'] = roll_angle_rate_eq
        outputs['pitch_angle_rate_eq'] = pitch_angle_rate_eq
        outputs['yaw_angle_rate_eq'] = yaw_angle_rate_eq
        outputs['dx_dt'] = dx_dt
        outputs['dy_dt'] = dy_dt
        outputs['dz_dt'] = dz_dt
        
    
    def compute_partials(self, inputs, J):
        mass = inputs['mass']
        u = inputs['u'] # u
        v = inputs['v'] # v
        w = inputs['w'] # w
        p = inputs['roll_angle_vel'] # p
        q = inputs['pitch_angle_vel'] # q
        r = inputs['yaw_ang_vel'] # r
        roll = inputs['roll'] # phi
        pitch = inputs['pitch'] # theta
        yaw = inputs['yaw'] # psi
        x = inputs['x'] # p1
        y = inputs['y'] # p2
        z = inputs['z'] # p3
        # time = inputs['time']
        g = inputs['g']
        Fx = inputs['Fx']
        Fy = inputs['Fy']
        Fz = inputs['Fz']
        lx = inputs['lx'] # l
        ly = inputs['ly'] # m
        lz = inputs['lz'] # n
        J_xz = inputs['J_xz']
        J_xx = inputs['J_xx']
        J_yy = inputs['J_yy']
        J_zz = inputs['J_zz']
        J_xy = inputs['J_xy']
        J_yz = inputs['J_yz']

        # for roll and yaw
        # Den = J_xx * J_zz - J_xz**2

        J['dx_accel', 'mass'] = -Fx / mass**2
        J['dx_accel', 'Fx'] = 1 / mass
        J['dx_accel', 'v'] = r
        J['dx_accel', 'w'] = -q
        J['dx_accel', 'yaw_ang_vel'] = v
        J['dx_accel', 'pitch_angle_vel'] = -w
        J['dx_accel', 'g'] = -np.sin(pitch)
        J['dx_accel', 'pitch'] = -np.cos(pitch) * g

        J['dy_accel', 'mass'] = -Fy / mass**2
        J['dy_accel', 'Fy'] = 1 / mass
        J['dy_accel', 'u'] = -r
        J['dy_accel', 'w'] = p
        J['dy_accel', 'yaw_ang_vel'] = -u
        J['dy_accel', 'roll_angle_vel'] = w
        J['dy_accel', 'g'] = np.sin(roll) * np.cos(pitch)
        J['dy_accel', 'roll'] = np.cos(roll) * np.cos(pitch) * g
        J['dy_accel', 'pitch'] = -np.sin(roll) * np.sin(pitch) * g

        J['dz_accel', 'mass'] = -Fz / mass**2
        J['dz_accel', 'Fz'] = 1 / mass
        J['dz_accel', 'v'] = -p
        J['dz_accel', 'u'] = q
        J['dz_accel', 'roll_angle_vel'] = -v
        J['dz_accel', 'pitch_angle_vel'] = u
        J['dz_accel', 'g'] = np.cos(roll) * np.cos(pitch)
        J['dz_accel', 'roll'] = -np.sin(roll) * np.cos(pitch) * g
        J['dz_accel', 'pitch'] = -np.cos(roll) * np.sin(pitch) * g

        J['roll_accel', 'J_xz'] = (
            (
                (
                    -1 * (Ixz)**(2) * Iyy + (
                        2 * Ixy * Ixz * Iyz + (
                            -1 * Ixx * (Iyz)**(2) + (
                                -1 * (Ixy)**(2) * Izz + Ixx * Iyy * Izz)))))**(-1) * (
                                    -1 * (-1 * (Iyz)**(2) + Iyy * Izz) * p * q + (
                                        (
                                            -1 * Ixz * Iyy + Ixy * Iyz) * q * r + (
                                                -1 * Iyy * (Mz + (
                                                    Ixx * p * q + (
                                                        -1 * Iyy * p * q + (Ixy * (
                                                            -1 * (p)**(2) + (q)**(2)) + (
                                                                -1 * Iyz * p * r + Ixz * q * r))))) + ((
                                                                    Ixz * Iyz + -1 * Ixy * Izz) * (
                                                                        (p)**(2) + -1 * (r)**(2)) + Iyz * (
                                                                            My + (Iyz * p * q + (
                                                                                -1 * Ixx * p * r + (
                                                                                    Izz * p * r + (
                                                                                        -1 * Ixy * q * r + Ixz * (
                                                                                            (p)**(2) + -1 * (r)**(2))))))))))) + -1 * (
                                                                                                -2 * Ixz * Iyy + 2 * Ixy * Iyz) * (
                                                                                                    (-1 * (Ixz)**(2) * Iyy + (
                                                                                                        2 * Ixy * Ixz * Iyz + (
                                                                                                            -1 * Ixx * (Iyz)**(2) + (
                                                                                                                -1 * (Ixy)**(2) * Izz + Ixx * Iyy * Izz)))))**(-2) * ((
                                                                                                                    -1 * Ixz * Iyy + Ixy * Iyz) * (Mz + (
                                                                                                                        Ixx * p * q + (
                                                                                                                            -1 * Iyy * p * q + (Ixy * (
                                                                                                                                -1 * (p)**(2) + (q)**(2)) + (
                                                                                                                                    -1 * Iyz * p * r + Ixz * q * r))))) + ((
                                                                                                                                        -1 * (Iyz)**(2) + Iyy * Izz) * (
                                                                                                                                            Mx + (-1 * Ixz * p * q + (
                                                                                                                                                -1 * Iyz * (q)**(2) + (
                                                                                                                                                    Ixy * p * r + (Iyy * q * r + (
                                                                                                                                                        -1 * Izz * q * r + Iyz * (r)**(2))))))) + (
                                                                                                                                                            Ixz * Iyz + -1 * Ixy * Izz) * (My + (
                                                                                                                                                                Iyz * p * q + (-1 * Ixx * p * r + (
                                                                                                                                                                    Izz * p * r + (-1 * Ixy * q * r + Ixz * (
                                                                                                                                                                        (p)**(2) + -1 * (r)**( 2 ))))))))))
        J['roll_accel', 'J_xx'] = (Den * (
            J_xz * p * q
        ) - (J_xz * (J_xx - J_yy + J_zz) * p * q - 
                   (J_zz * (J_zz - J_yy) + J_xz**2) * q * r + 
                   J_zz * lx + 
                   J_xz * lz) * J_zz) / Den**2
        J['roll_accel', 'J_yy'] = (-J_xz * p * q + J_zz * q * r) / Den
        J['roll_accel', 'J_zz'] = (Den * (
            J_xz * p * q - 2 * J_zz * q * r + J_yy * q * r + lx
        ) - (J_xz * (J_xx - J_yy + J_zz) * p * q - 
                   (J_zz * (J_zz - J_yy) + J_xz**2) * q * r + 
                   J_zz * lx + 
                   J_xz * lz) * J_xx) / Den**2
        J['roll_accel', 'roll_angle_vel'] = (J_xz * (J_xx - J_yy + J_zz) * q) / Den
        J['roll_accel', 'pitch_angle_vel'] = (J_xz * (J_xx - J_yy + J_zz) * p - (J_zz * (J_zz - J_yy) + J_xz**2) * r) / Den
        J['roll_accel', 'yaw_ang_vel'] = -((J_zz * (J_zz - J_yy) + J_xz**2) * q) / Den
        J['roll_accel', 'lx'] = J_zz / Den
        J['roll_accel', 'lz'] = J_xz / Den

        J['pitch_accel', 'J_xz'] = -(p**2 - r**2) / J_yy
        J['pitch_accel', 'J_xx'] = -(p * r) / J_yy
        J['pitch_accel', 'J_yy'] = -((J_zz - J_xx) * p * r - 
                    J_xz * (p**2 - r**2) + ly) / J_yy**2
        J['pitch_accel', 'J_zz'] = p * r / J_yy
        J['pitch_accel', 'roll_angle_vel'] = ((J_zz - J_xx) * r - 2 * J_xz * p) / J_yy
        J['pitch_accel', 'yaw_ang_vel'] = ((J_zz - J_xx) * p + 2 * J_xz * r) / J_yy
        J['pitch_accel', 'ly'] = 1 / J_yy

        J['yaw_accel', 'J_xz'] = (Den * (
            2 * J_xz * p * q + (J_xx - J_yy + J_zz) * q * r + lx + lz
        ) - ((J_xx * (J_xx - J_yy) + J_xz**2) * p * q + 
                  J_xz * (J_xx - J_yy + J_zz) * q * r + 
                  J_xz * lx + 
                  J_xz * lz) * -2 * J_xz) / Den**2
        J['yaw_accel', 'J_xx'] = (Den * (
            2 * J_xx * p * q - J_yy * p * q + J_xz * q * r
        ) - ((J_xx * (J_xx - J_yy) + J_xz**2) * p * q + 
                  J_xz * (J_xx - J_yy + J_zz) * q * r + 
                  J_xz * lx + 
                  J_xz * lz) * J_zz) / Den**2
        J['yaw_accel', 'J_yy'] = (-J_xx * p * q - J_xz * q * r) / Den
        J['yaw_accel', 'J_zz'] = (Den * (
            J_xz * q * r
        ) - ((J_xx * (J_xx - J_yy) + J_xz**2) * p * q + 
                  J_xz * (J_xx - J_yy + J_zz) * q * r + 
                  J_xz * lx + 
                  J_xz * lz) * J_xx) / Den**2
        J['yaw_accel', 'roll_angle_vel'] = ((J_xx * (J_xx - J_yy) + J_xz**2) * q) / Den
        J['yaw_accel', 'pitch_angle_vel'] = ((J_xx * (J_xx - J_yy) + J_xz**2) * p + J_xz * (J_xx - J_yy + J_zz) * r) / Den
        J['yaw_accel', 'yaw_ang_vel'] = (J_xz * (J_xx - J_yy + J_zz) * q) / Den
        J['yaw_accel', 'lx'] = J_xz / Den
        J['yaw_accel', 'lz'] = J_xz / Den

        J['roll_angle_rate_eq', 'roll_angle_vel'] = 1 
        J['roll_angle_rate_eq', 'pitch_angle_vel'] = np.sin(roll) * np.tan(pitch)
        J['roll_angle_rate_eq', 'yaw_ang_vel'] = np.cos(roll) * np.tan(pitch)
        J['roll_angle_rate_eq', 'roll'] = np.cos(roll) * np.tan(pitch) * q - np.sin(roll) * np.tan(pitch) * r
        J['roll_angle_rate_eq', 'pitch'] = np.sin(roll) * (1 / np.cos(pitch)**2) * q + np.cos(roll) * (1 / np.cos(pitch)**2) * r
        
        J['pitch_angle_rate_eq', 'pitch_angle_vel'] = np.cos(roll)
        J['pitch_angle_rate_eq', 'yaw_ang_vel'] = -np.sin(roll)
        J['pitch_angle_rate_eq', 'roll'] = -np.sin(roll) * q - np.cos(roll) * r

        J['yaw_angle_rate_eq', 'pitch_angle_vel'] = np.sin(roll) / np.cos(pitch)
        J['yaw_angle_rate_eq', 'yaw_ang_vel'] = np.cos(roll) / np.cos(pitch)
        J['yaw_angle_rate_eq', 'roll'] = np.cos(roll) / np.cos(pitch) * q - np.sin(roll) / np.cos(pitch) * r
        J['yaw_angle_rate_eq', 'pitch'] = np.sin(roll) * (np.tan(pitch) / np.cos(pitch)) * q + np.cos(roll) * (np.tan(pitch) / np.cos(pitch)) * r

        # note: d/dx tan(x) = sec^2(x) = 1 / cos^2(x)
        # note: d/dx 1 / cos(x) = d/dx sec(x) = sec(x)tan(x) = tan(x) / cos(x)

        J['dx_dt', 'u'] = np.cos(pitch) * np.cos(yaw)
        J['dx_dt', 'v'] = -np.cos(roll) * np.sin(yaw) + np.sin(roll) * np.sin(pitch) * np.cos(yaw)
        J['dx_dt', 'w'] = np.sin(roll) * np.sin(yaw) + np.cos(roll) * np.sin(pitch) * np.cos(yaw)
        J['dx_dt', 'roll'] = (np.sin(roll) * np.sin(yaw) + np.cos(roll) * np.sin(pitch) * np.cos(yaw)) * v + \
                             (np.cos(roll) * np.sin(yaw) - np.sin(roll) * np.sin(pitch) * np.cos(yaw)) * w
        J['dx_dt', 'pitch'] = -np.sin(pitch) * np.cos(yaw) * u + \
                              np.sin(roll) * np.cos(pitch) * np.cos(yaw) * v + \
                              np.cos(roll) * np.cos(pitch) * np.cos(yaw) * w
        J['dx_dt', 'yaw'] = -np.cos(pitch) * np.sin(yaw) * u + \
                            (-np.cos(roll) * np.cos(yaw) - np.sin(roll) * np.sin(pitch) * np.sin(yaw)) * v + \
                            (np.sin(roll) * np.cos(yaw) - np.cos(roll) * np.sin(pitch) * np.sin(yaw)) * w
        
        J['dy_dt', 'u'] = np.cos(pitch) * np.sin(yaw)
        J['dy_dt', 'v'] = np.cos(roll) * np.cos(yaw) + np.sin(roll) * np.sin(pitch) * np.sin(yaw)
        J['dy_dt', 'w'] = -np.sin(roll) * np.cos(yaw) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)
        J['dy_dt', 'roll'] = (-np.sin(roll) * np.cos(yaw) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) * v + \
                             (-np.cos(roll) * np.cos(yaw) - np.sin(roll) * np.sin(pitch) * np.sin(yaw)) * w
        J['dy_dt', 'pitch'] = -np.sin(pitch) * np.sin(yaw) * u + \
                              np.sin(roll) * np.cos(pitch) * np.sin(yaw) * v + \
                              np.cos(roll) * np.cos(pitch) * np.sin(yaw) * w
        J['dy_dt', 'yaw'] = np.cos(pitch) * np.cos(yaw) * u + \
                            (-np.cos(roll) * np.sin(yaw) + np.sin(roll) * np.sin(pitch) * np.cos(yaw)) * v + \
                            (np.sin(roll) * np.sin(yaw) + np.cos(roll) * np.sin(pitch) * np.cos(yaw)) * w
        
        J['dz_dt', 'u'] = -np.sin(pitch)
        J['dz_dt', 'v'] = np.sin(roll) * np.cos(pitch)
        J['dz_dt', 'w'] = np.cos(roll) * np.cos(pitch)
        J['dz_dt', 'roll'] = np.cos(roll) * np.cos(pitch) * v - \
                             np.sin(roll) * np.cos(pitch) * w
        J['dz_dt', 'pitch'] = -np.cos(pitch) * u - \
                              np.sin(roll) * np.sin(pitch) * v - \
                              np.cos(roll) * np.sin(pitch) * w
                             




if __name__ == "__main__":

    p = om.Problem()
    p.model = om.Group()
    des_vars = p.model.add_subsystem('des_vars', om.J_ndepVarComp(), promotes=['*'])

    des_vars.add_output('mass', 3.0, units='kg')
    des_vars.add_output('u', 0.1, units='m/s')
    des_vars.add_output('v', 0.7, units='m/s')
    des_vars.add_output('w', 0.12, units='m/s')
    des_vars.add_output('p', 0.1, units='rad/s')
    des_vars.add_output('q', 0.9, units='rad/s')
    des_vars.add_output('r', 0.12, units='rad/s')
    des_vars.add_output('roll', 0.9, units='rad')
    des_vars.add_output('pitch', 0.19, units='rad')
    des_vars.add_output('yaw', 0.70, units='rad')
    des_vars.add_output('g', 9.81, units='m/s**2')
    des_vars.add_output('Fx', 0.1, units='N')
    des_vars.add_output('Fy', 0.9, units='N')
    des_vars.add_output('Fz', 0.12, units='N')
    des_vars.add_output('lx', 3.0, units='N*m')
    des_vars.add_output('ly', 4.0, units='N*m')
    des_vars.add_output('lz', 5.0, units='N*m')
    des_vars.add_output('J_xz', 9.0, units='kg*m**2')
    des_vars.add_output('J_xx', 50.0, units='kg*m**2')
    des_vars.add_output('J_yy', 51.0, units='kg*m**2')
    des_vars.add_output('J_zz', 52.0, units='kg*m**2')

    p.model.add_subsystem('SixDOF_EOl', SixDOF_EOl(num_nodes=1), promotes=['*'])

    p.setup(check=False, force_alloc_complex=True)

    p.run_model()

    dx_accel = p.get_val('dx_accel')
    dy_accel = p.get_val('dy_accel')
    dz_accel = p.get_val('dz_accel')
    roll_accel = p.get_val('roll_accel')
    pitch_accel = p.get_val('pitch_accel')
    yaw_accel = p.get_val('yaw_accel')
    roll_angle_rate_eq = p.get_val('roll_angle_rate_eq')
    pitch_angle_rate_eq = p.get_val('pitch_angle_rate_eq')
    yaw_angle_rate_eq = p.get_val('yaw_angle_rate_eq')
    dx_dt = p.get_val('dx_dt')
    dy_dt = p.get_val('dy_dt')
    dz_dt = p.get_val('dz_dt')

    print(f"Accelerations in x,y,z: {dx_accel}, {dy_accel}, {dz_accel}")
    print(f"Euler angle accels in roll, pitch, yaw: {roll_accel}, {pitch_accel}, {yaw_accel}")
    print(f"Euler angular rates: {roll_angle_rate_eq}, {pitch_angle_rate_eq}, {yaw_angle_rate_eq}")
    print(f"velocities: {dx_dt}, {dy_dt}, {dz_dt}")

    p.check_partials(compact_print=True, show_only_incorrect=True, method='cs')



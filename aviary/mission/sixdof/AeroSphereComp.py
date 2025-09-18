import numpy as np
import openmdao.api as om 
from openmdao.components.interp_util.interp import InterpND
from dymos.models.atmosphere.atmos_1976 import USatm1976Data

class AeroSphereComp(om.ExplicitComponent):
    """
    OpenMDAO component to compute scalar drag/lift/side for a spherical UAV. 
    Outputs 'drag', 'lift', and 'side' of shape (nn,)

    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # Inputs
        self.add_input('u', 
                       val=np.zeros(nn), 
                       units='m/s', 
                       desc="body x velocity")
        
        self.add_input('v',
                       val=np.zeros(nn),
                       units='m/s',
                       desc="body y velocity")
        
        self.add_input('w',
                       val=np.zeros(nn),
                       units='m/s',
                       desc="body z velocity")
        
        self.add_input('radius',
                       val=np.ones(nn) * 0.12,
                       units='m',
                       desc="sphere radius")
        
        self.add_input('Cd',
                       val=np.ones(nn) * 0.5,
                       desc="drag coefficient (sphere ~ 0.5)")
        
        self.add_input('h',
                       val=np.zeros(nn),
                       units='m',
                       desc="height/altitude used for rho calculation")
        
        # Outputs

        self.add_output('drag', 
                        val=np.zeros(nn),
                        units='N',
                        desc="drag magnitude (always positive)")
        
        self.add_output('lift',
                        val=np.zeros(nn),
                        units='N',
                        desc="lift magnitude (sphere = 0)")
        
        self.add_output('side',
                        val=np.zeros(nn),
                        units='N',
                        desc="side magnitude (sphere = 0)")
        
        self.declare_coloring(wrt='*', method='cs')

        alt_data = USatm1976Data.alt * om.unit_conversion('ft', 'm')[0]
        rho_data = USatm1976Data.rho * om.unit_conversion('slug/ft**3', 'kg/m**3')[0]
        self.rho_interp = InterpND(points=np.array(alt_data),
                                   values=np.array(rho_data),
                                   method='slinear').interpolate

    def compute(self, inputs, outputs):
        u = inputs['u']
        v = inputs['v']
        w = inputs['w']
        R = inputs['radius']
        Cd = inputs['Cd']
        h = inputs['h']

        # V_rel
        V = np.sqrt(u**2 + v**2 + w**2)

        # rho -- handling complex step
        if np.iscomplexobj(h):
            rho = self.rho_interp(inputs['h'])
        else:
            rho = self.rho_interp(inputs['h']).real

        # Divide by zero check
        if V == 0:
            V = 1e-8
        
        A = np.pi * R**2 

        outputs['drag'] = 0.5 * rho * Cd * A * V**2
        outputs['lift'] = np.zeros_like(outputs['drag'])
        outputs['side'] = np.zeros_like(outputs['drag'])

    def compute_partials(self, inputs, J):
        u = inputs['u']
        v = inputs['v']
        w = inputs['w']
        rho = inputs['rho']
        R = inputs['radius']
        Cd = inputs['Cd']
        nn = self.options['num_nodes']
        
        V = np.sqrt(u**2 + v**2 + w**2)

        if V == 0:
            V = 1e-8
        
        A = np.pi * R**2
        
        J['drag', 'u'] = rho * Cd * A * u
        J['drag', 'v'] = rho * Cd * A * v
        J['drag', 'w'] = rho * Cd * A * w
        J['drag', 'rho'] = 0.5 * Cd * A * V**2
        J['drag', 'Cd'] = 0.5 * rho * A * V**2
        J['drag', 'radius'] = rho * Cd * np.pi * R * V**2

        J['lift', 'u'] = np.zeros(nn)
        J['lift', 'v'] = np.zeros(nn)
        J['lift', 'w'] = np.zeros(nn)

        J['side', 'u'] = np.zeros(nn)
        J['side', 'v'] = np.zeros(nn)
        J['side', 'w'] = np.zeros(nn)

if __name__ == "__main__":
    p = om.Problem()
    p.model = om.Group()
    des_vars = p.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=['*'])

    des_vars.add_output('u', 1, units='m/s')
    des_vars.add_output('v', 5, units='m/s')
    des_vars.add_output('w', 7, units='m/s')
    des_vars.add_output('rho', 1.225, units='kg/m**3')
    des_vars.add_output('radius', 5, units='m')
    des_vars.add_output('Cd', 0.5)

    p.model.add_subsystem('AeroSphereComp', AeroSphereComp(num_nodes=1), promotes=['*'])

    p.setup(check=False, force_alloc_complex=True)

    p.run_model()

    p.check_partials(compact_print=True, show_only_incorrect=False, method='cs')
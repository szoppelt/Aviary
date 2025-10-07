import numpy as np
import openmdao.api as om

# Use the simplified force resolver from above
class SimplifiedForceResolver(om.ExplicitComponent):
    """Ultra-simplified for debugging"""
    
    def initialize(self):
        self.options.declare('num_nodes', types=int)
    
    def setup(self):
        nn = self.options['num_nodes']
        
        self.add_input('u', val=np.zeros(nn), units='m/s')
        self.add_input('v', val=np.zeros(nn), units='m/s')
        self.add_input('w', val=np.zeros(nn), units='m/s')
        self.add_input('drag', val=np.zeros(nn), units='N')
        self.add_input('thrust', val=np.zeros(nn), units='N')
        
        self.add_output('Fx', val=np.zeros(nn), units='N')
        self.add_output('Fy', val=np.zeros(nn), units='N')
        self.add_output('Fz', val=np.zeros(nn), units='N')
        
        ar = np.arange(nn)
        self.declare_partials(of='Fx', wrt=['u', 'v', 'w', 'drag'], rows=ar, cols=ar)
        self.declare_partials(of='Fy', wrt=['u', 'v', 'w', 'drag'], rows=ar, cols=ar)
        self.declare_partials(of='Fz', wrt=['u', 'v', 'w', 'drag', 'thrust'], rows=ar, cols=ar)
    
    def compute(self, inputs, outputs):
        u = inputs['u']
        v = inputs['v']
        w = inputs['w']
        D = inputs['drag']
        T = inputs['thrust']
        
        V = np.sqrt(u**2 + v**2 + w**2) + 1e-8
        
        outputs['Fx'] = -D * u / V
        outputs['Fy'] = -D * v / V
        outputs['Fz'] = -D * w / V + T
    
    def compute_partials(self, inputs, J):
        u = inputs['u']
        v = inputs['v']
        w = inputs['w']
        D = inputs['drag']
        
        V = np.sqrt(u**2 + v**2 + w**2) + 1e-8
        V3 = V**3
        
        J['Fx', 'u'] = -D/V + D*u**2/V3
        J['Fx', 'v'] = D*u*v/V3
        J['Fx', 'w'] = D*u*w/V3
        J['Fx', 'drag'] = -u/V
        
        J['Fy', 'u'] = D*v*u/V3
        J['Fy', 'v'] = -D/V + D*v**2/V3
        J['Fy', 'w'] = D*v*w/V3
        J['Fy', 'drag'] = -v/V
        
        J['Fz', 'u'] = D*w*u/V3
        J['Fz', 'v'] = D*w*v/V3
        J['Fz', 'w'] = -D/V + D*w**2/V3
        J['Fz', 'drag'] = -w/V
        J['Fz', 'thrust'] = 1.0

if __name__ == "__main__":
    p = om.Problem()
    p.model = om.Group()
    des_vars = p.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=['*'])

    des_vars.add_output('u', 0.5, units='m/s')
    des_vars.add_output('v', 0.6, units='m/s')
    des_vars.add_output('w', 0.7, units='m/s')
    des_vars.add_output('drag', 50, units='N')
    des_vars.add_output('thrust', 50, units='N')

    p.model.add_subsystem('ForceComponentResolver', SimplifiedForceResolver(num_nodes=1), promotes=['*'])

    p.setup(check=False, force_alloc_complex=True)

    p.run_model()

    p.check_partials(compact_print=True, show_only_incorrect=False, method='cs')
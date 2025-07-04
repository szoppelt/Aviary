import openmdao.api as om
import openmdao.jax as omj
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline
import os 

from aviary.variable_info.variables import Aircraft
from aviary.variable_info.functions import add_aviary_output, add_aviary_input

try:
    from quadax import quadgk
except ImportError:
    raise ImportError(
        "quadax package not found. You can install it by running 'pip install quadax'."
    )

from aviary.subsystems.mass.simple_mass.materials_database import materials

from aviary.utils.named_values import get_keys

Debug = True # set to enable printing

class TailMassAndCOG(om.JaxExplicitComponent):
    def initialize(self):
        #self.options['default_shape'] = () # Sets the default shape to scalar

        self.options.declare('tail_type',
                            values=['horizontal', 'vertical'],
                            desc="Type of tail: 'horizontal' or 'vertical'")
        
        self.options.declare('airfoil_type', 
                             default='NACA', 
                             values=['NACA', 'file'],
                             desc="Airfoil type: 'NACA' for 4-digit or 'file' for user-provided coordinates")
        
        if self.options['airfoil_type'] == 'NACA':
            self.options.declare('NACA_digits',
                                 default='2412',
                                 desc="4 digit code for NACA airfoil, if that is given.")
        
        self.options.declare('material', 
                             default='Balsa', 
                             values=list(get_keys(materials)),
                             desc="Material type")
        
        self.options.declare('airfoil_file', 
                             default=None, 
                             desc="File path for airfoil coordinates (if applicable)")

        self.options.declare('num_sections', 
                             default=10, 
                             desc="Number of sections for enumeration")
    
    def setup(self):
        self.options['use_jit'] = not(Debug)

        # Inputs
        add_aviary_input(self,
                            Aircraft.HorizontalTail.SPAN,
                            units='m', 
                            desc="Tail span")
    
        add_aviary_input(self,
                            Aircraft.HorizontalTail.ROOT_CHORD,
                            units='m', 
                            desc="Root chord length")
        #else:
        add_aviary_input(self,
                            Aircraft.VerticalTail.SPAN,
                            units='m', 
                            desc="Tail span")
    
        add_aviary_input(self,
                            Aircraft.VerticalTail.ROOT_CHORD,
                            units='m', 
                            desc="Root chord length")
            
        # The inputs below have no aviary input, so there is no distinction for now

        self.add_input('tip_chord_tail', 
                       val=0.8, 
                       units='m', 
                       desc="Tip chord length")
        
        self.add_input('thickness_ratio', 
                       val=0.12, 
                       desc="Max thickness to chord ratio for NACA airfoil")
        
        self.add_input('skin_thickness', 
                       val=0.002, 
                       units='m', 
                       desc="Skin panel thickness")
        
        self.add_input('twist_tail', 
                       val=jnp.zeros(self.options['num_sections']), 
                       units='deg', 
                       desc="Twist distribution")
        
        # Outputs
        add_aviary_output(self,
                            Aircraft.HorizontalTail.MASS,
                            units='kg', 
                            desc="Total mass of the tail")
        #else:
        add_aviary_output(self,
                            Aircraft.VerticalTail.MASS,
                            units='kg', 
                            desc="Total mass of the tail")

    def compute_primal(self, 
                       aircraft__horizontal_tail__span, 
                       aircraft__horizontal_tail__root_chord,
                       aircraft__vertical_tail__span, 
                       aircraft__vertical_tail__root_chord, 
                       tip_chord_tail, 
                       thickness_ratio, 
                       skin_thickness, 
                       twist_tail):
        tail_type = self.options["tail_type"]
        airfoil_type = self.options["airfoil_type"]
        material = self.options['material']
        density, _ = materials.get_item(material)
        airfoil_file = self.options['airfoil_file']
        num_sections = self.options['num_sections']
        NACA_digits = self.options['NACA_digits']

        # This is just so that the differentiation and unittest do not break. If tail_type = horizontal, ignore the vertical tail mass and vice versa.
        # TODO: Potentially write these tails as separate files. 
        aircraft__horizontal_tail__mass = 0.0 * thickness_ratio
        aircraft__vertical_tail__mass = 0.0 * thickness_ratio

        # File check
        if airfoil_type == 'file':
            if airfoil_type == 'file' and (airfoil_file is None or not os.path.isfile(airfoil_file)):
                raise FileNotFoundError(f"Airfoil file '{airfoil_file}' not found or not provided.")
            try: 
                airfoil_data = np.loadtxt(airfoil_file, skiprows=1) # Assume a header
                x_coords, y_coords = airfoil_data[:, 0], airfoil_data[:, 1]
            except Exception as e:
                raise ValueError(f"Error reading airfoil file: {e}")
        
        # Compute section airfoil geometry
        if airfoil_file and os.path.exists(airfoil_file):
            airfoil_data = np.loadtxt(airfoil_file)
            x_coords = airfoil_data[:, 0]
            y_coords = airfoil_data[:, 1]

            camber, camber_location, max_thickness = self.extract_airfoil_features(x_coords, y_coords)
        else:
            # Parse the NACA airfoil type (4-digit)
            camber = int(NACA_digits[0]) / 100.0 # Maximum camber
            camber_location = int(NACA_digits[1]) / 10.0 # Location of max camber
            max_thickness = int(NACA_digits[2:4]) / 100.0 # Max thickness
        
        # Tail type check
        if tail_type not in ['horizontal', 'vertical']:
            raise ValueError("Invalid tail_type. Must be 'horizontal' or 'vertical'.")

        if tail_type == 'horizontal':
            span_locations = jnp.linspace(0, aircraft__horizontal_tail__span, num_sections)
        elif tail_type == 'vertical':
            span_locations = jnp.linspace(0, aircraft__vertical_tail__span, num_sections)

        # Get x_points and dx for later
        x_points, dx = self.precompute_airfoil_geometry()

        # Thickness distribution
        thickness_dist = self.airfoil_thickness(x_points, max_thickness)

        if tail_type == 'horizontal':
            total_mass, _ = quadgk(
                lambda x: density * self.airfoil_thickness(x, max_thickness) * (
                    aircraft__horizontal_tail__root_chord - (aircraft__horizontal_tail__root_chord - tip_chord_tail) * (x / aircraft__horizontal_tail__span)
                    ) * aircraft__horizontal_tail__span, [0, 1], epsabs=1e-9, epsrel=1e-9)
            
            aircraft__horizontal_tail__mass = total_mass

        elif tail_type == 'vertical':
            total_mass, _ = quadgk(
                lambda x: density * self.airfoil_thickness(x, max_thickness) * (
                    aircraft__vertical_tail__root_chord - (aircraft__vertical_tail__root_chord - tip_chord_tail) * (x / aircraft__vertical_tail__span)
                    ) * aircraft__vertical_tail__span, [0, 1], epsabs=1e-9, epsrel=1e-9)

            aircraft__vertical_tail__mass = total_mass
        
        return aircraft__horizontal_tail__mass, aircraft__vertical_tail__mass
    
    def precompute_airfoil_geometry(self):
        num_sections = self.options['num_sections']
        n_points = num_sections
        x_points = jnp.linspace(0, 1, n_points)
        dx = 1 / (n_points - 1)
        return x_points, dx
    
    def compute_airfoil_geometry(self, chord, camber, camber_location, thickness_dist, x_points, dx):

        section_area = jnp.trapezoid(thickness_dist, x_points, dx=dx) 
        section_area *= chord

        centroid_x = jnp.trapezoid(x_points * thickness_dist, x_points, dx=dx)
        centroid_x = (centroid_x * chord) / section_area

        centroid_z = jnp.trapezoid(self.airfoil_camber_line(x_points, camber, camber_location) * thickness_dist, x_points, dx=dx)
        centroid_z = (centroid_z * chord) / section_area
        return section_area, centroid_x, centroid_z
    
    def airfoil_thickness(self, x, max_thickness):
        return 5 * max_thickness * (0.2969 * jnp.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    def airfoil_camber_line(self, x, camber, camber_location):
        camber_location = omj.ks_max(camber_location, 1e-9) # Divide by zero check
        return jnp.where(
            x < camber_location, 
            (camber / camber_location**2) * (2 * camber_location * x - x**2), 
        (camber / (1 - camber_location)**2) * ((1 - 2 * camber_location) + 2 * camber_location * x - x**2)
        )

    def extract_airfoil_features(self, x_coords, y_coords):
        """
        Extract camber, camber location, and max thickness from the given airfoil data.
        This method assumes x_coords are normalized (ranging from 0 to 1).
        """
        # Approximate the camber line and max thickness from the data
        # Assume the camber line is the line of symmetry between the upper and lower surfaces
        upper_surface = y_coords[:int(len(x_coords) // 2)]
        lower_surface = y_coords[int(len(x_coords) // 2):]
        x_upper = x_coords[:int(len(x_coords) // 2)]
        x_lower = x_coords[int(len(x_coords) // 2):]

        upper_spline = CubicSpline(x_upper, upper_surface, bc_type='natural')
        lower_spline = CubicSpline(x_lower, lower_surface, bc_type='natural')

        camber_line = (upper_spline(x_coords) + lower_spline(x_coords)) / 2.0

        thickness = upper_spline(x_coords) - lower_spline(x_coords)

        max_thickness_index = omj.ks_max(thickness)
        max_thickness_value = thickness[max_thickness_index]

        camber_slope = jnp.gradient(camber_line, x_coords)
        camber_location_index = omj.ks_max(omj.smooth_abs(camber_slope))
        camber_location = x_coords[camber_location_index]

        camber = camber_line[camber_location_index]

        return camber, camber_location, max_thickness_value

if __name__ == "__main__":
    prob = om.Problem()

    prob.model.add_subsystem('tail', TailMassAndCOG(), promotes_inputs=['*'], promotes_outputs=['*'])

    prob.setup()

    # Input values
    #tail_type = prob.model.tail.options['tail_type']
    prob.model.tail.options['tail_type'] = 'vertical'
    
    if prob.model.tail.options['tail_type'] == 'horizontal':
        prob.set_val(Aircraft.HorizontalTail.SPAN, 1.0)
        prob.set_val(Aircraft.HorizontalTail.ROOT_CHORD, 1.0)
    elif prob.model.tail.options['tail_type'] == 'vertical':
        prob.set_val(Aircraft.VerticalTail.SPAN, 1.0)
        prob.set_val(Aircraft.VerticalTail.ROOT_CHORD, 1.0)

    prob.set_val('tip_chord_tail', 0.5)
    prob.set_val('thickness_ratio', 0.12)
    prob.set_val('skin_thickness', 0.002)

    prob.model.tail.options['material'] = 'Balsa'

    prob.run_model()

    # Print
    if prob.model.tail.options['tail_type'] == 'horizontal':
        print(f"Total mass of the horizontal tail: {prob.get_val(Aircraft.HorizontalTail.MASS)} kg")
    elif prob.model.tail.options['tail_type'] == 'vertical':
        print(f"Total mass of the vertical tail: {prob.get_val(Aircraft.VerticalTail.MASS)} kg")
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class TransportAvionicsMass(om.ExplicitComponent):
    '''
    Calculates the mass of the avionics group using the transport/general aviation method.
    The methodology is based on the FLOPS weight equations, modified to output mass
    instead of weight.
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.Avionics.MASS_SCALER)
        add_aviary_input(self, Aircraft.Fuselage.PLANFORM_AREA)
        add_aviary_input(self, Mission.Design.RANGE)

        add_aviary_output(self, Aircraft.Avionics.MASS)

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        aviary_options: AviaryValues = self.options['aviary_options']
        crew = aviary_options.get_val(Aircraft.CrewPayload.NUM_FLIGHT_CREW)
        scaler = inputs[Aircraft.Avionics.MASS_SCALER]
        planform = inputs[Aircraft.Fuselage.PLANFORM_AREA]
        des_range = inputs[Mission.Design.RANGE]

        outputs[Aircraft.Avionics.MASS] = \
            15.8 * des_range**0.1 * crew**0.7 * planform**0.43 * scaler / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        aviary_options: AviaryValues = self.options['aviary_options']
        crew = aviary_options.get_val(Aircraft.CrewPayload.NUM_FLIGHT_CREW)
        scaler = inputs[Aircraft.Avionics.MASS_SCALER]
        planform = inputs[Aircraft.Fuselage.PLANFORM_AREA]
        des_range = inputs[Mission.Design.RANGE]

        des_range_exp = des_range**0.1
        crew_exp = crew**0.7
        planform_exp = planform**0.43

        J[Aircraft.Avionics.MASS, Aircraft.Avionics.MASS_SCALER] = 15.8 * \
            des_range_exp * crew_exp * planform_exp / GRAV_ENGLISH_LBM

        J[Aircraft.Avionics.MASS, Aircraft.Fuselage.PLANFORM_AREA] = 6.794 * \
            des_range_exp * crew_exp * planform**-0.57 * scaler / GRAV_ENGLISH_LBM

        J[Aircraft.Avionics.MASS, Mission.Design.RANGE] = 1.58 * \
            des_range**-0.9 * crew_exp * planform_exp * scaler / GRAV_ENGLISH_LBM

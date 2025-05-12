import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.models.large_single_aisle_1.V3_bug_fixed_IO import (
    V3_bug_fixed_non_metadata,
    V3_bug_fixed_options,
)
from aviary.subsystems.geometry.gasp_based.size_group import SizeGroup
from aviary.subsystems.mass.gasp_based.mass_premission import MassPremission
from aviary.utils.aviary_values import get_items
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults, is_option
from aviary.variable_info.variables import Aircraft, Mission


class MassSummationTestCase1(unittest.TestCase):
    """
    This is the large single aisle 1 V3 bug fixed test case.
    All values are from V3 bug fixed output (or hand calculated from output) unless
    otherwise specified.
    """

    def setUp(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'gasp_based_geom',
            SizeGroup(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=[
                'aircraft:*',
            ],
        )
        self.prob.model.add_subsystem(
            'total_mass',
            MassPremission(),
            promotes=['*'],
        )

        for key, (val, units) in get_items(V3_bug_fixed_options):
            if not is_option(key):
                self.prob.model.set_input_defaults(key, val=val, units=units)

        for key, (val, units) in get_items(V3_bug_fixed_non_metadata):
            self.prob.model.set_input_defaults(key, val=val, units=units)

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=0.0, units='ft')
        # Adjust WETTED_AREA_SCALER such that WETTED_AREA = 4000.0
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_SCALER, val=0.86215, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, val=0.15)
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, val=0.9)

        setup_model_options(self.prob, V3_bug_fixed_options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        # print(f"wetted_area: {self.prob[Aircraft.Fuselage.WETTED_AREA]}")

        tol = 5e-4
        # size values:
        assert_near_equal(self.prob['gasp_based_geom.fuselage.cabin_height'], 13.1, tol)
        assert_near_equal(self.prob['gasp_based_geom.fuselage.cabin_len'], 72.09722222222223, tol)
        assert_near_equal(self.prob['gasp_based_geom.fuselage.nose_height'], 8.6, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 17.63, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 16.54, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1397, tol
        )  # not exact GASP value from the output file, likely due to rounding error

        assert_near_equal(self.prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 9.6509873673743, tol)
        assert_near_equal(self.prob[Aircraft.VerticalTail.AVERAGE_CHORD], 16.96457870166355, tol)
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 14.7, tol)

        # fixed mass values:
        assert_near_equal(self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS], 6384.35, tol)
        assert_near_equal(
            self.prob['total_mass.fixed_mass.tail.loc_MAC_vtail'], 0.44959578484694906, tol
        )

        # wing values:
        assert_near_equal(self.prob['wing_mass.isolated_wing_mass'], 15758, tol)
        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 12606, tol)
        assert_near_equal(self.prob[Aircraft.Engine.ADDITIONAL_MASS], 1765 / 2, tol)

        # fuel values:
        # modified from GASP value to account for updated crew mass. GASP value is
        # 78843.6
        assert_near_equal(
            self.prob['total_mass.fuel_mass.fuel_and_oem.OEM_wingfuel_mass'], 78846.0, tol
        )
        # modified from GASP value to account for updated crew mass. GASP value is
        # 102408.05695930264
        assert_near_equal(self.prob['fuel_mass.fus_mass_full'], 102408.4, tol)
        # modified from GASP value to account for updated crew mass. GASP value is
        # 1757
        assert_near_equal(
            self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1756.7, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1757
        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 50319.9, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuselage.MASS], 18667, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 18814

        # modified from GASP value to account for updated crew mass. GASP value is
        # 42843.6
        assert_near_equal(
            self.prob[Mission.Design.FUEL_MASS_REQUIRED], 42846.3, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 42843.6
        assert_near_equal(
            self.prob[Aircraft.Propulsion.MASS], 16127.5, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 16127
        assert_near_equal(
            self.prob[Mission.Design.FUEL_MASS], 42846.3, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 42844.0
        assert_near_equal(
            self.prob['fuel_mass.fuel_mass_min'], 32806.3, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 32803.6
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 856.55, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 856.4910800459031
        assert_near_equal(
            self.prob['fuel_mass.fuel_and_oem.OEM_fuel_vol'], 1576.22, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1576.1710061411081
        assert_near_equal(
            self.prob[Aircraft.Design.OPERATING_MASS], 96553.7, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 96556.0
        # extra_fuel_mass calculated differently in this version, so test for fuel_mass.fuel_and_oem.payload_mass_max_fuel not included
        assert_near_equal(
            self.prob['total_mass.fuel_mass.fuel_and_oem.volume_wingfuel_mass'], 57066.3, tol
        )
        assert_near_equal(self.prob['fuel_mass.max_wingfuel_mass'], 57066.3, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 0, tol
        )  # always zero when no body tank
        assert_near_equal(
            self.prob['total_mass.fuel_mass.body_tank.extra_fuel_volume'], 0, tol
        )  # always zero when no body tank
        assert_near_equal(
            self.prob['total_mass.fuel_mass.body_tank.max_extra_fuel_mass'], 0, tol
        )  # always zero when no body tank

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=3e-10, rtol=1e-12)


class MassSummationTestCase2(unittest.TestCase):
    """
    This is the large single aisle 1 V3.5 test case.
    All values are from V3.5 output (or hand calculated from the output, and these cases
    are specified).
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=37500, units='ft')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, val=200, units='lbm')
        options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 6)
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.Fuselage.SEAT_PITCH, 29, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.14, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'size',
            SizeGroup(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=[
                'aircraft:*',
            ],
        )
        self.prob.model.add_subsystem(
            'GASP_mass',
            MassPremission(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=25, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.LOADING, val=128, units='lbf/ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.189, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.145, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')
        # Adjust WETTED_AREA_SCALER such that WETTED_AREA = 4000.0
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_SCALER, val=0.86215, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.2307, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, 2.362, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.75, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.REFERENCE_DIAMETER, 5.8, units='ft')
        # self.prob.model.set_input_defaults(
        #     Aircraft.Engine.REFERENCE_SLS_THRUST, 28690, units="lbf"
        # )
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, 1.02823, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.25, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 2, units='unitless')

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )
        self.prob.model.set_input_defaults(Aircraft.CrewPayload.CARGO_MASS, val=0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.Design.MAX_CARGO_MASS, val=10040, units='lbm'
        )
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.SWEEP, val=0, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_COEFFICIENT, val=0.232, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_COEFFICIENT, val=0.289, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, val=1.9, units='unitless'
        )  # Based onlarge single aisle 1for updated flaps mass model
        self.prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=2.817, units='unitless'
        )  # Based onlarge single aisle 1for updated flaps mass model
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=0.95, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=16.5, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS, val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.TOTAL_MASS, val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, val=0.04, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT, val=0.85, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CLEARANCE_RATIO, val=0.2, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2')
        self.prob.model.set_input_defaults(Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )

        self.prob.model.set_input_defaults(Aircraft.APU.MASS, val=928.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Instruments.MASS_COEFFICIENT, val=0.0736, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, val=0.112, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, val=0.14, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Avionics.MASS, val=1959.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT, val=1.65, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=551.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS, val=11192.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, val=5.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, val=3.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, val=50.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, val=7.6, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, val=12.0, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.MASS_COEFFICIENT, val=102.5, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.MASS_COEFFICIENT, val=128, units='unitless'
        )
        self.prob.model.set_input_defaults('fuel_mass.fus_and_struct.pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults(
            'fuel_mass.fus_and_struct.MAT', val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Design.STRUCTURAL_MASS_INCREMENT, val=0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.041, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=0, units='unitless')

        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=0.0, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, val=0.15)
        self.prob.model.set_input_defaults(Aircraft.Wing.FLAP_CHORD_RATIO, val=0.3)
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, val=0.9)

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        # size values:
        assert_near_equal(self.prob['size.fuselage.cabin_height'], 13.1, tol)
        assert_near_equal(self.prob['size.fuselage.cabin_len'], 72.1, tol)
        assert_near_equal(self.prob['size.fuselage.nose_height'], 8.6, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 17.49, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 16.41, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1397, tol
        )  # not exact GASP value from the output file, likely due to rounding error

        # note: this is not the value in the GASP output, because the output calculates
        # them differently. This was calculated by hand.
        assert_near_equal(self.prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 9.578314120156815, tol)
        # note: this is not the value in the GASP output, because the output calculates
        # them differently. This was calculated by hand.
        assert_near_equal(self.prob[Aircraft.VerticalTail.AVERAGE_CHORD], 16.828924591320984, tol)
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 14.7, tol)

        # fixed mass values:
        assert_near_equal(
            self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS], 6384.35, tol
        )  # calculated by hand

        # note: fixed_mass.tail.loc_MAC_vtail not included in v3.5

        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 12606, tol)
        assert_near_equal(self.prob[Aircraft.Engine.ADDITIONAL_MASS], 1765 / 2, tol)

        # wing values:
        assert_near_equal(self.prob['wing_mass.isolated_wing_mass'], 15653, tol)

        # fuel values:
        # modified from GASP value to account for updated crew mass. GASP value is
        # 79147.2
        assert_near_equal(
            self.prob['fuel_mass.fuel_and_oem.OEM_wingfuel_mass'], 79002.9, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 79147.2

        # calculated by hand,  #modified from GASP value to account for updated crew
        # mass. GASP value is 102321.45695930265
        assert_near_equal(self.prob['fuel_mass.fus_mass_full'], 102359.6, tol)
        # modified from GASP value to account for updated crew mass. GASP value is
        # 1769
        assert_near_equal(
            self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1763.1, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1769

        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 50186, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuselage.MASS], 18663, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 18787

        assert_near_equal(
            self.prob[Mission.Design.FUEL_MASS_REQUIRED], 43002.9, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 43147.2
        assert_near_equal(
            self.prob[Aircraft.Propulsion.MASS], 16133.9, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 16140
        assert_near_equal(
            self.prob[Mission.Design.FUEL_MASS], 43002.9, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 43147
        assert_near_equal(
            self.prob['fuel_mass.fuel_mass_min'], 32962.9, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 33107.2
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 859.68, tol
        )  # calculated by hand,  #modified from GASP value to account for updated crew mass. GASP value is 862.5603807559726
        assert_near_equal(
            self.prob['fuel_mass.fuel_and_oem.OEM_fuel_vol'], 1579.36, tol
        )  # calculated by hand,  #modified from GASP value to account for updated crew mass. GASP value is 1582.2403068511774
        assert_near_equal(
            self.prob[Aircraft.Design.OPERATING_MASS], 96397.1, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 96253.0
        # extra_fuel_mass calculated differently in this version, so fuel_mass.fuel_and_oem.payload_mass_max_fuel test not included
        assert_near_equal(self.prob['fuel_mass.fuel_and_oem.volume_wingfuel_mass'], 55725.1, tol)
        assert_near_equal(self.prob['fuel_mass.max_wingfuel_mass'], 55725.1, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 0, tol
        )  # always zero when no body tank
        assert_near_equal(
            self.prob['fuel_mass.body_tank.extra_fuel_volume'], 0, tol
        )  # always zero when no body tank
        assert_near_equal(
            self.prob['fuel_mass.body_tank.max_extra_fuel_mass'], 0, tol
        )  # always zero when no body tank

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-10, rtol=1e-12)


class MassSummationTestCase3(unittest.TestCase):
    """
    This is thelarge single aisle 1V3.6 test case with a fuel margin of 0%, a wing loading of 128 psf, and a SLS thrust of 29500 lbf
    All values are from V3.6 output (or hand calculated from the output, and these cases are specified).
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=37500, units='ft')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, val=200, units='lbm')
        options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 6)
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.Fuselage.SEAT_PITCH, 29, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.14, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'size',
            SizeGroup(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=[
                'aircraft:*',
            ],
        )
        self.prob.model.add_subsystem(
            'GASP_mass',
            MassPremission(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=25, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.LOADING, val=128, units='lbf/ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.189, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.145, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')
        # Adjust WETTED_AREA_SCALER such that WETTED_AREA = 4000.0
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_SCALER, val=0.86215, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.2307, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, 2.362, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.75, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.REFERENCE_DIAMETER, 5.8, units='ft')
        # self.prob.model.set_input_defaults(
        #     Aircraft.Engine.REFERENCE_SLS_THRUST, 28690, units="lbf"
        # )
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, 1.02823, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.25, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 2, units='unitless')

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )

        self.prob.model.set_input_defaults(Aircraft.CrewPayload.CARGO_MASS, val=0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.Design.MAX_CARGO_MASS, val=10040, units='lbm'
        )
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.SWEEP, val=0, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_COEFFICIENT, val=0.232, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_COEFFICIENT, val=0.289, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, val=1.9, units='unitless'
        )  # Based onlarge single aisle 1for updated flaps mass model
        self.prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=2.817, units='unitless'
        )  # Based on large single aisle 1 for updated flaps mass model
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=0.95, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=16.5, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS, val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.TOTAL_MASS, val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, val=0.04, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT, val=0.85, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CLEARANCE_RATIO, val=0.2, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2')
        self.prob.model.set_input_defaults(Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )

        self.prob.model.set_input_defaults(Aircraft.APU.MASS, val=928.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Instruments.MASS_COEFFICIENT, val=0.0736, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, val=0.112, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, val=0.14, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Avionics.MASS, val=1959.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT, val=1.65, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=551.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS, val=11192.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, val=5.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, val=3.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, val=50.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, val=7.6, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, val=12.0, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.MASS_COEFFICIENT, val=102.5, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.MASS_COEFFICIENT, val=128, units='unitless'
        )
        self.prob.model.set_input_defaults('fuel_mass.fus_and_struct.pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults(
            'fuel_mass.fus_and_struct.MAT', val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Design.STRUCTURAL_MASS_INCREMENT, val=0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.041, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=0, units='unitless')

        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=0.0, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, val=0.15)
        self.prob.model.set_input_defaults(Aircraft.Wing.FLAP_CHORD_RATIO, val=0.3)
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, val=0.9)

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        # size values:
        assert_near_equal(self.prob['size.fuselage.cabin_height'], 13.1, tol)
        assert_near_equal(self.prob['size.fuselage.cabin_len'], 72.1, tol)
        assert_near_equal(self.prob['size.fuselage.nose_height'], 8.6, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 17.49, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 16.41, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1397, tol
        )  # not exact value, likely due to rounding error

        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 9.578314120156815, tol
        )  # note: this is not the value in the GASP output, because the output calculates them differently. This was calculated by hand.
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.AVERAGE_CHORD], 16.828924591320984, tol
        )  # note: this is not the value in the GASP output, because the output calculates them differently. This was calculated by hand.
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 14.7, tol)

        # fixed mass values:
        assert_near_equal(
            self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS], 6384.349999999999, tol
        )  # calculated by hand

        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 12606, tol)
        assert_near_equal(self.prob[Aircraft.Engine.ADDITIONAL_MASS], 1765 / 2, tol)

        # wing values:
        assert_near_equal(self.prob['wing_mass.isolated_wing_mass'], 15653, tol)

        # fuel values:
        assert_near_equal(
            self.prob['fuel_mass.fuel_and_oem.OEM_wingfuel_mass'], 79002.9, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 79147.2

        assert_near_equal(
            self.prob['fuel_mass.fus_mass_full'], 102359.6, tol
        )  # calculated by hand,  #modified from GASP value to account for updated crew mass. GASP value is 102321.45695930265
        assert_near_equal(
            self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1763.1, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 102321.45695930265

        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 50186, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuselage.MASS], 18663, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 18787

        assert_near_equal(
            self.prob[Mission.Design.FUEL_MASS_REQUIRED], 43002.9, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 43147.2
        assert_near_equal(
            self.prob[Aircraft.Propulsion.MASS], 16133.9, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 16140
        assert_near_equal(
            self.prob[Mission.Design.FUEL_MASS], 43002.9, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 43147
        assert_near_equal(
            self.prob['fuel_mass.fuel_mass_min'], 32962.9, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 33107.2
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 859.68, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 862.6
        assert_near_equal(
            self.prob['fuel_mass.fuel_and_oem.OEM_fuel_vol'], 1579.36, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1582.2
        assert_near_equal(
            self.prob[Aircraft.Design.OPERATING_MASS], 96397.1, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 96253.0
        assert_near_equal(
            self.prob['fuel_mass.fuel_and_oem.payload_mass_max_fuel'], 36000, tol
        )  # note: value came from running the GASP code on my own and printing it out
        assert_near_equal(self.prob['fuel_mass.fuel_and_oem.volume_wingfuel_mass'], 55725.1, tol)
        assert_near_equal(self.prob['fuel_mass.max_wingfuel_mass'], 55725.1, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 0, tol
        )  # always zero when no body tank
        assert_near_equal(
            self.prob['fuel_mass.body_tank.extra_fuel_volume'], 0, tol
        )  # always zero when no body tank
        assert_near_equal(
            self.prob['fuel_mass.body_tank.max_extra_fuel_mass'], 0, tol
        )  # always zero when no body tank

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-10, rtol=1e-12)


class MassSummationTestCase4(unittest.TestCase):
    """
    This is the large single aisle 1V3.6 test case with a fuel margin of 10%, a wing loading of 128 psf, and a SLS thrust of 29500 lbf
    All values are from V3.6 output (or hand calculated from the output, and these cases are specified).
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=37500, units='ft')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, val=200, units='lbm')
        options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 6)
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.Fuselage.SEAT_PITCH, 29, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.14, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'size',
            SizeGroup(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=[
                'aircraft:*',
            ],
        )
        self.prob.model.add_subsystem(
            'GASP_mass',
            MassPremission(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=25, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.LOADING, val=128, units='lbf/ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.189, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.145, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')
        # Adjust WETTED_AREA_SCALER such that WETTED_AREA = 4000.0
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_SCALER, val=0.86215, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.2307, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, 2.362, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.75, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.REFERENCE_DIAMETER, 5.8, units='ft')
        # self.prob.model.set_input_defaults(
        #     Aircraft.Engine.REFERENCE_SLS_THRUST, 28690, units="lbf"
        # )
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, 1.02823, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.25, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 2, units='unitless')

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )

        self.prob.model.set_input_defaults(Aircraft.CrewPayload.CARGO_MASS, val=0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.Design.MAX_CARGO_MASS, val=10040, units='lbm'
        )
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.SWEEP, val=0, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_COEFFICIENT, val=0.232, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_COEFFICIENT, val=0.289, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, val=1.9, units='unitless'
        )  # Based onlarge single aisle 1for updated flaps mass model
        self.prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=2.817, units='unitless'
        )  # Based on large single aisle 1 for updated flaps mass model
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=0.95, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=16.5, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS, val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.TOTAL_MASS, val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, val=0.04, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT, val=0.85, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CLEARANCE_RATIO, val=0.2, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2')
        self.prob.model.set_input_defaults(Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )

        self.prob.model.set_input_defaults(Aircraft.APU.MASS, val=928.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Instruments.MASS_COEFFICIENT, val=0.0736, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, val=0.112, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, val=0.14, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Avionics.MASS, val=1959.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT, val=1.65, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=551.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS, val=11192.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, val=5.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, val=3.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, val=50.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, val=7.6, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, val=12.0, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.MASS_COEFFICIENT, val=102.5, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.MASS_COEFFICIENT, val=128, units='unitless'
        )
        self.prob.model.set_input_defaults('fuel_mass.fus_and_struct.pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults(
            'fuel_mass.fus_and_struct.MAT', val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Design.STRUCTURAL_MASS_INCREMENT, val=0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.041, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=10, units='unitless')

        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=0.0, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, val=0.15)
        self.prob.model.set_input_defaults(Aircraft.Wing.FLAP_CHORD_RATIO, val=0.3)
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, val=0.9)

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        # size values:
        assert_near_equal(self.prob['size.fuselage.cabin_height'], 13.1, tol)
        assert_near_equal(self.prob['size.fuselage.cabin_len'], 72.1, tol)
        assert_near_equal(self.prob['size.fuselage.nose_height'], 8.6, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 17.49, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 16.41, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1397, tol
        )  # slightly different from GASP value, likely numerical error

        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 9.578314120156815, tol
        )  # note: this is not the value in the GASP output, because the output calculates them differently. This was calculated by hand.
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.AVERAGE_CHORD], 16.828924591320984, tol
        )  # note: this is not the value in the GASP output, because the output calculates them differently. This was calculated by hand.
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 14.7, tol)

        # fixed mass values:
        assert_near_equal(
            self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS], 6384.349999999999, tol
        )  # calculated by hand

        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 12606, tol)
        assert_near_equal(self.prob[Aircraft.Engine.ADDITIONAL_MASS], 1765 / 2, tol)

        # wing values:
        assert_near_equal(self.prob['wing_mass.isolated_wing_mass'], 15653, tol)

        # fuel values:
        assert_near_equal(
            self.prob['fuel_mass.fuel_and_oem.OEM_wingfuel_mass'], 78823.0, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 78966.7

        assert_near_equal(
            self.prob['fuel_mass.fus_mass_full'], 102541.4, tol
        )  # calculated by hand,  #modified from GASP value to account for updated crew mass. GASP value is 102501.95695930265
        assert_near_equal(
            self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1931.3, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1938

        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 50198, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuselage.MASS], 18675, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 18799

        assert_near_equal(
            self.prob[Mission.Design.FUEL_MASS_REQUIRED], 42823.0, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 42966.7
        assert_near_equal(
            self.prob[Aircraft.Propulsion.MASS], 16302.1, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 16309
        assert_near_equal(
            self.prob[Mission.Design.FUEL_MASS], 42823.0, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 42967
        assert_near_equal(
            self.prob['fuel_mass.fuel_mass_min'], 32783.0, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 32926.7
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 941.69, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 944.8
        assert_near_equal(
            self.prob['fuel_mass.fuel_and_oem.OEM_fuel_vol'], 1575.76, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1578.6
        assert_near_equal(
            self.prob[Aircraft.Design.OPERATING_MASS], 96577.0, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 96433.0
        assert_near_equal(
            self.prob['fuel_mass.fuel_and_oem.payload_mass_max_fuel'], 36000, tol
        )  # note: value came from running the GASP code on my own and printing it out
        assert_near_equal(self.prob['fuel_mass.fuel_and_oem.volume_wingfuel_mass'], 55725.1, tol)
        assert_near_equal(self.prob['fuel_mass.max_wingfuel_mass'], 55725.1, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 0, tol
        )  # always zero when no body tank
        assert_near_equal(
            self.prob['fuel_mass.body_tank.extra_fuel_volume'], 0, tol
        )  # always zero when no body tank
        assert_near_equal(
            self.prob['fuel_mass.body_tank.max_extra_fuel_mass'], 0, tol
        )  # always zero when no body tank

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=2e-10, rtol=1e-12)


class MassSummationTestCase5(unittest.TestCase):
    """
    This is thelarge single aisle 1V3.6 test case with a fuel margin of 0%, a wing loading of 150 psf, and a SLS thrust of 29500 lbf
    All values are from V3.6 output (or hand calculated from the output, and these cases are specified).
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=37500, units='ft')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, val=200, units='lbm')
        options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 6)
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.Fuselage.SEAT_PITCH, 29, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.14, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'size',
            SizeGroup(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=[
                'aircraft:*',
            ],
        )
        self.prob.model.add_subsystem(
            'GASP_mass',
            MassPremission(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=25, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.LOADING, val=150, units='lbf/ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.189, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.145, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')
        # Adjust WETTED_AREA_SCALER such that WETTED_AREA = 4000.0
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_SCALER, val=0.86215, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.2307, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, 2.362, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.75, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.REFERENCE_DIAMETER, 5.8, units='ft')
        # self.prob.model.set_input_defaults(
        #     Aircraft.Engine.REFERENCE_SLS_THRUST, 28690, units="lbf"
        # )
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, 1.02823, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.25, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 2, units='unitless')

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )

        self.prob.model.set_input_defaults(Aircraft.CrewPayload.CARGO_MASS, val=0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.Design.MAX_CARGO_MASS, val=10040, units='lbm'
        )
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.SWEEP, val=0, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_COEFFICIENT, val=0.232, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_COEFFICIENT, val=0.289, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, val=1.9, units='unitless'
        )  # Based onlarge single aisle 1for updated flaps mass model
        self.prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=2.817, units='unitless'
        )  # Based on large single aisle 1 for updated flaps mass model
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=0.95, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=16.5, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS, val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.TOTAL_MASS, val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, val=0.04, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT, val=0.85, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CLEARANCE_RATIO, val=0.2, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2')
        self.prob.model.set_input_defaults(Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )

        self.prob.model.set_input_defaults(Aircraft.APU.MASS, val=928.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Instruments.MASS_COEFFICIENT, val=0.0736, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, val=0.112, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, val=0.14, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Avionics.MASS, val=1959.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT, val=1.65, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=551.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS, val=11192.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, val=5.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, val=3.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, val=50.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, val=7.6, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, val=12.0, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.MASS_COEFFICIENT, val=102.5, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.MASS_COEFFICIENT, val=128, units='unitless'
        )
        self.prob.model.set_input_defaults('fuel_mass.fus_and_struct.pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults(
            'fuel_mass.fus_and_struct.MAT', val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Design.STRUCTURAL_MASS_INCREMENT, val=0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.041, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=0.0, units='unitless')

        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=0.0, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, val=0.15)
        self.prob.model.set_input_defaults(Aircraft.Wing.FLAP_CHORD_RATIO, val=0.3)
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, val=0.9)

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        # size values:
        assert_near_equal(self.prob['size.fuselage.cabin_height'], 13.1, tol)
        assert_near_equal(self.prob['size.fuselage.cabin_len'], 72.1, tol)
        assert_near_equal(self.prob['size.fuselage.nose_height'], 8.6, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 16.16, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 15.1, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1394, tol
        )  # slightly different from GASP value, likely rounding error

        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 8.848695928254141, tol
        )  # note: this is not the value in the GASP output, because the output calculates them differently. This was calculated by hand.
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.AVERAGE_CHORD], 15.550266681026597, tol
        )  # note: this is not the value in the GASP output, because the output calculates them differently. This was calculated by hand.
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 14.7, tol)

        # fixed mass values:
        assert_near_equal(
            self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS],
            6384.349999999999,
            tol,
            # self.prob["fixed_mass.main_gear_mass"], 6384.349999999999, tol
        )  # calculated by hand

        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 12606, tol)
        assert_near_equal(self.prob[Aircraft.Engine.ADDITIONAL_MASS], 1765 / 2, tol)

        # wing values:
        assert_near_equal(self.prob['wing_mass.isolated_wing_mass'], 14631, tol)

        # fuel values:
        assert_near_equal(
            self.prob['fuel_mass.fuel_and_oem.OEM_wingfuel_mass'], 80475.9, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 81424.8

        assert_near_equal(self.prob['fuel_mass.fus_mass_full'], 102510.7, tol)  # calculated by hand
        assert_near_equal(
            self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1823.4, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1862

        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 48941, tol)
        assert_near_equal(self.prob[Aircraft.Fuselage.MASS], 18675, tol)

        assert_near_equal(
            self.prob[Mission.Design.FUEL_MASS_REQUIRED], 44472.9, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 45424.8
        assert_near_equal(
            self.prob[Aircraft.Propulsion.MASS], 16194.2, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 16233
        assert_near_equal(
            self.prob[Mission.Design.FUEL_MASS], 44472.9, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 45425
        assert_near_equal(
            self.prob['fuel_mass.fuel_mass_min'], 34432.9, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 35384.8
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 889.06, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 908.1
        assert_near_equal(
            self.prob['fuel_mass.fuel_and_oem.OEM_fuel_vol'], 1608.74, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1627.8
        assert_near_equal(
            self.prob[Aircraft.Design.OPERATING_MASS], 94927.1, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 93975
        assert_near_equal(
            self.prob['fuel_mass.fuel_and_oem.payload_mass_max_fuel'], 35380.5, tol
        )  # note: value came from running the GASP code on my own and printing it out,  #modified from GASP value to account for updated crew mass. GASP value is 34427.4
        assert_near_equal(self.prob['fuel_mass.fuel_and_oem.volume_wingfuel_mass'], 43852.1, tol)
        assert_near_equal(self.prob['fuel_mass.max_wingfuel_mass'], 43852.1, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 620.739, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1572.6
        assert_near_equal(
            self.prob['fuel_mass.body_tank.extra_fuel_volume'], 12.4092, tol
        )  # slightly different from GASP value, likely a rounding error,  #modified from GASP value to account for updated crew mass. GASP value is 31.43
        assert_near_equal(
            self.prob['fuel_mass.body_tank.max_extra_fuel_mass'], 620.736, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1572.6

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=3e-10, rtol=1e-12)


class MassSummationTestCase6(unittest.TestCase):
    """
    This is thelarge single aisle 1V3.6 test case with a fuel margin of 10%, a wing loading of 150 psf, and a SLS thrust of 29500 lbf
    All values are from V3.6 output (or hand calculated from the output, and these cases are specified).
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=37500, units='ft')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, val=200, units='lbm')
        options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 6)
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.Fuselage.SEAT_PITCH, 29, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.14, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'size',
            SizeGroup(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=[
                'aircraft:*',
            ],
        )
        self.prob.model.add_subsystem(
            'GASP_mass',
            MassPremission(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=10.13, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=25, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.15, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=175400, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.LOADING, val=150, units='lbf/ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units='lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.189, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.145, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')
        # Adjust WETTED_AREA_SCALER such that WETTED_AREA = 4000.0
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_SCALER, val=0.86215, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.2307, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, 2.362, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.75, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.REFERENCE_DIAMETER, 5.8, units='ft')
        # self.prob.model.set_input_defaults(
        #     Aircraft.Engine.REFERENCE_SLS_THRUST, 28690, units="lbf"
        # )
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, 1.02823, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.25, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 2, units='unitless')

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )

        self.prob.model.set_input_defaults(Aircraft.CrewPayload.CARGO_MASS, val=0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.Design.MAX_CARGO_MASS, val=10040, units='lbm'
        )
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.SWEEP, val=0, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_COEFFICIENT, val=0.232, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_COEFFICIENT, val=0.289, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, val=1.9, units='unitless'
        )  # Based onlarge single aisle 1for updated flaps mass model
        self.prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=2.817, units='unitless'
        )  # Based onlarge single aisle 1for updated flaps mass model
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=0.95, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=16.5, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS, val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.TOTAL_MASS, val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, val=0.04, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT, val=0.85, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CLEARANCE_RATIO, val=0.2, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.21366, units='lbm/lbf'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2')
        self.prob.model.set_input_defaults(Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )

        self.prob.model.set_input_defaults(Aircraft.APU.MASS, val=928.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Instruments.MASS_COEFFICIENT, val=0.0736, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, val=0.112, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, val=0.14, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Avionics.MASS, val=1959.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT, val=1.65, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=551.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS, val=11192.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, val=5.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, val=3.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, val=50.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, val=7.6, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, val=12.0, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.MASS_COEFFICIENT, val=102.5, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.MASS_COEFFICIENT, val=128, units='unitless'
        )
        self.prob.model.set_input_defaults('fuel_mass.fus_and_struct.pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults(
            'fuel_mass.fus_and_struct.MAT', val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Design.STRUCTURAL_MASS_INCREMENT, val=0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.041, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=10.0, units='unitless')

        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=0.0, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, val=0.15)
        self.prob.model.set_input_defaults(Aircraft.Wing.FLAP_CHORD_RATIO, val=0.3)
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, val=0.9)

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        # size values:
        assert_near_equal(self.prob['size.fuselage.cabin_height'], 13.1, tol)
        assert_near_equal(self.prob['size.fuselage.cabin_len'], 72.1, tol)
        assert_near_equal(self.prob['size.fuselage.nose_height'], 8.6, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 16.16, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 15.1, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1394, tol
        )  # note: not exact GASP value, likely rounding error

        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 8.848695928254141, tol
        )  # note: this is not the value in the GASP output, because the output calculates them differently. This was calculated by hand.
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.AVERAGE_CHORD], 15.550266681026597, tol
        )  # note: this is not the value in the GASP output, because the output calculates them differently. This was calculated by hand.
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 14.7, tol)

        # fixed mass values:
        assert_near_equal(
            self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS],
            6384.349999999999,
            tol,
            # self.prob["fixed_mass.main_gear_mass"], 6384.349999999999, tol
        )  # calculated by hand

        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 12606, tol)
        assert_near_equal(self.prob[Aircraft.Engine.ADDITIONAL_MASS], 1765 / 2, tol)

        # wing values:
        assert_near_equal(self.prob['wing_mass.isolated_wing_mass'], 14631, tol)

        # fuel values:
        assert_near_equal(
            self.prob['fuel_mass.fuel_and_oem.OEM_wingfuel_mass'], 80029.2, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 80982.7

        assert_near_equal(self.prob['fuel_mass.fus_mass_full'], 106913.6, tol)  # calculated by hand
        assert_near_equal(
            self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1985.7, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 2029

        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 49222, tol)
        assert_near_equal(self.prob[Aircraft.Fuselage.MASS], 18956, tol)

        assert_near_equal(
            self.prob[Mission.Design.FUEL_MASS_REQUIRED], 44029.2, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 44982.7
        assert_near_equal(
            self.prob[Aircraft.Propulsion.MASS], 16356.5, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 16399
        assert_near_equal(
            self.prob[Mission.Design.FUEL_MASS], 44029.2, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 44982.7
        assert_near_equal(
            self.prob['fuel_mass.fuel_mass_min'], 33989.2, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 34942.7
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 968.21, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 989.2
        assert_near_equal(
            self.prob['fuel_mass.fuel_and_oem.OEM_fuel_vol'], 1599.87, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1618.9
        assert_near_equal(
            self.prob[Aircraft.Design.OPERATING_MASS], 95370.8, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 94417
        assert_near_equal(
            self.prob['fuel_mass.fuel_and_oem.payload_mass_max_fuel'], 35823.0, tol
        )  # note: value came from running the GASP code on my own and printing it out,  #modified from GASP value to account for updated crew mass. GASP value is 34879.2
        assert_near_equal(self.prob['fuel_mass.fuel_and_oem.volume_wingfuel_mass'], 43852.1, tol)
        assert_near_equal(self.prob['fuel_mass.max_wingfuel_mass'], 43852.1, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 177.042, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1120.9
        assert_near_equal(
            self.prob['fuel_mass.body_tank.extra_fuel_volume'], 91.5585, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 112.3
        assert_near_equal(
            self.prob['fuel_mass.body_tank.max_extra_fuel_mass'], 4579.96, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 5618.2

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=3e-10, rtol=1e-12)


class MassSummationTestCase7(unittest.TestCase):
    """
    This is the Advanced Tube and Wing V3.6 test case
    All values are from V3.6 output, hand calculated from the output, or were printed out after running the code manually.
    Values not directly from the output are labeled as such.
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=154, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=154, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=37100, units='ft')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, val=200, units='lbm')
        options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 6)
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.Fuselage.SEAT_PITCH, 29, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.165, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'size',
            SizeGroup(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=[
                'aircraft:*',
            ],
        )
        self.prob.model.add_subsystem(
            'GASP_mass',
            MassPremission(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=11, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.33, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=25, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=145388.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.LOADING, val=104.50, units='lbf/ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=0, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=1.67, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=17000.0, units='lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.35, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, 0.475, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.189, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.09986, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 9.5, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 3, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_SCALER, 1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.2307, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, 2.1621, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.75, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.REFERENCE_DIAMETER, 7.36, units='ft')
        # self.prob.model.set_input_defaults(
        #     Aircraft.Engine.REFERENCE_SLS_THRUST, 28620.0, units="lbf"
        # )
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, 0.594, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.2095, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 1.715, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.FOLDED_SPAN, 118, units='ft')

        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )
        self.prob.model.set_input_defaults(Aircraft.CrewPayload.CARGO_MASS, val=0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.Design.MAX_CARGO_MASS, val=15970.0, units='lbm'
        )
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.SWEEP, val=0, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_COEFFICIENT, val=0.232, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_COEFFICIENT, val=0.289, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=0.12, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, val=1.9, units='unitless'
        )  # Based onlarge single aisle 1for updated flaps mass model
        self.prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=2.817, units='unitless'
        )  # Based on large single aisle 1 for updated flaps mass model
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=0.95, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=16.5, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS, val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.TOTAL_MASS, val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, val=0.04, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT, val=0.85, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CLEARANCE_RATIO, val=0.2, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.2355, units='lbm/lbf'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.MASS_SPECIFIC, val=3, units='lbm/ft**2')
        self.prob.model.set_input_defaults(Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0.15, units='unitless'
        )

        self.prob.model.set_input_defaults(Aircraft.APU.MASS, val=1014.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Instruments.MASS_COEFFICIENT, val=0.0736, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, val=0.085, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, val=0.105, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Avionics.MASS, val=1504.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT, val=1.65, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=126.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS, val=9114.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, val=5.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, val=3.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, val=0.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, val=10.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, val=12.0, units='unitless'
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.MASS_COEFFICIENT, val=85, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLD_MASS_COEFFICIENT, val=0.2, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.MASS_COEFFICIENT, val=128, units='unitless'
        )
        self.prob.model.set_input_defaults('fuel_mass.fus_and_struct.pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults(
            'fuel_mass.fus_and_struct.MAT', val=0, units='lbm'
        )  # note: not actually defined in program, likely an error
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Design.STRUCTURAL_MASS_INCREMENT, val=0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.041, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=10.0, units='unitless')

        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=0.0, units='ft')
        self.prob.model.set_input_defaults(Mission.Design.MACH, val=0.8, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, val=0.15)
        self.prob.model.set_input_defaults(Aircraft.Wing.FLAP_CHORD_RATIO, val=0.3)
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, val=0.9)

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        # size values:
        assert_near_equal(self.prob['size.fuselage.cabin_height'], 13.1, tol)
        assert_near_equal(self.prob['size.fuselage.cabin_len'], 61.6, tol)
        assert_near_equal(self.prob['size.fuselage.nose_height'], 8.6, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 16.91, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 16.01, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1132, tol
        )  # slightly different from GASP value, likely a rounding error

        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.AVERAGE_CHORD],
            9.6498,
            tol,
            # note: value came from running the GASP code on my own and printing it out (GASP output calculates this differently)
        )
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.AVERAGE_CHORD],
            13.4662,
            tol,
            # note: value came from running the GASP code on my own and printing it out (GASP output calculates this differently)
        )
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 11.77, tol)

        # fixed mass values:
        assert_near_equal(
            self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS],
            5219.3076,
            tol,
            # self.prob["fixed_mass.main_gear_mass"], 5219.3076, tol
        )  # note: value came from running the GASP code on my own and printing it out

        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 8007, tol)
        assert_near_equal(self.prob[Aircraft.Engine.ADDITIONAL_MASS], 1321 / 2, tol)

        # wing values:
        assert_near_equal(
            self.prob['wing_mass.isolated_wing_mass'], 13993, tol
        )  # calculated as difference between wing mass and fold mass, not an actual GASP variable

        # fuel values:
        assert_near_equal(
            self.prob['fuel_mass.fuel_and_oem.OEM_wingfuel_mass'], 63452.2, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 62427.2

        assert_near_equal(
            self.prob['fuel_mass.fus_mass_full'], 99396.7, tol
        )  # note: value came from running the GASP code on my own and printing it out
        assert_near_equal(
            self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1472.6, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1426

        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 45373.4, tol)
        assert_near_equal(self.prob[Aircraft.Fuselage.MASS], 18859.9, tol)

        assert_near_equal(
            self.prob[Mission.Design.FUEL_MASS_REQUIRED], 32652.2, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 31627.2
        assert_near_equal(
            self.prob[Aircraft.Propulsion.MASS], 10800.8, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 10755.0
        assert_near_equal(
            self.prob[Mission.Design.FUEL_MASS], 32652.2, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 31627.0
        assert_near_equal(
            self.prob['fuel_mass.fuel_mass_min'], 16682.2, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 15657.2
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 718.03, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 695.5
        assert_near_equal(
            self.prob['fuel_mass.fuel_and_oem.OEM_fuel_vol'], 1268.48, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1248.0
        assert_near_equal(
            self.prob[Aircraft.Design.OPERATING_MASS], 81935.8, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 82961.0
        assert_near_equal(
            self.prob['fuel_mass.fuel_and_oem.payload_mass_max_fuel'], 30800.0039, tol
        )  # note: value came from running the GASP code on my own and printing it out
        assert_near_equal(self.prob['fuel_mass.fuel_and_oem.volume_wingfuel_mass'], 33892.8, tol)
        assert_near_equal(self.prob['fuel_mass.max_wingfuel_mass'], 33892.8, tol)
        assert_near_equal(self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 0, tol)
        assert_near_equal(
            self.prob['fuel_mass.body_tank.extra_fuel_volume'], 40.4789, 0.005
        )  # note: higher tol because slightly different from GASP value, likely numerical issues,  #modified from GASP value to account for updated crew mass. GASP value is 17.9
        assert_near_equal(
            self.prob['fuel_mass.body_tank.max_extra_fuel_mass'], 2024.70, 0.003
        )  # note: higher tol because slightly different from GASP value, likely numerical issues,  #modified from GASP value to account for updated crew mass. GASP value is 897.2

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=3e-9, rtol=6e-11)


class MassSummationTestCase8(unittest.TestCase):
    """
    This is the Trans-sonic Truss-Braced Wing V3.6 test case
    All values are from V3.6 output, hand calculated from the output, or were printed out after running the code manually.
    Values not directly from the output are labeled as such.
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        options.set_val(Aircraft.Electrical.HAS_HYBRID_SYSTEM, val=False, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=154, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=154, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=43000, units='ft')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless')
        options.set_val(Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, val=200, units='lbm')
        options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 6)
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.Fuselage.SEAT_PITCH, 44.2, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.163, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'size',
            SizeGroup(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=[
                'aircraft:*',
            ],
        )
        self.prob.model.add_subsystem(
            'GASP_mass',
            MassPremission(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.13067, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.025, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, 3.0496, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.REFERENCE_DIAMETER, 7.642, units='ft')
        # self.prob.model.set_input_defaults(
        #     Aircraft.Engine.REFERENCE_SLS_THRUST, 28620.0, units="lbf"
        # )
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, 1.35255, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.2095, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 1.660, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 6.85, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 1.18, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_SCALER, 1.0, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.LOADING, val=87.5, units='lbf/ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )

        self.prob.model.set_input_defaults(Aircraft.APU.MASS, val=1014.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Instruments.MASS_COEFFICIENT, val=0.0736, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, val=0.085, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, val=0.105, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Avionics.MASS, val=1504.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT, val=1.65, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=126.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS, val=9114.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, val=5.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, val=3.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, val=0.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, val=10.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, val=12.0, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=21160.0, units='lbf'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, 0.73934, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, 0.5625, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=22.47, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.VERTICAL_MOUNT_LOCATION, val=0.1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=19.565, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Strut.ATTACHMENT_LOCATION, val=118, units='ft')
        self.prob.model.set_input_defaults(Aircraft.CrewPayload.CARGO_MASS, val=0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.Design.MAX_CARGO_MASS, val=15970.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.2470, units='lbm/lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.MASS_SPECIFIC, val=2.5, units='lbm/ft**2'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.2143, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=0.825, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.SWEEP, val=0, units='deg'
        )  # not in file
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_COEFFICIENT, val=0.2076, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_COEFFICIENT, val=0.2587, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=0.11, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=0.1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, val=1.9, units='unitless'
        )  # Based onlarge single aisle 1for updated flaps mass model
        self.prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=2.817, units='unitless'
        )  # Based on large single aisle 1 for updated flaps mass model
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=0.5936, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=30, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS, val=0, units='lbm'
        )  # not in file
        self.prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.TOTAL_MASS, val=0, units='lbm'
        )  # not in file
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, val=0.03390, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT, val=0.85, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=10.0, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.060, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.MASS_COEFFICIENT, val=89.66, units='unitless'
        )
        self.prob.model.set_input_defaults('fuel_mass.fus_and_struct.pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults(
            'fuel_mass.fus_and_struct.MAT', val=0, units='lbm'
        )  # not in file
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Design.STRUCTURAL_MASS_INCREMENT, val=0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=143100.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.MASS_COEFFICIENT, val=78.94, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.346, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.11, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.43, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.066, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLD_MASS_COEFFICIENT, val=0.2, units='unitless'
        )
        # self.prob.model.set_input_defaults(
        #     Aircraft.Strut.AREA, 523.337, units="ft**2"
        # )  # had to calculate by hand
        self.prob.model.set_input_defaults(Aircraft.Strut.MASS_COEFFICIENT, 0.238, units='unitless')

        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=0.0, units='ft')
        self.prob.model.set_input_defaults(Mission.Design.MACH, val=0.8, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, val=0.15)
        self.prob.model.set_input_defaults(Aircraft.Wing.FLAP_CHORD_RATIO, val=0.3)
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, val=0.9)

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        # size values:
        assert_near_equal(self.prob['size.fuselage.cabin_height'], 13.1, tol)
        assert_near_equal(self.prob['size.fuselage.cabin_len'], 93.9, tol)
        assert_near_equal(self.prob['size.fuselage.nose_height'], 8.6, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 13.59, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 13.15, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1068, tol
        )  # note:precision came from running code on my own and printing it out

        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 9.381, tol
        )  # note, printed out manually because calculated differently in output subroutine
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.AVERAGE_CHORD], 20.056, tol
        )  # note, printed out manually because calculated differently in output subroutine
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 13.19, tol)

        # fixed mass values:
        assert_near_equal(
            self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS],
            4123.4,
            tol,
            # self.prob["fixed_mass.main_gear_mass"], 4123.4, tol
        )  # note:printed out from GASP code

        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 10453.0, tol)
        assert_near_equal(self.prob[Aircraft.Engine.ADDITIONAL_MASS], 1704.0 / 2, tol)

        # wing values:
        assert_near_equal(self.prob['wing_mass.isolated_wing_mass'], 14040, tol)
        assert_near_equal(self.prob[Aircraft.Wing.MASS], 18031, tol)

        # fuel values:
        assert_near_equal(
            self.prob['fuel_mass.fuel_and_oem.OEM_wingfuel_mass'], 60410.9, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 59372.3

        assert_near_equal(
            self.prob['fuel_mass.fus_mass_full'], 97697.5, tol
        )  # note:printed out from GASP code
        assert_near_equal(
            self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1954.3, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1886.0

        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 43660.4, tol)
        assert_near_equal(self.prob[Aircraft.Fuselage.MASS], 14657.4, tol)

        assert_near_equal(
            self.prob[Mission.Design.FUEL_MASS_REQUIRED], 29610.9, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 28572.3
        assert_near_equal(
            self.prob[Aircraft.Propulsion.MASS], 14111.2, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 14043.0
        assert_near_equal(
            self.prob[Mission.Design.FUEL_MASS], 29610.9, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 28572.0
        assert_near_equal(
            self.prob['fuel_mass.fuel_mass_min'], 13640.9, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 12602.3
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 651.15, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 628.3
        assert_near_equal(
            self.prob['fuel_mass.fuel_and_oem.OEM_fuel_vol'], 1207.68, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1186.9
        assert_near_equal(
            self.prob[Aircraft.Design.OPERATING_MASS], 82689.1, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 83728.0
        assert_near_equal(
            self.prob['fuel_mass.fuel_and_oem.payload_mass_max_fuel'], 30800.0, tol
        )  # note:printed out from GASP code
        assert_near_equal(self.prob['fuel_mass.fuel_and_oem.volume_wingfuel_mass'], 31051.6, tol)
        assert_near_equal(self.prob['fuel_mass.max_wingfuel_mass'], 31051.6, tol)
        assert_near_equal(self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 0, tol)
        assert_near_equal(
            self.prob['fuel_mass.body_tank.extra_fuel_volume'], 30.3942, 0.009
        )  # note: higher tol because slightly different from GASP value, likely numerical issues, printed out from the GASP code,  #modified from GASP value to account for updated crew mass. GASP value is 7.5568
        assert_near_equal(
            self.prob['fuel_mass.body_tank.max_extra_fuel_mass'], 1520.384, 0.009
        )  # note: higher tol because slightly different from GASP value, likely numerical issues, printed out from the GASP code,  #modified from GASP value to account for updated crew mass. GASP value is 378.0062

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=3e-9, rtol=6e-11)


class MassSummationTestCase9(unittest.TestCase):
    """
    This is the electrified Trans-sonic Truss-Braced Wing V3.6 test case
    All values are from V3.6 output, hand calculated from the output, or were printed out after running the code manually.
    Values not directly from the output are labeled as such.
    """

    def setUp(self):
        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        options.set_val(Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=154, units='unitless')
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=154, units='unitless')
        options.set_val(Mission.Design.CRUISE_ALTITUDE, val=43000, units='ft')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')
        options.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless'
        )
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, val=True, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR, val=False, units='unitless')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES, val=True, units='unitless')
        options.set_val(Aircraft.CrewPayload.PASSENGER_MASS_WITH_BAGS, val=200, units='lbm')
        options.set_val(Aircraft.Fuselage.NUM_SEATS_ABREAST, 6)
        options.set_val(Aircraft.Fuselage.AISLE_WIDTH, 24, units='inch')
        options.set_val(Aircraft.Fuselage.NUM_AISLES, 1)
        options.set_val(Aircraft.Fuselage.SEAT_PITCH, 44.2, units='inch')
        options.set_val(Aircraft.Fuselage.SEAT_WIDTH, 20.2, units='inch')
        options.set_val(Aircraft.Engine.ADDITIONAL_MASS_FRACTION, 0.163, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'size',
            SizeGroup(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=[
                'aircraft:*',
            ],
        )
        self.prob.model.add_subsystem(
            'GASP_mass',
            MassPremission(),
            promotes=['*'],
        )

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.TAPER_RATIO, val=0.352, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MOMENT_RATIO, val=0.13067, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.ASPECT_RATIO, val=4.025, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MOMENT_RATIO, 3.0496, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.TAPER_RATIO, val=0.801, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.REFERENCE_DIAMETER, 8.425, units='ft')
        # self.prob.model.set_input_defaults(
        #     Aircraft.Engine.REFERENCE_SLS_THRUST, 28620, units="lbf"
        # )
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, 0.73934, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.2095, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Nacelle.FINENESS, 1.569, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.DELTA_DIAMETER, 4.5, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Fuselage.NOSE_FINENESS, 1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PILOT_COMPARTMENT_LENGTH, 6.85, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.TAIL_FINENESS, 1.18, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.WETTED_AREA_SCALER, 1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.LOADING, val=96.10, units='lbf/ft**2')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.MAX_STRUCTURAL_SPEED, val=402.5, units='mi/h'
        )

        self.prob.model.set_input_defaults(Aircraft.APU.MASS, val=1014.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Instruments.MASS_COEFFICIENT, val=0.0736, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.FLIGHT_CONTROL_MASS_COEFFICIENT, val=0.085, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Hydraulics.GEAR_MASS_COEFFICIENT, val=0.105, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Avionics.MASS, val=1504.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.AirConditioning.MASS_COEFFICIENT, val=1.65, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.AntiIcing.MASS, val=126.0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Furnishings.MASS, val=9114.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS_PER_PASSENGER, val=5.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.WATER_MASS_PER_OCCUPANT, val=3.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.EMERGENCY_EQUIPMENT_MASS, val=0.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.CATERING_ITEMS_MASS_PER_PASSENGER, val=10.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.UNUSABLE_FUEL_MASS_COEFFICIENT, val=12.0, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units='psi'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=23750.0, units='lbf'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.SCALE_FACTOR, 0.82984, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, 0.5936, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, val=22.47, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.VERTICAL_MOUNT_LOCATION, val=0.1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, val=19.565, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Strut.ATTACHMENT_LOCATION, val=118.0, units='ft'
        )
        self.prob.model.set_input_defaults(Aircraft.CrewPayload.CARGO_MASS, val=0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.CrewPayload.Design.MAX_CARGO_MASS, val=15970.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.MASS_SPECIFIC, val=0.2744, units='lbm/lbf'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Nacelle.MASS_SPECIFIC, val=2.5, units='lbm/ft**2'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.PYLON_FACTOR, val=1.25, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Engine.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Propulsion.MISC_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Engine.WING_LOCATIONS, val=0.2143, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_LOCATION, val=0, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.ASPECT_RATIO, val=0.825, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.VerticalTail.SWEEP, val=0, units='deg')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_COEFFICIENT, val=0.2076, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TAIL_HOOK_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_COEFFICIENT, val=0.2587, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=0.11, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=0.1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.HIGH_LIFT_MASS_COEFFICIENT, val=1.9, units='unitless'
        )  # Based onlarge single aisle 1for updated flaps mass model
        self.prob.model.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=2.817, units='unitless'
        )  # Based on large single aisle 1 for updated flaps mass model
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_COEFFICIENT, val=0.5936, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Design.COCKPIT_CONTROL_MASS_COEFFICIENT, val=30.0, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS, val=1, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.COCKPIT_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.SURFACE_CONTROL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Controls.STABILITY_AUGMENTATION_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Controls.TOTAL_MASS, val=0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MASS_COEFFICIENT, val=0.03390, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.MAIN_GEAR_MASS_COEFFICIENT, val=0.85, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuel.DENSITY, val=6.687, units='lbm/galUS')
        self.prob.model.set_input_defaults(Aircraft.Fuel.FUEL_MARGIN, val=0.0, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_COEFFICIENT, val=0.060, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.MASS_COEFFICIENT, val=96.94, units='unitless'
        )
        self.prob.model.set_input_defaults('fuel_mass.fus_and_struct.pylon_len', val=0, units='ft')
        self.prob.model.set_input_defaults('fuel_mass.fus_and_struct.MAT', val=0, units='lbm')
        self.prob.model.set_input_defaults(Aircraft.Wing.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Fuselage.MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Engine.POD_MASS_SCALER, val=1, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Design.STRUCTURAL_MASS_INCREMENT, val=0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.FUEL_SYSTEM_MASS_SCALER, val=1, units='unitless'
        )
        self.prob.model.set_input_defaults(Mission.Design.GROSS_MASS, val=166100.0, units='lbm')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.MASS_COEFFICIENT, val=78.94, units='unitless'
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.TAPER_RATIO, val=0.346, units='unitless')
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.11, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.VOLUME_COEFFICIENT, val=1.43, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.VOLUME_COEFFICIENT, 0.066, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLD_MASS_COEFFICIENT, val=0.2, units='unitless'
        )
        # self.prob.model.set_input_defaults(
        #     Aircraft.Strut.AREA, 553.1, units="ft**2"
        # )
        self.prob.model.set_input_defaults(Aircraft.Strut.MASS_COEFFICIENT, 0.238, units='unitless')
        self.prob.model.set_input_defaults('fixed_mass.augmentation.motor_power', 830, units='kW')
        self.prob.model.set_input_defaults('fixed_mass.augmentation.motor_voltage', 850, units='V')
        self.prob.model.set_input_defaults(
            'fixed_mass.augmentation.max_amp_per_wire', 260, units='A'
        )
        self.prob.model.set_input_defaults(
            'fixed_mass.augmentation.safety_factor', 1, units='unitless'
        )  # (not in this GASP code)
        self.prob.model.set_input_defaults(
            'fixed_mass.augmentation.wire_area', 0.0015, units='ft**2'
        )
        self.prob.model.set_input_defaults(
            'fixed_mass.augmentation.rho_wire', 565, units='lbm/ft**3'
        )
        self.prob.model.set_input_defaults(
            'fixed_mass.augmentation.battery_energy', 6077, units='MJ'
        )
        self.prob.model.set_input_defaults(
            'fixed_mass.augmentation.motor_eff', 0.98, units='unitless'
        )
        self.prob.model.set_input_defaults(
            'fixed_mass.augmentation.inverter_eff', 0.99, units='unitless'
        )
        self.prob.model.set_input_defaults(
            'fixed_mass.augmentation.transmission_eff', 0.975, units='unitless'
        )
        self.prob.model.set_input_defaults(
            'fixed_mass.augmentation.battery_eff', 0.975, units='unitless'
        )
        self.prob.model.set_input_defaults(
            'fixed_mass.augmentation.rho_battery', 0.5, units='kW*h/kg'
        )
        self.prob.model.set_input_defaults(
            'fixed_mass.augmentation.motor_spec_mass', 4, units='hp/lbm'
        )
        self.prob.model.set_input_defaults(
            'fixed_mass.augmentation.inverter_spec_mass', 12, units='kW/kg'
        )
        self.prob.model.set_input_defaults(
            'fixed_mass.augmentation.TMS_spec_mass', 0.125, units='lbm/kW'
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=0.0, units='ft')
        self.prob.model.set_input_defaults(Mission.Design.MACH, val=0.8, units='unitless')
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, val=0.15)
        self.prob.model.set_input_defaults(Aircraft.Wing.FLAP_CHORD_RATIO, val=0.3)
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, val=0.9)

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()

        tol = 5e-4
        # size values:
        assert_near_equal(self.prob['size.fuselage.cabin_height'], 13.1, tol)
        assert_near_equal(self.prob['size.fuselage.cabin_len'], 93.9, tol)
        assert_near_equal(self.prob['size.fuselage.nose_height'], 8.6, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 13.97, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 13.53, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1068, tol
        )  # (printed out from GASP code to get better precision)

        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 9.644, tol
        )  # (printed out from GASP code)
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.AVERAGE_CHORD], 20.618, tol
        )  # (printed out from GASP code)
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 14.56, tol)

        # fixed mass values:
        assert_near_equal(
            self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS], 4786.2, tol
        )  # (printed out from GASP code)

        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 13034.0, tol)
        assert_near_equal(self.prob[Aircraft.Engine.ADDITIONAL_MASS], 2124.5 / 2, tol)

        # wing values:
        assert_near_equal(self.prob['wing_mass.isolated_wing_mass'], 15895, tol)
        assert_near_equal(self.prob[Aircraft.Wing.MASS], 20461.7, tol)

        # fuel values:
        assert_near_equal(
            self.prob['fuel_mass.fuel_and_oem.OEM_wingfuel_mass'], 64594.0, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 63707.6

        assert_near_equal(
            self.prob['fuel_mass.fus_mass_full'], 108803.9, tol
        )  # (printed out from GASP code),  #modified from GASP value to account for updated crew mass. GASP value is 108754.4
        assert_near_equal(
            self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 2027.6, 0.00055
        )  # slightly above tol, due to non-integer number of wires,  #modified from GASP value to account for updated crew mass. GASP value is 1974.5

        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 49582, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuselage.MASS], 16313.0, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 16436.0

        assert_near_equal(
            self.prob[Mission.Design.FUEL_MASS_REQUIRED], 33794.0, 0.00058
        )  # slightly above tol, due to non-integer number of wires,  #modified from GASP value to account for updated crew mass. GASP value is 32907.6
        assert_near_equal(
            self.prob[Aircraft.Propulsion.MASS], 26565.2, 0.00054
        )  # slightly above tol, due to non-integer number of wires,  #modified from GASP value to account for updated crew mass. GASP value is 26527.0
        assert_near_equal(
            self.prob[Mission.Design.FUEL_MASS], 33794.0, 0.00056
        )  # slightly above tol, due to non-integer number of wires,  #modified from GASP value to account for updated crew mass. GASP value is 32908
        assert_near_equal(
            self.prob['fuel_mass.fuel_mass_min'], 17824.0, 0.0012
        )  # slightly above tol, due to non-integer number of wires,  #modified from GASP value to account for updated crew mass. GASP value is 16937.6
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 675.58, 0.00051
        )  # slightly above tol, due to non-integer number of wires,  #modified from GASP value to account for updated crew mass. GASP value is 657.9
        assert_near_equal(
            self.prob['fuel_mass.fuel_and_oem.OEM_fuel_vol'], 1291.31, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1273.6
        assert_near_equal(
            self.prob[Aircraft.Design.OPERATING_MASS], 101506.0, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 102392.0
        assert_near_equal(
            self.prob['fuel_mass.fuel_and_oem.payload_mass_max_fuel'], 30800.0, tol
        )  # (printed out from GASP code)
        assert_near_equal(self.prob['fuel_mass.fuel_and_oem.volume_wingfuel_mass'], 35042.1, tol)
        assert_near_equal(self.prob['fuel_mass.max_wingfuel_mass'], 35042.1, tol)
        assert_near_equal(self.prob[Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY], 0, tol)
        assert_near_equal(self.prob['fuel_mass.body_tank.extra_fuel_volume'], 0, tol)
        assert_near_equal(self.prob['fuel_mass.body_tank.max_extra_fuel_mass'], 0, tol)

        assert_near_equal(self.prob[Aircraft.Electrical.HYBRID_CABLE_LENGTH], 65.6, tol)
        assert_near_equal(
            self.prob['fixed_mass.aug_mass'], 9394.3, 0.0017
        )  # slightly above tol, due to non-integer number of wires

        partial_data = self.prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=3e-9, rtol=6e-11)


if __name__ == '__main__':
    unittest.main()

import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

from aviary.subsystems.propulsion.propulsion_builder import CorePropulsionBuilder
from aviary.subsystems.geometry.geometry_builder import CoreGeometryBuilder
from aviary.subsystems.mass.mass_builder import CoreMassBuilder
from aviary.subsystems.aerodynamics.aerodynamics_builder import CoreAerodynamicsBuilder
from aviary.subsystems.premission import CorePreMission
from aviary.utils.aviary_values import get_items, get_keys
from aviary.validation_cases.validation_tests import (
    get_flops_case_names, get_flops_inputs, get_flops_outputs
)
from aviary.variable_info.variables import Aircraft, Mission
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData
from aviary.models.large_single_aisle_1.V3_bug_fixed_IO import V3_bug_fixed_options, V3_bug_fixed_non_metadata
from aviary.utils.functions import set_aviary_initial_values
from aviary.utils.preprocessors import preprocess_options
from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.functions import setup_model_options
from aviary.subsystems.propulsion.utils import build_engine_deck


FLOPS = LegacyCode.FLOPS
GASP = LegacyCode.GASP

data_sets = get_flops_case_names()


def setup_options(priority_data, backfill_data):
    priority_keys = get_keys(priority_data)
    full_data = priority_data.deepcopy()

    # add options to priority data that exist in backfill data but not priorirty
    for (key, (val, units)) in get_items(backfill_data):
        if key not in priority_keys:
            (new_val, new_unit) = backfill_data.get_item(key)
            full_data.set_val(key, val=new_val, units=new_unit)

    return full_data


class PreMissionTestCase(unittest.TestCase):

    def setUp(self):

        # set up inputs such that GASP inputs take priority

        case_name = 'LargeSingleAisle1FLOPS'

        flops_inputs = get_flops_inputs(case_name)
        flops_outputs = get_flops_outputs(case_name)

        FLOPS_input = flops_inputs
        GASP_input = V3_bug_fixed_options

        self.prob = om.Problem()

        input_options = setup_options(GASP_input, FLOPS_input)

        # delete the options that would override values
        input_options.delete(Aircraft.Wing.AREA)
        input_options.delete(Aircraft.Fuselage.LENGTH)
        input_options.delete(Aircraft.Wing.SPAN)
        input_options.delete(Aircraft.Nacelle.AVG_DIAMETER)
        input_options.delete(Aircraft.Fuselage.WETTED_AREA)
        input_options.delete(Aircraft.Wing.ULTIMATE_LOAD_FACTOR)
        input_options.delete(Aircraft.Fuel.TOTAL_CAPACITY)
        input_options.delete(Aircraft.Nacelle.AVG_LENGTH)

        engine = build_engine_deck(input_options)

        prop = CorePropulsionBuilder('core_propulsion', BaseMetaData, engine)
        mass = CoreMassBuilder('core_mass', BaseMetaData, GASP)
        aero = CoreAerodynamicsBuilder('core_aerodynamics', BaseMetaData, FLOPS)
        geom = CoreGeometryBuilder('core_geometry',
                                   BaseMetaData,
                                   use_both_geometries=True,
                                   code_origin_to_prioritize=GASP)

        core_subsystems = [prop, geom, mass, aero]

        self.prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(aviary_options=input_options,
                           subsystems=core_subsystems),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        # set defaults for all the vars
        val, units = input_options.get_item(Aircraft.Engine.SCALED_SLS_THRUST)
        self.prob.model.pre_mission.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=val, units=units)

        for (key, (val, units)) in get_items(GASP_input):
            try:
                if not BaseMetaData[key]['option']:
                    self.prob.model.set_input_defaults(key, val, units)
            except KeyError:
                continue

        for (key, (val, units)) in get_items(V3_bug_fixed_non_metadata):
            self.prob.model.set_input_defaults(key, val=val, units=units)

        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=1.0, units='ft')
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_SPAN_RATIO, val=0.9)
        self.prob.model.set_input_defaults(Aircraft.Wing.SLAT_CHORD_RATIO, val=.15)

        setup_model_options(self.prob, input_options)

        self.prob.setup(check=False, force_alloc_complex=True)

        self.prob.set_solver_print(2)

        # Initial guess for gross mass.
        # We set it to an unconverged value to test convergence.
        self.prob.set_val(Mission.Design.GROSS_MASS, val=1000.0)

        # Set inital values for all variables.
        set_aviary_initial_values(self.prob, input_options)

        # Adjust WETTED_AREA_SCALER such that WETTED_AREA = 4000.0
        self.prob.set_val(
            Aircraft.Fuselage.WETTED_AREA_SCALER, val=0.86215, units="unitless")

    def test_GASP_mass_FLOPS_everything_else(self):
        self.prob.run_model()

        # check the outputs from GASP mass and geometry (FLOPS outputs are not tested)

        tol = 5e-4
        # size values:
        assert_near_equal(self.prob["gasp_based_geom.fuselage.cabin_height"], 13.1, tol)
        assert_near_equal(
            self.prob["gasp_based_geom.fuselage.cabin_len"], 72.09722222222223, tol)
        assert_near_equal(self.prob["gasp_based_geom.fuselage.nose_height"], 8.6, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 17.63, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 16.54, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1397, tol
        )  # not exact GASP value from the output file, likely due to rounding error

        assert_near_equal(
            self.prob[Aircraft.HorizontalTail.AVERAGE_CHORD], 9.6509873673743, tol
        )
        assert_near_equal(
            self.prob[Aircraft.VerticalTail.AVERAGE_CHORD], 16.96457870166355, tol
        )
        assert_near_equal(self.prob[Aircraft.Nacelle.AVG_LENGTH], 14.7, tol)

        # fixed mass values:
        assert_near_equal(
            self.prob[Aircraft.LandingGear.MAIN_GEAR_MASS], 6384.35, tol)
        assert_near_equal(
            self.prob["fixed_mass.tail.loc_MAC_vtail"], 0.44959578484694906, tol
        )

        # wing values:
        assert_near_equal(
            self.prob["wing_mass.isolated_wing_mass"], 16205, tol)
        assert_near_equal(self.prob[Aircraft.Propulsion.TOTAL_ENGINE_MASS], 12606, tol)
        assert_near_equal(self.prob[Aircraft.Engine.ADDITIONAL_MASS], 1765/2, tol)

        # fuel values:
        # modified from GASP value to account for updated crew mass. GASP value is
        # 78843.6
        assert_near_equal(
            self.prob["fuel_mass.fuel_and_oem.OEM_wingfuel_mass"], 78145.0, tol
        )
        # modified from GASP value to account for updated crew mass. GASP value is
        # 102408.05695930264
        assert_near_equal(
            self.prob["fuel_mass.fus_mass_full"], 102663.6529626, tol
        )
        # modified from GASP value to account for updated crew mass. GASP value is
        # 1757
        assert_near_equal(
            self.prob[Aircraft.Fuel.FUEL_SYSTEM_MASS], 1727.9, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1757
        assert_near_equal(self.prob[Aircraft.Design.STRUCTURE_MASS], 50931.4, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuselage.MASS], 18833.76678366, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 18814

        # modified from GASP value to account for updated crew mass. GASP value is
        # 42843.6
        assert_near_equal(
            self.prob[Mission.Design.FUEL_MASS_REQUIRED], 42145.0, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 42843.6
        assert_near_equal(
            self.prob[Aircraft.Propulsion.MASS], 16098.7, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 16127
        assert_near_equal(
            self.prob[Mission.Design.FUEL_MASS], 42145.0, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 42844.0
        assert_near_equal(
            self.prob["fuel_mass.fuel_mass_min"], 32105.0, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 32803.6
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_DESIGN], 842.53, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 856.4910800459031
        assert_near_equal(
            self.prob["fuel_mass.fuel_and_oem.OEM_fuel_vol"], 1562.21, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 1576.1710061411081
        assert_near_equal(
            self.prob[Aircraft.Design.OPERATING_MASS], 97255.0, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 96556.0
        # extra_fuel_mass calculated differently in this version, so test for fuel_mass.fuel_and_oem.payload_mass_max_fuel not included
        assert_near_equal(
            self.prob["fuel_mass.fuel_and_oem.volume_wingfuel_mass"], 57066.3, tol
        )
        assert_near_equal(
            self.prob["fuel_mass.max_wingfuel_mass"], 57066.3, tol)

        # This is not in the model because it has been overridden, but is not an
        # input to any other component in the GASP premission model.
        err_text = '\'<model> <class Group>: Variable "aircraft:fuel:auxiliary_fuel_capacity" not found.\''
        with self.assertRaises(KeyError) as cm:
            self.prob.get_val(Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY)
        self.assertEqual(str(cm.exception), err_text)

        assert_near_equal(
            self.prob["fuel_mass.body_tank.extra_fuel_volume"], 0, tol
        )  # always zero when no body tank
        assert_near_equal(
            self.prob["fuel_mass.body_tank.max_extra_fuel_mass"], 0, tol
        )  # always zero when no body tank

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=3e-10, rtol=1e-12)

    def test_manual_override(self):

        # Problem in setup is GASP prioritized, so shared inputs for FLOPS will be manually overriden.

        outs = self.prob.model.pre_mission.list_outputs(
            includes='*gasp*fuselage:avg_diam*', prom_name=True, out_stream=None)

        self.assertTrue(
            outs[0][0] == f'core_geometry.gasp_based_geom.fuselage.parameters.{Aircraft.Fuselage.AVG_DIAMETER}')
        self.assertTrue('MANUAL_OVERRIDE' not in outs[0][1]['prom_name'])

        outs = self.prob.model.pre_mission.list_outputs(
            includes='*flops*fuselage:avg_diam*', prom_name=True, out_stream=None)

        self.assertTrue(
            outs[0][0] == f'core_geometry.flops_based_geom.fuselage_prelim.{Aircraft.Fuselage.AVG_DIAMETER}')
        self.assertTrue('MANUAL_OVERRIDE' in outs[0][1]['prom_name'])

        outs = self.prob.model.pre_mission.list_outputs(
            includes='*gasp*fuselage:wetted_area*', prom_name=True, out_stream=None)

        self.assertTrue(
            outs[0][0] == f'core_geometry.gasp_based_geom.fuselage.size.{Aircraft.Fuselage.WETTED_AREA}')
        self.assertTrue('MANUAL_OVERRIDE' not in outs[0][1]['prom_name'])

        outs = self.prob.model.pre_mission.list_outputs(
            includes='*flops*fuselage:wetted_area*', prom_name=True, out_stream=None)

        self.assertTrue(
            outs[0][0] == f'core_geometry.flops_based_geom.fuselage.{Aircraft.Fuselage.WETTED_AREA}')
        self.assertTrue('MANUAL_OVERRIDE' in outs[0][1]['prom_name'])

        # Setup FLOPS prioritized problem.

        case_name = 'LargeSingleAisle1FLOPS'

        flops_inputs = get_flops_inputs(case_name)
        flops_outputs = get_flops_outputs(case_name)

        FLOPS_input = flops_inputs
        GASP_input = V3_bug_fixed_options

        prob = om.Problem()

        aviary_inputs = setup_options(GASP_input, FLOPS_input)

        aviary_inputs.delete(Aircraft.Fuselage.WETTED_AREA)
        engine = build_engine_deck(aviary_inputs)
        preprocess_options(aviary_inputs, engine_models=engine)

        prob = om.Problem()
        model = prob.model

        prop = CorePropulsionBuilder('core_propulsion', BaseMetaData, engine)
        mass = CoreMassBuilder('core_mass', BaseMetaData, GASP)
        aero = CoreAerodynamicsBuilder('core_aerodynamics', BaseMetaData, FLOPS)
        geom = CoreGeometryBuilder('core_geometry',
                                   BaseMetaData,
                                   use_both_geometries=True,
                                   code_origin_to_prioritize=FLOPS)

        core_subsystems = [prop, geom, mass, aero]

        model.add_subsystem(
            'pre_mission',
            CorePreMission(aviary_options=aviary_inputs,
                           subsystems=core_subsystems),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        val, units = aviary_inputs.get_item(Aircraft.Engine.SCALED_SLS_THRUST)
        prob.model.pre_mission.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=val, units=units)

        for (key, (val, units)) in get_items(GASP_input):
            try:
                if not BaseMetaData[key]['option']:
                    prob.model.set_input_defaults(key, val, units)
            except KeyError:
                continue

        for (key, (val, units)) in get_items(V3_bug_fixed_non_metadata):
            prob.model.set_input_defaults(key, val=val, units=units)

        setup_model_options(prob, aviary_inputs)

        prob.setup()

        # Problem in setup is FLOPS prioritized, so shared inputs for FLOPS will be manually overriden.

        outs = prob.model.pre_mission.list_outputs(
            includes='*gasp*fuselage:avg_diam*', prom_name=True, out_stream=None)

        self.assertTrue(
            outs[0][0] == f'core_geometry.gasp_based_geom.fuselage.parameters.{Aircraft.Fuselage.AVG_DIAMETER}')
        self.assertTrue('MANUAL_OVERRIDE' in outs[0][1]['prom_name'])

        outs = prob.model.pre_mission.list_outputs(
            includes='*flops*fuselage:avg_diam*', prom_name=True, out_stream=None)

        self.assertTrue(
            outs[0][0] == f'core_geometry.flops_based_geom.fuselage_prelim.{Aircraft.Fuselage.AVG_DIAMETER}')
        self.assertTrue('MANUAL_OVERRIDE' not in outs[0][1]['prom_name'])

        outs = prob.model.pre_mission.list_outputs(
            includes='*gasp*fuselage:wetted_area*', prom_name=True, out_stream=None)

        self.assertTrue(
            outs[0][0] == f'core_geometry.gasp_based_geom.fuselage.size.{Aircraft.Fuselage.WETTED_AREA}')
        self.assertTrue('MANUAL_OVERRIDE' in outs[0][1]['prom_name'])

        outs = prob.model.pre_mission.list_outputs(
            includes='*flops*fuselage:wetted_area*', prom_name=True, out_stream=None)

        self.assertTrue(
            outs[0][0] == f'core_geometry.flops_based_geom.fuselage.{Aircraft.Fuselage.WETTED_AREA}')
        self.assertTrue('MANUAL_OVERRIDE' not in outs[0][1]['prom_name'])


if __name__ == "__main__":
    unittest.main()
    # test = PreMissionTestCase()
    # test.setUp()
    # test.test_manual_override()
    # test.test_GASP_mass_FLOPS_everything_else()

"""
Test the conversion between phase info and phase builder to ensure
consistency and correctness.
"""

import unittest
from copy import deepcopy

from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.interface.default_phase_info.height_energy import phase_info as ph_in_height_energy
from aviary.interface.default_phase_info.height_energy import (
    phase_info_parameterization as phase_info_parameterization_height_energy,
)
from aviary.interface.default_phase_info.two_dof import phase_info as ph_in_two_dof
from aviary.interface.default_phase_info.two_dof import (
    phase_info_parameterization as phase_info_parameterization_two_dof,
)
from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.mission.phase_builder_base import PhaseBuilderBase as PhaseBuilder
from aviary.mission.phase_builder_base import phase_info_to_builder
from aviary.variable_info.variables import Mission


class TestPhaseInfo(unittest.TestCase):
    def _test_phase_info_dict(self, phase_info_dict, name):
        """Helper method to test a given phase_info dict."""
        _climb_info = (name, phase_info_dict[name])

        # Removing the 'fix_duration' key from user_options for test comparison
        _climb_info[1]['user_options'].pop('fix_duration')

        # Convert phase info to a phase builder
        _phase_builder: PhaseBuilder = phase_info_to_builder(*_climb_info)

        # Convert back the phase builder to phase info
        _phase_builder_info = _phase_builder.to_phase_info()

        if _climb_info != _phase_builder_info:
            lhs_name, lhs_info = _climb_info
            rhs_name, rhs_info = _phase_builder_info

            if lhs_name != rhs_name:
                raise RuntimeError(f'name mismatch: {lhs_name} != {rhs_name}')

            lhs_keys = set(sorted(set(lhs_info.keys())))
            rhs_keys = set(sorted(set(rhs_info.keys())))

            common = lhs_keys & rhs_keys
            lhs_unique = lhs_keys - common

            if lhs_keys != rhs_keys:
                if not common:
                    raise RuntimeError(f'key mismatch: no keys in common: {lhs_keys} != {rhs_keys}')
                if lhs_unique:
                    raise RuntimeError(
                        f'key mismatch: the new builder is missing the following keys: {lhs_unique}'
                    )

            # Loop through each key to compare values
            for key in lhs_keys:
                lhs_value = lhs_info[key]
                rhs_value = rhs_info[key]

                if lhs_value != rhs_value:
                    if key in ['user_options', 'initial_guesses']:
                        for name in lhs_value:
                            lhs_option = lhs_value[name]
                            rhs_option = rhs_value[name]

                            # Support for more compact format for unitless vars.
                            if lhs_option[1] == 'unitless' and lhs_option[0] == rhs_option:
                                continue

                            if lhs_option != rhs_option:
                                raise RuntimeError(
                                    f'value mismatch ({key}[{name}]): {lhs_option} != {rhs_option}'
                                )
                    else:
                        raise RuntimeError(f'value mismatch ({key}): {lhs_value} != {rhs_value}')

    def test_default_phase_height_energy(self):
        """Tests the roundtrip conversion for default_phase_info.height_energy."""
        from aviary.interface.default_phase_info.height_energy import phase_info

        local_phase_info = deepcopy(phase_info)
        self._test_phase_info_dict(local_phase_info, 'cruise')


@use_tempdirs
class TestParameterizePhaseInfo(unittest.TestCase):
    def test_phase_info_parameterization_two_dof(self):
        phase_info = deepcopy(ph_in_two_dof)

        prob = AviaryProblem()

        csv_path = 'models/small_single_aisle/small_single_aisle_GASP.csv'

        prob.load_inputs(csv_path, phase_info)
        prob.check_and_preprocess_inputs()

        # We can set some crazy vals, since we aren't going to optimize.
        prob.aviary_inputs.set_val(Mission.Design.RANGE, 5000, 'km')
        prob.aviary_inputs.set_val(Mission.Design.CRUISE_ALTITUDE, 31000, units='ft')
        prob.aviary_inputs.set_val(Mission.Design.GROSS_MASS, 120000, 'lbm')
        prob.aviary_inputs.set_val(Mission.Design.MACH, 0.6, 'unitless')

        prob.add_pre_mission_systems()
        prob.add_phases(phase_info_parameterization=phase_info_parameterization_two_dof)
        prob.add_post_mission_systems()

        prob.link_phases()

        prob.setup()
        prob.set_initial_guesses()

        prob.run_model()

        assert_near_equal(
            prob.get_val('traj.desc2.timeseries.input_values:distance', units='km')[-1], 5000.0
        )
        assert_near_equal(
            prob.get_val('traj.climb2.timeseries.input_values:altitude', units='ft')[-1], 31000.0
        )
        assert_near_equal(
            prob.get_val('traj.groundroll.timeseries.input_values:mass', units='lbm')[0], 120000.0
        )
        assert_near_equal(prob.get_val('traj.cruise.rhs.mach')[0], 0.6)

    def test_phase_info_parameterization_height_energy(self):
        phase_info = deepcopy(ph_in_height_energy)

        prob = AviaryProblem()

        csv_path = 'models/test_aircraft/aircraft_for_bench_FwFm.csv'

        prob.load_inputs(csv_path, phase_info)
        prob.check_and_preprocess_inputs()

        # We can set some crazy vals, since we aren't going to optimize.
        prob.aviary_inputs.set_val(Mission.Design.RANGE, 5000.0, 'km')
        prob.aviary_inputs.set_val(Mission.Design.CRUISE_ALTITUDE, 31000.0, units='ft')
        prob.aviary_inputs.set_val(Mission.Design.GROSS_MASS, 195000.0, 'lbm')
        prob.aviary_inputs.set_val(Mission.Summary.CRUISE_MACH, 0.6, 'unitless')

        prob.add_pre_mission_systems()
        prob.add_phases(phase_info_parameterization=phase_info_parameterization_height_energy)
        prob.add_post_mission_systems()

        prob.link_phases()

        prob.setup()
        prob.set_initial_guesses()

        prob.run_model()

        range_resid = prob.get_val(Mission.Constraints.RANGE_RESIDUAL, units='nmi')[-1]
        assert_near_equal(range_resid, 1906, tolerance=1e-3)
        assert_near_equal(prob.get_val('traj.cruise.timeseries.altitude', units='ft')[0], 31000.0)
        assert_near_equal(prob.get_val('traj.cruise.timeseries.mach')[0], 0.6)


# To run the tests
if __name__ == '__main__':
    unittest.main()
    # test = TestParameterizePhaseInfo()
    # test.test_phase_info_parameterization_two_dof()

from packaging import version
import unittest

import numpy as np

import openmdao
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from aviary.mission.gasp_based.ode.params import set_params_for_unit_tests
from aviary.mission.gasp_based.phases.landing_group import LandingSegment
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.test_utils.default_subsystems import get_default_mission_subsystems
from aviary.utils.test_utils.IO_test_util import check_prob_outputs
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Dynamic, Mission

# TODO - Why does this test landing instead of taxi?


class DLandTestCase(unittest.TestCase):
    """
    TODO: docstring
    """

    def setUp(self):

        self.prob = om.Problem()

        options = get_option_defaults()
        default_mission_subsystems = get_default_mission_subsystems(
            'GASP', build_engine_deck(options))

        self.prob.model = LandingSegment(
            aviary_options=options, core_subsystems=default_mission_subsystems)

    @unittest.skipIf(version.parse(openmdao.__version__) < version.parse("3.26"), "Skipping due to OpenMDAO version being too low (<3.26)")
    def test_dland(self):
        self.prob.setup(check=False, force_alloc_complex=True)

        set_params_for_unit_tests(self.prob)

        self.prob.set_val(Mission.Landing.AIRPORT_ALTITUDE, 0, units="ft")
        self.prob.set_val(Mission.Landing.INITIAL_MACH, 0.1, units="unitless")
        self.prob.set_val("alpha", 0, units="deg")  # doesn't matter
        self.prob.set_val(Mission.Landing.MAXIMUM_SINK_RATE, 900, units="ft/min")
        self.prob.set_val(Mission.Landing.GLIDE_TO_STALL_RATIO, 1.3, units="unitless")
        self.prob.set_val(Mission.Landing.MAXIMUM_FLARE_LOAD_FACTOR,
                          1.15, units="unitless")
        self.prob.set_val(Mission.Landing.TOUCHDOWN_SINK_RATE, 5, units="ft/s")
        self.prob.set_val(Mission.Landing.BRAKING_DELAY, 1, units="s")
        self.prob.set_val("mass", 165279, units="lbm")
        self.prob.set_val(Dynamic.Mission.THROTTLE, 0.0, units='unitless')

        self.prob.run_model()

        testvals = {
            Mission.Landing.INITIAL_VELOCITY: 240.9179994,  # ft/s (142.74 knot)
            "TAS_touchdown": 213.1197687,  # ft/s (126.27 knot)
            "theta": 0.06230825,  # rad (3.57 deg)
            "flare_alt": 20.8,
            "ground_roll_distance": 1798,
            Mission.Landing.GROUND_DISTANCE: 2980,
            "CL_max": 2.9533,
        }
        check_prob_outputs(self.prob, testvals, rtol=1e-2)

        partial_data = self.prob.check_partials(
            out_stream=None, method="cs", excludes=["*params*", "*aero*"]
        )
        assert_check_partials(partial_data, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()

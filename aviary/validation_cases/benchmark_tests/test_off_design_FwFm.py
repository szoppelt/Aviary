import unittest
import aviary.api as av

from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs
from openmdao.utils.assert_utils import assert_near_equal

from aviary.interface.default_phase_info.height_energy import phase_info_parameterization
from aviary.variable_info.enums import ProblemType, Verbosity


class HeightEnergyTestCase(unittest.TestCase):
    """
    Setup basic aircraft mass and range and select climb, cruise, and descent phases for simulation. 
    """

    def setUp(self) -> None:
        self.sized_mass = 175871.04745399
        self.sized_range = 3375
        phase_info = {
            "pre_mission": {"include_takeoff": True, "optimize_mass": True},
            "climb": {
                "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
                "user_options": {
                    'fix_initial': False,
                    'input_initial': True,
                    "optimize_mach": True,
                    "optimize_altitude": True,
                    "use_polynomial_control": False,
                    "num_segments": 6,
                    "order": 3,
                    "solve_for_distance": False,
                    "initial_mach": (0.3, "unitless"),
                    "final_mach": (0.79, "unitless"),
                    "mach_bounds": ((0.1, 0.8), "unitless"),
                    "initial_altitude": (35., "ft"),
                    "final_altitude": (35000.0, "ft"),
                    "altitude_bounds": ((0.0, 35000.0), "ft"),
                    "throttle_enforcement": "path_constraint",
                    "constrain_final": False,
                    "fix_duration": False,
                    "initial_bounds": ((0.0, 2.0), "min"),
                    "duration_bounds": ((5.0, 50.0), "min"),
                    "no_descent": False,
                    "add_initial_mass_constraint": False,
                },
                "initial_guesses": {"time": ([0, 40.0], "min")},
            },
            "cruise": {
                "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
                "user_options": {
                    "optimize_mach": True,
                    "optimize_altitude": True,
                    "polynomial_control_order": 1,
                    "use_polynomial_control": True,
                    "num_segments": 1,
                    "order": 3,
                    "solve_for_distance": False,
                    "initial_mach": (0.79, "unitless"),
                    "final_mach": (0.79, "unitless"),
                    "mach_bounds": ((0.79, 0.79), "unitless"),
                    "initial_altitude": (35000.0, "ft"),
                    "final_altitude": (35000.0, "ft"),
                    "altitude_bounds": ((35000.0, 35000.0), "ft"),
                    "throttle_enforcement": "boundary_constraint",
                    "fix_initial": False,
                    "constrain_final": False,
                    "fix_duration": False,
                    "initial_bounds": ((64.0, 192.0), "min"),
                    "duration_bounds": ((60.0, 720.0), "min"),
                },
                "initial_guesses": {"time": ([128, 113], "min")},
            },
            "descent": {
                "subsystem_options": {"core_aerodynamics": {"method": "computed"}},
                "user_options": {
                    "optimize_mach": True,
                    "optimize_altitude": True,
                    "use_polynomial_control": False,
                    "num_segments": 5,
                    "order": 3,
                    "solve_for_distance": False,
                    "initial_mach": (0.79, "unitless"),
                    "final_mach": (0.3, "unitless"),
                    "mach_bounds": ((0.2, 0.8), "unitless"),
                    "initial_altitude": (35000.0, "ft"),
                    "final_altitude": (35.0, "ft"),
                    "altitude_bounds": ((0.0, 35000.0), "ft"),
                    "throttle_enforcement": "path_constraint",
                    "fix_initial": False,
                    "constrain_final": True,
                    "fix_duration": False,
                    "initial_bounds": ((120., 800.), "min"),
                    "duration_bounds": ((5.0, 35.0), "min"),
                    "no_climb": True,
                },
                "initial_guesses": {"time": ([241, 30], "min")},
            },
            "post_mission": {
                "include_landing": True,
                "constrain_range": True,
                "target_range": (3375.0, "nmi"),
            },
        }

        self.phase_info = phase_info


@use_tempdirs
class TestOffDesign(HeightEnergyTestCase):
    """
    Build the model using a large single aisle commercial transport aircraft data using
    FLOPS mass method and HEIGHT_ENERGY mission method. Run a fallout mission to test off design.
    """

    @require_pyoptsparse(optimizer="IPOPT")
    def test_off_design_IPOPT(self):
        # Fallout Mission
        prob_fallout = av.AviaryProblem()
        prob_fallout.load_inputs('models/test_aircraft/aircraft_for_bench_FwFm.csv',
                                 self.phase_info, verbosity=Verbosity.QUIET)

        prob_fallout.problem_type = ProblemType.FALLOUT
        prob_fallout.aviary_inputs.set_val('problem_type', ProblemType.FALLOUT,
                                           units='unitless')
        prob_fallout.aviary_inputs.set_val(
            'mission:design:gross_mass', self.sized_mass, units='lbm')
        prob_fallout.aviary_inputs.set_val(
            'mission:summary:gross_mass', self.sized_mass, units='lbm')

        prob_fallout.check_and_preprocess_inputs()
        prob_fallout.add_pre_mission_systems()
        prob_fallout.add_phases(
            phase_info_parameterization=phase_info_parameterization)
        prob_fallout.add_post_mission_systems()
        prob_fallout.link_phases()
        prob_fallout.add_driver('IPOPT', max_iter=100)
        prob_fallout.add_design_variables()
        prob_fallout.add_objective()
        prob_fallout.setup()
        prob_fallout.set_initial_guesses()
        prob_fallout.run_aviary_problem()

        # Alternate Mission
        prob_alternate = av.AviaryProblem()
        prob_alternate.load_inputs(
            'models/test_aircraft/aircraft_for_bench_FwFm.csv', self.phase_info,
            verbosity=Verbosity.QUIET)
        prob_alternate.problem_type = ProblemType.ALTERNATE
        prob_alternate.aviary_inputs.set_val(
            'problem_type', ProblemType.ALTERNATE, units='unitless')

        prob_alternate.aviary_inputs.set_val(
            'mission:design:gross_mass', self.sized_mass, units='lbm')
        prob_alternate.aviary_inputs.set_val(
            'mission:summary:gross_mass', self.sized_mass, units='lbm')

        prob_alternate.check_and_preprocess_inputs()
        prob_alternate.add_pre_mission_systems()
        prob_alternate.add_phases(
            phase_info_parameterization=phase_info_parameterization)
        prob_alternate.add_post_mission_systems()
        prob_alternate.link_phases()
        prob_alternate.add_driver('IPOPT', max_iter=100)
        prob_alternate.add_design_variables()
        prob_alternate.add_objective()
        prob_alternate.setup()
        prob_alternate.set_initial_guesses()
        prob_alternate.run_aviary_problem()

        fallout_range = prob_fallout.get_val(av.Mission.Summary.RANGE)
        alternate_mass = prob_alternate.get_val(av.Mission.Summary.GROSS_MASS)
        assert_near_equal(fallout_range, self.sized_range, tolerance=0.02)
        assert_near_equal(alternate_mass, self.sized_mass, tolerance=0.02)

    @require_pyoptsparse(optimizer='SNOPT')
    def test_off_design_SNOPT(self):
        # Fallout Mission
        prob_fallout = av.AviaryProblem()
        prob_fallout.load_inputs('models/test_aircraft/aircraft_for_bench_FwFm.csv',
                                 self.phase_info, verbosity=Verbosity.QUIET)

        prob_fallout.problem_type = ProblemType.FALLOUT
        prob_fallout.aviary_inputs.set_val('problem_type', ProblemType.FALLOUT,
                                           units='unitless')
        prob_fallout.aviary_inputs.set_val(
            'mission:design:gross_mass', self.sized_mass, units='lbm')
        prob_fallout.aviary_inputs.set_val(
            'mission:summary:gross_mass', self.sized_mass, units='lbm')

        prob_fallout.check_and_preprocess_inputs()
        prob_fallout.add_pre_mission_systems()
        prob_fallout.add_phases(
            phase_info_parameterization=phase_info_parameterization)
        prob_fallout.add_post_mission_systems()
        prob_fallout.link_phases()
        prob_fallout.add_driver('SNOPT', max_iter=100)
        prob_fallout.add_design_variables()
        prob_fallout.add_objective()
        prob_fallout.setup()
        prob_fallout.set_initial_guesses()
        prob_fallout.run_aviary_problem()

        # Alternate Mission
        prob_alternate = av.AviaryProblem()
        prob_alternate.load_inputs(
            'models/test_aircraft/aircraft_for_bench_FwFm.csv', self.phase_info,
            verbosity=Verbosity.QUIET)
        prob_alternate.problem_type = ProblemType.ALTERNATE
        prob_alternate.aviary_inputs.set_val(
            'problem_type', ProblemType.ALTERNATE, units='unitless')

        prob_alternate.aviary_inputs.set_val(
            'mission:design:gross_mass', self.sized_mass, units='lbm')
        prob_alternate.aviary_inputs.set_val(
            'mission:summary:gross_mass', self.sized_mass, units='lbm')

        prob_alternate.check_and_preprocess_inputs()
        prob_alternate.add_pre_mission_systems()
        prob_alternate.add_phases(
            phase_info_parameterization=phase_info_parameterization)
        prob_alternate.add_post_mission_systems()
        prob_alternate.link_phases()
        prob_alternate.add_driver('SNOPT', max_iter=100)
        prob_alternate.add_design_variables()
        prob_alternate.add_objective()
        prob_alternate.setup()
        prob_alternate.set_initial_guesses()
        prob_alternate.run_aviary_problem()

        fallout_range = prob_fallout.get_val(av.Mission.Summary.RANGE)
        alternate_mass = prob_alternate.get_val(av.Mission.Summary.GROSS_MASS)
        assert_near_equal(fallout_range, self.sized_range, tolerance=0.02)
        assert_near_equal(alternate_mass, self.sized_mass, tolerance=0.02)


if __name__ == '__main__':
    unittest.main()
    # test = TestOffDesign()
    # test.setUp()
    # test.test_off_design_SNOPT()
    # test.test_off_design_IPOPT()

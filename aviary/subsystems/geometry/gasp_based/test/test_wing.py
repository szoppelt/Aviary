import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.geometry.gasp_based.wing import (WingFold, WingGroup,
                                                        WingParameters,
                                                        WingSize)
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Mission


class WingSizeTestCase1(
    unittest.TestCase
):  # actual GASP test case, input and output values based on large single aisle 1 v3 without bug fix
    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem("size", WingSize(), promotes=["*"])

        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, 175400, units="lbm"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, 128, units="lbf/ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ASPECT_RATIO, 10.13, units="unitless"
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 2e-4
        assert_near_equal(self.prob[Aircraft.Wing.AREA], 1370.3, tol)
        assert_near_equal(self.prob[Aircraft.Wing.SPAN], 117.8, tol)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class WingSizeTestCase2(unittest.TestCase):
    """
    Test mass-weight conversion
    """

    def setUp(self):
        import aviary.subsystems.geometry.gasp_based.wing as wing
        wing.GRAV_ENGLISH_LBM = 1.1

    def tearDown(self):
        import aviary.subsystems.geometry.gasp_based.wing as wing
        wing.GRAV_ENGLISH_LBM = 1.0

    def test_case1(self):
        self.prob = om.Problem()
        self.prob.model.add_subsystem("size", WingSize(), promotes=["*"])
        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, 175400, units="lbm"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, 128, units="lbf/ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ASPECT_RATIO, 10.13, units="unitless"
        )
        self.prob.setup(check=False, force_alloc_complex=True)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class WingParametersTestCase1(
    unittest.TestCase
):  # actual GASP test case, input and output values based on large single aisle 1 v3 without bug fix
    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "parameters", WingParameters(), promotes=["*"]
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, 1370.3, units="ft**2")
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, 117.8, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ASPECT_RATIO, 10.13, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.TAPER_RATIO, 0.33, units="unitless")
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, 25, units="deg")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.AVG_DIAMETER, 13.1, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units="unitless"
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 17.49, tol)
        assert_near_equal(self.prob[Aircraft.Wing.AVERAGE_CHORD], 12.615, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 16.41, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1397, tol
        )  # this is slightly different from the GASP output value, likely due to rounding error
        assert_near_equal(self.prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 1114, tol)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class WingParametersTestCase2(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "parameters", WingParameters(), promotes=["*"]
        )

        self.prob.model.set_input_defaults(Aircraft.Wing.AREA, 1370.3, units="ft**2")
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, 117.8, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.ASPECT_RATIO, 10.13, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.TAPER_RATIO, 0.33, units="unitless")
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, 25, units="deg")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.AVG_DIAMETER, 13.1, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units="unitless"
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 5e-4
        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 17.49, tol)
        assert_near_equal(self.prob[Aircraft.Wing.AVERAGE_CHORD], 12.615, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 16.41, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1397, tol
        )  # this is slightly different from the GASP output value, likely due to rounding error

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class WingFoldTestCase1(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')
        options.set_val(Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED,
                        val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group",
            WingFold(),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.TAPER_RATIO, 0.33, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units="unitless"
        )
        self.prob.model.set_input_defaults(
            "strut_y", val=25, units="ft"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1370.3, units="ft**2"
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units="unitless"
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4

        assert_near_equal(
            self.prob["nonfolded_taper_ratio"], 0.71561969, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Wing.FOLDING_AREA], 620.04352246, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["nonfolded_wing_area"], 750.25647754, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["tc_ratio_mean_folded"], 0.14363328, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["nonfolded_AR"], 3.33219382, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 712.3428037422319, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=1e-12, rtol=1e-12)


class WingFoldTestCase2(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED,
                        val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group",
            WingFold(),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.TAPER_RATIO, 0.33, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLDED_SPAN, val=25, units="ft"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1370.3, units="ft**2"
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units="unitless"
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4

        assert_near_equal(
            self.prob["nonfolded_taper_ratio"], 0.85780985, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Wing.FOLDING_AREA], 964.0812219, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["nonfolded_wing_area"], 406.2187781, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["tc_ratio_mean_folded"], 0.14681664, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["nonfolded_AR"], 1.53857978, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 406.64971668264957, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


class WingGroupTestCase1(
    unittest.TestCase
):  # actual GASP test case, input and output values based on large single aisle 1 v3 without bug fix
    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group", WingGroup(), promotes=["*"]
        )

        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, 175400, units="lbm"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, 128, units="lbf/ft**2"
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.ASPECT_RATIO, 10.13, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.TAPER_RATIO, 0.33, units="unitless")
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, 25, units="deg")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.AVG_DIAMETER, 13.1, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units="unitless"
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units="unitless"
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 5e-4

        assert_near_equal(self.prob[Aircraft.Wing.AREA], 1370.3, tol)
        assert_near_equal(self.prob[Aircraft.Wing.SPAN], 117.8, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 17.49, tol)
        assert_near_equal(self.prob[Aircraft.Wing.AVERAGE_CHORD], 12.615, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 16.41, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1397, tol
        )  # this is slightly different from the GASP output value, likely due to rounding error
        assert_near_equal(self.prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 1114, tol)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


class WingGroupTestCase2(unittest.TestCase):
    """
    Wing with both folds and struts which has fold dimensional location and strut dimensional location specified
    with the fold at the strut connection
    """

    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')
        options.set_val(Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED,
                        val=True, units='unitless')
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED,
                        val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group",
            WingGroup(),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, 175400, units="lbm"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, 128, units="lbf/ft**2"
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.ASPECT_RATIO, 10.13, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.TAPER_RATIO, 0.33, units="unitless")
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, 25, units="deg")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.AVG_DIAMETER, 13.1, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units="unitless"
        )

        self.prob.model.set_input_defaults(
            Aircraft.Strut.AREA_RATIO, val=0.02189, units="unitless"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Strut.ATTACHMENT_LOCATION, val=1.0, units="ft"
        )  # not actual GASP value

        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units="unitless"
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 5e-4

        assert_near_equal(self.prob[Aircraft.Wing.AREA], 1370.3, tol)
        assert_near_equal(self.prob[Aircraft.Wing.SPAN], 117.8, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 17.49, tol)
        assert_near_equal(self.prob[Aircraft.Wing.AVERAGE_CHORD], 12.615, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 16.41, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1397, tol
        )  # this is slightly different from the GASP output value, likely due to rounding error

        assert_near_equal(
            self.prob["fold.nonfolded_taper_ratio"], 0.9943133, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Wing.FOLDING_AREA], 1352.8724859, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["fold.nonfolded_wing_area"], 17.4400141, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["fold.tc_ratio_mean_folded"], 0.14987269, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["fold.nonfolded_AR"], 0.0573394, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 18.26837098, tol
        )  # not actual GASP value

        assert_near_equal(
            self.prob[Aircraft.Strut.LENGTH], 14.42957033, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Strut.CHORD], 1.03953199, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=7e-12, rtol=1e-12)


class WingGroupTestCase3(unittest.TestCase):
    """
    Wing with folds which has dimensional location specified
    """

    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED,
                        val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group",
            WingGroup(),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, 175400, units="lbm"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, 128, units="lbf/ft**2"
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.ASPECT_RATIO, 10.13, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.TAPER_RATIO, 0.33, units="unitless")
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, 25, units="deg")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.AVG_DIAMETER, 13.1, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLDED_SPAN, val=25, units="ft"
        )  # not actual GASP value

        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units="unitless"
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 5e-4

        assert_near_equal(self.prob[Aircraft.Wing.AREA], 1370.3, tol)
        assert_near_equal(self.prob[Aircraft.Wing.SPAN], 117.8, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 17.49, tol)
        assert_near_equal(self.prob[Aircraft.Wing.AVERAGE_CHORD], 12.615, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 16.41, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1397, tol
        )  # this is slightly different from the GASP output value, likely due to rounding error

        assert_near_equal(
            self.prob["fold.nonfolded_taper_ratio"], 0.85780985, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Wing.FOLDING_AREA], 964.14982163, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["fold.nonfolded_wing_area"], 406.16267837, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["fold.tc_ratio_mean_folded"], 0.14681715, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob["fold.nonfolded_AR"], 1.5387923, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 406.64971668264957, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


class WingGroupTestCase4(unittest.TestCase):
    """
    Wing with both folds and struts which has fold dimensional location and strut dimensional location specified
    """

    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_FOLD, val=True, units='unitless')
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        options.set_val(Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED,
                        val=True, units='unitless')
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED,
                        val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group",
            WingGroup(),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.FOLDED_SPAN, val=1, units="ft"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Strut.ATTACHMENT_LOCATION, val=0, units="ft"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Strut.AREA_RATIO, val=.2, units='unitless'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.AVG_DIAMETER, val=10.0, units='ft'
        )
        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=152000.0, units='lbm'
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, 128, units="lbf/ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, val=.6, units='unitless'
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.ASPECT_RATIO, val=10.13, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.TAPER_RATIO, val=0.33, units="unitless")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, val=0.11, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, val=0.1, units="unitless"
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        self.prob.run_model()
        tol = 5e-4

        assert_near_equal(self.prob[Aircraft.Wing.AREA], 1187.5, tol)
        assert_near_equal(self.prob[Aircraft.Wing.SPAN], 109.6785, tol)
        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 16.2814, tol)
        assert_near_equal(self.prob[Aircraft.Wing.AVERAGE_CHORD], 11.7430, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 15.4789, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1067, tol)
        assert_near_equal(self.prob["fold.nonfolded_taper_ratio"], 0.9939, tol)
        assert_near_equal(self.prob[Aircraft.Wing.FOLDING_AREA], 1171.2684, tol)
        assert_near_equal(self.prob["fold.nonfolded_wing_area"], 16.2316, tol)
        assert_near_equal(self.prob["fold.tc_ratio_mean_folded"], 0.10995, tol)
        assert_near_equal(self.prob["fold.nonfolded_AR"], 0.06161, tol)
        assert_near_equal(
            self.prob[Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX], 11.61131, tol)
        assert_near_equal(self.prob[Aircraft.Strut.LENGTH], 11.18034, tol)
        assert_near_equal(self.prob[Aircraft.Strut.CHORD], 10.62132, tol)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=3e-12, rtol=1e-13)


class WingGroupTestCase5(unittest.TestCase):
    """
    Wing with struts which has dimentional location specified
    """

    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Wing.HAS_STRUT, val=True, units='unitless')
        options.set_val(Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED,
                        val=True, units='unitless')
        options.set_val(Aircraft.Wing.CHOOSE_FOLD_LOCATION, val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group",
            WingGroup(),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, 175400, units="lbm"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.LOADING, 128, units="lbf/ft**2"
        )

        self.prob.model.set_input_defaults(
            Aircraft.Wing.ASPECT_RATIO, 10.13, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.TAPER_RATIO, 0.33, units="unitless")
        self.prob.model.set_input_defaults(Aircraft.Wing.SWEEP, 25, units="deg")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_ROOT, 0.15, units="unitless"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.AVG_DIAMETER, 13.1, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.THICKNESS_TO_CHORD_TIP, 0.12, units="unitless"
        )

        self.prob.model.set_input_defaults(
            Aircraft.Strut.AREA_RATIO, val=.2, units="unitless"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=150, units="ft**2"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Strut.ATTACHMENT_LOCATION, val=1.0, units="ft"
        )  # not actual GASP value
        self.prob.model.set_input_defaults(
            Aircraft.Strut.AREA_RATIO, val=0.021893, units="unitless"
        )

        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, 0.6, units="unitless"
        )

        setup_model_options(self.prob, options)

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 5e-4

        assert_near_equal(self.prob[Aircraft.Wing.AREA], 1370.3, tol)
        assert_near_equal(self.prob[Aircraft.Wing.SPAN], 117.8, tol)

        assert_near_equal(self.prob[Aircraft.Wing.CENTER_CHORD], 17.49, tol)
        assert_near_equal(self.prob[Aircraft.Wing.AVERAGE_CHORD], 12.615, tol)
        assert_near_equal(self.prob[Aircraft.Wing.ROOT_CHORD], 16.41, tol)
        assert_near_equal(
            self.prob[Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED], 0.1397, tol
        )  # this is slightly different from the GASP output value, likely due to rounding error

        assert_near_equal(
            self.prob[Aircraft.Strut.LENGTH], 14.42957033, tol
        )  # not actual GASP value
        assert_near_equal(
            self.prob[Aircraft.Strut.CHORD], 1.03953199, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=2e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()

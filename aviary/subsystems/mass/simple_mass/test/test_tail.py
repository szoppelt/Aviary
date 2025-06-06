import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

import numpy as np

from aviary.subsystems.mass.simple_mass.tail import TailMassAndCOG

class TailMassTestCase(unittest.TestCase):
    """
    Tail mass test case.

    """

    def setUp(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "Tail",
            TailMassAndCOG(),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )

        self.prob.model.set_input_defaults(
            "span_tail",
            val=1,
            units="m"
        )

        self.prob.model.set_input_defaults(
            "root_chord_tail",
            val=1,
            units="m"
        )

        self.prob.model.set_input_defaults(
            "tip_chord_tail",
            val=0.5,
            units="m"
        )

        self.prob.model.set_input_defaults(
            "thickness_ratio",
            val=0.12
        )

        self.prob.model.set_input_defaults(
            "skin_thickness",
            val=0.002,
            units="m"
        )

        self.prob.model.set_input_defaults(
            "twist_tail",
            val=np.zeros(10),
            units="deg"
        )

        self.prob.setup(
            check=False,
            force_alloc_complex=True)
    
    def test_case(self):
        
        self.prob.run_model()

        tol = 1e-4

        assert_near_equal(
            self.prob["mass"],
            4.22032, 
            tol)
        
        partial_data = self.prob.check_partials(
            out_stream=None,
            method="cs") 
        
        assert_check_partials(
            partial_data,
            atol=1e-15,
            rtol=1e-15)

if __name__ == "__main__":
    unittest.main()
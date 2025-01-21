import numpy as np
import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_output, add_aviary_option
from aviary.variable_info.variables import Aircraft


class EngineSize(om.ExplicitComponent):
    """
    GASP engine geometry calculation. It returns Aircraft.Nacelle.AVG_DIAMETER,
    Nacelle.AVG_LENGTH, and Aircraft.Nacelle.SURFACE_AREA.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        add_aviary_input(self, Aircraft.Engine.REFERENCE_DIAMETER,
                         shape=num_engine_type)
        add_aviary_input(self, Aircraft.Engine.SCALE_FACTOR, shape=num_engine_type)
        add_aviary_input(self, Aircraft.Nacelle.CORE_DIAMETER_RATIO,
                         shape=num_engine_type)
        add_aviary_input(self, Aircraft.Nacelle.FINENESS, shape=num_engine_type)

        add_aviary_output(self, Aircraft.Nacelle.AVG_DIAMETER,
                          shape=num_engine_type)
        add_aviary_output(self, Aircraft.Nacelle.AVG_LENGTH,
                          shape=num_engine_type)
        add_aviary_output(self, Aircraft.Nacelle.SURFACE_AREA,
                          shape=num_engine_type)

    def setup_partials(self):
        # derivatives w.r.t vectorized engine inputs have known sparsity pattern
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])
        shape = np.arange(num_engine_type)

        innames = [
            Aircraft.Engine.REFERENCE_DIAMETER,
            Aircraft.Engine.SCALE_FACTOR,
            Aircraft.Nacelle.CORE_DIAMETER_RATIO,
            Aircraft.Nacelle.FINENESS,
        ]

        self.declare_partials(Aircraft.Nacelle.AVG_DIAMETER,
                              innames[:-1], rows=shape, cols=shape, val=1.0)
        self.declare_partials(Aircraft.Nacelle.AVG_LENGTH, innames,
                              rows=shape, cols=shape, val=1.0)
        self.declare_partials(Aircraft.Nacelle.SURFACE_AREA,
                              innames, rows=shape, cols=shape, val=1.0)

    def compute(self, inputs, outputs):
        d_ref = inputs[Aircraft.Engine.REFERENCE_DIAMETER]
        scale_fac = inputs[Aircraft.Engine.SCALE_FACTOR]
        d_nac_eng = inputs[Aircraft.Nacelle.CORE_DIAMETER_RATIO]
        ld_nac = inputs[Aircraft.Nacelle.FINENESS]

        d_eng = d_ref * np.sqrt(scale_fac)
        outputs[Aircraft.Nacelle.AVG_DIAMETER] = d_eng * d_nac_eng
        outputs[Aircraft.Nacelle.AVG_LENGTH] = (
            ld_nac * outputs[Aircraft.Nacelle.AVG_DIAMETER]
        )
        outputs[Aircraft.Nacelle.SURFACE_AREA] = (
            np.pi
            * outputs[Aircraft.Nacelle.AVG_DIAMETER]
            * outputs[Aircraft.Nacelle.AVG_LENGTH]
        )

    def compute_partials(self, inputs, J):
        d_ref = inputs[Aircraft.Engine.REFERENCE_DIAMETER]
        scale_fac = inputs[Aircraft.Engine.SCALE_FACTOR]
        d_nac_eng = inputs[Aircraft.Nacelle.CORE_DIAMETER_RATIO]
        ld_nac = inputs[Aircraft.Nacelle.FINENESS]

        tr = np.sqrt(scale_fac)
        d_eng = d_ref * tr
        d_nac = d_eng * d_nac_eng
        l_nac = d_nac * ld_nac

        J[Aircraft.Nacelle.AVG_DIAMETER, Aircraft.Engine.REFERENCE_DIAMETER] = tr * d_nac_eng
        J[Aircraft.Nacelle.AVG_DIAMETER, Aircraft.Engine.SCALE_FACTOR] = (
            d_nac_eng * d_ref / (2 * tr)
        )
        J[Aircraft.Nacelle.AVG_DIAMETER, Aircraft.Nacelle.CORE_DIAMETER_RATIO] = d_eng

        for wrt in [
            Aircraft.Engine.REFERENCE_DIAMETER,
            Aircraft.Engine.SCALE_FACTOR,
            Aircraft.Nacelle.CORE_DIAMETER_RATIO,
        ]:
            J[Aircraft.Nacelle.AVG_LENGTH, wrt] = (
                J[Aircraft.Nacelle.AVG_DIAMETER, wrt] * ld_nac
            )
            J[Aircraft.Nacelle.SURFACE_AREA, wrt] = np.pi * (
                J[Aircraft.Nacelle.AVG_DIAMETER, wrt] * l_nac
                + J[Aircraft.Nacelle.AVG_LENGTH, wrt] * d_nac
            )

        J[Aircraft.Nacelle.AVG_LENGTH, Aircraft.Nacelle.FINENESS] = d_nac
        J[Aircraft.Nacelle.SURFACE_AREA, Aircraft.Nacelle.FINENESS] = (
            np.pi * J[Aircraft.Nacelle.AVG_LENGTH, Aircraft.Nacelle.FINENESS] * d_nac
        )

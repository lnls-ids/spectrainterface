"""SIRIUS parameters."""

import numpy as _np
import matplotlib.pyplot as _plt
import mathphys.constants as _constants
from spectrainterface.accelerator import StorageRingParameters

ECHARGE = _constants.elementary_charge
EREST = _constants.electron_rest_energy


class SIRIUS:
    """Class with SIRIUS parameters for radiation calculations."""

    class StorageRing(StorageRingParameters):

        extraction_dict = {
            "low_beta": {
                "betax": 1.499,
                "betay": 1.435,
                "alphax": 0,
                "alphay": 0,
                "etax": 0,
                "etay": 0,
                "etapx": 0,
                "etapy": 0,
            },

            "high_beta": {
                "betax": 17.20,
                "betay": 3.605,
                "alphax": 0,
                "alphay": 0,
                "etax": 0,
                "etay": 0,
                "etapx": 0,
                "etapy": 0,
            },

            "bc": {
                "betax": 0.338,
                "betay": 5.356,
                "alphax": 0.003,
                "alphay": 0,
                "etax": 0.002,
                "etay": 0,
                "etapx": 0,
                "etapy": 0,
            },
            
            "b1": {
                "betax": 1.660,
                "betay": 26.820,
                "alphax": 2.908,
                "alphay": -6.564,
                "etax": 0.122e-3,
                "etay": 0,
                "etapx": 3.211e-3,
                "etapy": 0,
            },

            "b2": {
                "betax": 1.265,
                "betay": 25.5,
                "alphax": 1.94,
                "alphay": 0,
                "etax": 0.025,
                "etay": 0,
                "etapx": 0,
                "etapy": 0,
            },
        }

        def __init__(self):
            """Class constructor."""
            self._energy = 3  # [GeV]
            self._current = 100  # [mA]
            self._sigmaz = 2.9  # [mm]
            self._nat_emittance = 2.5e-10  # [m rad]
            self._coupling_constant = 0.01
            self._energy_spread = 0.00084
            self._gamma = self._energy / (1e-9 * EREST / ECHARGE)
            self._betax = 1.499  # [m]
            self._betay = 1.435  # [m]
            self._alphax = 0
            self._alphay = 0
            self._etax = 0  # [m]
            self._etay = 0  # [m]
            self._etapx = 0
            self._etapy = 0
            self._extraction_point = "low_beta"

            self._zero_emittance = False
            self._zero_energy_spread = False
            self._injection_condition = "Align at Entrance"

            # BSC parameters
            self.set_current_bsc()

        @property
        def bsc0_h_highbeta(self):
            """Horizontal BSC at center of high beta section.

            Returns:
                float: Horizontal BSC High Beta [mm]
            """
            return self._bsc0_h_highbeta

        @property
        def bsc0_v_highbeta(self):
            """Vertical BSC at center of high beta section.

            Returns:
                float: Vertical BSC High Beta [mm]
            """
            return self._bsc0_v_highbeta

        @property
        def bsc0_h_lowbeta(self):
            """Horizontal BSC at center of low beta section.

            Returns:
                float: Horizontal BSC low Beta [mm]
            """
            return self._bsc0_h_lowbeta

        @property
        def bsc0_v_lowbeta(self):
            """Vertical BSC at center of low beta section.

            Returns:
                float: Vertical BSC low Beta [mm]
            """
            return self._bsc0_v_lowbeta

        def set_current_bsc(self):
            """Set current BSC (01/10/2024)."""
            self._bsc0_h_lowbeta = 3.4529
            self._bsc0_v_lowbeta = 1.8627
            self._bsc0_h_highbeta = 11.6952
            self._bsc0_v_highbeta = 2.9524
            self._update_bsc()

        def set_bsc_orion_reduction(self):
            """Set BSC after changes due to ORION."""
            self._bsc0_h_lowbeta = 3.4529
            self._bsc0_v_lowbeta = 1.38
            self._bsc0_h_highbeta = 11.6952
            self._bsc0_v_highbeta = 2.18
            self._update_bsc()

        def set_extraction_point(self, value):
            """Set extraction point."""
            self._extraction_point = value
            self.betax = self.extraction_dict[value]['betax']
            self.betay = self.extraction_dict[value]['betay']
            self.alphax = self.extraction_dict[value]['alphax']
            self.alphay = self.extraction_dict[value]['alphay']
            self.etax = self.extraction_dict[value]['etax']
            self.etay = self.extraction_dict[value]['etay']
            self.etapx = self.extraction_dict[value]['etapx']
            self.etapy = self.extraction_dict[value]['etapy']
            self._update_bsc()

        def _update_bsc(self):
            if self.extraction_point in ["low_beta", "high_beta"]:
                if self.extraction_point == "low_beta":
                    self._bsc0_h = self._bsc0_h_lowbeta
                    self._bsc0_v = self._bsc0_v_lowbeta
                elif self.extraction_point == "high_beta":
                    self._bsc0_h = self._bsc0_h_highbeta
                    self._bsc0_v = self._bsc0_v_highbeta
            else:
                raise ValueError("A valid beta section must be selected!")

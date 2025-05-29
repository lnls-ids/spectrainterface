"""ESRF parameters - only for tests."""

import numpy as _np
import matplotlib.pyplot as _plt
import mathphys.constants as _constants
import os
from spectrainterface.accelerator import StorageRingParameters
from spectrainterface import sources

ECHARGE = _constants.elementary_charge
EREST = _constants.electron_rest_energy


class ESRF:
    """Class with ESRF parameters for radiation calculations."""

    class StorageRing(StorageRingParameters):

        extraction_dict = {
            "even_id": {
                "betax": 35.2,
                "betay": 2.52,
                "alphax": 0,
                "alphay": 0,
                "etax": 0.137,
                "etay": 0,
                "etapx": 0,
                "etapy": 0,
                "bsc0_h": 5.0,
                "bsc0_v": 3.0,
            },

            "odd_id": {
                "betax": 0.5,
                "betay": 2.73,
                "alphax": 0,
                "alphay": 0,
                "etax": 0.037,
                "etay": 0,
                "etapx": 0,
                "etapy": 0,
                "bsc0_h": 5.0,
                "bsc0_v": 3.0,
            },

            "bm_3mrad": {
                "betax": 1.41,
                "betay": 34.9,
                "alphax": 0,
                "alphay": 0,
                "etax": 0.061,
                "etay": 0,
                "etapx": 0,
                "etapy": 0,
            },
            
            "bm_9mrad": {
                "betax": 0.99,
                "betay": 34.9,
                "alphax": 0,
                "alphay": 0,
                "etax": 0.045,
                "etay": 0,
                "etapx": 0,
                "etapy": 0,
            },

        }

        def __init__(self):
            """Class constructor."""
            self._energy = 6  # [GeV]
            self._current = 200  # [mA]
            self._sigmaz = 2.91  # [mm]
            self._nat_emittance = 133e-12  # [m rad]
            self._coupling_constant = 0.04
            self._energy_spread = 0.93e-3
            self._gamma = self._energy / (1e-9 * EREST / ECHARGE)
            self._betax = 0.5  # [m]
            self._betay = 2.73  # [m]
            self._alphax = 0
            self._alphay = 0
            self._etax = 0  # [m]
            self._etay = 0  # [m]
            self._etapx = 0
            self._etapy = 0
            self._extraction_point = "odd_id"
            self._bsc0_h = 5.0
            self._bsc0_v = 3.0

            self._zero_emittance = False
            self._zero_energy_spread = False
            self._injection_condition = None

    class Sources():

        class BM(sources.BendingMagnet):
            """BC class.

            Args:
                BendingMagnet (Bending magnet class): BM class
            """

            def __init__(self):
                """Class constructor."""
                super().__init__()
                self._b_peak = 0.47
                self._label = "BM"

        class U18(sources.IVU_NdFeB):
            """IVU18 class."""

            def __init__(self, period=18, length=2):
                """Class constructor."""
                super().__init__(period, length)
                self._label = "U18"
                self._br = 1.27
                self._gap = 6
                self._vc_thickness = 0
                self._vc_tolerance = 0.35
                self._polarization = "hp"
                self._halbach_coef = {
                    "hp": {"a": 2.29044642, "b": -3.71638253, "c": 0.34898287},
                }
                self._material = 'NdFeB'

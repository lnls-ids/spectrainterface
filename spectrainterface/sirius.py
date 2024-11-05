"""SIRIUS parameters."""

import numpy as _np
import matplotlib.pyplot as _plt
import mathphys.constants as _constants
import os
from spectrainterface.accelerator import StorageRingParameters
from spectrainterface import sources

ECHARGE = _constants.elementary_charge
EREST = _constants.electron_rest_energy
REPOS_PATH = os.path.dirname(os.path.abspath(__file__))


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
            self._bsc0_v_lowbeta = 1.5588
            self._bsc0_h_highbeta = 11.6952
            self._bsc0_v_highbeta = 2.4706
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

    class Sources():

        class BC(sources.BendingMagnet):
            """BC class.

            Args:
                BendingMagnet (Bending magnet class): BM class
            """

            def __init__(self):
                """Class constructor."""
                super().__init__()
                self._b_peak = 3.2
                self._label = "BC"
                self._meas_fname = REPOS_PATH + "/files/sirius/field_bc.txt"

        class B2(sources.BendingMagnet):
            """B2 class.

            Args:
                BendingMagnet (Bending magnet class): BM class
            """

            def __init__(self):
                """Class constructor."""
                super().__init__()
                self._b_peak = 0.5665
                self._label = "B2"
                self._meas_fname = REPOS_PATH + "/files/sirius/field_b2.txt"

        class B1(sources.BendingMagnet):
            """B1 class.

            Args:
                BendingMagnet (Bending magnet class): BM class
            """

            def __init__(self):
                """Class constructor."""
                super().__init__()
                self._b_peak = 0.5642
                self._label = "B1"
                self._meas_fname = REPOS_PATH + "/files/sirius/field_b1.txt"

        class UE44(sources.APPLE2):
            """UE44  class."""

            def __init__(self, period=44, length=3.4):
                """Class constructor."""
                super().__init__(period, length)
                self._material = 'NdFeB'
                self._label = "UE44"
                self._gap = 11.4
                self._br = 1.14

        class APU58(sources.APU):
            """APU58 class."""

            def __init__(self, period=58, length=1):
                """Class constructor."""
                super().__init__(period, length)
                self._label = "APU58"
                self._gap = 15.8
                self._br = 1.34
                self._z0 = 0
        
        class APU22_SAPUCAIA(sources.APU):
            """APU22 1991d class."""

            def __init__(self, period=22, length=1.2):
                """Class constructor."""
                super().__init__(period, length)
                self._label = "APU22"
                self._gap = 8
                self._br = 1.34
                self._z0 = 0.321
                self._efficiency = 0.9981

        class EPU50(sources.APPLE2):
            """EPU50 class."""

            def __init__(self, period=50, length=3):
                """Class constructor."""
                super().__init__(period, length)
                self._label = "EPU50"
                self._br = 1.24
                self._gap = 10.3

        class EPU50_UVX(sources.APPLE2):
            """EPU50 UVX class."""

            def __init__(self, period=50, length=2.7):
                """Class constructor."""
                super().__init__(period, length)
                self._label = "EPU50 (UVX)"
                self._br = 1.135
                self._gap = 22

        class IVU18_2(sources.IVU_NdFeB):
            """IVU18 class."""

            def __init__(self, period=18.5, length=2):
                """Class constructor."""
                super().__init__(period, length)
                self._label = "IVU18-2"
                self._br = 1.27
                self._gap = 4.5
                self.vc_thickness = 0
                self.vc_tolerance = 0.35
                self._polarization = "hp"
                self._halbach_coef = {
                    "hp": {"a": 2.26223181, "b": -3.69776472, "c": 0.32867209},
                }
                self._material = 'NdFeB'

        class IVU18_1(sources.IVU_NdFeB):
            """IVU18 class."""

            def __init__(self, period=18.5, length=2):
                """Class constructor."""
                super().__init__(period, length)
                self._label = "IVU18-1"
                self._br = 1.27
                self._gap = 4.5
                self._vc_thickness = 0
                self._vc_tolerance = 0.35
                self._polarization = "hp"
                self._halbach_coef = {
                    "hp": {"a": 2.29044642, "b": -3.71638253, "c": 0.34898287},
                }
                self._material = 'NdFeB'

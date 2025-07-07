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
        """SIRIUS Storage Ring."""

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
            self._injection_condition = None

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
            self.betax = self.extraction_dict[value]["betax"]
            self.betay = self.extraction_dict[value]["betay"]
            self.alphax = self.extraction_dict[value]["alphax"]
            self.alphay = self.extraction_dict[value]["alphay"]
            self.etax = self.extraction_dict[value]["etax"]
            self.etay = self.extraction_dict[value]["etay"]
            self.etapx = self.extraction_dict[value]["etapx"]
            self.etapy = self.extraction_dict[value]["etapy"]
            self._update_bsc()

        def _update_bsc(self):
            if self.extraction_point in ["low_beta", "high_beta"]:
                if self.extraction_point == "low_beta":
                    self._bsc0_h = self._bsc0_h_lowbeta
                    self._bsc0_v = self._bsc0_v_lowbeta
                elif self.extraction_point == "high_beta":
                    self._bsc0_h = self._bsc0_h_highbeta
                    self._bsc0_v = self._bsc0_v_highbeta

    class Sources:
        """SIRIUS Sources."""

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

        class UE44_IPE(sources.APPLE2):  # noqa: N801
            """UE44  class."""

            def __init__(self, period=44, length=3.4):
                """Class constructor."""
                super().__init__(period, length)
                self._material = "NdFeB"
                self._label = "UE44"
                self._gap = 11.4
                self._min_gap = 11.4
                self._br = 1.14

        class APU58_IPE(sources.APU):  # noqa: N801
            """APU58 class."""

            def __init__(self, period=58, length=1):
                """Class constructor."""
                super().__init__(period, length)
                self._label = "APU58"
                self._gap = 15.8
                self._min_gap = 15.8
                self._br = 1.34
                self._z0 = 0

        class APU22_SPU(sources.APU):  # noqa: N801
            """APU22 1991d class."""

            def __init__(self, period=22, length=1.2):
                """Class constructor."""
                super().__init__(period, length)
                self._label = "APU22"
                self._gap = 8
                self._min_gap = 8
                self._br = 1.34
                self._z0 = 0.321
                self._efficiency = 0.9981

        class APU22_MNC(sources.APU):  # noqa: N801
            """APU22 1991d class."""

            def __init__(self, period=22, length=1.2):
                """Class constructor."""
                super().__init__(period, length)
                self._label = "APU22"
                self._gap = 8
                self._min_gap = 8
                self._br = 1.34
                self._z0 = -0.300609
                self._efficiency = 1.0022029

        class EPU50(sources.APPLE2):
            """EPU50 class."""

            def __init__(self, period=50, length=3):
                """Class constructor."""
                super().__init__(period, length)
                self._label = "EPU50"
                self._br = 1.24
                self._gap = 10.3
                self._min_gap = 10.3

        class EPU50_UVX(sources.APPLE2):  # noqa: N801
            """EPU50 UVX class."""

            def __init__(self, period=50, length=2.7):
                """Class constructor."""
                super().__init__(period, length)
                self._label = "EPU50 (UVX)"
                self._br = 1.135
                self._gap = 22
                self._min_gap = 22

        class IVU18_EMA(sources.IVU_NdFeB):  # noqa: N801
            """IVU18-2 class (EMA beamline)."""

            def __init__(self, period=18.5, length=2):
                """Class constructor."""
                super().__init__(period, length)
                self._label = "IVU18-2"
                self._br = 1.27
                self._gap = 4.3
                self._min_gap = 4.3
                self.vc_thickness = 0
                self.vc_tolerance = 0.2501
                self._polarization = "hp"
                self._halbach_coef = {
                    "hp": {"a": 2.26223181, "b": -3.69776472, "c": 0.32867209},
                }
                self._material = "NdFeB"

        class IVU18_PNR(sources.IVU_NdFeB):  # noqa: N801
            """IVU18-1 class (PAINEIRA beamline)."""

            def __init__(self, period=18.5, length=2):
                """Class constructor."""
                super().__init__(period, length)
                self._label = "IVU18-1"
                self._br = 1.27
                self._gap = 4.3
                self._min_gap = 4.3
                self._vc_thickness = 0
                self._vc_tolerance = 0.2501
                self._polarization = "hp"
                self._halbach_coef = {
                    "hp": {"a": 2.29044642, "b": -3.71638253, "c": 0.34898287},
                }
                self._material = "NdFeB"

        class DELTA52_SAB(sources.Elliptic):  # noqa: N801
            """DELTA Undulator class."""

            def __init__(self, period=52.5, length=1.2):
                """Class constructor.

                Args:
                    period (float, optional): Undulator period [mm].
                    length (float, optional): Undulator length [m].
                """
                super().__init__()
                self._undulator_type = "DELTA"
                self._label = "DELTA"
                self._br = 1.39
                self._polarization = "hp"
                self._phase_coef = {
                    "hp": {"ef": 1.00566, "z0": 25.89527},
                    "vp": {"ef": 0.99032, "z0": 26.08821},
                    "cp": {"ef": 0.71497, "z0": 25.89593},
                }
                self._gap = 13.6
                self._min_gap = 13.6
                self._phase = 0
                self._halbach_coef = {
                    "hp": {"a": 1.696, "b": -2.349, "c": -0.658},
                    "vp": {"a": 1.696, "b": -2.349, "c": -0.658},
                    "cp": {"a": 1.696, "b": -2.349, "c": -0.658},
                }
                self._material = "NdFeB"
                self._period = period
                self._source_length = length
                self._source_type = "ellipticundulator"

            @property
            def phase(self):
                """Undulator phase [mm].

                Returns:
                    float: Phase [mm]
                """
                return self._phase

            @property
            def phase_coef(self):
                """Undulator calibration coefficients."""
                return self._phase_coef

            @phase.setter
            def phase(self, value):
                """Undulator phase setter [mm]."""
                self._phase = value

            @phase_coef.setter
            def phase_coef(self, value):
                """Undulator calibration coefficients setter."""
                self._phase_coef = value

            def calc_min_gap(
                self,
                si_parameters=None,
                vc_thickness=None,
                vc_tolerance=None,
            ):
                """Calculate minimum gap of undulator.

                Args:
                si_parameters (StorageRingParameters, optional): StorageRingParameters
                 object. Defaults to None.
                vc_thickness (float, optional): Vacuum chamber thickness.
                 Defaults to None.
                vc_tolerance (float, optional): Extra delta in gap. Defaults to None.

                Returns:
                  float: (min gap vertical, min gap horizontal) minimum gap allowed.
                """
                pos = self.source_length / 2

                if si_parameters is None:
                    raise ValueError("Accelerator must be selected")
                else:
                    acc = si_parameters

                if vc_thickness is None:
                    vc_thickness = self.vc_thickness
                if vc_tolerance is None:
                    vc_tolerance = self.vc_tolerance

                bsch, bscv = acc.calc_beam_stay_clear(pos)
                gap = _np.sqrt(2 * (bsch**2 + bscv**2))
                gap = gap + vc_thickness + vc_tolerance

                return gap, gap

            def get_beff(self, gap_over_period, phase=None):
                """Get peak magnetic field for a given device and gap.

                Args:
                    gap_over_period (float): gap normalized by the undulator period.

                Returns:
                    _type_: _description_
                """
                phase = self.phase if phase is None else phase
                br = self.br
                z0 = self.phase_coef[self.polarization]["z0"]
                a = self.halbach_coef[self.polarization]["a"]
                b = self.halbach_coef[self.polarization]["b"]
                c = self.halbach_coef[self.polarization]["c"]
                efficiency = self.phase_coef[self.polarization]["ef"]
                return (
                    efficiency
                    * self.beff_function(
                        gap_over_period=gap_over_period, br=br, a=a, b=b, c=c
                    )
                    * _np.abs(_np.cos(_np.pi / self._period * (phase - z0)))
                )

            def calc_max_k(self, si_parameters):
                """Calc max K achieved by undulator.

                Args:
                    si_parameters (StorageRingParameters): StorageRingParameters
                    object.
                """
                if self.gap != 0:
                    phase0 = self.phase
                    self.phase = self.phase_coef[self.polarization]["z0"]
                    k_max = self.get_k()
                    self.phase = phase0
                else:
                    gap_minv, gap_minh = self.calc_min_gap(si_parameters)
                    gap_min = gap_minv if self.polarization == "hp" else gap_minh
                    phase0 = self.phase
                    self.phase = self.phase_coef[self.polarization]["z0"]
                    b_max = self.get_beff(gap_min / self.period)
                    k_max = self.undulator_b_to_k(b_max, self.period)
                    self.phase = phase0
                return k_max

        class CPMU13_HIB(sources.CPMU_PrFeB_HEPS):  # noqa: N801
            """Cpmu PrFeB Undulator class (HEPS).

            Args:
            Undulator (Undulator class): Undulator class
            """

            def __init__(self, period=13.6, length=2.03):
                """Class constructor.

                Args:
                    period (float, optional): Undulator period [mm]
                    length (float, optional): Undulator length [m]
                """
                super().__init__(period, length)
                self._undulator_type = "CPMU"
                self._label = "CPMU 13.6"
                self._gap = 4.84
                self._min_gap = 4.84
                self.vc_tolerance = 0.160

        class CPMU15_TIB(sources.CPMU_PrFeB_HEPS):  # noqa: N801
            """Cpmu PrFeB Undulator class (HEPS).

            Args:
            Undulator (Undulator class): Undulator class
            """

            def __init__(self, period=15.8, length=2.03):
                """Class constructor.

                Args:
                    period (float, optional): Undulator period [mm]
                    length (float, optional): Undulator length [m]
                """
                super().__init__(period, length)
                self._undulator_type = "CPMU"
                self._label = "CPMU 15.8"
                self._gap = 3.7
                self._min_gap = 3.7
                self.vc_tolerance = 0.210

        class VPU29_CNB(sources.VPU):  # noqa: N801
            """VPU29b / 2386b (CARNAUBA) class."""

            def __init__(self, period=29.0, length=1.54):
                """Class constructor."""
                super().__init__(period, length)
                self._label = "VPU29-CNB"
                self._br = 1.304
                self._gap = 9.7
                self._min_gap = 9.7
                self.vc_thickness = 0.5
                self.vc_tolerance = 0.468
                self._polarization = "vp"
                self._halbach_coef = {
                    "vp": {"a": 2.03304573, "b": -3.4431994, "c": 0.18171406},
                }
                self._material = "NdFeB"

        class VPU29_CAT(sources.VPU):  # noqa: N801
            """VPU29a (CATERETE) class."""

            def __init__(self, period=29.0, length=1.54):
                """Class constructor."""
                super().__init__(period, length)
                self._label = "VPU29-CAT"
                self._br = 1.304
                self._gap = 9.7
                self._min_gap = 9.7
                self.vc_thickness = 0.5
                self.vc_tolerance = 0.468
                self._polarization = "vp"
                self._halbach_coef = {
                    "vp": {
                        "a": 2.03304573,
                        "b": -3.4431994,
                        "c": 0.18171406,
                    },  # must to be update
                }
                self._material = "NdFeB"

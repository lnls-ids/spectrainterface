"""Undulator tools to use with spectra."""

import mathphys.constants as _constants
import numpy as _np
from spectrainterface.tools import SourceFunctions
from spectrainterface.accelerator import StorageRingParameters

ECHARGE = _constants.elementary_charge
EMASS = _constants.electron_mass
LSPEED = _constants.light_speed
PLACK = _constants.reduced_planck_constant


class BendingMagnet(SourceFunctions):
    """Main class for bending magnets."""

    def __init__(self):
        """Class constructor."""
        super().__init__()
        self._b_peak = 1
        self._source_type = "bendingmagnet"
        self._label = "label"

    @property
    def b_peak(self):
        """Field peak of bending magnet.

        Returns:
            float: Peak field [T]
        """
        return self._b_peak

    @property
    def source_type(self):
        """Source type.

        Returns:
            str: Type of source.
        """
        return self._source_type

    @property
    def label(self):
        """BM label.

        Returns:
            str: Undulator label
        """
        return self._label

    @b_peak.setter
    def b_peak(self, value):
        self._b_peak = value

    @label.setter
    def label(self, value):
        self._label = value


class BC(BendingMagnet):
    """BC class.

    Args:
        BendingMagnet (Bending magnet class): BM class
    """

    def __init__(self):
        """Class constructor."""
        super().__init__()
        self._b_peak = 3.2
        self._label = "BC"


class B2(BendingMagnet):
    """B2 class.

    Args:
        BendingMagnet (Bending magnet class): BM class
    """

    def __init__(self):
        """Class constructor."""
        super().__init__()
        self._b_peak = 0.5642
        self._label = "B2"


class Undulator(SourceFunctions):
    """Main class for undulators.

    Args:
        SourceFunctions (SourceFunctions class): S. F. functions
    """

    def __init__(self):
        """Class constructor."""
        super().__init__()
        self._undulator_type = "planar"
        self._gap = 0
        self._br = 1.37
        self._period = 50
        self._efficiency = 1.0
        self._label = "label"
        self._polarization = "hp"
        self._halbach_coef = dict()
        self._vc_thickness = 0.5
        self._vc_tolerance = 0.1

    @property
    def undulator_type(self):
        """Undulator type.

        Returns:
            str: Undulator type.
        """
        return self._undulator_type

    @property
    def gap(self):
        """Undulator gap [mm].

        Returns:
            float: Gap [mm]
        """
        return self._gap

    @property
    def br(self):
        """Remanent magnetization.

        Returns:
            float: Remanent mag [T]
        """
        return self._br

    @property
    def period(self):
        """Undulator period.

        Returns:
            float: Period [mm]
        """
        return self._period

    @property
    def efficiency(self):
        """Undulator efficiency.

        Returns:
            float: Efficiency
        """
        return self._efficiency

    @property
    def label(self):
        """Undulator label.

        Returns:
            str: Undulator label
        """
        return self._label

    @property
    def polarization(self):
        """Polarization.

        Returns:
            str: Undulator polarization
        """
        return self._polarization

    @property
    def halbach_coef(self):
        """Halbach coefficients.

        Returns:
            dict: dictionary with halbach coeffs for each polarization.
        """
        return self._halbach_coef

    @property
    def vc_thickness(self):
        """Vacuum chamber thickness.

        Returns:
            float: Thickness of vacuum chamber.
        """
        return self._vc_thickness

    @property
    def vc_tolerance(self):
        """Tolerance space between undulator and vacuum chamber.

        Returns:
            float: Tolarance between id and vacuum chamber
        """
        return self._vc_tolerance

    @undulator_type.setter
    def undulator_type(self, value):
        self._undulator_type = value

    @gap.setter
    def gap(self, value):
        if value < 0:
            raise ValueError("Gap must be positive.")
        else:
            self._gap = value

    @br.setter
    def br(self, value):
        self._br = value

    @period.setter
    def period(self, value):
        self._period = value

    @efficiency.setter
    def efficiency(self, value):
        self._efficiency = value

    @label.setter
    def label(self, value):
        self._label = value

    @polarization.setter
    def polarization(self, value):
        allowed_pol = self._get_list_of_pol(self.undulator_type)
        if value in allowed_pol:
            self._polarization = value
        else:
            raise ValueError("Polarization not allowed.")

    @halbach_coef.setter
    def halbach_coef(self, value):
        self._halbach_coef = value

    @vc_thickness.setter
    def vc_thickness(self, value):
        self._vc_thickness = value

    @vc_tolerance.setter
    def vc_tolerance(self, value):
        self._vc_tolerance = value

    def get_beff(self, gap_over_period):
        """Get peak magnetic field for a given device and gap.

        Args:
            gap_over_period (float): gap normalized by the undulator period.

        Returns:
            _type_: _description_
        """
        br = self.br
        a = self.halbach_coef[self.polarization]["a"]
        b = self.halbach_coef[self.polarization]["b"]
        c = self.halbach_coef[self.polarization]["c"]
        efficiency = self.efficiency

        return efficiency * SourceFunctions.beff_function(
            gap_over_period=gap_over_period, br=br, a=a, b=b, c=c
        )

    def calc_min_gap(
        self,
        si_parameters=None,
        section="SB",
        vc_thickness=None,
        vc_tolerance=None,
    ):
        """Calculate minimum gap of undulator.

        Args:
        si_parameters (StorageRingParameters, optional): StorageRingParameters
         object. Defaults to None.
        section (str, optional): Straight section (SB, SP or SA).
         Defaults to 'SB'.
        vc_thickness (float, optional): Vacuum chamber thickness.
         Defaults to None.
        vc_tolerance (float, optional): Extra delta in gap. Defaults to None.

        Returns:
            float: (min gap vertical, min gap horizontal) minimum gap allowed.
        """
        pos = self.source_length / 2
        section = section.lower()

        if si_parameters is None:
            acc = StorageRingParameters()
            acc.set_bsc_with_ivu18()
            if section == "sb" or section == "sp":
                acc.set_low_beta_section()
            elif section == "sa":
                acc.set_high_beta_section()
            else:
                raise ValueError("Section not defined.")
        else:
            acc = si_parameters

        if vc_thickness is None:
            vc_thickness = self.vc_thickness
        if vc_tolerance is None:
            vc_tolerance = self.vc_tolerance

        bsch, bscv = acc.calc_beam_stay_clear(pos)
        gaph = 2 * bsch + vc_thickness + vc_tolerance
        gapv = 2 * bscv + vc_thickness + vc_tolerance

        return gapv, gaph

    def calc_max_length(
        self,
        si_parameters=None,
        section="SB",
        vc_thickness=None,
        vc_tolerance=None,
    ):
        """Calc max length for given gap.

        Args:
        si_parameters (StorageRingParameters, optional): StorageRingParameters
         object. Defaults to None.
        section (str, optional): Straight section (SB, SP or SA).
         Defaults to 'SB'.
        vc_thickness (float, optional): Vacuum chamber thickness.
         Defaults to None.
        vc_tolerance (float, optional): Extra delta in gap. Defaults to None.

        Raises:
            ValueError: Section not defined

        Returns:
            float: (max length conventional undulator, max length vertical
             undulator).
        """
        pos = self.source_length / 2
        pos = _np.linspace(0, 3.5, 2001)
        section = section.lower()

        if si_parameters is None:
            acc = StorageRingParameters()
            acc.set_bsc_with_ivu18()
            if section == "sb" or section == "sp":
                acc.set_low_beta_section()
            elif section == "sa":
                acc.set_high_beta_section()
            else:
                raise ValueError("Section not defined.")
        else:
            acc = si_parameters

        if vc_thickness is None:
            vc_thickness = self.vc_thickness
        if vc_tolerance is None:
            vc_tolerance = self.vc_tolerance

        bsch, bscv = acc.calc_beam_stay_clear(pos)
        gaph = 2 * bsch + vc_thickness + vc_tolerance
        gapv = 2 * bscv + vc_thickness + vc_tolerance

        idxv = _np.argmin(_np.abs(self.gap - gapv))
        idxh = _np.argmin(_np.abs(self.gap - gaph))

        length_verticalid = 2 * pos[idxh]
        length_conventional = 2 * pos[idxv]

        return length_conventional, length_verticalid

    def calc_max_k(self, si_parameters):
        """Calc max K achieved by undulator.

        Args:
            si_parameters (StorageRingParameters): StorageRingParameters
             object.
        """
        gap_minv, _ = self.calc_min_gap(si_parameters)
        b_max = self.get_beff(gap_minv / self.period)
        k_max = self.undulator_b_to_k(b_max, self.period)
        return k_max

    def get_k(self):
        """Get K for configured gap.

        Returns:
            float: K value
        """
        beff = self.get_beff(self.gap / self.period)
        k = self.undulator_b_to_k(beff, self.period)
        return k


class Wiggler(Undulator):
    """Wiggler Undulator class.

    Args:
        Undulator (Undulator class): Undulator class
    """

    def __init__(self, period, length):
        """Class constructor.

        Args:
            period (float, optional): Undulator period [mm]
            length (float, optional): Undulator length [m]
        """
        super().__init__()
        self._undulator_type = "wiggler"
        self._label = "Wiggler"
        self._br = 1.37
        self._polarization = "hp"
        self._efficiency = 1
        self._halbach_coef = {"hp": {"a": 1.732, "b": -3.238, "c": 0.0}}
        self._period = period
        self._source_length = length
        self._source_type = "wiggler"


class Planar(Undulator):
    """Planar Undulator class.

    Args:
        Undulator (Undulator class): Undulator class
    """

    def __init__(self, period, length):
        """Class constructor.

        Args:
            period (float, optional): Undulator period [mm]
            length (float, optional): Undulator length [m]
        """
        super().__init__()
        self._undulator_type = "planar"
        self._label = "Planar"
        self._br = 1.37
        self._polarization = "hp"
        self._efficiency = 1
        self._halbach_coef = {"hp": {"a": 1.732, "b": -3.238, "c": 0.0}}
        self._period = period
        self._source_length = length
        self._source_type = "linearundulator"


class Apple2(Undulator):
    """Apple2 Undulator class.

    Args:
        Undulator (Undulator class): Undulator class
    """

    def __init__(self, period, length):
        """Class constructor.

        Args:
            period (float, optional): Undulator period [mm].
            length (float, optional): Undulator length [m].
        """
        super().__init__()
        self._undulator_type = "apple2"
        self._label = "Apple-II"
        self._br = 1.37
        self._polarization = "hp"
        self._efficiency = 1
        self._halbach_coef = {
            "hp": {"a": 1.732, "b": -3.238, "c": 0.0},
            "vp": {"a": 1.926, "b": -5.629, "c": 1.448},
            "cp": {"a": 1.356, "b": -4.875, "c": 0.947},
        }
        self._period = period
        self._source_length = length
        self._source_type = "ellipticundulator"

class Ue44(Apple2):
    """Ue44  class."""
    def __init__(self, period=44, length=3.4):
        """Class constructor."""
        super().__init__(period, length)
        self._label = 'Ue44'
        self._gap = 11.4
        self._br = 1.14

class Epu50(Apple2):
    """Epu50 class."""
    def __init__(self, period=50, length=3):
        """Class constructor."""
        super().__init__(period, length)
        self._label = 'Epu50'
        self._br = 1.24
        self._gap = 10.3
        

class Epu50_uvx(Apple2):
    """Epu50 uvx class."""
    def __init__(self, period=50, length=2.7):
        """Class constructor."""
        super().__init__(period, length)
        self._label = 'Epu50 (UVX)'
        self._br = 1.135
        self._gap = 22
        

class Delta(Undulator):
    """Delta Undulator class.

    Args:
        Undulator (Undulator class): Undulator class
    """

    def __init__(self, period, length):
        """Class constructor.

        Args:
            period (float, optional): Undulator period [mm].
            length (float, optional): Undulator length [m].
        """
        super().__init__()
        self._undulator_type = "delta"
        self._label = "Delta"
        self._br = 1.37
        self._polarization = "hp"
        self._efficiency = 1
        self._halbach_coef = {
            "hp": {"a": 1.696, "b": -2.349, "c": -0.658},
            "vp": {"a": 1.696, "b": -2.349, "c": -0.658},
            "cp": {"a": 1.193, "b": -2.336, "c": -0.667},
        }
        self._period = period
        self._source_length = length
        self._source_type = "ellipticundulator"


class Hybrid(Undulator):
    """Hybrid Undulator class.

    Args:
        Undulator (Undulator class): Undulator class
    """

    def __init__(self, period, length):
        """Class constructor.

        Args:
            period (float, optional): Undulator period [mm]
            length (float, optional): Undulator length [m]
        """
        super().__init__()
        self._undulator_type = "hybrid"
        self._label = "Hybrid (Nd)"
        self._br = 1.24
        self._polarization = "hp"
        self._efficiency = 1
        self._halbach_coef = {"hp": {"a": 2.552, "b": -4.431, "c": 1.101}}
        self._period = period
        self._source_length = length
        self._source_type = "linearundulator"


class Hybrid_smco(Undulator):
    """Hybrid smco Undulator class.

    Args:
        Undulator (Undulator class): Undulator class
    """

    def __init__(self, period, length):
        """Class constructor.

        Args:
            period (float, optional): Undulator period [mm]
            length (float, optional): Undulator length [m]
        """
        super().__init__()
        self._undulator_type = "hybrid_smco"
        self._label = "Hybrid (SmCo)"
        self._br = 1.24
        self._polarization = "hp"
        self._efficiency = 1
        self._halbach_coef = {"hp": {"a": 2.789, "b": -4.853, "c": 1.550}}
        self._period = period
        self._source_length = length
        self._source_type = "linearundulator"


class Cpmu_nd(Undulator):
    """Cpmu nd Undulator class.

    Args:
        Undulator (Undulator class): Undulator class
    """

    def __init__(self, period, length):
        """Class constructor.

        Args:
            period (float, optional): Undulator period [mm]
            length (float, optional): Undulator length [m]
        """
        super().__init__()
        self._undulator_type = "cpmu_nd"
        self._label = "CPMU (Nd)"
        self._br = 1.5
        self._polarization = "hp"
        self._efficiency = 0.9
        self._halbach_coef = {"hp": {"a": 2.268, "b": -3.895, "c": 0.554}}
        self._period = period
        self._source_length = length
        self._source_type = "linearundulator"


class Cpmu_pr_nd(Undulator):
    """Cpmu pr nd Undulator class.

    Args:
        Undulator (Undulator class): Undulator class
    """

    def __init__(self, period, length):
        """Class constructor.

        Args:
            period (float, optional): Undulator period [mm]
            length (float, optional): Undulator length [m]
        """
        super().__init__()
        self._undulator_type = "cpmu_prnd"
        self._label = "CPMU (Pr,Nd)"
        self._br = 1.62
        self._polarization = "hp"
        self._efficiency = 0.9
        self._halbach_coef = {"hp": {"a": 2.132, "b": -3.692, "c": 0.391}}
        self._period = period
        self._source_length = length
        self._source_type = "linearundulator"


class Cpmu_pr(Undulator):
    """Cpmu pr Undulator class.

    Args:
        Undulator (Undulator class): Undulator class
    """

    def __init__(self, period, length):
        """Class constructor.

        Args:
            period (float, optional): Undulator period [mm]
            length (float, optional): Undulator length [m]
        """
        super().__init__()
        self._undulator_type = "cpmu_pr"
        self._label = "CPMU (Pr)"
        self._br = 1.67
        self._polarization = "hp"
        self._efficiency = 0.9
        self._halbach_coef = {"hp": {"a": 2.092, "b": -3.655, "c": 0.376}}
        self._period = period
        self._source_length = length
        self._source_type = "linearundulator"

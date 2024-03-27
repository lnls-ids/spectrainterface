"""Undulator tools to use with spectra."""

import mathphys.constants as _constants
import numpy as _np
from spectrainterface.tools import SourceFunctions
from spectrainterface.accelerator import StorageRingParameters
import os

REPOS_PATH = os.path.dirname(os.path.abspath(__file__))
ECHARGE = _constants.elementary_charge
EMASS = _constants.electron_mass
LSPEED = _constants.light_speed
PLANCK = _constants.reduced_planck_constant


class BendingMagnet(SourceFunctions):
    """Main class for bending magnets."""

    def __init__(self):
        """Class constructor."""
        super().__init__()
        self._b_peak = 1
        self._source_type = "bendingmagnet"
        self._label = "label"
        self._meas_fname = None

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

    @property
    def meas_fname(self):
        """Measured field file.

        Returns:
            str: Filename with field measurements.
        """
        return self._meas_fname

    @b_peak.setter
    def b_peak(self, value):
        self._b_peak = value

    @label.setter
    def label(self, value):
        self._label = value

    def get_meas_field(self):
        """Get measured field.

        Returns:
            numpy array: First column contains longitudinal spatial
            coordinate (z) [mm], second column contais vertical field
            [T], and third column constais horizontal field [T].
        """
        z, bx, by = _np.genfromtxt(self.meas_fname, unpack=True, skip_header=1)
        field = _np.zeros((len(z), 3))
        field[:, 0] = z
        field[:, 1] = bx
        field[:, 2] = by
        return field


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
        self._meas_fname = REPOS_PATH + "/files/field_bc.txt"


class B2(BendingMagnet):
    """B2 class.

    Args:
        BendingMagnet (Bending magnet class): BM class
    """

    def __init__(self):
        """Class constructor."""
        super().__init__()
        self._b_peak = 0.5665
        self._label = "B2"
        self._meas_fname = REPOS_PATH + "/files/field_b2.txt"


class B1(BendingMagnet):
    """B1 class.

    Args:
        BendingMagnet (Bending magnet class): BM class
    """

    def __init__(self):
        """Class constructor."""
        super().__init__()
        self._b_peak = 0.5642
        self._label = "B1"
        self._meas_fname = REPOS_PATH + "/files/field_b1.txt"


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

    def find_adjusted_br(self, gap=0, b_max=[0, 0, 0]):
        """Get adjust br with values of b max.

        Args:
            gap (float): physical min gap of undulator
             If not defined. Default to the minimum BSC will be get for the
              undulator.
            b_max (float list): B Max to polarization hp:
                b_max[0], vp: b_max[1], cp: b_max[2].

        Returns:
            float: br value
        """
        b_max = {"hp": b_max[0], "vp": b_max[1], "cp": b_max[2]}

        br0 = self._br
        gap0 = self._gap
        polarization0 = self._polarization

        if gap0 == 0:
            self._gap, *_ = self.calc_min_gap() if gap == 0 else (gap, 0)

        brs = _np.linspace(0, 2 * self._br, 5001)
        bpeak = _np.ones(len(brs))

        br2 = 1
        n = 0

        allowed_pol = self._get_list_of_pol(self.undulator_type)
        for j in b_max:
            if b_max[j] != 0 and (j in allowed_pol):
                self._polarization = j
                for i, br in enumerate(brs):
                    self._br = br
                    bpeak[i] = self.get_beff(self._gap / self._period)

                idx = _np.argmin(_np.abs(bpeak - b_max[j]))
                br2 *= brs[idx]
                n += 1

        self._br = br0
        self._gap = gap0
        self._polarization = polarization0

        return br2 ** (1 / n) if br2 != 1 else br0


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


class Elliptic(Undulator):
    """Class for undulators that allow elliptic polarizations.

    Args:
        Undulator (Undulator class): Undulator class
    """

    def __init__(self):
        """Class consteuctor."""
        super().__init__()
        self._fields_ratio = 1
        self._source_type = "ellipticundulator"

    @property
    def fields_ratio(self):
        """Ratio By_peak / Bx_peak.

        Returns:
            float: Ratio of peak fields.
        """
        return self._fields_ratio

    @fields_ratio.setter
    def fields_ratio(self, value):
        self._fields_ratio = value


class Apple2(Elliptic):
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


class Delta(Elliptic):
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
        self._halbach_coef = {
            "hp": {"a": 2.552, "b": -4.431, "c": 1.101},
            "vp": {"a": 2.552, "b": -4.431, "c": 1.101},
        }
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


class Ue44(Apple2):
    """Ue44  class."""

    def __init__(self, period=44, length=3.4):
        """Class constructor."""
        super().__init__(period, length)
        self._label = "UE44"
        self._gap = 11.4
        self._br = 1.14


class Vpu(Hybrid):
    """Vpu class."""

    def __init__(self, period=29, length=1.5):
        """Class constructor."""
        super().__init__(period, length)
        self._polarization = "vp"
        self._source_type = "verticalundulator"
        self._label = "Vpu"
        self._gap = 9.7


class Apu58(Planar):
    """Apu58 class."""

    def __init__(self, period=58, length=1):
        """Class constructor."""
        super().__init__(period, length)
        self._label = "Apu58"
        self._gap = 15.8
        self._br = 1.34


class Epu50(Apple2):
    """Epu50 class."""

    def __init__(self, period=50, length=3):
        """Class constructor."""
        super().__init__(period, length)
        self._label = "Epu50"
        self._br = 1.24
        self._gap = 10.3


class Epu50_uvx(Apple2):
    """Epu50 uvx class."""

    def __init__(self, period=50, length=2.7):
        """Class constructor."""
        super().__init__(period, length)
        self._label = "Epu50 (UVX)"
        self._br = 1.135
        self._gap = 22

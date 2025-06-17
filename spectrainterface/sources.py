"""Undulator tools to use with spectra."""

import mathphys.constants as _constants
import numpy as _np
from spectrainterface.tools import SourceFunctions
from spectrainterface.accelerator import StorageRingParameters

ECHARGE = _constants.elementary_charge
EMASS = _constants.electron_mass
LSPEED = _constants.light_speed
PLANCK = _constants.reduced_planck_constant
PI = _np.pi
VACUUM_PERMITTICITY = _constants.vacuum_permitticity


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

    def calc_total_power(self, gamma, acceptance=0.230, current=100):
        """Calculate total power from bending magnet.

        Args:
            gamma (float): Lorentz fator
            acceptance (float): slit acceptance [mrad]
            current (float): electron beam current [mA]
        Returns:
            float: Total power of source light [kW]
        """
        const = ((ECHARGE**3) * (gamma**3)) / (
            6 * PI * VACUUM_PERMITTICITY * EMASS * LSPEED
        )

        total_power = (
            const
            * self._b_peak
            * (acceptance * 1e-3)
            * (current * 1e-3)
            / (1e3 * ECHARGE)
        )

        return total_power


class Undulator(SourceFunctions):
    """Main class for undulators.

    Args:
        SourceFunctions (SourceFunctions class): S. F. functions
    """

    def __init__(self):
        """Class constructor."""
        super().__init__()
        self._undulator_type = "Halbach"
        self._gap = 0
        self._br = 1.37
        self._period = 50
        self._efficiency = 1.0
        self._label = "label"
        self._polarization = "hp"
        self._halbach_coef = dict()
        self._material = None
        self._vc_thickness = 0.5
        self._vc_tolerance = 0.1
        self._add_phase_errors = False
        self._use_rec_params = True

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
    def material(self):
        """Undulator magnets material.

        Returns:
            str: String with material name.
        """
        return self._material

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

    @property
    def add_phase_errors(self):
        """Add phase errors.

        Returns:
            bool: If true, phase errors will be added
        """
        return self._add_phase_errors

    @property
    def use_recovery_params(self):
        """Use recovery parameters to calc phase error.

        Returns:
            bool: If true, rec params will be used
        """
        return self._use_rec_params

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

    @add_phase_errors.setter
    def add_phase_errors(self, value):
        if type(value) is not bool:
            raise ValueError("Add phase error must be a boolean")  # noqa: E501
        else:
            self._add_phase_errors = value

    @use_recovery_params.setter
    def use_recovery_params(self, value):
        if type(value) is not bool:
            raise ValueError("Use recovery params must be a boolean")  # noqa: E501
        else:
            self._use_rec_params = value

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
            raise ValueError("Accelerator must be selected.")
        else:
            acc = si_parameters

        if vc_thickness is None:
            vc_thickness = self.vc_thickness
        if vc_tolerance is None:
            vc_tolerance = self.vc_tolerance

        bsch, bscv = acc.calc_beam_stay_clear(pos)
        gaph = 2 * (bsch + vc_thickness + vc_tolerance)
        gapv = 2 * (bscv + vc_thickness + vc_tolerance)

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
        gap_minv, gap_minh = self.calc_min_gap(si_parameters)
        gap_min = gap_minv if self.polarization == 'hp' else gap_minh
        b_max = self.get_beff(gap_min / self.period)
        k_max = self.undulator_b_to_k(b_max, self.period)
        return k_max

    def energy_to_gaps(
        self,
        target_energy: float,
        si_parameters: StorageRingParameters,
        h_max: int = 31,
    ):
        """Calc max K achieved by undulator.

        Args:
            target_energy (float): Target Energy
            si_parameters (StorageRingParameters): StorageRingParameters
             object.
            h_max (int): max harmonic to search gap
        Return:
            Harmonic with gaps matrix (array numpy)
        """
        n = _np.arange(1, h_max, 2)

        k_max = self.calc_max_k(si_parameters)
        ks = self.calc_k_target(
            si_parameters.gamma, n, self.period, target_energy
        )
        isnan = _np.isnan(ks)
        idcs_nan = _np.argwhere(~isnan)
        idcs_max = _np.argwhere(ks < k_max)
        idcs_kmin = _np.argwhere(ks > 0)
        idcs = _np.intersect1d(
            idcs_nan.ravel(),
            _np.intersect1d(idcs_max.ravel(), idcs_kmin.ravel()),
        )
        kres = ks[idcs]
        gaps = self.undulator_k_to_gap(
            k=kres,
            period=self.period,
            br=self.br,
            a=self.halbach_coef["hp"]["a"],
            b=self.halbach_coef["hp"]["b"],
            c=self.halbach_coef["hp"]["c"],
        )
        harms = n[idcs]
        return _np.array([harms, gaps]).T

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

    def calc_total_power(self, gamma, b, current=100):
        """Calculate total power from an source light.

        Args:
            gamma (float): Lorentz fator
            b (float): Field amplitude [T]
            current (float): electron beam current [mA]
        Returns:
            float: Total power of source light [kW]
        """

        b = _np.sqrt(2 * b**2) if self._polarization == "cp" else b

        const = ((ECHARGE**4) * (gamma**2)) / (
            12 * PI * VACUUM_PERMITTICITY * (EMASS**2) * (LSPEED**2)
        )

        total_power = (
            const
            * (b**2)
            * self._source_length
            * (current * 1e-3)
            / (1e3 * ECHARGE)
        )

        return total_power


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


class Halbach(Undulator):
    """Halbach Planar Undulator class.

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
        self._undulator_type = "Halbach"
        self._label = "Halbach"
        self._br = 1.37
        self._polarization = "hp"
        self._efficiency = 1
        self._halbach_coef = {"hp": {"a": 1.732, "b": -3.238, "c": 0.0}}
        self._period = period
        self._source_length = length
        self._source_type = "linearundulator"


class APU(Halbach):
    """APU class."""

    def __init__(self, period=22, length=1):
        """Class constructor."""
        super().__init__(period, length)
        self._undulator_type = "APU"
        self._phase = 0
        self._label = "APU"
        self._gap = 0
        self._br = 1.34
        self._phase_coef = {
            "hp": {"ef": 1, "z0": 0},
        }

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

    @phase.setter
    def phase(self, value):
        """Undulator phase setter [mm]."""
        self._phase = value

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
            * SourceFunctions.beff_function(
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
            self.phase = self.phase_coef[self.polarization]['z0']
            k_max = self.get_k()
            self.phase = phase0
        else:
            gap_minv, gap_minh = self.calc_min_gap(si_parameters)
            gap_min = gap_minv if self.polarization == 'hp' else gap_minh
            phase0 = self.phase
            self.phase = self.phase_coef[self.polarization]['z0']
            b_max = self.get_beff(gap_min / self.period)
            k_max = self.undulator_b_to_k(b_max, self.period)
            self.phase = phase0
        return k_max


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


class APPLE2(Elliptic):
    """APPLE2 Undulator class.

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
        self._undulator_type = "APPLE2"
        self._label = "APPLE-II"
        self._br = 1.37
        self._polarization = "hp"
        self._efficiency = 1
        self._halbach_coef = {
            "hp": {"a": 1.732, "b": -3.238, "c": 0.0},
            "vp": {"a": 1.926, "b": -5.629, "c": 1.448},
            "cp": {"a": 1.356, "b": -4.875, "c": 0.947},
        }
        self._phase_coef = {
            "hp": {"ef": 1, "z0": 0},
            "vp": {"ef": 1, "z0": 0},
            "cp": {"ef": 1, "z0": 0},
        }
        self._phase = 0
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
    
    def get_beff(self, gap_over_period, phase=None):
        """Get peak magnetic field for a given device and gap.

        Args:
            gap_over_period (float): gap normalized by the undulator period.

        Returns:
            _type_: _description_
        """
        phase = self.phase if phase is None else phase
        br = self.br
        z0 = self.phase_coef[self.polarization]['z0']
        a = self.halbach_coef[self.polarization]["a"]
        b = self.halbach_coef[self.polarization]["b"]
        c = self.halbach_coef[self.polarization]["c"]
        efficiency = self.phase_coef[self.polarization]['ef']
        return (
            efficiency
            * SourceFunctions.beff_function(
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
        b_max = self.get_beff(self.gap / self.period)
        k_max = self.undulator_b_to_k(b_max, self.period)
        return k_max


class Hybrid_Nd(Undulator):
    """Hybrid_Nd Undulator class.

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
        self._undulator_type = "Hybrid"
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


class VPU(Hybrid_Nd):
    """VPU class."""

    def __init__(self, period=29, length=1.5):
        """Class constructor."""
        super().__init__(period, length)
        self._undulator_type = "VPU"
        self._material = "NdFeB"
        self._polarization = "vp"
        self._source_type = "verticalundulator"
        self._label = "VPU"
        self._gap = 9.7


class Hybrid_SmCo(Undulator):
    """Hybrid SmCo Undulator class.

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
        self._undulator_type = "Hybrid"
        self._label = "Hybrid (SmCo)"
        self._br = 1.24
        self._polarization = "hp"
        self._efficiency = 1
        self._halbach_coef = {"hp": {"a": 2.789, "b": -4.853, "c": 1.550}}
        self._period = period
        self._source_length = length
        self._source_type = "linearundulator"


class IVU_NdFeB(Hybrid_Nd):
    """IVU NdFeB Undulator class.

    Args:
        Undulator (Undulator class): Undulator class
    """

    def __init__(self, period, length):
        """Class constructor.

        Args:
            period (float, optional): Undulator period [mm]
            length (float, optional): Undulator length [m]
        """
        super().__init__(period, length)
        self._label = "IVU (Nd)"
        self.vc_thickness = 0
        self.vc_tolerance = 0.2


class CPMU_Nd(IVU_NdFeB):
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
        super().__init__(period, length)
        self._undulator_type = "CPMU"
        self._label = "CPMU (Nd)"
        self._br = 1.5
        self._polarization = "hp"
        self._efficiency = 0.9
        self._halbach_coef = {"hp": {"a": 2.268, "b": -3.895, "c": 0.554}}
        self._source_type = "linearundulator"


class CPMU_PrNd(IVU_NdFeB):
    """CPMU Pr Nd Undulator class.

    Args:
        Undulator (Undulator class): Undulator class
    """

    def __init__(self, period, length):
        """Class constructor.

        Args:
            period (float, optional): Undulator period [mm]
            length (float, optional): Undulator length [m]
        """
        super().__init__(period, length)
        self._undulator_type = "CPMU"
        self._label = "CPMU (PrNd)"
        self._br = 1.62
        self._polarization = "hp"
        self._efficiency = 0.9
        self._halbach_coef = {"hp": {"a": 2.132, "b": -3.692, "c": 0.391}}
        self._source_type = "linearundulator"


class CPMU_Pr(IVU_NdFeB):
    """CPMU Pr Undulator class.

    Args:
        Undulator (Undulator class): Undulator class
    """

    def __init__(self, period, length):
        """Class constructor.

        Args:
            period (float, optional): Undulator period [mm]
            length (float, optional): Undulator length [m]
        """
        super().__init__(period, length)
        self._undulator_type = "CPMU"
        self._label = "CPMU (Pr)"
        self._br = 1.67
        self._polarization = "hp"
        self._efficiency = 0.9
        self._halbach_coef = {"hp": {"a": 2.092, "b": -3.655, "c": 0.376}}
        self._source_type = "linearundulator"


class CPMU_PrFeB_HEPS(IVU_NdFeB):
    """Cpmu PrFeB Undulator class (HEPS).

    Args:
        Undulator (Undulator class): Undulator class
    """

    def __init__(self, period, length):
        """Class constructor.

        Args:
            period (float, optional): Undulator period [mm]
            length (float, optional): Undulator length [m]
        """
        super().__init__(period, length)
        self._undulator_type = "CPMU"
        self._label = "CPMU (PrFeB)"
        self._br = 1.71
        self._polarization = "hp"
        self._efficiency = 1.0
        self._halbach_coef = {
            "hp": {"a": 1.797533, "b": -2.87665627, "c": -0.4065176}
        }
        self._material = "PrFeB"
        self._source_type = "linearundulator"


class CPMU_PrFeB_HEPS_model(IVU_NdFeB):
    """Cpmu PrFeB Undulator class (HEPS).

    Args:
        Undulator (Undulator class): Undulator class
    """

    def __init__(self, period, length):
        """Class constructor.

        Args:
            period (float, optional): Undulator period [mm]
            length (float, optional): Undulator length [m]
        """
        super().__init__(period, length)
        self._undulator_type = "CPMU"
        self._label = "CPMU (PrFeB)"
        self._br = 1.689
        self._polarization = "hp"
        self._efficiency = 1.0
        self._halbach_coef = {
            "hp": {"a": 2.42609676, "b": -4.20036671, "c": 0.79735306}
        }
        self._material = "PrFeB"
        self._source_type = "linearundulator"

"""General tools to use with spectra."""

import numpy as _np
import matplotlib.pyplot as _plt
import mathphys.constants as _constants
from scipy.integrate import cumtrapz
from scipy.optimize import minimize

ECHARGE = _constants.elementary_charge
EMASS = _constants.electron_mass
LSPEED = _constants.light_speed
PLACK = _constants.reduced_planck_constant


class SourceFunctions:
    """Class with generic source methods."""

    @staticmethod
    def undulator_b_to_k(b, period):
        """Given field and period it returns k.

        Args:
            b (float): Field amplitude [T].
            period (float): ID's period [mm].

        Returns:
            float: K (deflection parameter)
        """
        return ECHARGE * 1e-3 * period * b / (2 * _np.pi * EMASS * LSPEED)

    @staticmethod
    def undulator_k_to_b(k, period):
        """Given K and period it returns b.

        Args:
            k (float): Deflection parameter.
            period (float): ID's period [mm].

        Returns:
            float: Field amplitude [T].
        """
        return 2 * _np.pi * EMASS * LSPEED * k / (ECHARGE * 1e-3 * period)

    @staticmethod
    def _generate_field(a, peak, period, nr_periods, pts_period):
        x_1period = _np.linspace(-period / 2, period / 2, pts_period)
        y = peak * _np.sin(2 * _np.pi / period * x_1period)

        x_nperiods = _np.linspace(
            -nr_periods * period / 2,
            nr_periods * period / 2,
            nr_periods * pts_period,
        )
        mid = peak * _np.sin(2 * _np.pi / period * x_nperiods)

        tanh = _np.tanh(a * 2 * _np.pi / period * x_1period)
        if nr_periods % 2 == 1:
            term0 = (tanh + 1) / 2
            term1 = (-tanh + 1) / 2
        else:
            term0 = -(tanh + 1) / 2
            term1 = (tanh - 1) / 2

        out = _np.concatenate((term0 * y, mid, term1 * y))
        x_out = _np.linspace(
            -(2 + nr_periods) * period / 2,
            (2 + nr_periods) * period / 2,
            1 * (2 + nr_periods) * len(x_1period),
        )

        return x_out, out

    @staticmethod
    def _calc_field_integral(a, *args):
        peak = args[0]
        period = args[1]
        nr_periods = args[2]
        pts_period = args[3]
        s, out = SourceFunctions._generate_field(
            a, peak, period, nr_periods, pts_period
        )
        ds = s[1] - s[0]
        i1 = cumtrapz(dx=ds, y=-out)
        i2 = _np.trapz(dx=ds, y=i1)
        return _np.abs(i2)

    @staticmethod
    def create_field_profile(
        nr_periods, period, bx=None, by=None, pts_period=1001
    ):
        """Create a sinusoidal field with first and second integrals zero.

        Args:
            nr_periods (int): Number of periods.
            period (float): ID's period [mm]
            bx (float, optional): horizontal field amplitude. Defaults to None.
            by (float, optional): vertical field amplitude. Defaults to None.
            pts_period (int, optional): Pts per period. Defaults to 1001.

        Returns:
            numpy array: First column contains longitudinal spatial
            coordinate (z) [m], second column contais vertical field
            [T], and third column constais horizontal field [T].
        """
        field = _np.zeros(((nr_periods + 2) * pts_period, 3))

        if by is not None:
            result = minimize(
                SourceFunctions._calc_field_integral,
                0.4,
                args=(by, period, nr_periods, pts_period),
            )
            s, by = SourceFunctions._generate_field(
                result.x, by, period, nr_periods, pts_period
            )
            field[:, 1] = by

        if bx is not None:
            result = minimize(
                SourceFunctions._calc_field_integral,
                0.4,
                args=(bx, period, nr_periods, pts_period),
            )
            s, bx = SourceFunctions._generate_field(
                result.x, bx, period, nr_periods, pts_period
            )
            field[:, 2] = bx

        field[:, 0] = 1e-3 * s
        return field

    @staticmethod
    def get_harmonic_wavelength(n, gamma, theta, period, k):
        """Get harmonic wavelength.

        Args:
            n (int): harmonic number
            gamma (float): lorentz factor
            theta (float): Observation angle [mrad]
            period (float): Undulator period [mm]
            k (float): Deflection parameter

        Returns:
            float: Harmonic wavelength [m].
        """
        return (
            1e-3
            * period
            / (n * 2 * gamma**2)
            * (1 + 0.5 * k**2 + (gamma * 1e-3 * theta) ** 2)
        )

    @staticmethod
    def get_harmonic_energy(n, gamma, theta, period, k):
        """Get harmonic energy.

        Args:
            n (int): harmonic number
            gamma (float): lorentz factor
            theta (float): Observation angle [mrad]
            period (float): Undulator period [mm]
            k (float): Deflection parameter

        Returns:
            float: Harmonic energy [eV].
        """
        lamb = SourceFunctions.get_harmonic_wavelength(
            n, gamma, theta, period, k
        )
        energy = PLACK * 2 * _np.pi * LSPEED / lamb / ECHARGE
        return energy

    @staticmethod
    def calc_k_given_1sth(energy, gamma, period):
        """Calc k given first harmonic.

        Args:
            energy (float): energy of first [eV]
            gamma (float): lorentz factor
            period (float): Undulator period [mm]
        """
        k = _np.sqrt(8 * PLACK * _np.pi * gamma**2 / (energy * period) - 2)
        return k

    @staticmethod
    def get_hybrid_devices():
        """Get a list of all hybrid devices.

        Returns:
            list of str: list of all hybrid devices
        """
        hybrids = [
            "cpmu_pr",
            "cpmu_prnd",
            "cpmu_nd",
            "ivu",
            "hybrid",
            "ivu_smco",
            "hybrid_smco",
        ]
        return hybrids

    @staticmethod
    def get_ppm_devices():
        """Get a list of all pure permanent magnet (PPM) devices.

        Returns:
            list of str: list of all PPM devices
        """
        ppms = [
            "planar",
            "apple2",
            "delta",
            "delta_prototype",
        ]
        return ppms

    @staticmethod
    def get_planar_devices():
        """Get a list of all planar devices.

        Returns:
            list of str: list of all planar devices
        """
        planars = [
            "cpmu_pr",
            "cpmu_prnd",
            "cpmu_nd",
            "ivu",
            "hybrid",
            "ivu_smco",
            "hybrid_smco",
            "planar",
        ]
        return planars

    @staticmethod
    def get_invacuum_devices():
        """Get a list of all in-vacuum devices.

        Returns:
            list of str: list of all in-vacuum devices
        """
        ivus = [
            "ivu",
            "ivu_smco",
            "cpmu_nd",
            "cpmu_pr",
            "cpmu_prnd",
        ]
        return ivus

    @staticmethod
    def get_cpmu_devices():
        """Get a list of all cryogenic permanent magnet devices.

        Returns:
            list of str: list of all CPMU devices
        """
        cpmus = [
            "cpmu_nd",
            "cpmu_pr",
            "cpmu_prnd",
        ]
        return cpmus


    @staticmethod
    def calc_min_gap(
        device,
        length,
        polarization,
        section,
        dg_out_vacuum=1.0,
        dg_in_vacuum=0.2,
        dg_delta_prototype=0.6,
        delta_gap=0.0,
    ):
        """Calculate minimum gap due to beam stay clear, vacuum chamber and tolerances, given a device type and length.

        Args:
            device (str): device label
            length (float): length of the device in [m]
            polarization (str): light polarization 'hp', 'vp' or 'cp'
            section (str): straight section 'sb', 'sp' or 'sa'
            dg_out_vacuum (float, optional): Vacuum chamber wall thickness (x2) in [mm]. Defaults to 1.0.
            dg_in_vacuum (float, optional): tolerances + sheet thickness. Defaults to 0.2.
            dg_delta_prototype (float, optional): Vacuum chamber wall thickness (x2) in [mm]. Defaults to 0.6.
            delta_gap (float, optional): Addicional tolerance for the gap. Defaults to 0.0.

        Returns:
            float: minimum gap in [mm]
        """

        pos = length / 2

        if device == "delta_prototype":
            delta_prototype_chamber = True
            pos = pos + 0.1
        else:
            delta_prototype_chamber = False

        bsc_h, bsc_v = SourceFunctions.calc_beam_stay_clear(
            pos,
            section=section,
            delta_prototype_chamber=delta_prototype_chamber,
        )

        planar_devices = SourceFunctions.get_planar_devices()

        ivu_devices = SourceFunctions.get_invacuum_devices()

        if device in planar_devices:
            if polarization.lower() == "vp":
                gap0 = 2 * bsc_h
            else:
                gap0 = 2 * bsc_v
        elif device == "apple2":
            gap0 = 2 * bsc_v
        elif device in ("delta", "delta_prototype"):
            gap0 = _np.sqrt(2 * (bsc_v**2 + bsc_h**2))
        else:
            raise Exception("device not found: ", device)

        if device in ivu_devices:
            gap = gap0 + dg_in_vacuum
        elif device == "delta_prototype":
            gap = gap0 + dg_delta_prototype
        else:
            gap = gap0 + dg_out_vacuum

        return gap + delta_gap

    @staticmethod
    def beff_function(gap_over_period, br, a, b, c):
        """Calculate peak magnetic field for a given gap (Halbach equation).

        Args:
            gap_over_period (float): gap normalized by the undulator period.
            br (float): Remanent field in [T]
            a (float): Halbach coefficient 'a'.
            b (float): Halbach coefficient 'b'.
            c (float): Halbach coefficient 'c'.

        Returns:
            float: Effective field B_eff
        """
        return a * br * _np.exp(b * gap_over_period + c * (gap_over_period**2))

    @staticmethod
    def _get_list_of_pol(undulator_type):
        polarizations = dict()
        polarizations["apple2"] = ["hp", "vp", "cp"]
        polarizations["delta"] = ["hp", "vp", "cp"]
        polarizations["planar"] = ["hp"]
        polarizations["hybrid"] = ["hp"]
        polarizations["cpmu_nd"] = ["hp"]
        polarizations["cpmu_prnd"] = ["hp"]
        polarizations["cpmu_pr"] = ["hp"]

        return polarizations[undulator_type]

    @staticmethod
    def get_polarization_label(polarization):
        """Get polarization string.

        Args:
            polarization (str): Light polarization 'hp', 'vp' or 'cp'.

        Returns:
            str: polarization label
        """
        if polarization == "hp":
            label = "Horizontal Polarization"

        elif polarization == "vp":
            label = "Vertical Polarization"

        elif polarization == "cp":
            label = "Circular Polarization"

        return label

    @staticmethod
    def get_undulator_prefix(device, polarization):
        """Generate a string prefix for a device and polarization.

        Args:
            device (str): Device label.
            polarization (str): Light polarization 'hp', 'vp' or 'cp'.

        Returns:
            str: prefix string
        """
        return "und_" + device + "_" + polarization


class Undulator(SourceFunctions):
    """Main class for undulators.

    Args:
        SourceFunctions (SourceFunctions class): S. F. functions
    """

    def __init__(self):
        """Class constructor."""
        self._undulator_type = "planar"
        self._br = 1.37
        self._period = 50
        self._length = 1.0
        self._efficiency = 1.0
        self._label = "label"
        self._polarization = "hp"
        self._halbach_coef = dict()

    @property
    def undulator_type(self):
        """Undulator type.

        Returns:
            str: Undulator type.
        """
        return self._undulator_type

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
    def length(self):
        """Undulator length.

        Returns:
            float: Length [m]
        """
        return self._length

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

    @undulator_type.setter
    def undulator_type(self, value):
        self._undulator_type = value

    @br.setter
    def br(self, value):
        self._br = value

    @period.setter
    def period(self, value):
        self._period = value

    @length.setter
    def length(self, value):
        self._length = value

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
        self._label = 'Planar'
        self._br = 1.37
        self._polarization = "hp"
        self._efficiency = 1
        self._halbach_coef = {"hp": {"a": 1.732, "b": -3.238, "c": 0.0}}
        self._period = period
        self._length = length


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
        self._label = 'Apple-II'
        self._br = 1.37
        self._polarization = "hp"
        self._efficiency = 1
        self._halbach_coef = {
                "hp": {"a": 1.732, "b": -3.238, "c": 0.0},
                "vp": {"a": 1.926, "b": -5.629, "c": 1.448},
                "cp": {"a": 1.356, "b": -4.875, "c": 0.947},
            }
        self._period = period
        self._length = length


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
        self._label = 'Delta'
        self._br = 1.37
        self._polarization = "hp"
        self._efficiency = 1
        self._halbach_coef = {
                "hp": {"a": 1.696, "b": -2.349, "c": -0.658},
                "vp": {"a": 1.696, "b": -2.349, "c": -0.658},
                "cp": {"a": 1.193, "b": -2.336, "c": -0.667},
            }
        self._period = period
        self._length = length


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
        self._label = 'Hybrid (Nd)'
        self._br = 1.24
        self._polarization = "hp"
        self._efficiency = 1
        self._halbach_coef = {"hp": {"a": 2.552, "b": -4.431, "c": 1.101}}
        self._period = period
        self._length = length


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
        self._label = 'Hybrid (SmCo)'
        self._br = 1.24
        self._polarization = "hp"
        self._efficiency = 1
        self._halbach_coef = {"hp": {"a": 2.789, "b": -4.853, "c": 1.550}}
        self._period = period
        self._length = length


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
        self._label = 'CPMU (Nd)'
        self._br = 1.5
        self._polarization = "hp"
        self._efficiency = 0.9
        self._halbach_coef = {"hp": {"a": 2.268, "b": -3.895, "c": 0.554}}
        self._period = period
        self._length = length


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
        self._label = 'CPMU (Pr,Nd)'
        self._br = 1.62
        self._polarization = "hp"
        self._efficiency = 0.9
        self._halbach_coef = {"hp": {"a": 2.132, "b": -3.692, "c": 0.391}}
        self._period = period
        self._length = length


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
        self._label = 'CPMU (Pr)'
        self._br = 1.67
        self._polarization = "hp"
        self._efficiency = 0.9
        self._halbach_coef = {"hp": {"a": 2.092, "b": -3.655, "c": 0.376}}
        self._period = period
        self._length = length

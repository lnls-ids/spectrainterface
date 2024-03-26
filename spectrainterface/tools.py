"""General tools to use with spectra."""

import numpy as _np
import matplotlib.pyplot as _plt
import mathphys.constants as _constants
from scipy.integrate import cumtrapz
from scipy.optimize import minimize
from scipy.special import erf
from spectrainterface.accelerator import StorageRingParameters

ECHARGE = _constants.elementary_charge
EMASS = _constants.electron_mass
LSPEED = _constants.light_speed
PLANCK = _constants.reduced_planck_constant
PI = _np.pi


class SourceFunctions:
    """Class with generic source methods."""

    def __init__(self):
        """Class constructor."""
        self._source_type = None
        self._source_length = None

    @staticmethod
    def undulator_b_to_k(b, period):
        """Given field and period it returns k.

        Args:
            b (float): Field amplitude [T].
            period (float): ID's period [mm].

        Returns:
            float: K (deflection parameter)
        """
        return ECHARGE * 1e-3 * period * b / (2 * PI * EMASS * LSPEED)

    @staticmethod
    def undulator_k_to_b(k, period):
        """Given K and period it returns b.

        Args:
            k (float): Deflection parameter.
            period (float): ID's period [mm].

        Returns:
            float: Field amplitude [T].
        """
        return 2 * PI * EMASS * LSPEED * k / (ECHARGE * 1e-3 * period)

    @staticmethod
    def _generate_field(a, peak, period, nr_periods, pts_period):
        x_1period = _np.linspace(-period / 2, period / 2, pts_period)
        y = peak * _np.sin(2 * PI / period * x_1period)

        x_nperiods = _np.linspace(
            -nr_periods * period / 2,
            nr_periods * period / 2,
            nr_periods * pts_period,
        )
        mid = peak * _np.sin(2 * PI / period * x_nperiods)

        tanh = _np.tanh(a * 2 * PI / period * x_1period)
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
    def create_field_profile(nr_periods, period, bx=None, by=None, pts_period=1001):
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
        lamb = SourceFunctions.get_harmonic_wavelength(n, gamma, theta, period, k)
        energy = PLANCK * 2 * PI * LSPEED / lamb / ECHARGE
        return energy

    @staticmethod
    def calc_k_given_1sth(energy, gamma, period):
        """Calc k given first harmonic.

        Args:
            energy (float): energy of first [eV]
            gamma (float): lorentz factor
            period (float): Undulator period [mm]
        """
        k = _np.sqrt(8 * PLANCK * PI * gamma**2 / (energy * period) - 2)
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
        polarizations["hybrid"] = ["hp", "vp"]
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

    @property
    def source_length(self):
        """Source length.

        Returns:
            float: Length [m]
        """
        return self._source_length

    @source_length.setter
    def source_length(self, value):
        self._source_length = value

    @property
    def source_type(self):
        """Source type.

        Returns:
            String: String according to spectrainterface.calc.source_type
        """
        return self._source_type

    @staticmethod
    def calc_beam_size_and_div(
        emittance, beta, energy_spread, und_length, und_period, photon_energy, harmonic
    ):
        """Calculates the RMS size and divergence of the undulator radiation for given
        electron beam parameters, taking into account the beam energy spread effect on 
        the harmonics.

        Args:
            emittance (float): electron beam emittance (hor. or vert.) in [m.rad].
            beta (float): betraton function (hor. or vert.) in [m].
            energy_spread (float): electron beam relative energy spread.
            und_length (float): undulator length in [m].
            und_period (float): undulator period in [mm].
            photon_energy (float): photon energy in [eV].
            harmonic (int): radiation harmonic number.

        Ref:
            T. Tanaka and H. Kitamura, J. Synchrotron Rad. (2009). 16, 380-386.

        Returns:
            float, float: RMS size [m], RMS divergence [rad]
        """

        hc = PLANCK * 2 * PI * LSPEED / ECHARGE

        x = 2 * PI * harmonic * und_length / und_period * 1e-3 * energy_spread
        a1 = _np.sqrt(2 * PI) * x * erf(_np.sqrt(2) * x)
        Qax = _np.sqrt(2 * x**2 / (-1 + _np.exp(-2 * x**2) + a1))
        div_sigma = _np.sqrt(
            emittance / beta + (hc / photon_energy) / (2 * und_length) * Qax**2
        )

        a1s = _np.sqrt(2 * PI) * (x / 4) * erf(_np.sqrt(2) * (x / 4))
        Qas = (
            _np.sqrt(2 * (x / 4) ** 2 / (-1 + _np.exp(-2 * (x / 4) ** 2) + a1s))
        ) ** (2 / 3.0)
        size_sigma = _np.sqrt(
            emittance * beta
            + und_length * hc / (2 * (PI**2) * photon_energy) * (Qas) ** 2
        )

        return size_sigma, div_sigma

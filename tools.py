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

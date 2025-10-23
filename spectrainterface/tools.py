"""General tools to use with spectra."""

import numpy as _np
import mathphys.constants as _constants
from scipy.integrate import cumtrapz
from scipy.optimize import minimize
from scipy.special import erf

ECHARGE = _constants.elementary_charge
EMASS = _constants.electron_mass
LSPEED = _constants.light_speed
PLANCK = _constants.reduced_planck_constant
PI = _np.pi
VACUUM_PERMITTICITY = _constants.vacuum_permitticity


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
    def undulator_k_to_gap(k, period, br, a, b, c):
        """Calculate undulator gap for a k value (Halbach equation).

        Args:
            k (float): K value deflection parameter.
            period (float): ID's period [mm].
            br (float): Remanent field in [T].
            a (float): Halbach coefficient 'a'.
            b (float): Halbach coefficient 'b'.
            c (float): Halbach coefficient 'c'.

        Returns:
            float: gap [mm]
        """
        beff = 2 * PI * EMASS * LSPEED * k / (ECHARGE * 1e-3 * period)
        delta = b**2 + 4 * c * _np.log(beff / (a * br))

        if c != 0:
            gap1 = period * (-b + _np.sqrt(delta)) / (2 * c)
            gap2 = period * (-b - _np.sqrt(delta)) / (2 * c)

            if type(k) is float or type(k) is _np.float64:
                if gap1 > 0 and gap2 > 0:
                    return _np.min([gap1, gap2])
                else:
                    return _np.max([gap1, gap2])
            elif type(k) is _np.ndarray:
                gap1_mask = _np.zeros(gap1.shape)
                gap1_mask[gap1 > 0] = 1

                gap2_mask = _np.zeros(gap2.shape)
                gap2_mask[gap2 > 0] = 1

                idx_major = _np.where(gap1_mask * gap2_mask == 1)[0]
                idx_minor = _np.where(gap1_mask * gap2_mask == 0)[0]

                gaps_major = _np.array([gap1[idx_major], gap2[idx_major]])
                gaps_major = _np.min(gaps_major, axis=0)

                gaps_minor = _np.array([gap1[idx_minor], gap2[idx_minor]])
                gaps_minor = _np.max(gaps_minor, axis=0)

                new_gaps = _np.zeros(k.shape)
                if idx_major.shape[0] > 0:
                    new_gaps[idx_major] = gaps_major
                if idx_minor.shape[0] > 0:
                    new_gaps[idx_minor] = gaps_minor

                return new_gaps
            else:
                raise ValueError('k must be array numpy or number')
        else:
            return (period / b) * _np.log(beff / (a * br))

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
            'cpmu_pr',
            'cpmu_prnd',
            'cpmu_nd',
            'ivu',
            'hybrid',
            'ivu_smco',
            'hybrid_smco',
        ]
        return hybrids

    @staticmethod
    def get_ppm_devices():
        """Get a list of all pure permanent magnet (PPM) devices.

        Returns:
            list of str: list of all PPM devices
        """
        ppms = [
            'planar',
            'apple2',
            'delta',
            'delta_prototype',
        ]
        return ppms

    @staticmethod
    def get_planar_devices():
        """Get a list of all planar devices.

        Returns:
            list of str: list of all planar devices
        """
        planars = [
            'cpmu_pr',
            'cpmu_prnd',
            'cpmu_nd',
            'ivu',
            'hybrid',
            'ivu_smco',
            'hybrid_smco',
            'planar',
        ]
        return planars

    @staticmethod
    def get_invacuum_devices():
        """Get a list of all in-vacuum devices.

        Returns:
            list of str: list of all in-vacuum devices
        """
        ivus = [
            'ivu',
            'ivu_smco',
            'cpmu_nd',
            'cpmu_pr',
            'cpmu_prnd',
        ]
        return ivus

    @staticmethod
    def get_cpmu_devices():
        """Get a list of all cryogenic permanent magnet devices.

        Returns:
            list of str: list of all CPMU devices
        """
        cpmus = [
            'cpmu_nd',
            'cpmu_pr',
            'cpmu_prnd',
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
        polarizations['APPLE2'] = ['hp', 'vp', 'cp', 'lp54y', 'lp54x']
        polarizations['DELTA'] = ['hp', 'vp', 'cp']
        polarizations['Halbach'] = ['hp']
        polarizations['Hybrid'] = ['hp', 'vp']
        polarizations['CPMU'] = ['hp']
        polarizations['wiggler'] = ['hp']
        polarizations['APU'] = ['hp']
        polarizations['VPU'] = ['vp']

        return polarizations[undulator_type]

    @staticmethod
    def get_polarization_label(polarization):
        """Get polarization string.

        Args:
            polarization (str): Light polarization 'hp', 'vp' or 'cp'.

        Returns:
            str: polarization label
        """
        if polarization == 'hp':
            label = 'Horizontal Polarization'

        elif polarization == 'vp':
            label = 'Vertical Polarization'

        elif polarization == 'cp':
            label = 'Circular Polarization'

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
        return 'und_' + device + '_' + polarization

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
        emittance,
        beta,
        energy_spread,
        und_length,
        und_period,
        photon_energy,
        harmonic,
    ):
        """Calculates the RMS size and divergence of the undulator radiation.

        Args:
            emittance (float): electron beam emittance (hor. or vert.)
                in [m.rad].
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

        x = (
            2
            * PI
            * harmonic
            * und_length
            / (und_period * 1e-3)
            * energy_spread
        )
        a1 = _np.sqrt(2 * PI) * x * erf(_np.sqrt(2) * x)
        Qax = _np.sqrt(2 * x**2 / (-1 + _np.exp(-2 * x**2) + a1))
        div_sigma = _np.sqrt(
            emittance / beta + (hc / photon_energy) / (2 * und_length) * Qax**2
        )

        a1s = _np.sqrt(2 * PI) * (x / 4) * erf(_np.sqrt(2) * (x / 4))
        Qas = (
            _np.sqrt(
                2 * (x / 4) ** 2 / (-1 + _np.exp(-2 * (x / 4) ** 2) + a1s)
            )
        ) ** (2 / 3.0)
        size_sigma = _np.sqrt(
            emittance * beta
            + und_length * hc / (2 * (PI**2) * photon_energy) * (Qas) ** 2
        )

        return size_sigma, div_sigma

    @staticmethod
    def get_min_or_max_k(
        period, photon_energy, k_extreme, what_harmonic='max', si_energy=3.0
    ):
        """Get max or min K-value and harmonic number for a given energy.

        Args:
            period (float): Undulator period in [mm].
            photon_energy (float): Photon energy in [eV].
            k_extreme (float): Max. or Min. K-value
            what_harmonic (str, optional): Either 'min', 'max' or 'first'.
                Defaults to 'max'.
            si_energy (float, optional): Storage ring energy in [GeV].
                 Defaults to 3.0.

        Returns:
            int, float, float: harmonic number, K-value, B (field) in [T].
        """
        gamma = si_energy * 1e9 * ECHARGE / (EMASS * LSPEED**2)
        n = [2 * i + 1 for i in range(50)]
        harmonic = _np.nan
        k = _np.nan
        for h_n in n:
            K2 = (
                8
                * h_n
                * PI
                * (PLANCK / ECHARGE)
                * LSPEED
                * (gamma**2)
                / (period * 1e-3)
                / photon_energy
                - 2
            )
            if K2 > 0:
                if what_harmonic == 'max':
                    if K2**0.5 < k_extreme:
                        harmonic = h_n
                        k = K2**0.5
                    else:
                        break
                elif what_harmonic == 'min':
                    if K2**0.5 > k_extreme:
                        harmonic = h_n
                        k = K2**0.5
                        break
                elif what_harmonic == 'first':
                    harmonic = h_n
                    k = K2**0.5
                    break

        if _np.isnan(k):
            B = _np.nan
            harmonic = _np.nan
        else:
            B = 2 * PI * EMASS * LSPEED * k / (ECHARGE * period * 1e-3)

        return harmonic, k, B

    @staticmethod
    def calc_total_power(gamma, field_profile, current):
        """Calculate total power from an source light.

        Args:
            gamma (float): lorentz factor
            field_profile (numpy matrix float): First column contains
                longitudinal spatial coordinate (z) [m]
                (end position of each segment);
                Second column contais vertical field [T];
                Third column constais horizontal field [T].
            current (float): current of beam [mA]
        Ref:
            James A. Clarke. The science and technology of undulators and
                wigglers. (2004), 47-48.

        Returns:
            float: Total power of source light [kW]
        """
        s_f = field_profile[:, 0]
        by = field_profile[:, 1]
        bx = field_profile[:, 2]

        s_i = _np.delete(_np.append([0], s_f), -1)
        b = _np.sqrt(by**2 + bx**2)

        ds = s_f - s_i

        energy = gamma * (EMASS * LSPEED**2) / ECHARGE
        b_rho = (energy * ECHARGE) / (LSPEED * ECHARGE)
        const = 1e-3 * (ECHARGE * gamma**4) / (6 * PI * VACUUM_PERMITTICITY)

        iradius = b / b_rho

        darg_integral = iradius**2 * ds
        integral = _np.sum(darg_integral)

        return const * integral * (current * 1e-3)

    @staticmethod
    def calc_k_target(
        gamma: float, n: int, period: float, target_energy: float
    ):
        """Calc k for target energy given harmonic number and period.

        Args:
            gamma (float): Lorentz factor
            n (int): harmonic number.
            period (float): undulator period [mm].
            target_energy (float): target energy of radiation [eV].

        Returns:
            float: K value
        """
        arg = (
            2
            * n
            * gamma**2
            * PLANCK
            * 2
            * _np.pi
            * LSPEED
            / (target_energy * ECHARGE * 1e-3 * period)
            - 1
        )
        return _np.sqrt(2) * _np.sqrt(arg)

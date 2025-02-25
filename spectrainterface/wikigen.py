"""Graphs generator functions."""

import numpy as _np
import matplotlib.pyplot as _plt
from matplotlib import colors
import matplotlib.patches as _patches
from spectrainterface.interface import SpectraInterface
import copy
import mathphys
import importlib
import inspect

ECHARGE = mathphys.constants.elementary_charge
EMASS = mathphys.constants.electron_mass
LSPEED = mathphys.constants.light_speed
ECHARGE_MC = ECHARGE / (2 * _np.pi * EMASS * LSPEED)
PLANCK = mathphys.constants.reduced_planck_constant
VACUUM_PERMITTICITY = mathphys.constants.vacuum_permitticity
PI = _np.pi


class FunctionsManipulation:
    """Manipulation generic Spectra Interface functions to generate graphs"""

    @staticmethod
    def process_flux_distribuition_2d(spectra, source, calc_params):
        distance_from_source = calc_params.distance_from_source
        slit_shape = calc_params.slit_shape
        slit_acceptance = calc_params.slit_acceptance
        slit_acceptance = (
            calc_params.slit_acceptance[0] / distance_from_source,
            calc_params.slit_acceptance[1] / distance_from_source,
        )
        slit_position = calc_params.slit_position
        slit_position = (
            slit_position[0] / distance_from_source,
            slit_position[1] / distance_from_source,
        )
        target_energy = calc_params.target_energy
        x_range = calc_params.x_range
        x_range = (
            x_range[0] / distance_from_source,
            x_range[1] / distance_from_source,
        )
        y_range = calc_params.y_range
        y_range = (
            y_range[0] / distance_from_source,
            y_range[1] / distance_from_source,
        )
        x_nr_pts = calc_params.x_nr_pts
        y_nr_pts = calc_params.y_nr_pts

        savefig = calc_params.savefig
        figsize = calc_params.figsize
        dpi = calc_params.dpi

        target_k = calc_params.target_k
        target_k = target_k if source.source_type != "bendingmagnet" else 0

        spectra_calc: SpectraInterface = copy.deepcopy(spectra)
        result = spectra_calc.calc_flux_distribuition_2d(
            source,
            target_energy=target_energy,
            target_k=target_k,
            x_range=x_range,
            x_nr_pts=x_nr_pts,
            y_range=y_range,
            y_nr_pts=y_nr_pts,
            distance_from_source=distance_from_source,
        )
        flux_total = spectra_calc.calc_partial_flux(
            source,
            target_energy=target_energy,
            target_k=target_k,
            slit_shape=slit_shape,
            slit_acceptance=slit_acceptance,
            slit_position=slit_position,
            distance_from_source=distance_from_source,
        )
        if source.source_type != "bendingmagnet":
            title = "Flux Density\nEnergy: {:.2f} keV, z = {:.1f}\n{:} ({:.1f} m, {:.2f} mm)".format(  # noqa: E501
                target_energy * 1e-3,
                distance_from_source,
                source.label,
                source.source_length,
                source.period,
            )
            figname = "flux_density_{:}_{:.0f}m_{:.0f}mm_{:.0f}keV".format(
                source.label,
                source.source_length,
                source.period,
                target_energy * 1e-3,
            )
        else:
            title = "Flux Density\nEnergy: {:.2f} keV, z = {:.1f}\n{:}".format(  # noqa: E501
                target_energy * 1e-3,
                distance_from_source,
                source.label,
            )
            figname = "flux_density_{:}_{:.0f}keV".format(
                source.label, target_energy * 1e-3
            )
        fig = _plt.figure(figsize=(figsize[0], figsize[0]))
        ax = fig.add_subplot(111)

        ax.set_title(title, fontsize=9)
        im = ax.imshow(
            result,
            extent=[
                x_range[0],
                x_range[1],
                y_range[0],
                y_range[1],
            ],
            aspect="equal",
        )
        ax.text(
            x=x_range[0] * (1 - 0.05),
            y=y_range[1] * (1 - 0.13),
            s="Tot.Flux: {:.2e} [ph/s/0.1%/100mA]".format(flux_total[0]),
            fontsize=8,
            c="white",
        )

        if slit_shape == "retslit":
            ax.text(
                x=x_range[0] * (1 - 0.05),
                y=y_range[0] * (1 - 0.05),
                s="{:.3f} x {:.3f} mm²".format(
                    slit_acceptance[0] * distance_from_source,
                    slit_acceptance[1] * distance_from_source,
                ),
                fontsize=8,
                c="white",
            )
            patch = _patches.Rectangle(
                (
                    (slit_position[0] - slit_acceptance[0] / 2),
                    (slit_position[1] - slit_acceptance[1] / 2),
                ),
                slit_acceptance[0],
                slit_acceptance[1],
                fc=(0, 0, 0, 0),
                ec="black",
                lw=1,
                ls=":",
            )
            ax.add_patch(patch)
        else:
            ax.text(
                x=x_range[0] * (1 - 0.05),
                y=y_range[0] * (1 - 0.17),
                s=r"$R_1:$"
                + "{:.1f} mm".format(
                    slit_acceptance[0] * distance_from_source
                ),
                fontsize=8,
                c="white",
            )
            ax.text(
                x=x_range[0] * (1 - 0.05),
                y=y_range[0] * (1 - 0.05),
                s=r"$R_2:$"
                + "{:.1f} mm".format(
                    slit_acceptance[1] * distance_from_source
                ),
                fontsize=8,
                c="white",
            )
            patch = _patches.Circle(
                (slit_position[0], slit_position[1]),
                slit_acceptance[0],
                fc=(0, 0, 0, 0),
                ec="black",
                lw=1,
                ls=":",
            )
            ax.add_patch(patch)
            patch = _patches.Circle(
                (slit_position[0], slit_position[1]),
                slit_acceptance[1],
                fc=(0, 0, 0, 0),
                ec="black",
                lw=1,
                ls=":",
            )
            ax.add_patch(patch)
        ax.tick_params(labelsize=8)
        sm = _plt.cm.ScalarMappable(
            _plt.Normalize(
                vmin=_np.min(result / (distance_from_source**2)),
                vmax=_np.max(result / (distance_from_source**2)),
            )
        )
        sm.set_array(result / (distance_from_source**2))

        cbar = fig.colorbar(
            sm,
            ax=ax,
            format="%.1e",
            shrink=0.5,
        )
        cbar.set_label(label="Flux Density [ph/s/mm²/0.1%/100mA]", size=8)
        cbar.set_ticks(
            _np.linspace(
                _np.min(result / (distance_from_source**2)),
                _np.max(result / (distance_from_source**2)),
                5,
            )
        )
        cbar.ax.tick_params(labelsize=8)
        ax.set_xlabel("X [mm]", fontsize=8)
        ax.set_ylabel("Y [mm]", fontsize=8)
        _plt.tight_layout()
        if savefig:
            _plt.savefig(figname, dpi=dpi)

    @staticmethod
    def process_beam_size(spectra, source, calc_params):
        spectra_calc = copy.deepcopy(spectra)
        if (
            source.source_type == "bendingmagnet"
            or source.source_type == "wiggler"
        ):
            return 0
        title = "Beam Size\n{:} ({:.1f} m, {:.2f} mm)".format(
                source.label, source.source_length, source.period
            )
        xlim = calc_params.e_range
        xscale = "linear"
        yscale = "linear"
        linewidth = calc_params.linewidth
        savefig = calc_params.savefig
        figsize = calc_params.figsize
        dpi = calc_params.dpi

        _plt.figure(figsize=figsize)

        source_period = source.period
        source_length = source.source_length
        if source.gap == 0:
            source_k_max = source.calc_max_k(spectra_calc.accelerator)
        else:
            source_k_max = source.get_k()

        coupling_const = spectra_calc.accelerator.coupling_constant
        nat_emittance = spectra_calc.accelerator.nat_emittance
        x_emittance = (1 / (1 + coupling_const)) * nat_emittance
        y_emittance = (coupling_const / (1 + coupling_const)) * nat_emittance
        beta_x = spectra_calc.accelerator.betax
        beta_y = spectra_calc.accelerator.betay
        energy_spread = spectra_calc.accelerator.energy_spread

        ne = 5001
        dimensions_beam = _np.zeros((ne, 4))
        energies = _np.linspace(100, xlim[1] * 1e3, ne)

        for i in range(ne):
            energy = energies[i]
            h, K, B = source.get_min_or_max_k(
                source_period, energy, source_k_max, "max"
            )
            sizex, divx = source.calc_beam_size_and_div(
                x_emittance,
                beta_x,
                energy_spread,
                source_length,
                source_period,
                energy,
                h,
            )
            sizey, divy = source.calc_beam_size_and_div(
                y_emittance,
                beta_y,
                energy_spread,
                source_length,
                source_period,
                energy,
                h,
            )
            dimensions_beam[i, 0] = sizex
            dimensions_beam[i, 1] = sizey
            dimensions_beam[i, 2] = divx
            dimensions_beam[i, 3] = divy
        _plt.title(title)
        _plt.plot(
            energies * 1e-3,
            dimensions_beam[:, 0] * 1e6,
            "-C0",
            label=r"$\sigma_{x}$",
            linewidth=linewidth,
        )
        _plt.plot(
            energies * 1e-3,
            dimensions_beam[:, 1] * 1e6,
            "-C1",
            label=r"$\sigma_{y}$",
            linewidth=linewidth,
        )
        y_lim = max(dimensions_beam[-1, 0], dimensions_beam[-1, 1])
        y_lim *= 1e6

        _plt.xlabel("Energy [keV]")
        _plt.ylabel("RMS beam size [\u03bcm]")
        _plt.legend(loc=1, ncol=2, fontsize=9)
        _plt.xlim(*xlim)
        _plt.xscale(xscale)
        _plt.ylim(0, y_lim - y_lim % 5 + 15)
        _plt.yscale(yscale)
        _plt.tick_params(
            which="both", axis="both", direction="in", right=True, top=True
        )
        _plt.tight_layout()
        _plt.grid(which="major", alpha=0.3)
        _plt.grid(which="minor", alpha=0.1)
        _plt.minorticks_on()
        if savefig:
            _plt.savefig(
                "beam_size_{:}_{:.0f}m_{:.0f}mm.png".format(
                    source.label, source.source_length, source.period
                ),
                dpi=dpi,
            )

    @staticmethod
    def process_beam_divergence(spectra, source, calc_params):
        spectra_calc = copy.deepcopy(spectra)
        if (
            source.source_type == "bendingmagnet"
            or source.source_type == "wiggler"
        ):
            return 0
        title = "Beam Divergence\n{:} ({:.1f} m, {:.2f} mm)".format(
                source.label, source.source_length, source.period
            )
        xlim = calc_params.e_range
        xscale = "linear"
        yscale ="linear"
        linewidth = calc_params.linewidth
        savefig = calc_params.savefig
        figsize = calc_params.figsize

        _plt.figure(figsize=figsize)

        source_period = source.period
        source_length = source.source_length
        if source.gap == 0:
            source_k_max = source.calc_max_k(spectra_calc.accelerator)
        else:
            source_k_max = source.get_k()

        coupling_const = spectra_calc.accelerator.coupling_constant
        nat_emittance = spectra_calc.accelerator.nat_emittance
        x_emittance = (1 / (1 + coupling_const)) * nat_emittance
        y_emittance = (coupling_const / (1 + coupling_const)) * nat_emittance
        beta_x = spectra_calc.accelerator.betax
        beta_y = spectra_calc.accelerator.betay
        energy_spread = spectra_calc.accelerator.energy_spread

        ne = 5001
        dimensions_beam = _np.zeros((ne, 4))
        energies = _np.linspace(100, xlim[1] * 1e3, ne)

        for i in range(ne):
            energy = energies[i]
            h, K, B = source.get_min_or_max_k(
                source_period, energy, source_k_max, "max"
            )
            sizex, divx = source.calc_beam_size_and_div(
                x_emittance,
                beta_x,
                energy_spread,
                source_length,
                source_period,
                energy,
                h,
            )
            sizey, divy = source.calc_beam_size_and_div(
                y_emittance,
                beta_y,
                energy_spread,
                source_length,
                source_period,
                energy,
                h,
            )
            dimensions_beam[i, 0] = sizex
            dimensions_beam[i, 1] = sizey
            dimensions_beam[i, 2] = divx
            dimensions_beam[i, 3] = divy

        _plt.title(title)
        _plt.plot(
            energies * 1e-3,
            dimensions_beam[:, 2] * 1e6,
            "-C0",
            label=r"$\sigma'_{x}$",
            linewidth=linewidth,
        )
        _plt.plot(
            energies * 1e-3,
            dimensions_beam[:, 3] * 1e6,
            "-C1",
            label=r"$\sigma'_{y}$",
            linewidth=linewidth,
        )
        y_lim = max(dimensions_beam[-1, 2], dimensions_beam[-1, 3])
        y_lim *= 1e6

        _plt.xlabel("Energy [keV]")
        _plt.ylabel("RMS beam divergence [\u03bcrad]")
        _plt.legend(loc=1, ncol=2, fontsize=9)
        _plt.xlim(*xlim)
        _plt.xscale(xscale)
        _plt.ylim(0, y_lim - y_lim % 5 + 15)
        _plt.yscale(yscale)
        _plt.tick_params(
            which="both", axis="both", direction="in", right=True, top=True
        )
        _plt.tight_layout()
        _plt.minorticks_on()
        _plt.grid(which="major", alpha=0.3)
        _plt.grid(which="minor", alpha=0.1)
        if savefig:
            _plt.savefig(
                "beam_div_{:}_{:.0f}m_{:.0f}mm.png".format(
                    source.label, source.source_length, source.period
                ),
                dpi=300,
            )

    @staticmethod
    def process_table_parameters(spectra, source, calc_params):
        spectra_calc = copy.deepcopy(spectra)
        figsize = calc_params.figsize
        savefig = calc_params.savefig

        if source.source_type != "bendingmagnet":
            fig, ax = _plt.subplots(figsize=figsize)
            if source.gap == 0:
                source_k_max = source.calc_max_k(spectra_calc.accelerator)
            else:
                source_k_max = source.get_k()
            rows = 7
            col = 1.3
            ax.set_ylim(0, rows)
            ax.set_xlim(0, col + 0.2)
            data = dict()
            data["Máx. B [T]"] = _np.round(
                source.undulator_k_to_b(source_k_max, source.period), 2
            )
            data["Máx. K"] = _np.round(source_k_max, 2)
            if source.undulator_type != "APU":
                gapv, gaph = source.calc_min_gap(spectra_calc.accelerator)
                data["Min. gap [mm]"] = _np.round(gapv, 2)
            else:
                gapv = source.gap
                data["Gap [mm]"] = _np.round(gapv, 2)
            data["Polarization"] = source.polarization
            data["Length [m]"] = _np.round(source.source_length, 2)
            data["Period [mm]"] = _np.round(source.period, 2)
            data["Source"] = source.material

            for i, info in enumerate(data):
                ax.text(
                    x=col, y=0.5 + i, s=data[info], va="center", ha="right"
                )
                ax.text(col - 1.1, 0.5 + i, info, weight="bold", ha="left")
            ax.plot([0, col + 1], [rows - 1, rows - 1], lw=".5", c="black")
            ax.plot([0, col + 1], [rows, rows], lw=".5", c="black")
            ax.plot([0.7, 0.7], [0, rows], ls=":", lw=".5", c="grey")
            for row in range(rows):
                ax.plot([0, col + 1], [row, row], ls=":", lw=".5", c="grey")
            rect = _patches.Rectangle(
                (0, rows - 1), 2, 1, ec="none", fc="blue", alpha=0.4, zorder=-1
            )
            ax.add_patch(rect)
            for row in range(int(rows / 2)):
                rect = _patches.Rectangle(
                    (0, rows - 2 * row - 3),
                    2,
                    1,
                    ec="none",
                    fc="blue",
                    alpha=0.2,
                    zorder=-1,
                )
                ax.add_patch(rect)

            ax.set_title(
                "Source Parameters\n{:}".format(source.label),
                loc="center",
                fontsize=12,
            )
            ax.axis("off")
            _plt.tight_layout()
            if savefig:
                _plt.savefig(
                    "parameters_{:}_{:.0f}m_{:.0f}mm.png".format(
                        source.label, source.source_length, source.period
                    ),
                    dpi=300,
                )
        else:
            fig, ax = _plt.subplots(figsize=(figsize[0], figsize[1] / 1.7))
            rows = 1
            col = 1.2
            ax.set_ylim(0, rows)
            ax.set_xlim(0, col + 0.2)
            data = {
                "Máx. B [T]": _np.round(source.b_peak, 4)
            }
            for i, info in enumerate(data):
                ax.text(
                    x=col, y=0.5 + i, s=data[info], va="center", ha="right"
                )
                ax.text(col - 1, 0.5 + i, info, weight="bold", ha="left")
            ax.plot([0, col + 1], [rows - 1, rows - 1], lw=".5", c="black")
            ax.plot([0, col + 1], [rows, rows], lw=".5", c="black")
            ax.plot([0.7, 0.7], [0, rows], ls=":", lw=".5", c="grey")
            for row in range(rows):
                ax.plot([0, col + 1], [row, row], ls=":", lw=".5", c="grey")

            rect = _patches.Rectangle(
                (0, rows - 1), 2, 1, ec="none", fc="blue", alpha=0.4, zorder=-1
            )
            ax.add_patch(rect)

            ax.set_title(
                "Source Parameters\n{:}".format(source.label),
                loc="center",
                fontsize=12,
            )
            ax.axis("off")
            _plt.tight_layout()
            if savefig:
                _plt.savefig(
                    "parameters_{:}.png".format(source.label),
                    dpi=300,
                )

    @staticmethod
    def process_gap_energy(spectra, source, calc_params):
        spectra_calc = copy.deepcopy(spectra)
        if (
            source.source_type == "bendingmagnet"
            or source.source_type == "wiggler"
        ):
            return 0
        xlim = calc_params.e_range
        title = "Gap vs Energy\n{:} ({:.1f} m, {:.2f} mm)".format(
                source.label, source.source_length, source.period
            )
        xscale = "linear"
        yscale =  "linear"
        linewidth = calc_params.linewidth
        savefig = calc_params.savefig
        figsize = calc_params.figsize
        dpi = calc_params.dpi

        gapmax = 25
        gapv, gaph = source.calc_min_gap(spectra_calc.accelerator)
        gaps = _np.linspace(gapv, gapmax, 501)
        Bs = source.get_beff(gaps / source.period)
        Ks = (ECHARGE * Bs * source.period * 1e-3) / (EMASS * LSPEED * 2 * PI)
        gamma = spectra_calc.accelerator.gamma

        _plt.figure(figsize=figsize)
        _plt.title(title)
        for i in range(17):
            Es = source.get_harmonic_energy(
                n=2 * i + 1, gamma=gamma, theta=0, period=source.period, k=Ks
            )
            _plt.plot(
                Es * 1e-3,
                gaps,
                "-C0",
                linewidth=linewidth,
            )

        _plt.plot(
            [*xlim],
            [gaps[0], gaps[0]],
            "--C1",
            linewidth=linewidth,
            label="Min. gap: {:.2f} mm".format(gaps[0]),
        )

        _plt.xlabel("Energy [keV]")
        _plt.xscale(xscale)
        _plt.ylabel("Gap [mm]")
        _plt.yscale(yscale)
        _plt.legend(loc=4, ncol=1, fontsize=9)
        _plt.grid(which="major", alpha=0.3)
        _plt.grid(which="minor", alpha=0.1)
        _plt.xlim(*xlim)
        _plt.ylim(0, gapmax)
        _plt.tick_params(
            which="both", axis="both", direction="in", right=True, top=True
        )
        _plt.minorticks_on()
        _plt.tight_layout()
        if savefig:
            _plt.savefig(
                "gap_energy_{:}_{:.0f}m_{:.0f}mm.png".format(
                    source.label, source.source_length, source.period
                ),
                dpi=dpi,
            )

        # Fundamental Energy
        gaps = _np.linspace(gapv, gapmax, 501)
        Bs = source.get_beff(gaps / source.period)
        Ks = (ECHARGE * Bs * source.period * 1e-3) / (EMASS * LSPEED * 2 * PI)
        Es = source.get_harmonic_energy(
            n=1, gamma=gamma, theta=0, period=source.period, k=Ks
        )

        _plt.figure(figsize=figsize)
        _plt.title(title)
        _plt.plot(
            gaps,
            Es * 1e-3,
            "-C0",
            linewidth=linewidth,
        )
        _plt.plot(
            [gaps[0], gaps[0]],
            [0, 30],
            "--C1",
            linewidth=linewidth,
            label="Min. gap: {:.2f} mm".format(gaps[0]),
        )
        _plt.ylabel("Energy [keV]")
        _plt.xlabel("Gap [mm]")
        _plt.legend(loc=4, ncol=1, fontsize=9)
        _plt.grid(which="major", alpha=0.3)
        _plt.grid(which="minor", alpha=0.1)
        _plt.ylim(0, int(Es[-1] * 1e-3) + 1)
        _plt.yscale(yscale)
        _plt.xlim(0, gapmax)
        _plt.xscale(xscale)
        _plt.tick_params(
            which="both", axis="both", direction="in", right=True, top=True
        )
        _plt.minorticks_on()
        _plt.tight_layout()
        if savefig:
            _plt.savefig(
                "gap_fundamental_energy_{:}_{:.0f}m_{:.0f}mm.png".format(
                    source.label, source.source_length, source.period
                ),
                dpi=dpi,
            )

    @staticmethod
    def process_gap_k(spectra, source, calc_params):
        spectra_calc = copy.deepcopy(spectra)
        if (
            source.source_type == "bendingmagnet"
            or source.source_type == "wiggler"
        ):
            return 0
        title = "Gap vs K\n{:} ({:.1f} m, {:.2f} mm)".format(
                source.label, source.source_length, source.period
            )
        xscale = "linear"
        yscale = "linear"
        linewidth = calc_params.linewidth
        savefig = calc_params.savefig
        figsize = calc_params.figsize
        dpi = calc_params.dpi

        gapmax = 25
        gapv, gaph = source.calc_min_gap(spectra_calc.accelerator)
        gaps = _np.linspace(gapv, gapmax, 501)
        Bs = source.get_beff(gaps / source.period)
        Ks = (ECHARGE * Bs * source.period * 1e-3) / (EMASS * LSPEED * 2 * PI)

        _plt.figure(figsize=figsize)
        _plt.title(title)
        _plt.plot(
            gaps,
            Ks,
            "-C0",
            linewidth=linewidth,
        )
        _plt.plot(
            [gaps[0], gaps[0]],
            [0, Ks[0]],
            "--C1",
            linewidth=linewidth,
            label="Min. gap: {:.2f} mm".format(gaps[0]),
        )
        _plt.ylabel("K")
        _plt.xlabel("Gap [mm]")
        _plt.legend(loc=1, ncol=1, fontsize=9)
        _plt.grid(which="major", alpha=0.3)
        _plt.grid(which="minor", alpha=0.1)
        _plt.ylim(0, Ks[0])
        _plt.yscale(yscale)
        _plt.xlim(0, gapmax)
        _plt.xscale(xscale)
        _plt.tick_params(
            which="both", axis="both", direction="in", right=True, top=True
        )
        _plt.minorticks_on()
        _plt.tight_layout()
        if savefig:
            _plt.savefig(
                "gap_k_parameter_{:}_{:.0f}m_{:.0f}mm.png".format(
                    source.label, source.source_length, source.period
                ),
                dpi=dpi,
            )

    @staticmethod
    def process_gap_field(spectra, source, calc_params):
        spectra_calc = copy.deepcopy(spectra)
        if (
            source.source_type == "bendingmagnet"
            or source.source_type == "wiggler"
        ):
            return 0
        title =  "Gap vs B\n{:} ({:.1f} m, {:.2f} mm)".format(
                source.label, source.source_length, source.period
            )
        xscale = "linear"
        yscale = "linear"
        linewidth = calc_params.linewidth
        savefig = calc_params.savefig
        figsize = calc_params.figsize
        dpi = calc_params.dpi

        gapmax = 25
        gapv, gaph = source.calc_min_gap(spectra_calc.accelerator)
        gaps = _np.linspace(gapv, gapmax, 501)
        Bs = source.get_beff(gaps / source.period)

        _plt.figure(figsize=figsize)
        _plt.title(title)
        _plt.plot(
            gaps,
            Bs,
            "-C0",
            linewidth=linewidth,
        )
        _plt.plot(
            [gaps[0], gaps[0]],
            [0, Bs[0]],
            "--C1",
            linewidth=linewidth,
            label="Min. gap: {:.2f} mm".format(gaps[0]),
        )
        _plt.ylabel("B [T]")
        _plt.xlabel("Gap [mm]")
        _plt.legend(loc=1, ncol=1, fontsize=9)
        _plt.grid(which="major", alpha=0.3)
        _plt.grid(which="minor", alpha=0.1)
        _plt.ylim(0, Bs[0])
        _plt.yscale(yscale)
        _plt.xlim(0, gapmax)
        _plt.xscale(xscale)
        text = r"$B(g)=B_r\cdot a\cdot \exp{\left(b\frac{g}{\lambda_u}+c\frac{g^2}{\lambda_u^2}\right)}$"
        text += "\n"
        text += r"$B_r=$ {:.2f}        a={:.4f}".format(
            source.br, source.halbach_coef["hp"]["a"]
        )
        text += "\n"
        text += r"$b=$ {:.4f}    c={:.4f}".format(
            source.halbach_coef["hp"]["b"], source.halbach_coef["hp"]["c"]
        )
        _plt.text(x=gapmax / 3, y=(Bs[0] - Bs[-1]) / 2, s=text, fontsize=10)
        _plt.tick_params(
            which="both", axis="both", direction="in", right=True, top=True
        )
        _plt.minorticks_on()
        _plt.tight_layout()
        if savefig:
            _plt.savefig(
                "gap_field_{:}_{:.0f}m_{:.0f}mm.png".format(
                    source.label, source.source_length, source.period
                ),
                dpi=dpi,
            )

    @staticmethod
    def process_phase_field(spectra, source, calc_params):
        spectra_calc = copy.deepcopy(spectra)
        if (
            source.source_type == "bendingmagnet"
            or source.source_type == "wiggler"
        ):
            return 0
        title = "Phase vs B\n{:} ({:.1f} m, {:.2f} mm)".format(
                source.label, source.source_length, source.period
            )
        
        xscale = "linear"
        yscale = "linear"
        linewidth = calc_params.linewidth
        savefig = calc_params.savefig
        figsize = calc_params.figsize
        dpi = calc_params.dpi

        phases = _np.linspace(0, source.period / 2, 501)
        Bs = source.get_beff(
            gap_over_period=source.gap / source.period, phase=phases
        )

        _plt.figure(figsize=figsize)
        _plt.title(title)

        _plt.title(
            "Phase vs B \nAPU22 ({:.1f} m, {:.1f} mm)".format(
                source.source_length, source.period
            )
        )
        _plt.plot(
            phases,
            Bs,
            "-C0",
            linewidth=linewidth,
        )
        text = r"$B(z) = B_0|\cos\left(\frac{\pi}{\lambda_u}(z-z_0)\right)|$"
        text += "\n" + r"$B_0 = $" + "{:.4f}   ".format(Bs[0])
        text += r"$z_0 = $" + "{:.4f}".format(source._z0)
        _plt.text(
            x=1.9 * source.period / 2 / 5,
            y=14 * (Bs[0] - Bs[-1]) / 15,
            s=text,
            fontsize=11,
        )
        _plt.ylabel("B [T]")
        _plt.yscale(yscale)
        _plt.xlabel("Phase [mm]")
        _plt.xscale(xscale)
        _plt.grid(which="major", alpha=0.3)
        _plt.grid(which="minor", alpha=0.1)
        _plt.tick_params(
            which="both", axis="both", direction="in", right=True, top=True
        )
        _plt.xlim(0, source.period / 2)
        _plt.xticks(range(0, int(source.period / 2), 1))
        _plt.ylim(0, _np.round(_np.max(Bs) + 0.2, 1))
        _plt.minorticks_on()
        _plt.tight_layout()
        if savefig:
            _plt.savefig(
                "phase_field_{:}_{:.0f}m_{:.0f}mm.png".format(
                    source.label, source.source_length, source.period
                ),
                dpi=dpi,
            )

    @staticmethod
    def process_phase_k(spectra, source, calc_params):
        spectra_calc = copy.deepcopy(spectra)
        if (
            source.source_type == "bendingmagnet"
            or source.source_type == "wiggler"
        ):
            return 0
        title = "Phase vs K\n{:} ({:.1f} m, {:.2f} mm)".format(
                source.label, source.source_length, source.period
            )
        xscale = "linear"
        yscale = "linear"
        linewidth = calc_params.linewidth
        savefig = calc_params.savefig
        figsize = calc_params.figsize
        dpi = calc_params.dpi

        phases = _np.linspace(0, source.period / 2, 501)
        Bs = source.get_beff(
            gap_over_period=source.gap / source.period, phase=phases
        )
        Ks = (ECHARGE * Bs * source.period * 1e-3) / (EMASS * LSPEED * 2 * PI)

        _plt.figure(figsize=figsize)
        _plt.title(title)

        _plt.title(
            "Phase vs K \nAPU22 ({:.1f} m, {:.1f} mm)".format(
                source.source_length, source.period
            )
        )
        _plt.plot(
            phases,
            Ks,
            "-C0",
            linewidth=linewidth,
        )
        _plt.ylabel("K")
        _plt.yscale(yscale)
        _plt.xlabel("Phase [mm]")
        _plt.xscale(xscale)
        _plt.xticks(range(0, int(source.period / 2), 1))
        _plt.grid(which="major", alpha=0.3)
        _plt.grid(which="minor", alpha=0.1)
        _plt.tick_params(
            which="both", axis="both", direction="in", right=True, top=True
        )
        _plt.xlim(0, source.period / 2)
        _plt.ylim(0, _np.round(Ks[0] + 0.2, 1))
        _plt.minorticks_on()
        _plt.tight_layout()
        if savefig:
            _plt.savefig(
                "phase_k_{:}_{:.0f}m_{:.0f}mm.png".format(
                    source.label, source.source_length, source.period
                ),
                dpi=dpi,
            )

    @staticmethod
    def process_phase_energy(spectra, source, calc_params):
        spectra_calc = copy.deepcopy(spectra)
        if (
            source.source_type == "bendingmagnet"
            or source.source_type == "wiggler"
        ):
            return 0
        xlim = calc_params.e_range
        title = "Phase vs Energy\n{:} ({:.1f} m, {:.2f} mm)".format(
                source.label, source.source_length, source.period
            )

        xscale = "linear"
        yscale = "linear"
        linewidth = calc_params.linewidth
        savefig = calc_params.savefig
        figsize = calc_params.figsize
        dpi = calc_params.dpi

        phases = _np.linspace(0, source.period / 2, 501)
        Bs = source.get_beff(
            gap_over_period=source.gap / source.period, phase=phases
        )
        Ks = (ECHARGE * Bs * source.period * 1e-3) / (EMASS * LSPEED * 2 * PI)
        gamma = spectra_calc.accelerator.gamma

        _plt.figure(figsize=figsize)
        _plt.title(title)
        for i in range(17):
            Es = source.get_harmonic_energy(
                n=2 * i + 1, gamma=gamma, theta=0, period=source.period, k=Ks
            )
            _plt.plot(
                Es * 1e-3,
                phases,
                "-C0",
                linewidth=linewidth,
            )
        _plt.xlabel("Energy [keV]")
        _plt.xscale(xscale)
        _plt.ylabel("Phase [mm]")
        _plt.yscale(yscale)
        _plt.grid(which="major", alpha=0.3)
        _plt.grid(which="minor", alpha=0.1)
        _plt.xlim(*xlim)
        _plt.ylim(0, source.period / 2)
        _plt.tick_params(
            which="both", axis="both", direction="in", right=True, top=True
        )
        _plt.minorticks_on()
        _plt.tight_layout()
        if savefig:
            _plt.savefig(
                "phase_energy_{:}_{:.0f}m_{:.0f}mm.png".format(
                    source.label, source.source_length, source.period
                ),
                dpi=dpi,
            )

        # Fundamental Energy
        phases = _np.linspace(0, source.period / 2, 501)
        Bs = source.get_beff(
            gap_over_period=source.gap / source.period, phase=phases
        )
        Ks = (ECHARGE * Bs * source.period * 1e-3) / (EMASS * LSPEED * 2 * PI)
        Es = source.get_harmonic_energy(
            n=1, gamma=gamma, theta=0, period=source.period, k=Ks
        )

        _plt.figure(figsize=figsize)
        _plt.title(title)
        _plt.plot(
            phases,
            Es * 1e-3,
            "-C0",
            linewidth=linewidth,
        )
        _plt.ylabel("Energy [keV]")
        _plt.xlabel("Phase [mm]")
        _plt.grid(which="major", alpha=0.3)
        _plt.grid(which="minor", alpha=0.1)
        _plt.ylim(int(Es[0]) * 1e-3 - 0.1, int(Es[-1] * 1e-3) + 1)
        _plt.yscale(yscale)
        _plt.xlim(0, source.period / 2)
        _plt.xscale(xscale)
        _plt.tick_params(
            which="both", axis="both", direction="in", right=True, top=True
        )
        _plt.minorticks_on()
        _plt.tight_layout()
        if savefig:
            _plt.savefig(
                "phase_fundamental_energy_{:}_{:.0f}m_{:.0f}mm.png".format(
                    source.label, source.source_length, source.period
                ),
                dpi=dpi,
            )

    @staticmethod
    def process_flux(spectra, source, calc_params):
        spectra_calc = copy.deepcopy(spectra)
        if source.source_type != "bendingmagnet":
            if source.use_recovery_params and source.add_phase_errors:
                spectra_calc.use_recovery_params = True
        distance_from_source = calc_params.distance_from_source
        slit_acceptance = calc_params.slit_acceptance
        slit_acceptance = [i / distance_from_source for i in slit_acceptance]
        slit_shape = calc_params.slit_shape
        xlim = calc_params.e_range
        energy_range = [xlim[0] * 1e3, xlim[1] * 1e3]
        
        nr_pts_k = 11
        kmin = 0.2
        gamma = spectra_calc.accelerator.gamma
        if (
            source.source_type != "wiggler"
            and source.source_type != "bendingmagnet"
        ):
            if source.gap == 0:
                source_k_max = source.calc_max_k(spectra_calc.accelerator)
            else:
                source_k_max = source.get_k()
            first_hamonic_energy = source.get_harmonic_energy(
                1, gamma, 0, source.period, source_k_max
            )
            n = int(xlim[1] * 1e3 / first_hamonic_energy)
            n_harmonic = n - 1 if n % 2 == 0 else n
            if source.polarization == "cp":
                n_harmonic = 1
        else:
            n_harmonic = 1
        harmonic_range =  [1, n_harmonic]
        process_curves =  True
        superp_value = 250
        title = "Flux curve"
        xscale = "linear"
        yscale = "log"
        figname = "flux_curve_{:}.png".format(source.label) if source.source_type == "bendingmagnet" else (
                    "flux_curve_{:}_{:.0f}m_{:.0f}mm.png".format(
                        source.label, source.source_length, source.period
                    )
                )
        linewidth = calc_params.linewidth
        savefig = calc_params.savefig
        figsize = calc_params.figsize
        dpi = calc_params.dpi
        legend_fs = 10
        legend_properties = True
        
        spectra_calc.sources = [source]
        spectra_calc.calc_flux_curve(
            energy_range=energy_range,
            harmonic_range=[harmonic_range],
            nr_pts_k=nr_pts_k,
            kmin=kmin,
            slit_shape=slit_shape,
            slit_acceptances=[slit_acceptance],
            extraction_points=[spectra_calc.accelerator.extraction_point],
        )
        if (
            source.source_type != "wiggler"
            and source.source_type != "bendingmagnet"
        ):
            idx_xlim = _np.argmin(
                _np.abs(spectra_calc._energies[0, -1, :] - (xlim[1] * 1e3))
            )
            min_flux = float(
                10 ** int(_np.log10(spectra_calc._fluxes[0, -1, idx_xlim - 1]))
            )
            max_flux = float(
                10 ** (int(_np.log10(_np.max(spectra_calc._fluxes))) + 1)
            )
        else:
            min_flux = float(
                10 ** (int(_np.log10(spectra_calc._fluxes[0, -1])) - 1)
            )
            max_flux = float(
                10 ** (int(_np.log10(_np.max(spectra_calc._fluxes))) + 1)
            )
        ylim = [min_flux, max_flux]
        spectra_calc.plot_flux_curve(
            process_curves=process_curves,
            superp_value=superp_value,
            title=title,
            xscale=xscale,
            yscale=yscale,
            linewidth=linewidth,
            savefig=savefig,
            figname=figname,
            dpi=dpi,
            figsize=figsize,
            legend_fs=legend_fs,
            legend_properties=legend_properties,
            xlim=xlim,
            ylim=ylim,
        )
        del spectra_calc

    @staticmethod
    def process_brilliance(spectra, source, calc_params):
        spectra_calc = copy.deepcopy(spectra)
        if source.source_type != "bendingmagnet":
            if source.use_recovery_params and source.add_phase_errors:
                spectra_calc.use_recovery_params = True
        nr_pts_k = 11
        kmin = 0.2
        xlim =calc_params.e_range
        emax = xlim[1] * 1e3
        gamma = spectra_calc.accelerator.gamma
        if (
            source.source_type != "wiggler"
            and source.source_type != "bendingmagnet"
        ):
            if source.gap == 0:
                source_k_max = source.calc_max_k(spectra_calc.accelerator)
            else:
                source_k_max = source.get_k()
            first_hamonic_energy = source.get_harmonic_energy(
                1, gamma, 0, source.period, source_k_max
            )
            n = int(xlim[1] * 1e3 / first_hamonic_energy)
            n_harmonic = n - 1 if n % 2 == 0 else n
            if source.polarization == "cp":
                n_harmonic = 1
        else:
            n_harmonic = 1
        harmonic_range = [[1, n_harmonic]]
        x_accep = 1

        process_curves =  True
        superp_value =  250
        title = "Brilliance curve"
        xscale = "linear"
        yscale = "log"
        linewidth = calc_params.linewidth
        savefig = calc_params.savefig
        figsize = calc_params.figsize
        figname = (
                "brilliance_curve_{:}.png".format(source.label)
                if source.source_type == "bendingmagnet"
                else (
                    "brilliance_curve_{:}_{:.0f}m_{:.0f}mm.png".format(
                        source.label, source.source_length, source.period
                    )
                )
            )
        dpi = calc_params.dpi
        legend_fs = 10
        legend_properties =  True
        spectra_calc.sources = [source]
        spectra_calc.calc_brilliance_curve(
            harmonic_range=harmonic_range,
            nr_pts_k=nr_pts_k,
            kmin=kmin,
            emax=emax,
            x_accep=x_accep,
            extraction_points=[spectra_calc.accelerator.extraction_point],
        )
        if (
            source.source_type != "wiggler"
            and source.source_type != "bendingmagnet"
        ):
            idx_xlim = _np.argmin(
                _np.abs(spectra_calc._energies[0, -1, :] - (xlim[1] * 1e3))
            )
            min_brilliance = float(
                10
                ** int(
                    _np.log10(spectra_calc._brilliances[0, -1, idx_xlim - 1])
                )
            )
            max_brilliance = float(
                10 ** (int(_np.log10(_np.max(spectra_calc._brilliances))) + 1)
            )
        else:
            min_brilliance = float(
                10 ** (int(_np.log10(spectra_calc._brilliances[0, -1])) - 1)
            )
            max_brilliance = float(
                10 ** (int(_np.log10(_np.max(spectra_calc._brilliances))) + 1)
            )
        ylim = [min_brilliance, max_brilliance]
        spectra_calc.plot_brilliance_curve(
            process_curves=process_curves,
            superp_value=superp_value,
            title=title,
            xscale=xscale,
            yscale=yscale,
            xlim=xlim,
            ylim=ylim,
            linewidth=linewidth,
            savefig=savefig,
            figsize=figsize,
            figname=figname,
            dpi=dpi,
            legend_fs=legend_fs,
            legend_properties=legend_properties,
        )
        del spectra_calc

    @staticmethod
    def process_degree_polarization(spectra, source, calc_params):
        spectra_calc: SpectraInterface = copy.deepcopy(spectra)
        distance_from_source = calc_params.distance_from_source
        slit_acceptance = calc_params.slit_acceptance
        slit_acceptance = (
            slit_acceptance[0] / distance_from_source,
            slit_acceptance[1] / distance_from_source,
        )
        slit_shape = calc_params.slit_shape
        xlim = calc_params.e_range
        energy_range = (xlim[0] * 1e3, xlim[1] * 1e3)
        slit_position = calc_params.slit_position
        slit_position = (
            slit_position[0] / distance_from_source,
            slit_position[1] / distance_from_source,
        )
        title = "Polarization Degree"
        xscale = "linear"
        yscale = "linear"
        linewidth = calc_params.linewidth
        savefig = calc_params.savefig
        figsize = calc_params.figsize
        figname = (
                "degree_polarization_{:}.png".format(source.label)
                if source.source_type == "bendingmagnet"
                else (
                    "degree_polarization_{:}_{:.0f}m_{:.0f}mm.png".format(
                        source.label, source.source_length, source.period
                    )
                )
            )
        dpi = calc_params.dpi
        legend_fs = 9

        energies, degree_pl, degree_pc = spectra_calc.calc_degree_polarization(
            source=source,
            slit_shape=slit_shape,
            slit_position=slit_position,
            slit_acceptance=slit_acceptance,
            distance_from_source=distance_from_source,
            energy_range=energy_range,
        )

        _plt.figure(figsize=figsize)
        _plt.title(title)
        _plt.plot(
            energies * 1e-3,
            degree_pl**2,
            "-C0",
            label="PL²",
            linewidth=linewidth,
        )
        _plt.plot(
            energies * 1e-3,
            degree_pc**2,
            "-C1",
            label="PC²",
            linewidth=linewidth,
        )
        _plt.xlabel("Energy [keV]")
        _plt.ylabel("Polarization Degree")
        _plt.legend(loc=5, ncol=3, fontsize=legend_fs)
        _plt.minorticks_on()
        _plt.grid(which="major", alpha=0.3)
        _plt.grid(which="minor", alpha=0.1)
        _plt.xlim(*xlim)
        _plt.xscale(xscale)
        _plt.yscale(yscale)
        _plt.tick_params(
            which="both", axis="both", direction="in", right=True, top=True
        )
        _plt.tight_layout()
        if savefig:
            _plt.savefig(
                figname,
                dpi=dpi,
            )

    @staticmethod
    def process_degree_coherence(spectra, source, calc_params):
        pass

    @staticmethod
    def process_power(spectra, source, calc_params):
        spectra = copy.deepcopy(spectra)
        k_max = source.calc_max_k(spectra.accelerator)
        distance_from_source = calc_params.distance_from_source
        x_nr_pts = calc_params.x_nr_pts
        y_nr_pts = calc_params.y_nr_pts
        slit_shape = calc_params.slit_shape
        slit_acceptance = calc_params.slit_acceptance
        slit_acceptance = (
            slit_acceptance[0] / distance_from_source,
            slit_acceptance[1] / distance_from_source,
        )
        slit_position = calc_params.slit_position
        slit_position = (
            slit_position[0] / distance_from_source,
            slit_position[1] / distance_from_source,
        )
        figsize = calc_params.figsize
        savefig = calc_params.savefig
        dpi = calc_params.dpi
        figname = (
                "partial_power_{:}.png".format(source.label)
                if source.source_type == "bendingmagnet"
                else (
                    "partial_power_{:}_{:.0f}m_{:.0f}mm.png".format(
                        source.label, source.source_length, source.period
                    )
                )
            )
        if source.source_type == "bendingmagnet":
            x_range = (-1, 1)
            y_range = (-0.3, 0.3)
        else:
            x_range = (-0.3, 0.3)
            y_range = (-0.3, 0.3)

        # Calc Power Density and Partial Power
        spectra_calc: SpectraInterface = copy.deepcopy(spectra)
        power_densities = spectra_calc.calc_power_density(
            source=source,
            x_range=x_range,
            x_nr_pts=x_nr_pts,
            y_range=y_range,
            y_nr_pts=y_nr_pts,
            distance_from_source=distance_from_source,
            current=350,
        )
        partial_power = spectra_calc.calc_partial_power(
            source=source,
            slit_shape=slit_shape,
            slit_position=slit_position,
            slit_acceptance=slit_acceptance,
            distance_from_source=distance_from_source,
            current=350,
        )
        del spectra_calc

        # Plot
        fig = _plt.figure(figsize=(figsize[0], figsize[0]))
        ax = fig.add_subplot(111)
        title = (
            "Power Density @ 350 mA\n{:} ({:.1f} m, {:.2f} mm)\nK máx: {:.3f}".format(
                source.label, source.source_length, source.period, k_max
            )
            if source.source_type != "bendingmagnet"
            else "Power Density @ 350 mA\n{:}\n".format(source.label)
        )
        ax.set_title(title, fontsize=11)
        ax.text(
            x=x_range[0] + 0.015,
            y=y_range[1] - 0.035,
            s="Partial Power: {:.2f} W".format(partial_power * 1e3),
            fontsize=9,
            c="white",
        )
        im = ax.imshow(
            power_densities,
            extent=[*x_range, *y_range],
            aspect="auto" if source.source_type == "bendingmagnet" else "equal",
            norm=colors.Normalize(
                vmin=_np.min(power_densities), vmax=_np.max(power_densities)
            ),
        )
        if source.source_type == "bendingmagnet":
            ax.set_xticks([-1, -0.7, -0.4, 0.0, 0.4, 0.7, 1])
            ax.set_yticks([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
        else:
            ax.set_xticks([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
            ax.set_yticks([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
        ax.tick_params(labelsize=9)
        sm = _plt.cm.ScalarMappable(
            _plt.Normalize(
                vmin=_np.min(power_densities * distance_from_source**2),
                vmax=_np.max(power_densities * distance_from_source**2),
            )
        )
        sm.set_array(power_densities * distance_from_source**2)
        cbar = fig.colorbar(
            sm,
            ax=ax,
            label="Power Density [kW/mrad²]",
            format="%.2f",
            shrink=0.5,
        )
        cbar.set_ticks(
            _np.linspace(
                _np.min(power_densities * distance_from_source**2),
                _np.max(power_densities * distance_from_source**2),
                5,
            )
        )
        cbar.ax.tick_params(labelsize=9)
        ax.set_xlabel("X [mrad]")
        ax.set_ylabel("Y [mrad]")
        if slit_shape == "retslit":
            ax.text(
                x=x_range[0] * (1 - 0.05),
                y=y_range[0] * (1 - 0.05),
                s="{:.1f} x {:.1f} \u03bcrad²".format(
                    slit_acceptance[0] * 1e3, slit_acceptance[1] * 1e3
                ),
                fontsize=8,
                c="white",
            )
            patch = _patches.Rectangle(
                (
                    slit_position[0] - slit_acceptance[0] / 2,
                    slit_position[1] - slit_acceptance[1] / 2,
                ),
                slit_acceptance[0],
                slit_acceptance[1],
                fc=(0, 0, 0, 0),
                ec="black",
                lw=1,
                ls=":",
            )
            ax.add_patch(patch)
        else:
            ax.text(
                x=x_range[0] * (1 - 0.05),
                y=y_range[0] * (1 - 0.05) + 0.05,
                s=r"$R_1:$"
                + "{:.1f} \u03bcrad".format(slit_acceptance[0] * 1e3),
                fontsize=8,
                c="white",
            )
            ax.text(
                x=x_range[0] * (1 - 0.05),
                y=y_range[0] * (1 - 0.05),
                s=r"$R_2:$"
                + "{:.1f} \u03bcrad".format(slit_acceptance[1] * 1e3),
                fontsize=8,
                c="white",
            )
            patch = _patches.Circle(
                (slit_position[0], slit_position[1]),
                slit_acceptance[0],
                fc=(0, 0, 0, 0),
                ec="black",
                lw=1,
                ls=":",
            )
            ax.add_patch(patch)
            patch = _patches.Circle(
                (slit_position[0], slit_position[1]),
                slit_acceptance[1],
                fc=(0, 0, 0, 0),
                ec="black",
                lw=1,
                ls=":",
            )
            ax.add_patch(patch)
        _plt.tight_layout()
        if savefig:
            _plt.savefig(
                figname,
                dpi=dpi,
            )

    @staticmethod
    def process_beam_div_size_wigner(spectra, source, calc_params):
        spectra_calc: SpectraInterface = copy.deepcopy(spectra)

        if (
            source.source_type == "bendingmagnet"
            or source.source_type == "wiggler"
        ):
            return 0

        xlim = calc_params.e_range
        x_nr_pts = calc_params.e_nr_pts
        linewidth = calc_params.linewidth
        savefig = calc_params.savefig
        figsize = calc_params.figsize
        dpi = calc_params.dpi

        div_size_x, energies = spectra_calc.calc_numerical_div_size_wigner(
            source,
            emax=xlim[1] * 1e3,
            e_pts=x_nr_pts,
            direction="horizontal",
        )

        div_size_y, energies = spectra_calc.calc_numerical_div_size_wigner(
            source,
            emax=xlim[1] * 1e3,
            e_pts=x_nr_pts,
            direction="vertical",
        )

        # Plot Beam Divergence
        _plt.figure(figsize=figsize)
        _plt.title(
            "Beam Divergence ({:})\n{:} ({:.2f} m, {:.2f} mm)".format(
                spectra_calc.accelerator._extraction_point,
                source.label,
                source.source_length,
                source.period,
            )
        )
        _plt.plot(
            energies * 1e-3,
            div_size_y[:, 0] * 1e3,
            "-C1",
            label=r"$\sigma'_y$",
            linewidth=linewidth,
        )
        _plt.plot(
            energies * 1e-3,
            div_size_x[:, 0] * 1e3,
            "-C0",
            label=r"$\sigma'_x$",
            linewidth=linewidth,
        )
        _plt.legend()
        valmax = (
            max(_np.nanmax(div_size_x[:, 0]), _np.nanmax(div_size_y[:, 0]))
            * 1e3
        )
        valmax = (int(valmax / 2) + 2) * 2
        _plt.ylim(0, valmax)
        _plt.xlim(0, xlim[1])
        _plt.ylabel("RMS beam divergence [\u03bcrad]")
        _plt.xlabel("Energy [keV]")
        _plt.minorticks_on()
        _plt.grid(which="major", alpha=0.3)
        _plt.grid(which="minor", alpha=0.1)
        _plt.tick_params(
            which="both", axis="both", direction="in", right=True, top=True
        )
        _plt.tight_layout()
        if savefig:
            _plt.savefig(
                "numerical_beam_divergence{:}_{:.0f}m_{:.0f}mm_{:}_beta.png".format(
                    source.label,
                    source.source_length,
                    source.period,
                    spectra_calc.accelerator._extraction_point,
                ),
                dpi=dpi,
            )

        # Plot Beam Size
        _plt.figure(figsize=figsize)
        _plt.title(
            "Beam Size ({:})\n{:} ({:.2f} m, {:.2f} mm)".format(
                spectra_calc.accelerator._extraction_point,
                source.label,
                source.source_length,
                source.period,
            )
        )
        _plt.plot(
            energies * 1e-3,
            div_size_y[:, 1] * 1e3,
            "-C1",
            label=r"$\sigma_y$",
            linewidth=linewidth,
        )
        _plt.plot(
            energies * 1e-3,
            div_size_x[:, 1] * 1e3,
            "-C0",
            label=r"$\sigma_x$",
            linewidth=linewidth,
        )
        _plt.legend()

        valmax = (
            max(_np.nanmax(div_size_x[:, 1]), _np.nanmax(div_size_y[:, 1]))
            * 1e3
        )
        valmax = (int(valmax / 5) + 2) * 5

        _plt.ylim(0, valmax)
        _plt.xlim(0, xlim[1])
        _plt.ylabel("RMS beam size [\u03bcm]")
        _plt.xlabel("Energy [keV]")
        _plt.minorticks_on()
        _plt.grid(which="major", alpha=0.3)
        _plt.grid(which="minor", alpha=0.1)
        _plt.tick_params(
            which="both", axis="both", direction="in", right=True, top=True
        )
        _plt.tight_layout()
        if savefig:
            _plt.savefig(
                "numerical_beam_size_{:}_{:.0f}m_{:.0f}mm_{:}_beta.png".format(
                    source.label,
                    source.source_length,
                    source.period,
                    spectra_calc.accelerator._extraction_point,
                ),
                dpi=dpi,
            )

    @staticmethod
    def process_flux_curve_generic(spectra, source, calc_params):
        spectra_calc: SpectraInterface = copy.deepcopy(spectra)
        if source.source_type != "bendingmagnet":
            if source.use_recovery_params and source.add_phase_errors:
                spectra_calc.use_recovery_params = True

        if (
            source.source_type == "bendingmagnet"
            or source.source_type == "wiggler"
        ):
            return 0

        distance_from_source = calc_params.distance_from_source
        slit_acceptance = calc_params.slit_acceptance
        slit_acceptance = [i / distance_from_source for i in slit_acceptance]
        slit_position = calc_params.slit_position
        slit_shape = calc_params.slit_shape
        xlim = calc_params.e_range
        figsize = calc_params.figsize
        savefig = calc_params.savefig
        linewidth = calc_params.linewidth
        dpi = calc_params.dpi
        superb = 701
        deltak = 0.99
        k_nr_pts = 11

        fs_at_res, es_at_res = spectra_calc.calc_flux_curve_generic(
            source,
            emax=xlim[1] * 1e3,
            slit_shape=slit_shape,
            slit_acceptance=(slit_acceptance[0], slit_acceptance[1]),
            observation_angle=(slit_position[0], slit_position[1]),
            distance_from_source=distance_from_source,
            k_nr_pts=1,
            deltak=0.99,
            even_harmonic=False,
            superb=superb,
        )

        fs_out_res, es_out_res = spectra_calc.calc_flux_curve_generic(
            source,
            emax=xlim[1] * 1e3,
            slit_shape=slit_shape,
            slit_acceptance=(slit_acceptance[0], slit_acceptance[1]),
            observation_angle=(slit_position[0], slit_position[1]),
            distance_from_source=distance_from_source,
            k_nr_pts=k_nr_pts,
            deltak=deltak,
            even_harmonic=False,
            superb=superb,
        )

        valmin = float(
            10
            ** int(
                _np.log10(
                    max(
                        fs_at_res[
                            _np.argmin(
                                [
                                    _np.min(_np.abs(i - xlim[1] * 1e3))
                                    for i in _np.array(
                                        es_at_res, dtype="object"
                                    )
                                ]
                            )
                        ]
                    )
                )
            )
        )
        valmax = float(
            10
            ** int(
                _np.log10(
                    max(
                        [
                            i.max()
                            for i in _np.array(fs_out_res, dtype="object")
                        ]
                    )
                )
                + 1
            )
        )

        # Plot flux curve
        _plt.figure(figsize=figsize)
        _plt.title(
            "Flux curve ({:})\n{:} ({:.2f} m, {:.2f} mm)".format(
                spectra_calc.accelerator._extraction_point,
                source.label,
                source.source_length,
                source.period,
            ),
            fontsize=10,
        )

        _plt.plot(
            _np.array(es_at_res[0]) * 1e-3,
            fs_at_res[0],
            "-C0",
            label="At Ressonance",
            linewidth=linewidth,
        )
        _plt.plot(
            _np.array(es_out_res[0]) * 1e-3,
            fs_out_res[0],
            "-C1",
            label="At Peak Flux",
            linewidth=linewidth,
            alpha=0.6,
        )
        for i in range(1, len(es_at_res)):
            _plt.plot(
                _np.array(es_at_res[i]) * 1e-3,
                fs_at_res[i],
                "-C0",
                linewidth=linewidth,
            )
            _plt.plot(
                _np.array(es_out_res[i]) * 1e-3,
                fs_out_res[i],
                "-C1",
                linewidth=linewidth,
                alpha=0.6,
            )

        _plt.legend(fontsize=8)
        _plt.ylim(valmin, valmax)
        _plt.xlim(0, xlim[1])
        _plt.ylabel("Flux [ph/s/0.1%/100mA]", fontsize=10)
        _plt.yscale("log")
        _plt.xlabel("Energy [keV]", fontsize=10)
        _plt.minorticks_on()
        _plt.grid(which="major", alpha=0.3)
        _plt.grid(which="minor", alpha=0.1)
        _plt.tick_params(
            which="both", axis="both", direction="in", right=True, top=True
        )
        _plt.tight_layout()
        if savefig:
            _plt.savefig(
                "flux_curve_generic_{:}_{:.0f}m_{:.0f}mm_{:}_beta.png".format(
                    source.label,
                    source.source_length,
                    source.period,
                    spectra_calc.accelerator._extraction_point,
                ),
                dpi=dpi,
            )


class IDParameters:

    def __init__(self):
        self._polarization = None
        self._type = None
        self._material = None
        self._period = None
        self._length = None
        self._phase_error = None
        self._label = None
        self._vc_tolerance = None
    
    @property
    def polarization(self):
        return self._polarization

    @property
    def type(self):
        return self._type

    @property
    def material(self):
        return self._material

    @property
    def period(self):
        return self._period

    @property
    def length(self):
        return self._length

    @property
    def phase_error(self):
        return self._phase_error

    @property
    def label(self):
        return self._label

    @property
    def vc_tolerance(self):
        return self._vc_tolerance

    @polarization.setter
    def polarization(self, value):
        self._polarization = value

    @type.setter
    def type(self, value):
        self._type = value

    @material.setter
    def material(self, value):
        self._material = value

    @period.setter
    def period(self, value):
        self._period = value

    @length.setter
    def length(self, value):
        self._length = value

    @phase_error.setter
    def phase_error(self, value):
        self._phase_error = value

    @label.setter
    def label(self, value):
        self._label = value

    @vc_tolerance.setter
    def vc_tolerance(self, value):
        self._vc_tolerance = value


class Calculations:

    def __init__(self):
        self._gap_energy = False
        self._gap_field = False
        self._gap_k = False
        self._phase_energy = False
        self._phase_field = False
        self._phase_k = False
        self._table_parameters = False
        self._flux = False
        self._flux_curve_generic = False
        self._brilliance = False
        self._beam_size = False
        self._beam_divergence = False
        self._beam_div_size_wigner = False
        self._power = False
        self._degree_polarization = False
        self._degree_coherence = False
        self._flux_distribuition_2d = False
    
    @property
    def gap_energy(self):
        return self._gap_energy
    
    @property
    def gap_field(self):
        return self._gap_field

    @property
    def gap_k(self):
        return self._gap_k

    @property
    def phase_energy(self):
        return self._phase_energy

    @property
    def phase_k(self):
        return self._phase_k
    
    @property
    def phase_field(self):
        return self._phase_field

    @property 
    def table_parameters(self):
        return self._table_parameters

    @property 
    def flux(self):
        return self._flux

    @property 
    def flux_curve_generic(self):
        return self._flux_curve_generic

    @property 
    def brilliance(self):
        return self._brilliance

    @property 
    def beam_size(self):
        return self._beam_size

    @property 
    def beam_divergence(self):
        return self._beam_divergence

    @property 
    def beam_div_size_wigner(self):
        return self._beam_div_size_wigner

    @property 
    def power(self):
        return self._power

    @property 
    def degree_polarization(self):
        return self._degree_polarization

    @property 
    def degree_coherence(self):
        return self._degree_coherence

    @property 
    def flux_distribuition_2d(self):
        return self._flux_distribuition_2d

    @gap_energy.setter
    def gap_energy(self, value):
        self._gap_energy = value
    
    @gap_field.setter
    def gap_field(self, value):
        self._gap_field = value
    
    @gap_k.setter
    def gap_k(self, value):
        self._gap_k = value

    @phase_energy.setter
    def phase_energy(self, value):
        self._phase_energy = value

    @phase_field.setter
    def phase_field(self, value):
        self._phase_field = value

    @phase_k.setter
    def phase_k(self, value):
        self._phase_k = value

    @table_parameters.setter
    def table_parameters(self, value):
        self._table_parameters = value

    @flux.setter
    def flux(self, value):
        self._flux = value

    @flux_curve_generic.setter
    def flux_curve_generic(self, value):
        self._flux_curve_generic = value

    @brilliance.setter
    def brilliance(self, value):
        self._brilliance = value

    @beam_size.setter
    def beam_size(self, value):
        self._beam_size = value

    @beam_divergence.setter
    def beam_divergence(self, value):
        self._beam_divergence = value

    @beam_div_size_wigner.setter
    def beam_div_size_wigner(self, value):
        self._beam_div_size_wigner = value

    @power.setter
    def power(self, value):
        self._power = value

    @degree_polarization.setter
    def degree_polarization(self, value):
        self._degree_polarization = value

    @degree_coherence.setter
    def degree_coherence(self, value):
        self._degree_coherence = value

    @flux_distribuition_2d.setter
    def flux_distribuition_2d(self, value):
        self._flux_distribuition_2d = value


class CalcParameters:

    def __init__(self):
        self._beta_section = None
        self._target_energy = None
        self._target_k = None
        self._distance_from_source = None
        self._slit_shape = None
        self._slit_acceptance = None
        self._slit_position = None
        self._x_range = None
        self._x_nr_pts = None
        self._y_range = None
        self._y_nr_pts = None
        self._e_range = None
        self._e_nr_pts = None
        self._figsize = None
        self._savefig = None
        self._linewidth = None
        self._dpi = None

    @property
    def beta_section(self):
        return self._beta_section

    @property
    def target_energy(self):
        return self._target_energy

    @property
    def target_k(self):
        return self._target_k

    @property
    def distance_from_source(self):
        return self._distance_from_source

    @property
    def slit_shape(self):
        return self._slit_shape

    @property
    def slit_acceptance(self):
        return self._slit_acceptance

    @property
    def slit_position(self):
        return self._slit_position

    @property
    def x_range(self):
        return self._x_range

    @property
    def x_nr_pts(self):
        return self._x_nr_pts

    @property
    def y_range(self):
        return self._y_range

    @property
    def y_nr_pts(self):
        return self._y_nr_pts

    @property
    def e_range(self):
        return self._e_range

    @property
    def e_nr_pts(self):
        return self._e_nr_pts

    @property
    def figsize(self):
        return self._figsize

    @property
    def savefig(self):
        return self._savefig

    @property
    def linewidth(self):
        return self._linewidth

    @property
    def dpi(self):
        return self._dpi
    
    @beta_section.setter
    def beta_section(self, value):
        self._beta_section = value

    @target_energy.setter
    def target_energy(self, value):
        self._target_energy = value

    @target_k.setter
    def target_k(self, value):
        self._target_k = value

    @distance_from_source.setter
    def distance_from_source(self, value):
        self._distance_from_source = value

    @slit_shape.setter
    def slit_shape(self, value):
        self._slit_shape = value

    @slit_acceptance.setter
    def slit_acceptance(self, value):
        self._slit_acceptance = value

    @slit_position.setter
    def slit_position(self, value):
        self._slit_position = value

    @x_range.setter
    def x_range(self, value):
        self._x_range = value

    @x_nr_pts.setter
    def x_nr_pts(self, value):
        self._x_nr_pts = value

    @y_range.setter
    def y_range(self, value):
        self._y_range = value

    @y_nr_pts.setter
    def y_nr_pts(self, value):
        self._y_nr_pts = value

    @e_range.setter
    def e_range(self, value):
        self._e_range = value

    @e_nr_pts.setter
    def e_nr_pts(self, value):
        self._e_nr_pts = value

    @figsize.setter
    def figsize(self, value):
        self._figsize = value

    @savefig.setter
    def savefig(self, value):
        self._savefig = value

    @linewidth.setter
    def linewidth(self, value):
        self._linewidth = value

    @dpi.setter
    def dpi(self, value):
        self._dpi = value


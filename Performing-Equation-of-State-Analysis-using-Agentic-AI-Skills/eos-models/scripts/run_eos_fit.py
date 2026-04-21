#!/usr/bin/env python3
"""Reference EOS workflow with class-method parity.

Implements the legacy-style API:
    - plot_eos
    - fitting
    - _get_peos
    - _get_eeos

Supports:
    - eostype="energy" from E(V)
    - eostype="pressure" from P(V)
    - eostype="enthalpy" (per-phase best-model selection)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.integrate import cumulative_trapezoid
except ImportError:  # pragma: no cover
    from scipy.integrate import cumtrapz as cumulative_trapezoid


class EOSModel:
    """EOS fitting and plotting class with option/method parity."""

    ENERGY_MODELS = ("Vinet", "Birch", "Murnaghan", "Birch-Murnaghan")

    def plot_eos(
        self,
        infile=None,
        eostype="energy",
        natoms=1,
        au=False,
        vlim=None,
        model=None,
        raw_data=True,
        export=True,
        savefig=True,
    ):
        """Fit EOS and generate plots/reports.

        Parameters
        ----------
        infile : str | list[str]
            Input path; for enthalpy, may be a list of files.
        eostype : str
            "energy", "pressure", or "enthalpy".
        natoms : int
            Number of atoms for per-atom plotting.
        au : bool
            Use atomic units in E(V) plot labels and values.
        vlim : list[float] | tuple[float, float] | None
            Optional [vmin, vmax] interpolation range.
        model : str | None
            Model name; if None, plots all supported models.
        raw_data : bool
            Include raw points in plots.
        export : bool
            Retained for API parity; only the TXT report is written.
        savefig : bool
            Save PNG files.
        """

        self.infile = infile
        self.eostype = eostype
        self.natoms = int(natoms)
        self.au = bool(au)
        self.vlim = vlim
        self.model = model
        self.raw_data = bool(raw_data)
        self.export = bool(export)
        self.savefig = bool(savefig)

        if self.eostype == "energy":
            self.volume, self.energy = self._load_two_column(self.infile, "energy")
            self.coeffs0 = self._initial_guess_energy(self.volume, self.energy)
            self.vol_array = self._volume_grid(self.volume)

            self.fitting()
            self._get_peos()

            self._plot_energy_from_energy()
            self._plot_pressure_from_energy()

            report = self._build_energy_report()
            self._write_reports(report)
            self._print_energy_summary(report)

        elif self.eostype == "pressure":
            self.volume, self.pressure = self._load_two_column(self.infile, "pressure")
            self.coeffs0 = self._initial_guess_pressure(self.volume, self.pressure)
            self.vol_array = self._volume_grid(self.volume)

            self.fitting()
            self._get_eeos()

            self._plot_pressure_from_pressure()
            self._plot_energy_from_pressure()

            report = self._build_pressure_report()
            self._write_reports(report)
            self._print_pressure_summary(report)

        elif self.eostype == "enthalpy":
            self.infiles = self._normalize_infiles(self.infile)
            self.volume = []
            self.energy = []
            self.coeffs0 = []
            self.vol_array = []

            for path in self.infiles:
                v, e = self._load_two_column(path, "energy")
                self.volume.append(v)
                self.energy.append(e)
                self.coeffs0.append(self._initial_guess_energy(v, e))
                self.vol_array.append(self._volume_grid(v))

            self.fitting()
            self._get_peos()

            report = self._build_enthalpy_report()
            self._write_reports(report)
            self._print_enthalpy_summary(report)

        else:
            raise ValueError("eostype must be one of: energy, pressure, enthalpy")

    def fitting(self):
        """Fit coefficients using least squares and compute MSE."""

        if self.eostype == "energy":
            self.energy_fits = {}
            for name, fn in self._energy_model_map().items():
                res = least_squares(
                    lambda coeffs, y, x: y - fn(coeffs, x),
                    self.coeffs0,
                    args=(self.energy, self.volume),
                )
                self.energy_fits[name] = {
                    "result": res,
                    "coeffs": res.x,
                    "mse": float(np.mean(res.fun**2)),
                }

            self.eos_vinet_fitted = self.energy_fits["Vinet"]["result"]
            self.eos_birch_fitted = self.energy_fits["Birch"]["result"]
            self.eos_murnaghan_fitted = self.energy_fits["Murnaghan"]["result"]
            self.eos_birch_murnaghan_fitted = self.energy_fits["Birch-Murnaghan"][
                "result"
            ]

            self.best_model = min(
                self.energy_fits, key=lambda k: self.energy_fits[k]["mse"]
            )

        elif self.eostype == "pressure":
            res = least_squares(
                lambda coeffs, y, x: y - self.eos_birch_murnaghan_pressure(coeffs, x),
                self.coeffs0,
                args=(self.pressure, self.volume),
            )
            self.pressure_fit = {
                "name": "Birch-Murnaghan",
                "result": res,
                "coeffs": res.x,
                "mse": float(np.mean(res.fun**2)),
            }
            self.eos_birch_murnaghan_pressure_fitted = res

        elif self.eostype == "enthalpy":
            self.selected_coeffs = []
            self.model = []
            self.phase_fits = []
            energy_models = self._energy_model_map()

            print("\nDetermining best model for each phase...")
            for idx, path in enumerate(self.infiles):
                fits = {}
                for name, fn in energy_models.items():
                    res = least_squares(
                        lambda coeffs, y, x: y - fn(coeffs, x),
                        self.coeffs0[idx],
                        args=(self.energy[idx], self.volume[idx]),
                    )
                    fits[name] = {
                        "result": res,
                        "coeffs": res.x,
                        "mse": float(np.mean(res.fun**2)),
                    }

                best = min(fits, key=lambda k: fits[k]["mse"])
                self.selected_coeffs.append(fits[best]["coeffs"])
                self.model.append(best)
                self.phase_fits.append(fits)
                print(f"{path} : {best}")

    def _get_peos(self, dv=1e-3):
        """Differentiate energy EOS to pressure using central differences."""

        def pressure_from_energy(model_fn, coeffs, vol):
            e1 = model_fn(coeffs, vol - dv)
            e2 = model_fn(coeffs, vol + dv)
            return -1.0 * (e2 - e1) / (2.0 * dv) * 1.60217e02

        if self.eostype != "enthalpy":
            v = self.vol_array
            self.P_murnaghan = pressure_from_energy(
                self.eos_murnaghan, self.eos_murnaghan_fitted.x, v
            )
            self.P_birch_murnaghan = pressure_from_energy(
                self.eos_birch_murnaghan, self.eos_birch_murnaghan_fitted.x, v
            )
            self.P_birch = pressure_from_energy(
                self.eos_birch, self.eos_birch_fitted.x, v
            )
            self.P_vinet = pressure_from_energy(
                self.eos_vinet, self.eos_vinet_fitted.x, v
            )
        else:
            self.pressure = []
            for idx, coeffs in enumerate(self.selected_coeffs):
                v = self.vol_array[idx]
                name = self.model[idx]
                if name == "Vinet":
                    p = pressure_from_energy(self.eos_vinet, coeffs, v)
                elif name == "Birch":
                    p = pressure_from_energy(self.eos_birch, coeffs, v)
                elif name == "Murnaghan":
                    p = pressure_from_energy(self.eos_murnaghan, coeffs, v)
                else:
                    p = pressure_from_energy(self.eos_birch_murnaghan, coeffs, v)
                self.pressure.append(p)

    def _get_eeos(self):
        """Integrate pressure EOS to relative energy."""

        v = np.asarray(self.volume, dtype=float)
        p = self.eos_birch_murnaghan_pressure(
            self.eos_birch_murnaghan_pressure_fitted.x, v
        )

        idx = np.argsort(v)
        v_sorted = v[idx]
        p_sorted = p[idx]
        e_sorted = -cumulative_trapezoid(p_sorted, v_sorted, initial=0.0) * 6.241509e-3

        self.E_birch_murnaghan = np.empty_like(e_sorted)
        self.E_birch_murnaghan[idx] = e_sorted

        self.vol_array_energy = self._volume_grid(v)
        p_curve = self.eos_birch_murnaghan_pressure(
            self.eos_birch_murnaghan_pressure_fitted.x, self.vol_array_energy
        )
        self.E_birch_murnaghan_curve = (
            -cumulative_trapezoid(p_curve, self.vol_array_energy, initial=0.0)
            * 6.241509e-3
        )

    # EOS model definitions
    def eos_murnaghan(self, coeffs, vol):
        "Ref: Phys. Rev. B 28, 5480 (1983)"
        E0, B0, Bp, V0 = coeffs
        return (
            E0
            + B0 / Bp * vol * ((V0 / vol) ** Bp / (Bp - 1.0) + 1.0)
            - V0 * B0 / (Bp - 1.0)
        )

    def eos_birch_murnaghan(self, coeffs, vol):
        "Ref: Phys. Rev. B 70, 224107"
        E0, B0, Bp, V0 = coeffs
        eta = (vol / V0) ** (1.0 / 3.0)
        return E0 + 9.0 * B0 * V0 / 16.0 * (eta**2 - 1.0) ** 2 * (
            6.0 + Bp * (eta**2 - 1.0) - 4.0 * eta**2
        )

    def eos_birch(self, coeffs, vol):
        """
        Ref: Michael J. Mehl; Barry M. Klein; Dimitris A. Papaconstantopoulos. First-Principles Calculation of Elastic Properties. In Intermetallic Compounds; John Wiley & Sons Ltd, 1994; Vol. 1.
        """
        E0, B0, Bp, V0 = coeffs
        term = (V0 / vol) ** (2.0 / 3.0) - 1.0
        return (
            E0
            + 9.0 / 8.0 * B0 * V0 * term**2
            + 9.0 / 16.0 * B0 * V0 * (Bp - 4.0) * term**3
        )

    def eos_vinet(self, coeffs, vol):
        "Ref: Phys. Rev. B 70, 224107"
        E0, B0, Bp, V0 = coeffs
        eta = (vol / V0) ** (1.0 / 3.0)
        return E0 + 2.0 * B0 * V0 / (Bp - 1.0) ** 2 * (
            2.0
            - (5.0 + 3.0 * Bp * (eta - 1.0) - 3.0 * eta)
            * np.exp(-3.0 * (Bp - 1.0) * (eta - 1.0) / 2.0)
        )

    def eos_birch_murnaghan_pressure(self, coeffs, vol):
        """
        Ref: doi:10.3390/min9120745
        """
        _P0, K0, Kp, V0 = coeffs
        eta = (V0 / vol) ** (1.0 / 3.0)
        return (
            (3.0 / 2.0)
            * K0
            * (eta**7 - eta**5)
            * (1.0 + 0.75 * (Kp - 4.0) * (eta**2 - 1.0))
        )

    # Plotting
    def _plot_energy_from_energy(self):
        fig, ax = plt.subplots(figsize=(10, 7))
        v_conv, e_conv = self._unit_converters()
        vol = v_conv * self.vol_array / self.natoms

        for name in self._selected_energy_models():
            y = self._energy_model_map()[name](
                self.energy_fits[name]["coeffs"], self.vol_array
            )
            style = self._style(name)
            ax.plot(
                vol,
                e_conv * y / self.natoms,
                color=style[0],
                linestyle=style[1],
                linewidth=2,
                label=name,
            )

        if self.raw_data:
            ax.scatter(
                v_conv * self.volume / self.natoms,
                e_conv * self.energy / self.natoms,
                s=50,
                facecolors="none",
                edgecolors="black",
                label="Raw Data",
            )

        if self.au:
            ax.set_xlabel("Volume (Bohr^3/atom)")
            ax.set_ylabel("Energy (Ha/atom)")
        else:
            ax.set_xlabel("Volume (Angstrom^3/atom)")
            ax.set_ylabel("Energy (eV/atom)")
        ax.set_title("Energy vs Volume")
        ax.legend(loc="best")
        fig.tight_layout()
        if self.savefig:
            fig.savefig("EvsV.png", dpi=300)
        plt.close(fig)

    def _plot_pressure_from_energy(self):
        fig, ax = plt.subplots(figsize=(10, 7))
        v_conv, _e_conv = self._unit_converters()
        vol = v_conv * self.vol_array / self.natoms

        pressure_map = {
            "Vinet": self.P_vinet,
            "Birch": self.P_birch,
            "Murnaghan": self.P_murnaghan,
            "Birch-Murnaghan": self.P_birch_murnaghan,
        }
        for name in self._selected_energy_models():
            style = self._style(name)
            ax.plot(
                vol,
                pressure_map[name],
                color=style[0],
                linestyle=style[1],
                linewidth=2,
                label=name,
            )

        if self.au:
            ax.set_xlabel("Volume (Bohr^3/atom)")
        else:
            ax.set_xlabel("Volume (Angstrom^3/atom)")
        ax.set_ylabel("Pressure (GPa)")
        ax.set_title("Pressure vs Volume")
        ax.legend(loc="best")
        fig.tight_layout()
        if self.savefig:
            fig.savefig("PvsV.png", dpi=300)
        plt.close(fig)

    def _plot_pressure_from_pressure(self):
        fig, ax = plt.subplots(figsize=(10, 7))
        v_conv, _e_conv = self._unit_converters()
        vol = v_conv * self.vol_array / self.natoms
        p_fit = self.eos_birch_murnaghan_pressure(
            self.pressure_fit["coeffs"], self.vol_array
        )

        ax.plot(vol, p_fit, color="orangered", linewidth=2, label="Birch-Murnaghan")
        if self.raw_data:
            ax.scatter(
                v_conv * self.volume / self.natoms,
                self.pressure / self.natoms,
                s=50,
                facecolors="none",
                edgecolors="black",
                label="Raw Data",
            )

        if self.au:
            ax.set_xlabel("Volume (Bohr^3/atom)")
        else:
            ax.set_xlabel("Volume (Angstrom^3/atom)")
        ax.set_ylabel("Pressure (GPa)")
        ax.set_title("Pressure vs Volume")
        ax.legend(loc="best")
        fig.tight_layout()
        if self.savefig:
            fig.savefig("PvsV.png", dpi=300)
        plt.close(fig)

    def _plot_energy_from_pressure(self):
        fig, ax = plt.subplots(figsize=(10, 7))
        v_conv, e_conv = self._unit_converters()

        ax.plot(
            v_conv * self.vol_array_energy / self.natoms,
            e_conv * self.E_birch_murnaghan_curve / self.natoms,
            color="black",
            linewidth=2,
            label="Integrated from P(V)",
        )

        if self.au:
            ax.set_xlabel("Volume (Bohr^3/atom)")
            ax.set_ylabel("Relative Energy (Ha/atom)")
        else:
            ax.set_xlabel("Volume (Angstrom^3/atom)")
            ax.set_ylabel("Relative Energy (eV/atom)")
        ax.set_title("Energy vs Volume (from pressure integration)")
        ax.legend(loc="best")
        fig.tight_layout()
        if self.savefig:
            fig.savefig("EvsV.png", dpi=300)
        plt.close(fig)

    # Reporting
    def _build_energy_report(self):
        models = []
        for name in self.ENERGY_MODELS:
            models.append(
                {
                    "name": name,
                    "coeffs": [float(x) for x in self.energy_fits[name]["coeffs"]],
                    "mse": float(self.energy_fits[name]["mse"]),
                }
            )
        return {
            "infile": str(self.infile),
            "eostype": "energy",
            "natoms": self.natoms,
            "au": self.au,
            "models": models,
            "best_model": self.best_model,
            "files": ["EOS_fit_results.txt", "EvsV.png", "PvsV.png"],
        }

    def _build_pressure_report(self):
        return {
            "infile": str(self.infile),
            "eostype": "pressure",
            "natoms": self.natoms,
            "au": self.au,
            "models": [
                {
                    "name": "Birch-Murnaghan",
                    "coeffs": [float(x) for x in self.pressure_fit["coeffs"]],
                    "mse": float(self.pressure_fit["mse"]),
                }
            ],
            "best_model": "Birch-Murnaghan",
            "files": ["EOS_fit_results.txt", "PvsV.png", "EvsV.png"],
        }

    def _build_enthalpy_report(self):
        phases = []
        for idx, path in enumerate(self.infiles):
            best = self.model[idx]
            fits = self.phase_fits[idx]
            phase_models = []
            for name in self.ENERGY_MODELS:
                phase_models.append(
                    {
                        "name": name,
                        "coeffs": [float(x) for x in fits[name]["coeffs"]],
                        "mse": float(fits[name]["mse"]),
                    }
                )
            phases.append(
                {
                    "infile": str(path),
                    "best_model": best,
                    "models": phase_models,
                }
            )

        return {
            "infile": [str(x) for x in self.infiles],
            "eostype": "enthalpy",
            "natoms": self.natoms,
            "au": self.au,
            "phases": phases,
            "files": ["EOS_fit_results.txt"],
        }

    def _write_reports(self, report):
        self._write_txt(report)

    def _write_txt(self, report):
        with Path("EOS_fit_results.txt").open("w", encoding="utf-8") as f:
            if report["eostype"] == "energy":
                f.write(f"EOS fit report for {report['infile']}\n")
                f.write(
                    "Columns: [E0 (eV), B0 (eV/Angstrom^3), Bp, V0 (Angstrom^3)]\n\n"
                )
                for item in report["models"]:
                    coeffs = item["coeffs"]
                    f.write(f"{item['name']}:\n")
                    f.write(
                        f"  coeffs = [{coeffs[0]: .10f}, {coeffs[1]: .10f}, {coeffs[2]: .10f}, {coeffs[3]: .10f}]\n"
                    )
                    f.write(f"  MSE    = {item['mse']:.6e}\n\n")
                f.write(f"Best model by MSE: {report['best_model']}\n")

            elif report["eostype"] == "pressure":
                item = report["models"][0]
                coeffs = item["coeffs"]
                f.write(f"EOS fit report for {report['infile']}\n")
                f.write("Model: Birch-Murnaghan pressure EOS\n")
                f.write("Columns: [P0 (GPa), K0 (GPa), Kp, V0 (Angstrom^3)]\n\n")
                f.write(
                    f"coeffs = [{coeffs[0]: .10f}, {coeffs[1]: .10f}, {coeffs[2]: .10f}, {coeffs[3]: .10f}]\n"
                )
                f.write(f"MSE    = {item['mse']:.6e}\n")

            else:
                f.write("EOS fit report for enthalpy workflow\n\n")
                for phase in report["phases"]:
                    f.write(f"{phase['infile']}\n")
                    f.write(f"Best model: {phase['best_model']}\n")
                    for item in phase["models"]:
                        coeffs = item["coeffs"]
                        f.write(
                            f"  {item['name']}: coeffs=[{coeffs[0]: .10f}, {coeffs[1]: .10f}, {coeffs[2]: .10f}, {coeffs[3]: .10f}], MSE={item['mse']:.6e}\n"
                        )
                    f.write("\n")

    # Summaries
    def _print_energy_summary(self, report):
        print("Fitting complete.")
        print(f"Best model: {report['best_model']}")
        for item in report["models"]:
            print(f"{item['name']} MSE: {item['mse']:.6e}")
        self._print_written_files(report)

    def _print_pressure_summary(self, report):
        item = report["models"][0]
        print("Fitting complete.")
        print("Model: Birch-Murnaghan pressure EOS")
        print(f"MSE: {item['mse']:.6e}")
        self._print_written_files(report)

    def _print_enthalpy_summary(self, report):
        print("Fitting complete.")
        for phase in report["phases"]:
            print(f"{phase['infile']}: {phase['best_model']}")
        self._print_written_files(report)

    def _print_written_files(self, report):
        print("Wrote: EOS_fit_results.txt")
        if self.savefig:
            for filename in report.get("files", []):
                if filename.endswith(".png"):
                    print(f"Wrote: {filename}")

    # Utilities
    def _energy_model_map(self):
        return {
            "Vinet": self.eos_vinet,
            "Birch": self.eos_birch,
            "Murnaghan": self.eos_murnaghan,
            "Birch-Murnaghan": self.eos_birch_murnaghan,
        }

    def _selected_energy_models(self):
        if self.model is None:
            return self.ENERGY_MODELS
        if self.model not in self.ENERGY_MODELS:
            raise ValueError(f"Unsupported model: {self.model}")
        return (self.model,)

    def _style(self, name):
        styles = {
            "Vinet": ("orangered", "-"),
            "Birch": ("limegreen", "-"),
            "Murnaghan": ("magenta", "--"),
            "Birch-Murnaghan": ("black", "--"),
        }
        return styles[name]

    def _unit_converters(self):
        if self.au:
            return 6.748376012323316, 0.0367493
        return 1.0, 1.0

    def _normalize_infiles(self, infile):
        if isinstance(infile, (list, tuple)):
            if not infile:
                raise ValueError("For enthalpy, provide one or more input files")
            return [str(x) for x in infile]
        if infile is None:
            raise ValueError("For enthalpy, provide input files")
        text = str(infile)
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if not parts:
            raise ValueError("For enthalpy, provide input files")
        return parts

    def _load_two_column(self, infile, second_name):
        if infile is None:
            raise ValueError("infile is required")
        data = np.loadtxt(infile)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"Expected at least two columns: volume and {second_name}")
        return data[:, 0].astype(float), data[:, 1].astype(float)

    def _volume_grid(self, volume):
        if self.vlim is not None:
            if len(self.vlim) != 2:
                raise ValueError("vlim must contain [vmin, vmax]")
            return np.linspace(float(self.vlim[0]), float(self.vlim[1]), 1000)
        return np.linspace(float(np.min(volume)), float(np.max(volume)), 1000)

    def _initial_guess_energy(self, volume, energy):
        a, b, c = np.polyfit(volume, energy, 2)
        if np.isclose(a, 0.0):
            raise ValueError(
                "Quadratic fit produced near-zero curvature; cannot build stable initial guess"
            )
        V0 = -b / (2 * a)
        E0 = a * V0**2 + b * V0 + c
        B0 = 2 * a * V0
        Bp = 4.0
        return np.array([E0, B0, Bp, V0], dtype=float)

    def _initial_guess_pressure(self, volume, pressure):
        a, b, c = np.polyfit(volume, pressure, 2)
        if np.isclose(a, 0.0):
            raise ValueError(
                "Quadratic fit produced near-zero curvature; cannot build stable initial guess"
            )
        V0 = -b / (2 * a)
        P0 = a * V0**2 + b * V0 + c
        K0 = 2 * a * V0
        Kp = 4.0
        return np.array([P0, K0, Kp, V0], dtype=float)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Reference EOS model fitting workflow")
    parser.add_argument(
        "--infile",
        default="EvsV.txt",
        help="Input file path; comma-separated list for enthalpy",
    )
    parser.add_argument(
        "--eostype", default="energy", choices=["energy", "pressure", "enthalpy"]
    )
    parser.add_argument("--natoms", type=int, default=1)
    parser.add_argument(
        "--au", action="store_true", help="Use atomic units in E(V) plots"
    )
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument(
        "--model",
        default=None,
        choices=[None, "Vinet", "Birch", "Murnaghan", "Birch-Murnaghan"],
    )
    parser.add_argument("--no-raw-data", action="store_true")
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--no-savefig", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if (args.vmin is None) ^ (args.vmax is None):
        raise ValueError("Provide both --vmin and --vmax, or neither")
    vlim = None if args.vmin is None else [args.vmin, args.vmax]

    eos = EOSModel()
    eos.plot_eos(
        infile=args.infile,
        eostype=args.eostype,
        natoms=args.natoms,
        au=args.au,
        vlim=vlim,
        model=args.model,
        raw_data=not args.no_raw_data,
        export=not args.no_export,
        savefig=not args.no_savefig,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)

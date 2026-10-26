from dataclasses import dataclass
from pathlib import Path
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from scipy.ndimage import gaussian_filter
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import warnings


@dataclass
class TAPlotConfig:
    max_rows: int = 513
    cmap: str = "bwr"
    robust_percentile: float = 99
    color_vmin: list[float] | float | None = None
    color_vmax: list[float] | float | None = None
    symmetric_color_scale: bool = True
    smooth_sigma: tuple[float, float] | None = (1.2, 1.2)
    xlim: tuple[float, float] = (420, 600)
    lower_ylim: tuple[float, float] = (-1, 10)
    upper_ylim: tuple[float, float] = (10.1, 1000)
    upper_log_scale: bool = True
    upper_yticks: tuple[float, ...] = (100, 1000)
    figsize: tuple[float, float] = (8, 6)
    height_ratios: tuple[float, float] = (2.5, 1.0)
    hspace: float = 0.01
    scale_to_milli: bool = True
    dpi: int = 300
    transparent: bool = True
    tick_labelsize: int = 18
    label_fontsize: int = 18
    title_fontsize: int = 16
    show_titles: bool = True
    background_correction: bool = True
    background_time_range: tuple[float, float] = (-5.0, -0.5)
    background_wavelength_range: tuple[float, float] | None = None
    chirp_correction: bool = False
    chirp_points: tuple[tuple[float, float], ...] | None = None
    chirp_poly_coefficients: tuple[float, ...] | None = None
    chirp_poly_degree: int = 2
    chirp_reference_wavelength: float | None = None
    chirp_reference_time: float = 0.0
    normalize_global: bool = False
    line_profile_ymin: float | list[float] | None = None
    line_profile_ymax: float | list[float] | None = None
    line_profile_ylim: list[tuple[float, float]] | tuple[float, float] | None = None
    global_normalization_reference: str = "minimum"
    colorbar_title: str = r"$\Delta A$ (mOD)"
    normalized_colorbar_title: str = "Normalized " + r"$\Delta A$"


class TADataset:
    def __init__(self, file_path, label=None, max_rows=513):
        self.file_path = Path(file_path)
        self.label = label or self.file_path.stem
        self.max_rows = max_rows
        self.delay_time = None
        self.wavelength = None
        self.signal = None

    def load(self):
        data = np.genfromtxt(
            self.file_path,
            delimiter=",",
            max_rows=self.max_rows,
        )
        self.delay_time = data[0, 1:]
        self.wavelength = data[1:, 0]
        self.signal = data[1:, 1:]
        return self

    @classmethod
    def from_matrix(cls, matrix, label="Dataset", max_rows=513):
        """Build a dataset from the editable project matrix without rereading a file."""
        data = np.asarray(matrix, dtype=float)
        if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
            raise ValueError("TAS data must include a delay-time row and wavelength column.")
        dataset = cls("<editable-project-data>", label=label, max_rows=max_rows)
        dataset.delay_time = data[0, 1:].copy()
        dataset.wavelength = data[1:, 0].copy()
        dataset.signal = data[1:, 1:].copy()
        return dataset

    def crop_valid_region(self):
        if self.signal is None:
            self.load()

        valid_rows = np.isfinite(self.signal).any(axis=1)
        valid_cols = np.isfinite(self.signal).any(axis=0)

        self.delay_time = self.delay_time[valid_cols]
        self.wavelength = self.wavelength[valid_rows]
        self.signal = self.signal[np.ix_(valid_rows, valid_cols)]
        return self

    def smooth(self, smooth_sigma):
        if smooth_sigma is None:
            return self

        finite_mask = np.isfinite(self.signal)
        signal_filled = np.nan_to_num(self.signal, nan=0.0)

        weight = gaussian_filter(finite_mask.astype(float), sigma=smooth_sigma)
        signal_smooth = gaussian_filter(signal_filled, sigma=smooth_sigma)

        self.signal = np.divide(
            signal_smooth,
            weight,
            out=np.full_like(signal_smooth, np.nan),
            where=weight > 0,
        )
        return self

    def subtract_background(self, time_range, wavelength_range=None):
        if time_range is None:
            return self

        t_min, t_max = time_range
        time_mask = (self.delay_time >= t_min) & (self.delay_time <= t_max)

        if wavelength_range is not None:
            wl_min, wl_max = wavelength_range
            wavelength_mask = (
                (self.wavelength >= wl_min)
                & (self.wavelength <= wl_max)
            )
        else:
            wavelength_mask = np.ones_like(self.wavelength, dtype=bool)

        if not np.any(time_mask):
            raise ValueError(
                "background_time_range does not overlap the delay-time axis."
            )
        if not np.any(wavelength_mask):
            raise ValueError(
                "background_wavelength_range does not overlap the wavelength axis."
            )

        # Some instrument exports contain entirely missing wavelength rows.
        # Preserve them as NaN without emitting a warning during every render.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
            background = np.nanmedian(
                self.signal[np.ix_(wavelength_mask, time_mask)],
                axis=1,
            )

        if wavelength_range is not None:
            full_background = np.zeros(self.signal.shape[0], dtype=float)
            full_background[wavelength_mask] = background
            self.signal[wavelength_mask, :] = (
                self.signal[wavelength_mask, :]
                - full_background[wavelength_mask, None]
            )
        else:
            self.signal = self.signal - background[:, None]

        return self

    def chirp_correct(
        self,
        chirp_points=None,
        chirp_poly_coefficients=None,
        chirp_poly_degree=3,
        reference_wavelength=None,
        reference_time=0.0,
    ):
        """
        Apply wavelength-dependent chirp correction.
    
        Parameters
        ----------
        chirp_points : sequence of (wavelength, t0)
            Measured time-zero values used to fit the chirp.
        chirp_poly_coefficients : array-like, optional
            Precomputed polynomial coefficients.
        chirp_poly_degree : int, default=3
            Degree of polynomial used to fit chirp.
        reference_wavelength : float, optional
            Wavelength whose time-zero is defined as reference_time.
        reference_time : float, default=0.0
            Desired time-zero at the reference wavelength.
        """
    
        if chirp_poly_coefficients is None and chirp_points is None:
            return self
    
        # -------------------------------------------------------------
        # Determine polynomial coefficients
        # -------------------------------------------------------------
        if chirp_poly_coefficients is not None:
    
            coefficients = np.asarray(chirp_poly_coefficients, dtype=float)
    
        else:
    
            chirp_points = np.asarray(chirp_points, dtype=float)
    
            if chirp_points.ndim != 2 or chirp_points.shape[1] != 2:
                raise ValueError(
                    "chirp_points must contain wavelength/time-zero pairs, "
                    "e.g. ((450,-0.2),(520,0.0),(600,0.8))."
                )
    
            wavelengths = chirp_points[:, 0]
            time_zero = chirp_points[:, 1]
    
            # Use cubic whenever possible
            degree = min(chirp_poly_degree, len(chirp_points) - 1)
    
            coefficients = np.polyfit(
                wavelengths,
                time_zero,
                degree,
            )
    
        # -------------------------------------------------------------
        # Evaluate chirp curve
        # -------------------------------------------------------------
        t0 = np.polyval(coefficients, self.wavelength)
    
        # -------------------------------------------------------------
        # Reference wavelength
        # -------------------------------------------------------------
        if reference_wavelength is None:
            reference_wavelength = np.nanmedian(self.wavelength)
    
        reference_t0 = np.polyval(coefficients, reference_wavelength)
    
        t0 = t0 - reference_t0 + reference_time
    
        # -------------------------------------------------------------
        # Shift every wavelength independently
        # -------------------------------------------------------------
        corrected = np.full_like(self.signal, np.nan, dtype=float)
    
        for row_index, wavelength_t0 in enumerate(t0):
    
            corrected_time = self.delay_time - wavelength_t0
    
            valid = (
                np.isfinite(corrected_time)
                & np.isfinite(self.signal[row_index])
            )
    
            if np.count_nonzero(valid) < 4:
                continue
    
            order = np.argsort(corrected_time[valid])
    
            f = interp1d(
                corrected_time[valid][order],
                self.signal[row_index][valid][order],
                kind="cubic",
                bounds_error=False,
                fill_value=np.nan,
                assume_sorted=True,
            )
    
            corrected[row_index] = f(self.delay_time)
    
        self.signal = corrected
    
        return self

    def prepare(self, config):
        self.load()
        self.crop_valid_region()

        if config.background_correction:
            self.subtract_background(
                config.background_time_range,
                wavelength_range=config.background_wavelength_range,
            )

        if config.chirp_correction:
            self.chirp_correct(
                chirp_points=config.chirp_points,
                chirp_poly_coefficients=config.chirp_poly_coefficients,
                chirp_poly_degree=config.chirp_poly_degree,
                reference_wavelength=config.chirp_reference_wavelength,
                reference_time=config.chirp_reference_time,
            )

        self.smooth(config.smooth_sigma)

        if config.scale_to_milli:
            self.signal = self.signal * 1000

        return self


class TAPlotter:
    def __init__(self, config=None):
        self.config = config or TAPlotConfig()
        self.datasets = []

    def load_files(self, file_paths, labels=None):
        labels = labels or [None] * len(file_paths)
        self.datasets = [
            TADataset(path, label=label, max_rows=self.config.max_rows).prepare(
                self.config
            )
            for path, label in zip(file_paths, labels)
        ]
        if self.config.normalize_global:
            self._normalize_datasets_global(self.datasets)
        return self.datasets

    def _normalize_datasets_global(self, datasets):
        amplitudes = [
            np.nanmax(np.abs(dataset.signal))
            for dataset in datasets
            if np.isfinite(dataset.signal).any()
        ]
        amplitudes = [value for value in amplitudes if value > 0]

        if not amplitudes:
            return

        if self.config.global_normalization_reference == "minimum":
            reference = min(amplitudes)
        elif self.config.global_normalization_reference == "maximum":
            reference = max(amplitudes)
        else:
            raise ValueError(
                'global_normalization_reference must be "minimum" or "maximum".'
            )

        if reference > 0:
            for dataset in datasets:
                dataset.signal = dataset.signal / reference
    def _dataset_color_limits(self, datasets, index):
    
        if isinstance(self.config.color_vmin, (list, tuple)):
            vmin = self.config.color_vmin[index]
        else:
            vmin = self.config.color_vmin
    
        if isinstance(self.config.color_vmax, (list, tuple)):
            vmax = self.config.color_vmax[index]
        else:
            vmax = self.config.color_vmax
    
        if vmin is None and vmax is None:
            return self._color_limits([datasets[index]])
    
        if vmin is None:
            vmin = -abs(vmax)
    
        if vmax is None:
            vmax = abs(vmin)
    
        return float(vmin), float(vmax)

    @staticmethod
    def _add_break_marks(ax_top, ax_bottom, size=0.007):
        kwargs = dict(color="black", clip_on=False, linewidth=1.2)

        ax_top.plot(
            (-size, size),
            (-size, size),
            transform=ax_top.transAxes,
            **kwargs,
        )
        ax_top.plot(
            (1 - size, 1 + size),
            (-size, size),
            transform=ax_top.transAxes,
            **kwargs,
        )
        ax_bottom.plot(
            (-size, size),
            (1 - size, 1 + size),
            transform=ax_bottom.transAxes,
            **kwargs,
        )
        ax_bottom.plot(
            (1 - size, 1 + size),
            (1 - size, 1 + size),
            transform=ax_bottom.transAxes,
            **kwargs,
        )

    def _color_limits(self, datasets):
        if self.config.normalize_global:
            return -1.0, 1.0

        all_values = np.concatenate([
            d.signal[np.isfinite(d.signal)].ravel()
            for d in datasets
        ])

        vmax = np.nanpercentile(
            np.abs(all_values),
            self.config.robust_percentile,
        )

        if self.config.symmetric_color_scale:
            vmin = -vmax
        else:
            vmin = np.nanpercentile(
                all_values,
                100 - self.config.robust_percentile,
            )

        return vmin, vmax

    def _colorbar_title(self):
        if self.config.normalize_global:
            return self.config.normalized_colorbar_title
        return self.config.colorbar_title

    def _plot_dataset_on_axes(self, dataset, ax_top, ax_bottom, vmin, vmax):
        cfg = self.config

        mesh_top = ax_top.pcolormesh(
            dataset.wavelength,
            dataset.delay_time,
            dataset.signal.T,
            shading="auto",
            cmap=cfg.cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax_bottom.pcolormesh(
            dataset.wavelength,
            dataset.delay_time,
            dataset.signal.T,
            shading="auto",
            cmap=cfg.cmap,
            vmin=vmin,
            vmax=vmax,
        )

        ax_top.set_ylim(*cfg.upper_ylim)
        ax_bottom.set_ylim(*cfg.lower_ylim)
        ax_bottom.set_xlim(*cfg.xlim)

        if cfg.upper_log_scale:
            ax_top.set_yscale("log")
            ax_top.set_yticks(cfg.upper_yticks)
            ax_top.set_yticklabels([f"{tick:g}" for tick in cfg.upper_yticks])
            ax_top.yaxis.set_minor_formatter(NullFormatter())

        ax_top.spines["bottom"].set_visible(False)
        ax_bottom.spines["top"].set_visible(False)
        ax_top.tick_params(labelbottom=False, bottom=False)
        ax_bottom.xaxis.tick_bottom()
        ax_bottom.axhline(0, color="black", linestyle="--", linewidth=1)
        ax_top.text(
            0.05,
            0.92,
            dataset.label,
            transform=ax_top.transAxes,
            fontsize=cfg.title_fontsize,
            fontweight="bold",
            ha="left",
            va="top",
            bbox=dict(
                facecolor="white",
                alpha=0.7,
                edgecolor="none",
                pad=2,
            ),
        )
        self._add_break_marks(ax_top, ax_bottom)

        for ax in (ax_top, ax_bottom):
            ax.tick_params(axis="both", labelsize=cfg.tick_labelsize)

        return mesh_top

    def _add_single_colorbar(self, fig, ax_top, ax_bottom, mesh):
        fig.subplots_adjust(left=0.12, right=0.84, bottom=0.12, top=0.94)

        top_pos = ax_top.get_position()
        bottom_pos = ax_bottom.get_position()
        cax = fig.add_axes(
            [
                0.87,
                bottom_pos.y0,
                0.025,
                top_pos.y1 - bottom_pos.y0,
            ]
        )
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.ax.set_title(
            self._colorbar_title(),
            fontsize=self.config.label_fontsize,
            fontweight="bold",
            pad=10,
        )
        cbar.ax.tick_params(labelsize=self.config.tick_labelsize)
        return cbar

    def plot_matrix(
        self,
        file_paths,
        labels=None,
        ncols=2,
        figsize=(12, 8),
        output_path=None,
        show=True,
        share_colorbar=True,
        nrows=None,
        horizontal_spacing=0.36,
        vertical_spacing=0.28,
    ):
        datasets = self.load_files(file_paths, labels=labels)
        nfiles = len(datasets)
        nrows = nrows or int(np.ceil(nfiles / ncols))

        shared_colorbar = share_colorbar
        shared_color_limits = share_colorbar
        
        if shared_color_limits:
            vmin, vmax = self._color_limits(datasets)

        fig = plt.figure(figsize=figsize)
        outer = fig.add_gridspec(
            nrows,
            ncols,
            wspace=horizontal_spacing,
            hspace=vertical_spacing,
            left=0.08,
            right=0.88 if shared_colorbar else 0.96,
            bottom=0.08,
            top=0.95,
        )

        first_mesh = None
        axes_groups = []

        for index, dataset in enumerate(datasets):
            row = index // ncols
            col = index % ncols

            inner = outer[row, col].subgridspec(
                2,
                1,
                height_ratios=self.config.height_ratios,
                hspace=self.config.hspace,
            )
            ax_top = fig.add_subplot(inner[0, 0])
            ax_bottom = fig.add_subplot(inner[1, 0], sharex=ax_top)

            if shared_color_limits:
                dataset_vmin, dataset_vmax = vmin, vmax
            else:
                dataset_vmin, dataset_vmax = self._dataset_color_limits(datasets, index)

            mesh = self._plot_dataset_on_axes(
                dataset,
                ax_top,
                ax_bottom,
                dataset_vmin,
                dataset_vmax,
            )
            if first_mesh is None:
                first_mesh = mesh

            if index + ncols >= nfiles:
                ax_bottom.set_xlabel(
                    "Wavelength (nm)",
                    fontsize=self.config.label_fontsize,
                    fontweight="bold",
                )
            if col == 0:
                ax_top.set_ylabel(
                    "Delay time (ps)",
                    fontsize=self.config.label_fontsize,
                    fontweight="bold",
                )

            axes_groups.append((ax_top, ax_bottom))

            if not shared_colorbar:
                self._add_matrix_item_colorbar(fig, ax_top, ax_bottom, mesh)

        for empty_index in range(nfiles, nrows * ncols):
            row = empty_index // ncols
            col = empty_index % ncols
            ax = fig.add_subplot(outer[row, col])
            ax.axis("off")

        if shared_colorbar and first_mesh is not None:
            cax = fig.add_axes([0.91, 0.12, 0.022, 0.76])
            cbar = fig.colorbar(first_mesh, cax=cax)
            cbar.ax.set_title(
                self._colorbar_title(),
                fontsize=self.config.label_fontsize,
                fontweight="bold",
                pad=10,
            )
            cbar.ax.tick_params(labelsize=self.config.tick_labelsize)

        if output_path is not None:
            fig.savefig(
                output_path,
                dpi=self.config.dpi,
                transparent=self.config.transparent,
                bbox_inches="tight",
            )

        if show:
            plt.show()

        return fig, axes_groups

    def plot_line_profile_matrix(
        self,
        file_paths,
        labels=None,
        delays=1.0,
        ncols=3,
        figsize=(15, 4),
        xlim=None,
        ylim=None,
        output_path=None,
        show=True,
        nrows=None,
        delays_by_file=None,
    ):
        if np.isscalar(delays):
            delays = [delays]
    
        datasets = self.load_files(file_paths, labels=labels)
    
        nfiles = len(datasets)
        nrows = nrows or int(np.ceil(nfiles / ncols))
    
        cmap = plt.get_cmap(self.config.cmap)
    
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            squeeze=False,
            sharex=True,
            sharey=False,          # <-- allow each subplot its own y-axis
        )
    
        default_delays = delays
        for index, dataset in enumerate(datasets):
            delays = (delays_by_file[index] if delays_by_file and delays_by_file[index] is not None else default_delays)
            delays = np.atleast_1d(delays)

            row = index // ncols
            col = index % ncols
            ax = axes[row, col]
    
            # -----------------------------------------
            # Per-dataset colour scale
            # -----------------------------------------
            vmin, vmax = self._dataset_color_limits(
                datasets,
                index,
            )
    
            norm = Normalize(
                vmin=vmin,
                vmax=vmax,
            )
    
            legend_handles = []
            legend_labels = []
    
            for delay in delays:
                delay_index = np.nanargmin(
                    np.abs(dataset.delay_time - delay)
                )
                
                # Use the nearest available point in the dataset
                profile = dataset.signal[:, delay_index]
                
                x = dataset.wavelength
                y = profile
                
                valid = np.isfinite(x) & np.isfinite(y)
                
                x = x[valid]
                y = y[valid]
                
                if len(x) < 2:
                    continue
                
                points = np.array([x, y]).T.reshape(-1, 1, 2)
                
                segments = np.concatenate(
                    [points[:-1], points[1:]],
                    axis=1,
                )
                
                line = LineCollection(
                    segments,
                    cmap=cmap,
                    norm=norm,
                    linewidth=2,
                )
                
                line.set_array(y)
                
                ax.add_collection(line)
                
                mean_color = cmap(norm(np.nanmean(y)))
                
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        color=mean_color,
                        linewidth=2,
                    )
                )
                
                # Always display the requested delay, not the nearest dataset value
                legend_labels.append(
                    f"{delay:.1f} ps"
                )
    
            ax.axhline(
                0,
                color="gray",
                linestyle="--",
                linewidth=1,
            )
    
            ax.text(
                0.05,
                0.10,
                dataset.label,
                transform=ax.transAxes,
                fontsize=self.config.title_fontsize,
                fontweight="bold",
                ha="left",
                va="top",
            )
    
            ax.tick_params(
                axis="both",
                labelsize=self.config.tick_labelsize,
            )
    
            ax.grid(
                alpha=0.25,
                linestyle="--",
            )
    
            if len(legend_handles) > 2:
                legend_handles = [
                    legend_handles[0],
                    legend_handles[-1],
                ]
    
                legend_labels = [
                    legend_labels[0],
                    legend_labels[-1],
                ]
    
            ax.legend(
                legend_handles,
                legend_labels,
                fontsize=16,
                frameon=True,
            )
    
            # -----------------------------------------
            # x limits
            # -----------------------------------------
            if xlim is not None:
                ax.set_xlim(*xlim)
            else:
                ax.set_xlim(*self.config.xlim)
    
            # -----------------------------------------
            # y limits
            # -----------------------------------------
            if ylim is not None:
    
                ax.set_ylim(*ylim)
    
            elif self.config.line_profile_ylim is not None:
    
                if isinstance(
                    self.config.line_profile_ylim,
                    (list, tuple),
                ) and isinstance(
                    self.config.line_profile_ylim[0],
                    tuple,
                ):
    
                    ax.set_ylim(
                        *self.config.line_profile_ylim[index]
                    )
    
                else:
    
                    ax.set_ylim(
                        *self.config.line_profile_ylim
                    )
    
            else:
    
                profiles = []
    
                for delay in delays:
    
                    delay_index = np.nanargmin(
                        np.abs(
                            dataset.delay_time - delay
                        )
                    )
    
                    profiles.append(
                        dataset.signal[:, delay_index]
                    )
    
                profiles = np.concatenate(profiles)
    
                ymin = np.nanpercentile(
                    profiles,
                    1,
                )
    
                ymax = np.nanpercentile(
                    profiles,
                    99,
                )
    
                ax.set_ylim(
                    ymin,
                    ymax,
                )
    
            # -----------------------------------------
            # labels
            # -----------------------------------------
            if index + ncols >= nfiles:
    
                ax.set_xlabel(
                    "Wavelength (nm)",
                    fontsize=self.config.label_fontsize,
                    fontweight="bold",
                )
    
            if col == 0:
    
                ax.set_ylabel(
                    self._colorbar_title(),
                    fontsize=self.config.label_fontsize,
                    fontweight="bold",
                )
    

    
        # -----------------------------------------
        # Hide unused axes
        # -----------------------------------------
        for empty_index in range(
            nfiles,
            nrows * ncols,
        ):
    
            row = empty_index // ncols
            col = empty_index % ncols
    
            axes[row, col].axis("off")
    
        fig.tight_layout()
    
        if output_path is not None:
    
            fig.savefig(
                output_path,
                dpi=self.config.dpi,
                transparent=self.config.transparent,
                bbox_inches="tight",
            )
    
        if show:
            plt.show()
    
        return fig, axes

    def _add_matrix_item_colorbar(self, fig, ax_top, ax_bottom, mesh):
        top_pos = ax_top.get_position()
        bottom_pos = ax_bottom.get_position()

        cax = fig.add_axes(
            [
                top_pos.x1 + 0.008,
                bottom_pos.y0,
                0.010,
                top_pos.y1 - bottom_pos.y0,
            ]
        )
        cbar = fig.colorbar(mesh, cax=cax)
        cbar.ax.set_title(
            self._colorbar_title(),
            fontsize=self.config.label_fontsize,
            fontweight="bold",
            pad=8,
        )
        cbar.ax.tick_params(labelsize=self.config.tick_labelsize)
        return cbar
    def _dataset_line_ylim(self, datasets, index):
    
        ylim = self.config.line_profile_ylim
    
        if ylim is None:
            return None
    
        if isinstance(ylim, list):
            if len(ylim) != len(datasets):
                raise ValueError(
                    f"Expected {len(datasets)} y-axis limits, "
                    f"got {len(ylim)}."
                )
            return ylim[index]
    
        return ylim

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class GSBLifetimeFitter:
    """
    Extract GSB kinetics using the original mytas preprocessing,
    then fit each trace with n exponential lifetime components.
    """

    def __init__(self, config):
        self.config = config
        self.datasets = []
        self.fit_results = []

    def load_files(self, file_paths, labels=None):
        ta_plotter = TAPlotter(self.config)
        self.datasets = ta_plotter.load_files(file_paths, labels=labels)
        return self.datasets

    @staticmethod
    def _find_gsb_wavelength(dataset, wavelength_range, time_range):
        wl_min, wl_max = wavelength_range
        time_min, time_max = time_range

        wl_mask = (
            (dataset.wavelength >= wl_min)
            & (dataset.wavelength <= wl_max)
        )
        time_mask = (
            (dataset.delay_time >= time_min)
            & (dataset.delay_time <= time_max)
        )

        if not np.any(wl_mask) or not np.any(time_mask):
            raise ValueError(
                f"GSB search range does not overlap {dataset.label}."
            )

        mean_spectrum = np.nanmean(
            dataset.signal[np.ix_(wl_mask, time_mask)],
            axis=1,
        )

        minimum_index = np.nanargmin(mean_spectrum)
        return float(dataset.wavelength[wl_mask][minimum_index])

    @staticmethod
    def _extract_gsb_trace(dataset, center_wavelength, half_width):
        wl_mask = (
            np.abs(dataset.wavelength - center_wavelength)
            <= half_width
        )

        if not np.any(wl_mask):
            nearest_index = np.nanargmin(
                np.abs(dataset.wavelength - center_wavelength)
            )
            wl_mask = np.zeros(len(dataset.wavelength), dtype=bool)
            wl_mask[nearest_index] = True

        return (
            dataset.delay_time.copy(),
            np.nanmean(dataset.signal[wl_mask, :], axis=0),
        )

    @staticmethod
    def _multi_exponential(time, *parameters):
        """
        parameters = [offset, A1, A2, ..., tau1, tau2, ...]
        """
        number_of_components = (len(parameters) - 1) // 2

        offset = parameters[0]
        amplitudes = parameters[1:1 + number_of_components]
        lifetimes = parameters[1 + number_of_components:]

        signal = np.full_like(time, offset, dtype=float)

        for amplitude, lifetime in zip(amplitudes, lifetimes):
            signal += amplitude * np.exp(-time / lifetime)

        return signal

    def _fit_trace(self, time, signal, number_of_components, fit_range):
        fit_min, fit_max = fit_range

        valid = (
            np.isfinite(time)
            & np.isfinite(signal)
            & (time >= fit_min)
            & (time <= fit_max)
        )

        fit_time = time[valid]
        fit_signal = signal[valid]

        required_points = 2 * number_of_components + 2

        if len(fit_time) < required_points:
            raise ValueError(
                f"Not enough points for a {number_of_components}-component fit."
            )

        offset_guess = np.nanmedian(fit_signal[-max(3, len(fit_signal) // 10):])
        amplitude_guess = max(
            (fit_signal[0] - offset_guess) / number_of_components,
            1e-12,
        )

        tau_guess = np.geomspace(
            max(fit_time.min() * 2, 0.01),
            fit_time.max() / 2,
            number_of_components,
        )

        initial_guess = (
            [offset_guess]
            + [amplitude_guess] * number_of_components
            + list(tau_guess)
        )

        lower_bounds = (
            [-np.inf]                       # offset
            + [0.0] * number_of_components  # positive decay amplitudes
            + [1e-6] * number_of_components # positive lifetimes
        )
        upper_bounds = (
            [np.inf] * (number_of_components + 1)
            + [fit_time.max() * 1e4] * number_of_components
        )

        parameters, covariance = curve_fit(
            self._multi_exponential,
            fit_time,
            fit_signal,
            p0=initial_guess,
            bounds=(lower_bounds, upper_bounds),
            maxfev=100000,
        )

        parameter_error = np.sqrt(np.diag(covariance))

        lifetimes = parameters[1 + number_of_components:]
        lifetime_errors = parameter_error[1 + number_of_components:]

        # Sort from fast to slow component.
        sort_index = np.argsort(lifetimes)

        return {
            "parameters": parameters,
            "parameter_errors": parameter_error,
            "lifetimes": lifetimes[sort_index],
            "lifetime_errors": lifetime_errors[sort_index],
            "fit_time": fit_time,
            "fit_signal": fit_signal,
        }
    @staticmethod
    def _resolve_component_counts(number_of_components, nfiles):
        """
        Accept one integer for all files, or a list/tuple with one integer per file.
        """
        if np.isscalar(number_of_components):
            component_counts = [int(number_of_components)] * nfiles
        else:
            component_counts = list(number_of_components)

            if len(component_counts) != nfiles:
                raise ValueError(
                    "number_of_components must contain one value for each "
                    f"loaded file: expected {nfiles}, got {len(component_counts)}."
                )

            component_counts = [int(value) for value in component_counts]

        if any(value < 1 for value in component_counts):
            raise ValueError("Each component count must be at least 1.")

        return component_counts
    @staticmethod
    def _resolve_fit_ranges(fit_range, nfiles):
        """
        Accept one range for all files, or one (start, end) range per file.
        """
        # One range for every file: (1, 1000)
        if (
            isinstance(fit_range, tuple)
            and len(fit_range) == 2
            and np.isscalar(fit_range[0])
            and np.isscalar(fit_range[1])
        ):
            fit_ranges = [fit_range] * nfiles

        # Individual ranges: [(1, 1000), (0.1, 500)]
        else:
            fit_ranges = [tuple(value) for value in fit_range]

            if len(fit_ranges) != nfiles:
                raise ValueError(
                    "fit_range must contain one range for each loaded file: "
                    f"expected {nfiles}, got {len(fit_ranges)}."
                )

        for fit_min, fit_max in fit_ranges:
            if fit_min <= 0 or fit_max <= fit_min:
                raise ValueError(
                    "Each fit range must be (positive_start, end), "
                    "for example (0.05, 1000)."
                )

        return fit_ranges
    def plot_matrix(
        self,
        file_paths,
        labels=None,
        gsb_wavelength=None,
        gsb_search_wavelength_range=(450, 600),
        gsb_search_time_range=(0.0, 1000),
        wavelength_half_width=1.0,
        number_of_components=2,
        fit_range=(0.05, 1000),
        ncols=3,
        figsize=(15, 4),
        ylim=None,
        output_path=None,
        show=True,
        nrows=None,
    ):
        datasets = self.load_files(file_paths, labels=labels)

        nfiles = len(datasets)
        component_counts = self._resolve_component_counts(
            number_of_components,
            nfiles,
        )
        fit_ranges = self._resolve_fit_ranges(
            fit_range,
            nfiles,
        )
        nrows = nrows or int(np.ceil(nfiles / ncols))

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            squeeze=False,
        )

        self.fit_results = []

        for index, dataset in enumerate(datasets):
            row, col = divmod(index, ncols)
            ax = axes[row, col]
            components_for_this_file = component_counts[index]
            fit_range_for_this_file = fit_ranges[index]
            if gsb_wavelength is None:
                selected_wavelength = self._find_gsb_wavelength(
                    dataset,
                    gsb_search_wavelength_range,
                    gsb_search_time_range,
                )
            else:
                selected_wavelength = gsb_wavelength

            time, signal = self._extract_gsb_trace(
                dataset,
                selected_wavelength,
                wavelength_half_width,
            )

            positive_time = time > 0
            time = time[positive_time]
            signal = signal[positive_time]
            
            # Convert negative GSB ΔA into a positive bleach amplitude that decays to zero.
            late_time_mask = time >= 0.8 * fit_range_for_this_file[1]
            
            if np.any(late_time_mask):
                baseline = np.nanmedian(signal[late_time_mask])
            else:
                baseline = np.nanmedian(signal[-max(3, len(signal) // 10):])
            
            signal = baseline - signal
            
            # Ensure the early-time GSB amplitude is positive and decays downward.
            early_points = max(3, len(signal) // 10)
            
            if np.nanmedian(signal[:early_points]) < 0:
                signal = -signal

            result = self._fit_trace(
                time,
                signal,
                components_for_this_file,
                fit_range_for_this_file,
            )
            
            result["number_of_components"] = components_for_this_file

            smooth_time = np.geomspace(
                fit_range_for_this_file[0],
                fit_range_for_this_file[1],
                1000,
            )
            fitted_signal = self._multi_exponential(
                smooth_time,
                *result["parameters"],
            )

            ax.scatter(
                time,
                signal,
                s=14,
                color="tab:blue",
                alpha=0.7,
                label="Ground State Bleaching (GSB)",
            )
            ax.plot(
                smooth_time,
                fitted_signal,
                color="black",
                linewidth=2.2,
                label="Multi-exponential fit",
            )

            lifetime_text = "\n".join(
                [
                    rf"$\tau_{i + 1}$ = {tau:.2g} ± {err:.1g} ps"
                    for i, (tau, err) in enumerate(
                        zip(
                            result["lifetimes"],
                            result["lifetime_errors"],
                        )
                    )
                ]
            )

            ax.text(
                0.04,
                0.05,
                lifetime_text,
                transform=ax.transAxes,
                fontsize=self.config.title_fontsize - 2,
                va="bottom",
                bbox=dict(
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="none",
                ),
            )

            ax.axhline(0, color="gray", linestyle="--", linewidth=1)
            ax.set_xlim(0, fit_range_for_this_file[1])

            if ylim is not None:
                ax.set_ylim(*ylim)

            ax.grid(alpha=0.25, linestyle="--")
            ax.tick_params(
                axis="both",
                labelsize=self.config.tick_labelsize,
            )
            ax.set_title(
                f"{dataset.label}\nGSB: {selected_wavelength:.1f} nm",
                fontsize=self.config.title_fontsize,
                fontweight="bold",
            )
            ax.legend(fontsize=14, frameon=False)

            if index + ncols >= nfiles:
                ax.set_xlabel(
                    "Delay time (ps)",
                    fontsize=self.config.label_fontsize,
                    fontweight="bold",
                )

            if col == 0:
                ax.set_ylabel(
                    r"GSB $\Delta A$ (mOD)",
                    fontsize=self.config.label_fontsize,
                    fontweight="bold",
                )

            result["label"] = dataset.label
            result["gsb_wavelength"] = selected_wavelength
            self.fit_results.append(result)

        for empty_index in range(nfiles, nrows * ncols):
            row, col = divmod(empty_index, ncols)
            axes[row, col].axis("off")

        fig.tight_layout()

        if output_path is not None:
            fig.savefig(
                output_path,
                dpi=self.config.dpi,
                transparent=self.config.transparent,
                bbox_inches="tight",
            )

        if show:
            plt.show()

        return fig, axes

    def print_average_lifetimes(self):
        """
        Print amplitude-weighted average lifetimes after plot_matrix() fitting.
        """
        if not self.fit_results:
            raise RuntimeError(
                "Run plot_matrix() first so lifetime fitting is performed."
            )

        print("Amplitude-weighted average lifetimes:")

        for result in self.fit_results:
            parameters = result["parameters"]
            number_of_components = (len(parameters) - 1) // 2

            amplitudes = parameters[1:1 + number_of_components]
            lifetimes = parameters[1 + number_of_components:]

            average_lifetime = (
                np.sum(amplitudes * lifetimes)
                / np.sum(amplitudes)
            )

            component_text = ", ".join(
                [
                    f"τ{i + 1} = {tau:.3g} ps"
                    for i, tau in enumerate(lifetimes)
                ]
            )

            print(
                f"{result['label']} "
                f"(GSB {result['gsb_wavelength']:.1f} nm): "
                f"<τ> = {average_lifetime:.3g} ps "
                f"[{component_text}]"
            )

"""Portable TAS projects and rendering; no executable objects in saved files."""
import ast
from io import BytesIO
import copy
import json
import os
from pathlib import Path
import tempfile
import zipfile
import hashlib
import uuid
from dataclasses import asdict
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.artist import ArtistInspector
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Bbox
from matplotlib import font_manager
from matplotlib.lines import Line2D
from mytas import TAPlotConfig, TAPlotter, GSBLifetimeFitter, TADataset

KINDS = ['TA heatmap', 'Spectral profiles', 'GSB lifetimes']
_FONT_CACHE = None

def literal(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def default_state():
    return dict(config=asdict(TAPlotConfig()), options=dict(
        ncols=3, nrows=0, figsize=(12, 5), share_colorbar=False,
        snapshot_colors=[], lifetime_data_color='tab:blue', lifetime_fit_color='black',
        horizontal_spacing=0.5, vertical_spacing=0.5,
        delays=[0, 1, 2, 5, 10], gsb_wavelength=None,
        gsb_search_wavelength_range=(450, 600),
        gsb_search_time_range=(1, 1000), wavelength_half_width=1.0,
        number_of_components=2, fit_range=(1, 1000), ylim=None,
        combined=False, log_x=False),
        style=dict(font_family='DejaVu Sans', font_size=12,
                   font_weight='normal', title_size=16, label_size=14,
                   tick_size=11, legend_size=11), rc={}, edits={}, views=[], annotations=[])


def default_manual():
    """A portable, spreadsheet-like data sheet for manually entered plots."""
    return dict(columns=['X', 'Y1'], rows=[], settings=dict(
        x_column=0, y_columns=[1], kind='Line', title='', xlabel='', ylabel='',
        colors=[], grid=True, legend=True))


def packed(value):
    # Preserve tuples because the original plotter distinguishes tuples and lists.
    if isinstance(value, tuple):
        return {'__tuple__': [packed(x) for x in value]}
    if isinstance(value, list):
        return [packed(x) for x in value]
    if isinstance(value, dict):
        return {str(k): packed(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return packed(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def unpacked(value):
    if isinstance(value, dict):
        if set(value) == {'__tuple__'}:
            return tuple(unpacked(x) for x in value['__tuple__'])
        return {k: unpacked(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unpacked(x) for x in value]
    return value


class Project:
    def __init__(self):
        self.sources = []
        self.fonts = []
        self.states = [default_state() for _ in KINDS]
        self.manual = default_manual()
        self._prepared_cache = {}

    def add(self, path):
        path = Path(path)
        self.sources.append(dict(id=uuid.uuid4().hex, name=path.name, label=path.stem, data=path.read_bytes()))

    @staticmethod
    def source_matrix(source, max_rows=513):
        """Read the same valid matrix region as the original CSV reader.

        Instrument exports may contain notes or auxiliary rows after the 513-row
        signal matrix.  `max_rows` keeps those rows from corrupting reloads.
        """
        matrix = np.genfromtxt(BytesIO(source['data']), delimiter=",", max_rows=max_rows)
        if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
            raise ValueError("The embedded data is not a valid TAS matrix.")
        return matrix

    def update_source_matrix(self, source, matrix):
        stream = BytesIO()
        np.savetxt(stream, np.asarray(matrix, dtype=float), delimiter=",", fmt="%.12g")
        source['data'] = stream.getvalue()
        self.clear_cache(source)

    def clear_cache(self, source=None):
        if source is None:
            self._prepared_cache.clear()
            return
        source_id = source.get('id')
        self._prepared_cache = {key: value for key, value in self._prepared_cache.items()
                                if key[0] != source_id}

    def register_font(self, name, data):
        # Qt/FreeType may retain handles: keep fonts available for the whole session.
        global _FONT_CACHE
        if _FONT_CACHE is None:
            _FONT_CACHE = tempfile.TemporaryDirectory(prefix='tas_fonts_', ignore_cleanup_errors=True)
        path = Path(_FONT_CACHE.name) / (hashlib.sha256(data).hexdigest() + Path(name).suffix)
        if not path.exists():
            path.write_bytes(data)
        font_manager.fontManager.addfont(str(path))
        family = font_manager.FontProperties(fname=str(path)).get_name()
        self.fonts.append(dict(name=name, family=family, data=data, path=str(path)))
        return self.fonts[-1]

    def save(self, path):
        path = Path(path)
        manifest = dict(format='TAS Studio', version=3, states=self.states, manual=self.manual,
                        sources=[{k:v for k,v in s.items() if k != 'data'} for s in self.sources],
                        fonts=[dict(name=f['name']) for f in self.fonts])
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
        os.close(fd)
        try:
            with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as z:
                z.writestr('project.json', json.dumps(packed(manifest), allow_nan=False))
                for i, s in enumerate(self.sources):
                    z.writestr(f'data/{i}.csv', s['data'])
                for i, f in enumerate(self.fonts):
                    z.writestr(f'fonts/{i}', f['data'])
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    @classmethod
    def load(cls, path):
        obj = cls()
        with zipfile.ZipFile(path) as z:
            m = unpacked(json.loads(z.read('project.json')))
            if m.get('format') != 'TAS Studio' or m.get('version') not in (1, 2, 3):
                raise ValueError('Unsupported TAS project format or version.')
            if len(m['states']) != 3:
                raise ValueError('Expected three plot states.')
            obj.states = m['states']
            for state in obj.states:
                for name, value in default_state()['options'].items():
                    state['options'].setdefault(name, copy.deepcopy(value))
            manual = m.get('manual', {})
            obj.manual = default_manual()
            if isinstance(manual, dict):
                obj.manual.update({k: v for k, v in manual.items() if k in obj.manual})
                if isinstance(manual.get('settings'), dict):
                    obj.manual['settings'].update(manual['settings'])
            if not isinstance(obj.manual['columns'], list) or not obj.manual['columns']:
                obj.manual['columns'] = ['X', 'Y1']
            obj.manual['columns'] = [str(value) for value in obj.manual['columns']]
            if not isinstance(obj.manual['rows'], list):
                obj.manual['rows'] = []
            obj.manual['rows'] = [
                [str(value) if value is not None else '' for value in row]
                for row in obj.manual['rows'] if isinstance(row, list)
            ]
            obj.sources = [dict(s, id=s.get('id', uuid.uuid4().hex), data=z.read(f'data/{i}.csv'))
                           for i, s in enumerate(m['sources'])]
            for i, font in enumerate(m.get('fonts', [])):
                obj.register_font(font['name'], z.read(f'fonts/{i}'))
        return obj


class SourcePlotter(TAPlotter):
    """Apply optional per-file preprocessing using the original dataset reader."""
    def __init__(self, config, sources, project):
        super().__init__(config)
        self.sources = sources
        self.project = project

    def load_files(self, file_paths, labels=None):
        self.datasets = []
        for path, source, label in zip(file_paths, self.sources, labels):
            values = asdict(self.config)
            values.update(source.get('processing', {}))
            cfg = TAPlotConfig(**values)
            key = (source['id'], json.dumps(packed(asdict(cfg)), sort_keys=True))
            cached = self.project._prepared_cache.get(key)
            if cached is None:
                matrix = self.project.source_matrix(source, cfg.max_rows)
                dataset = TADataset.from_matrix(matrix, label=label, max_rows=cfg.max_rows)
                dataset.crop_valid_region()
                if cfg.background_correction:
                    dataset.subtract_background(cfg.background_time_range, cfg.background_wavelength_range)
                if cfg.chirp_correction:
                    dataset.chirp_correct(cfg.chirp_points, cfg.chirp_poly_coefficients, cfg.chirp_poly_degree,
                                          cfg.chirp_reference_wavelength, cfg.chirp_reference_time)
                dataset.smooth(cfg.smooth_sigma)
                if cfg.scale_to_milli:
                    dataset.signal *= 1000
                cached = dataset
                self.project._prepared_cache[key] = cached
            dataset = copy.copy(cached)
            dataset.delay_time = cached.delay_time.copy()
            dataset.wavelength = cached.wavelength.copy()
            dataset.signal = cached.signal.copy()
            dataset.label = label
            self.datasets.append(dataset)
        if self.config.normalize_global:
            self._normalize_datasets_global(self.datasets)
        return self.datasets


def artists(fig):
    # Semantic addresses stay stable when tick counts change during zooming.
    out = {'figure': fig, 'figure/background': fig.patch}
    for j, text in enumerate(fig.texts):
        out[f'figure/text/{j}'] = text
    for i, ax in enumerate(fig.axes):
        p = f'axes/{i}'
        out[p] = ax
        for name in ['title', 'xaxis', 'yaxis', 'patch']:
            out[f'{p}/{name}'] = getattr(ax, name)
        out[f'{p}/xlabel'] = ax.xaxis.label
        out[f'{p}/ylabel'] = ax.yaxis.label
        for direction, axis in [('x', ax.xaxis), ('y', ax.yaxis)]:
            out[f'{p}/{direction}offset'] = axis.get_offset_text()
            for minor in (False, True):
                ticks = axis.get_minor_ticks() if minor else axis.get_major_ticks()
                for tick in ticks:
                    for side in ('label1', 'label2'):
                        key = f'{p}/tick/{direction}/{int(minor)}/{float(tick.get_loc()):.15g}/{side}'
                        out[key] = getattr(tick, side)
        for group in ['lines', 'collections', 'texts', 'images']:
            for j, a in enumerate(getattr(ax, group)):
                key = a.get_gid()
                out[key if key and key.startswith('annotation/') else f'{p}/{group}/{j}'] = a
        for name, spine in ax.spines.items():
            out[f'{p}/spines/{name}'] = spine
        leg = ax.get_legend()
        if leg:
            out[f'{p}/legend'] = leg
            out[f'{p}/legend/title'] = leg.get_title()
            for j, t in enumerate(leg.get_texts()):
                out[f'{p}/legend/text/{j}'] = t
    return out


def apply_artist_edit(fig, key, props):
    """Tick text must use a formatter or Matplotlib overwrites it on redraw."""
    obj = artists(fig).get(key)
    if '/tick/' in key and 'text' in props:
        parts = key.split('/')
        axis = getattr(fig.axes[int(parts[1])], parts[3] + 'axis')
        minor, location = bool(int(parts[4])), float(parts[5])
        formatter = axis.get_minor_formatter() if minor else axis.get_major_formatter()
        if not hasattr(formatter, '_tas_labels'):
            original = formatter
            replacements = {}
            def label(value, pos):
                return replacements.get(f'{float(value):.15g}', original(value, pos))
            formatter = FuncFormatter(label)
            formatter.get_offset = original.get_offset
            formatter._tas_labels = replacements
            (axis.set_minor_formatter if minor else axis.set_major_formatter)(formatter)
        formatter._tas_labels[f'{location:.15g}'] = props['text']
    if key.endswith(('/xoffset', '/yoffset')) and 'text' in props:
        parts = key.split('/')
        axis = getattr(fig.axes[int(parts[1])], parts[2][0] + 'axis')
        axis.get_major_formatter().get_offset = lambda: props['text']
    if obj is not None:
        obj.set(**props)
    elif '/tick/' not in key:
        raise ValueError(f'Saved artist no longer exists: {key}. Reset figure edits to regenerate.')


def fit_figure_labels(fig, padding_points=7):
    """Reserve measured outer margins, including manually positioned colorbars.

    Move all axes together so broken axes and their colorbars stay aligned.
    Unlike bbox_inches='tight', this preserves the requested export dimensions.
    Only adjust when content overflows; saved manual layouts otherwise stay intact.
    """
    for _ in range(12):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bounds = [ax.get_tightbbox(renderer) for ax in fig.axes if ax.get_visible()]
        bounds = [b for b in bounds if b is not None and np.isfinite(b.extents).all()]
        if not bounds:
            return
        box = Bbox.union(bounds).transformed(fig.transFigure.inverted())
        width, height = fig.get_size_inches()
        px, py = padding_points / (72 * width), padding_points / (72 * height)
        left, right = min(box.x0, px), max(box.x1, 1-px)
        bottom, top = min(box.y0, py), max(box.y1, 1-py)
        if (left >= px-1e-4 and right <= 1-px+1e-4
                and bottom >= py-1e-4 and top <= 1-py+1e-4):
            return
        sx, sy = (1-2*px)/(right-left), (1-2*py)/(top-bottom)
        for ax in fig.axes:
            pos = ax.get_position()
            ax.set_position([px+(pos.x0-left)*sx, py+(pos.y0-bottom)*sy,
                             pos.width*sx, pos.height*sy])
    fig.canvas.draw()


def render(project, index):
    if not project.sources:
        raise ValueError('Add at least one TAS CSV file first.')
    state = project.states[index]
    config = TAPlotConfig(**state['config'])
    o = state['options']
    spacing = {name: o.get(name, 0.5) for name in ('horizontal_spacing', 'vertical_spacing')}
    if any(not isinstance(v, (int, float)) or not np.isfinite(v) or not 0 <= v <= 3
           for v in spacing.values()):
        raise ValueError('Plot spacing must be between 0 and 3.')
    if not isinstance(o['ncols'], int) or o['ncols'] < 1:
        raise ValueError('ncols must be a positive integer.')
    rows = o.get('nrows', 0)
    if not isinstance(rows, int) or rows < 0:
        raise ValueError('Matrix rows must be zero (automatic) or a positive integer.')
    if rows and rows * o['ncols'] < len(project.sources) and not (index == 2 and o['combined']):
        raise ValueError('The matrix is too small for the loaded files. Increase rows or columns, or set rows to Auto.')
    if len(o['figsize']) != 2 or min(o['figsize']) <= 0:
        raise ValueError('figsize must contain two positive values in inches.')
    before = set(plt.get_fignums())
    try:
        with mpl.rc_context(state['rc']):
            paths = [f"<project:{source['id']}>" for source in project.sources]
            common = dict(labels=[s['label'] for s in project.sources],
                          ncols=o['ncols'], nrows=rows or None, figsize=o['figsize'], show=False)
            plotter = SourcePlotter(config, project.sources, project)
            fitter = None
            if index == 0:
                fig, groups = plotter.plot_matrix(paths, share_colorbar=o['share_colorbar'], **spacing, **common)
                for i, (source, group) in enumerate(zip(project.sources, groups)):
                    style = source.get('plot_style', {})
                    if o['share_colorbar'] and any(style.get(k) is not None for k in ('cmap', 'vmin', 'vmax')):
                        raise ValueError('Turn off Shared colorbar to use per-file heatmap colors or limits.')
                    for ax in group:
                        for mesh in ax.collections:
                            if style.get('cmap'):
                                mesh.set_cmap(style['cmap'])
                            mesh.set_clim(style.get('vmin'), style.get('vmax'))
            elif index == 1:
                fig, _ = plotter.plot_line_profile_matrix(paths, delays=o['delays'], ylim=o['ylim'],
                    delays_by_file=[s.get('plot_style', {}).get('delays') for s in project.sources], **common)
                for source, ax in zip(project.sources, fig.axes):
                    colors = source.get('plot_style', {}).get('snapshot_colors') or o.get('snapshot_colors', [])
                    if colors:
                        if isinstance(colors, str):
                            colors = [colors]
                        for i, line in enumerate(ax.collections):
                            line.set_array(None)
                            line.set_color(colors[i % len(colors)])
                        delays = np.atleast_1d(source.get('plot_style', {}).get('delays') or o['delays'])
                        ax.legend([Line2D([], [], color=colors[i % len(colors)], lw=2) for i in range(len(delays))],
                                  [f'{delay:g} ps' for delay in delays])
            else:
                fitter = GSBLifetimeFitter(config)
                fitter.load_files = plotter.load_files
                kwargs = {k: o[k] for k in ['gsb_wavelength', 'gsb_search_wavelength_range',
                    'gsb_search_time_range', 'wavelength_half_width', 'number_of_components', 'fit_range', 'ylim']}
                counts = fitter._resolve_component_counts(kwargs['number_of_components'], len(paths))
                ranges = fitter._resolve_fit_ranges(kwargs['fit_range'], len(paths))
                kwargs['number_of_components'] = [s.get('plot_style', {}).get('components') or n for s,n in zip(project.sources, counts)]
                kwargs['fit_range'] = [s.get('plot_style', {}).get('fit_range') or r for s,r in zip(project.sources, ranges)]
                if o['combined']:
                    common['nrows'] = None
                fig, _ = fitter.plot_matrix(paths, **common, **kwargs)
                for source, ax in zip(project.sources, fig.axes):
                    style = source.get('plot_style', {})
                    ax.collections[0].set_color(style.get('lifetime_data_color') or o.get('lifetime_data_color', 'tab:blue'))
                    ax.lines[0].set_color(style.get('lifetime_fit_color') or o.get('lifetime_fit_color', 'black'))
                if o['combined']:
                    plt.close(fig)
                    fig, ax = plt.subplots(figsize=o['figsize'])
                    for source, r in zip(project.sources, fitter.fit_results):
                        params = r['parameters'].copy()
                        y = r['fit_signal'] - params[0]
                        scale = np.max(y)
                        if not np.isfinite(scale) or scale <= 0:
                            raise ValueError('Cannot normalize a GSB trace with no positive peak.')
                        params[0] = 0
                        n = r['number_of_components']
                        params[1:1+n] /= scale
                        t = r['fit_time']
                        style = source.get('plot_style', {})
                        line, = ax.plot(t, y / scale, '.', label=r['label'],
                                        color=style.get('lifetime_data_color') or o.get('lifetime_data_color', 'tab:blue'))
                        tt = np.geomspace(t.min(), t.max(), 1000)
                        ax.plot(tt, fitter._multi_exponential(tt, *params),
                                color=style.get('lifetime_fit_color') or o.get('lifetime_fit_color', 'black'))
                    ax.set(xlabel='Delay time (ps)', ylabel='Normalized GSB amplitude', ylim=(-.05, 1.08))
                    ax.legend()
                    fig.tight_layout()
                if o['log_x']:
                    for ax in fig.axes:
                        ax.set_xscale('log')
                        positive = min(r['fit_time'].min() for r in fitter.fit_results)
                        ax.set_xlim(left=positive)
            sty = state['style']
            for t in fig.findobj(match=Text):
                t.set(fontfamily=sty['font_family'], fontsize=sty['font_size'], fontweight=sty['font_weight'])
            for ax in fig.axes:
                ax.title.set_fontsize(sty['title_size'])
                ax.xaxis.label.set_fontsize(sty['label_size'])
                ax.yaxis.label.set_fontsize(sty['label_size'])
                ax.tick_params(labelsize=sty['tick_size'])
                if ax.get_legend():
                    for t in ax.get_legend().get_texts():
                        t.set_fontsize(sty['legend_size'])
            if index != 0:
                fig.tight_layout()
                # Keep room for labels when embedded in a narrower desktop canvas.
                fig.subplots_adjust(left=max(fig.subplotpars.left, .12), bottom=max(fig.subplotpars.bottom, .14),
                                    wspace=spacing['horizontal_spacing'], hspace=spacing['vertical_spacing'])
            for ax, view in zip(fig.axes, state['views']):
                ax.set(xscale=view['xscale'], yscale=view['yscale'], xlim=view['xlim'], ylim=view['ylim'])
                ax.set_position(view['position'])
            for note in state.get('annotations', []):
                ax = fig.axes[note['axes']]
                ax.text(*note['position'], note['text'], transform=ax.transAxes,
                        gid=note['id'], **note['style'])
            # Resolve tick positions before replaying edits addressed by tick value.
            fig.canvas.draw()
            for key, props in state['edits'].items():
                apply_artist_edit(fig, key, props)
            fit_figure_labels(fig)
            plt.close(fig)  # Drop pyplot's hidden manager; the Qt canvas owns the figure.
            return fig, fitter.fit_results if fitter else []
    except Exception:
        for number in set(plt.get_fignums()) - before:
            plt.close(number)
        raise


def capture_views(fig):
    return [dict(xlim=ax.get_xlim(), ylim=ax.get_ylim(), xscale=ax.get_xscale(),
                 yscale=ax.get_yscale(), position=tuple(ax.get_position().bounds)) for ax in fig.axes]

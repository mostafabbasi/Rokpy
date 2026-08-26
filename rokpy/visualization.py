"""
Well-log visualization and rock optimization tools
This module provides classes and functions for visualizing well-log data in
a structured track format, as well as tools for optimizing rock physics models.
"""
from typing import Literal
from matplotlib import pyplot as plt
from rokpy.constants import PropertyTemplate, PropertyTemplates
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import Divider, Size
import numpy as np
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import Slider
from matplotlib.widgets import MultiCursor
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class Track(Axes):
    """
    A composite layout representing a single well-log track.

    Internally divided vertically into title, header, and plot axes.
    Inherits from `matplotlib.axes.Axes`.

    Parameters
    ----------
    sheet : Sheet
        The parent sheet figure that contains this track.
    template : PropertyTemplate or str
        The property template defining the track type and plot range,
        or a string specifying only the type.
    sharey : matplotlib.axes.Axes, optional
        Axes to share the y-axis with (typically the depth axis).

    Attributes
    ----------
    type : str
        The type of log or property shown in this track.
    plot_xlim : tuple or None
        The x-axis limits for the plot area.
    sheet : Sheet
        Reference to the parent sheet.
    plot_list : list
        List of plotted line objects.
    title_ax, header_ax, plot_ax : matplotlib.axes.Axes
        Internal axes for title, legend/header, and data plotting.
    """

    def __init__(self, sheet, template: PropertyTemplate, sharey=None):
        if isinstance(template, PropertyTemplate):
            self.type = template.type
            self.plot_xlim = template.plot_range
        elif isinstance(template, str):
            self.type = template
            self.plot_xlim = None
        self.sheet = sheet
        self.rect = [0, 0, 1, 1]
        self.plot_list = []
        self.divider = self.setup_divider()
        self.setup_title_ax(sheet)
        self.setup_header_ax(sheet)
        self.setup_plot_ax(sheet, sharey)
        self.title_ax.set_axes_locator(self.divider.new_locator(nx=0, ny=2))
        self.header_ax.set_axes_locator(self.divider.new_locator(nx=0, ny=1))
        self.plot_ax.set_axes_locator(self.divider.new_locator(nx=0, ny=0))
        self.plot_ax.curser_text = self.plot_ax.text(0.02, 0.95, '', ha='left', va='bottom', fontsize=10)
        self.plot_ax.invert_yaxis()

 
    @property
    def plot_xlim(self):
        return self._plot_xlim
    @plot_xlim.setter
    def plot_xlim(self, value):
        self._plot_xlim = value

    def clear_track(self):
        """Clear all artists from the plot and header axes."""
        self.clear_axes(self.plot_ax)
        self.clear_axes(self.header_ax)

    def clear_axes(self, ax):
        """
        Remove all artists (lines, patches, texts, collections) from a given axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to clear.
        """
        artists = ax.get_lines() + ax.patches + ax.texts + ax.collections
        for line in artists:
            line.remove()

    def attach_to_sheet(self, sheet, nx, ny):
        """
        Attach this track to a specific grid cell in the sheet layout.

        Parameters
        ----------
        sheet : Sheet
            The parent sheet.
        nx : int
            Column index in the sheet layout.
        ny : int
            Row index in the sheet layout.
        """
        locator = sheet.divider.new_locator(nx=nx, ny=ny)
        bbox = locator(sheet, None)             # returns a Bbox
        self.divider.set_position(bbox.bounds)  # convert to tuple

    def add_marker(self, marker_name, depth, color='#ff0000'):
        """
        Add a horizontal marker line at a specified depth.

        Parameters
        ----------
        marker_name : str
            Label for the marker.
        depth : float
            Depth (y-coordinate) at which to place the marker.
        color : str, optional
            Color of the marker line and text (default: '#ff0000').
        """
        self.plot_ax.hlines(depth, self.plot_ax.get_xlim()[0], self.plot_ax.get_xlim()[1], linewidth=2, color=color)
        self.plot_ax.text(self.plot_xlim[0], depth, marker_name, ha="left", va="bottom", fontsize=9, color=color)

    def plot(self, depth, log_data, linewidth=1.5, color='#000000', handle=None, **kwargs):
        """
        Plot a well log on the track's plot axis.

        Parameters
        ----------
        depth : array-like
            Depth values (y-axis).
        log_data : array-like
            Log values (x-axis).
        linewidth : float, optional
            Line width (default: 0.7).
        color : str, optional
            Line color (default: '#000000').
        handle : object, optional
            Reserved for future use.
        **kwargs
            Additional keyword arguments passed to `matplotlib.axes.Axes.plot`.

        Returns
        -------
        matplotlib.lines.Line2D
            The plotted line object.
        """
        log_plot, = self.plot_ax.plot(log_data, depth, linewidth=linewidth, color=color, **kwargs)
        try:
            log_plot.set_label(kwargs["label"])
        except:
            log_plot.set_label(self.type)
        self.plot_list.append(log_plot)
        self.set_track_range()
        self.update_log_legend()
        return log_plot

    def update_log_legend(self):
        """Update the header axis to display a legend based on current plots."""
        for line in self.header_ax.lines:
            line.remove()
        attrs = [
            "color", "linestyle", "linewidth",
            "marker", "markersize",
            "markerfacecolor", "markeredgecolor"
        ]
        for plot_idx, log_plot in enumerate(self.plot_list):
            legend_plot = self.header_ax.plot(np.array(self.plot_xlim), plot_idx + np.array([1, 1]))
            self.header_ax.text(self.plot_xlim[0], plot_idx + 1, "{:.2f}".format(self.plot_xlim[0]), ha="left", va="top", fontsize=9, color=log_plot._color)
            self.header_ax.text(self.plot_xlim[1], plot_idx + 1, "{:.2f}".format(self.plot_xlim[1]), ha="right", va="top", fontsize=9, color=log_plot._color)
            self.header_ax.text((self.plot_xlim[0] + self.plot_xlim[1]) / 2, plot_idx + 1, log_plot.get_label(), ha="center", va="bottom", fontsize=9, color=log_plot._color)
            for attr in attrs:
                getattr(legend_plot[0], f"set_{attr}")(getattr(log_plot, f"get_{attr}")())

    def plot_fraction_set(self, depth, fraction_set):
        """
        Plot stacked fractional data (e.g., lithology fractions).

        Parameters
        ----------
        depth : array-like
            Depth values.
        fraction_set : array-like, shape (N, M)
            Fraction data where columns represent cumulative fractions.
        """
        stacked_fractions = np.cumsum(list(fraction_set.values()), axis=0).T
        n_fract = np.size(stacked_fractions, 1)
        for idx, component in enumerate(fraction_set.keys()):
            if idx == 0:
                base = 0
            else:
                base = stacked_fractions[:, idx - 1]
            self.plot_ax.fill_betweenx(depth, base, stacked_fractions[:, idx], color=component.color.decimal)
            self.header_ax.text(np.mean(self.plot_xlim), 4.7-0.6*idx, component.properties.type,
                                         fontsize=8, ha="center", va="top", fontfamily='monospace',
                                         bbox=dict(facecolor=component.color.decimal, alpha=1, linewidth=0, pad=3))

    def plot_component_set(self, depth, component_set):
        """
        Plot component fractions with colors from rock component definitions.

        Parameters
        ----------
        depth : array-like
            Depth values.
        component_set : object
            Object with attributes `stacked_fractions` and `components`.
        """
        stacked_fractions = component_set.stacked_fractions.T.copy()
        self.plot_fraction_set(depth, component_set.fraction_set)
        # n_fract = np.size(stacked_fractions, 1)
        # for idx, component in enumerate(component_set.components):
        #     if idx == 0:
        #         base = 0
        #     else:
        #         base = stacked_fractions[:, idx - 1]
        #     self.plot_ax.fill_betweenx(depth, base, stacked_fractions[:, idx], color=component.color.decimal)
        #     legend = self.header_ax.text(idx / n_fract + (5 / self.sheet.dpi), 4.5, component.properties.type,
        #                                  fontsize=8, ha="left", va="top", fontfamily='monospace',
        #                                  bbox=dict(facecolor=component.color.decimal, alpha=1, linewidth=0, pad=5))

    def wiggle(self, seis_data, depth, x=None, scale=1, color='black', fill_color='black', linewidth=0.5):
        """
        Plot seismic wiggle traces.

        Parameters
        ----------
        seis_data : array-like, shape (Nz, Nx)
            Seismic data matrix.
        depth : array-like
            Depth or time values (z-axis).
        x : array-like, optional
            Horizontal positions (default: 1-based index).
        scale : float, optional
            Scaling factor for trace amplitude (default: 1).
        color : str, optional
            Line color (default: 'black').
        fill_color : str, optional
            Fill color for positive amplitudes (default: 'black').
        linewidth : float, optional
            Line width (default: 0.5).
        """
        self.plot_ax.autoscale(enable=True, axis='x')
        SeisPlot.wiggle(self.plot_ax, seis_data, z=depth, x=x, scale=scale, color=color, fill_color=fill_color, linewidth=linewidth)
        self.plot_ax.tick_params(axis="x", direction="out", which="both", 
                                 color='black', 
                                 top=True, 
                                 labeltop=True, 
                                 labelbottom=False, 
                                 rotation=90, 
                                 bottom=False,
                                 labelsize=8)
        self.plot_ax.set_xticks(ticks=x)
        self.plot_ax.grid(axis='x', visible=False)

    def set_track_range(self):
        """Set and synchronize x-axis limits for plot and header axes."""
        try:
            self.plot_ax.set_xlim(self.plot_xlim)
            self.plot_xlim = self.plot_ax.get_xlim()
            self.header_ax.set_xlim(self.plot_xlim)
        except:
            self.plot_ax.relim()
            self.plot_ax.autoscale(enable=True, axis='x')
            self.plot_xlim = self.plot_ax.get_xlim()
            self.header_ax.set_xlim(self.plot_xlim)

    def setup_title_ax(self, sheet):
        """
        Initialize the title axis.

        Parameters
        ----------
        sheet : Sheet
            Parent sheet.
        """
        self.title_ax = sheet.add_axes(self.rect)
        self.title_ax.set_xticks([])
        self.title_ax.set_yticks([])
        self.title_ax.set_frame_on(True)
        self.title_ax.set_facecolor("#f0f0f0")
        title = self.title_ax.text(0.5, 0.5, self.type, ha="center", va="center", fontsize=9)
        title.set_fontfamily("monospace")

    def setup_header_ax(self, sheet):
        """
        Initialize the header (legend) axis.

        Parameters
        ----------
        sheet : Sheet
            Parent sheet.
        """
        self.header_ax = sheet.add_axes(self.rect)
        self.header_ax.set_xticks([])
        self.header_ax.set_yticks([])
        self.header_ax.set_frame_on(True)
        self.header_ax.set_ylim(0, 5)

    def setup_plot_ax(self, sheet, sharey):
        """
        Initialize the main plot axis.

        Parameters
        ----------
        sheet : Sheet
            Parent sheet.
        sharey : matplotlib.axes.Axes, optional
            Axes to share y-axis with.
        """
        self.plot_ax = sheet.add_axes(self.rect, sharey=sharey)
        self.plot_ax.invert_yaxis()
        self.plot_ax.grid(axis='both', color='#BBBBBB', linewidth=0.5, linestyle='-')
        self.plot_ax.grid(axis='both', which='minor', color='#BBBBBB', linewidth=0.5, linestyle=':')
        self.plot_ax.tick_params(axis="y", direction="inout", which="both", color='gray', right=True)
        self.plot_ax.tick_params(axis="x", direction="in", which="both", color='gray', top=True, labeltop=False, labelbottom=False)
        self.plot_ax.set_xlim(self.plot_xlim)

    def setup_divider(self):
        """
        Set up layout divider for internal axes arrangement.

        Returns
        -------
        mpl_toolkits.axes_grid1.Divider
            Configured divider object.
        """
        horiz = [Size.Scaled(2)]
        vert = [Size.Scaled(4), self.sheet.header_h, self.sheet.title_h]
        divider = Divider(self.sheet, self.rect, horiz, vert, aspect=False)
        return divider

class Sheet(Figure):
    """
    A composite figure for displaying well-log tracks and depth axis.

    Inherits from `matplotlib.figure.Figure`.

    Parameters
    ----------
    figsize : tuple, optional
        Figure size in inches (default: (10, 6)).
    **kwargs
        Additional keyword arguments passed to `Figure`.

    Attributes
    ----------
    tracks : list of Track
        List of track objects.
    track_widths : list
        Width specifications for each track.
    depth_ax : matplotlib.axes.Axes
        Axis for depth scale.
    """

    def __init__(self, figsize=(10, 6), **kwargs):
        super().__init__(figsize=figsize, **kwargs)
        self.depth_domain = "Depth"
        self.depth_unit = "m"
        self.tracks = []
        self.track_widths = []
        self._setup_layout()

    @property
    def plot_axes(self):
        """list of matplotlib.axes.Axes: Get list of all plot axes in tracks."""
        return list(tr.plot_ax for tr in self.tracks)

    @property
    def depth_domain(self):
        """str: Get the depth type ('Depth' or 'Time'), capitalized."""
        return self._depth_type.capitalize()
    @depth_domain.setter
    def depth_domain(self, value: Literal['DEPTH', 'TIME']):
        self._depth_type = value.capitalize()

    @property
    def depth_unit(self):
        """str: Get the unit of depth (e.g., 'm', 'ft', 'ms')."""
        return self._depth_unit
    @depth_unit.setter
    def depth_unit(self, value: Literal['m', 'ft', 'ms', 's', 'sec', 'msec']):
        self._depth_unit = value

    def update_depth_label(self):
        """Update the y-axis label of the depth axis with current type and unit."""
        self.depth_ax.set_ylabel(f"{self.depth_domain} ({self.depth_unit})")

    def set_depth_label(self, domain: Literal['DEPTH', 'TIME'], unit: Literal['m', 'ft', 'ms', 's', 'sec', 'msec']):
        """Set the depth domain and unit and update the depth label.
        Parameters
        ----------
        domain : str
            The domain of the depth (e.g., 'Depth', 'Time').
        unit : str
            The unit of the depth (e.g., 'm', 'ft', 'ms', 's', 'sec', 'msec').
        """
        self.depth_domain = domain
        self.depth_unit = unit
        self.update_depth_label()

    def _setup_layout(self):
        """Initialize the figure layout with margin and axis dividers."""
        margin = 0.02
        rect = [margin, margin, 1 - 2 * margin, 1 - 2 * margin]
        self.depth_width = Size.Fixed(0.6)
        self.title_h = Size.Fixed(0.3)
        self.header_h = Size.Fixed(1.5)
        self.h = [self.depth_width] + self.track_widths
        self.v = [Size.Scaled(1), self.header_h, self.title_h]
        self.divider = Divider(self, rect, self.h, [Size.Scaled(1)], aspect=False)
        self.depth_tr_divider = Divider(self, rect, self.h, self.v, aspect=False)
        if not hasattr(self, "depth_ax"):
            self.setup_depth_ax()
            self.depth_ax.set_axes_locator(self.depth_tr_divider.new_locator(nx=0, ny=0))

    def add_track(self, type: PropertyTemplate = PropertyTemplates().General):
        """
        Add a new track to the sheet.

        Parameters
        ----------
        type : PropertyTemplate, optional
            Template defining track properties (default: general template).

        Returns
        -------
        Track
            The newly created track.
        """
        self.track_widths.append(Size.Fixed(1.5))
        self._setup_layout()
        nx = len(self.track_widths)
        tr = Track(self, type, sharey=self.depth_ax)
        tr.attach_to_sheet(self, nx=nx, ny=0)
        self.tracks.append(tr)
        self.depth_ax.yaxis.set_tick_params(labelright=True)
        for trk in self.tracks:
            trk.plot_ax.tick_params(labelleft=False)
        self.hor_cursor = MultiCursor(self.canvas, self.plot_axes, color='r', lw=0.5, horizOn=True, vertOn=False, useblit=True)
        return tr

    def get_track_bytype(self, type):
        """
        Retrieve a track by its type.

        Parameters
        ----------
        type : str
            Track type to search for.

        Returns
        -------
        Track or None
            Matching track, or None if not found.
        """
        for track in self.tracks:
            if track.type == type:
                return track

    def add_empty_axes(self, width=2):
        """
        Add a blank axes area (e.g., for controls or labels).

        Parameters
        ----------
        width : float, optional
            Relative width (default: 2).

        Returns
        -------
        matplotlib.axes.Axes
            The created empty axes.
        """
        ax = self.add_axes([0, 0, 1, 1])
        self.track_widths.append(Size.Fixed(0.1))
        self.track_widths.append(Size.Scaled(width))
        self._setup_layout()
        nx = len(self.track_widths)
        locator = self.divider.new_locator(nx=nx, ny=0)
        bbox = locator(self, None)
        ax.set_position(bbox.bounds)
        ax.set_facecolor('0.9')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.invert_yaxis()
        return ax

    def setup_depth_ax(self):
        """Initialize the depth axis with appropriate ticks, labels, and appearance."""
        self.depth_ax = self.add_axes([0, 0, 1, 1])
        self.depth_ax.yaxis.set_major_locator(MaxNLocator(prune='both'))
        self.depth_ax.set_xticks([])
        self.depth_ax.tick_params(axis="y", which='minor', length=3)
        self.depth_ax.tick_params(axis="y", direction="in", which="both", pad=-30, color='gray', right=True, labelleft=False, labelright=True)
        self.depth_ax.yaxis.tick_right()
        self.depth_ax.minorticks_on()
        self.depth_ax.set_facecolor("#f0f0f0")
        self.depth_ax.yaxis.set_ticks_position('right')
        self.depth_ax.yaxis.set_label_position('left')
        self.update_depth_label()

    def set_depth_range(self, top, bottom):
        """
        Set the depth (y-axis) range for all tracks.

        Parameters
        ----------
        top : float
            Shallow depth limit.
        bottom : float
            Deep depth limit.
        """
        self.depth_ax.set_ylim(bottom, top)
        for tr in self.tracks:
            tr.plot_ax.set_ylim(bottom, top)

    def on_curser_move(self, event):
        """Handle cursor movement events (currently disabled)."""
        pass

class EmptyAxes(Axes):
    """
    An empty axes mainly used to display rock optimizer handles.

    Inherits from `matplotlib.axes.Axes`.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Parent figure.
    rect : list or tuple
        Position rectangle [left, bottom, width, height].
    """

    def __init__(self, fig, rect):
        super().__init__(fig, rect)
        self.set_xticks([])
        self.tick_params(axis="y", direction="inout", which="both", color='gray', right=True)
        self.tick_params(axis="x", direction="in", which="both", color='gray', top=True, labeltop=False, labelbottom=False)
        self.invert_yaxis()

class SeisPlot:
    """
    Static utility class for seismic wiggle plotting.
    """

    @staticmethod
    def wiggle(ax, traces, z=None, x=None, scale=1, color='black', fill_color='black', linewidth=0.5):
        """
        Plot seismic traces as wiggles on a given axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes.
        traces : array-like, shape (Nz, Nx)
            Seismic data.
        z : array-like, optional
            Vertical coordinates (default: 0 to Nz-1).
        x : array-like, optional
            Horizontal coordinates (default: 1 to Nx).
        scale : float, optional
            Amplitude scaling factor (default: 1).
        color : str, optional
            Line color (default: 'black').
        fill_color : str, optional
            Fill color for positive lobes (default: 'black').
        linewidth : float, optional
            Line width (default: 0.5).
        """
        Nz = np.shape(traces)[0]
        Nx = np.shape(traces)[1]
        if z is None:
            z = np.arange(Nz)
        if x is None:
            x = 1 + np.arange(Nx)
        nrm = np.linalg.norm(traces, ord=1) / Nz
        disp_scale = scale / nrm
        for i, xi in enumerate(x):
            trace = traces[:, i] * disp_scale
            ax.fill_betweenx(z, x[i] + trace, x[i],
                             where=(x[i] + trace > x[i]),
                             interpolate=True,
                             color=fill_color,
                             label=x[i],
                             linewidth=0)
            ax.plot(x[i] + trace, z, label=f'{xi}', color=color, linewidth=linewidth)

class RockOptimizer:
    """
    Interactive optimizer for rock physics model parameters.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes for hosting sliders.
    sheet : Sheet
        Parent sheet containing log tracks.
    rock : object
        Rock physics model object.
    depth : array-like
            Depth values for plotting.
    """

    def __init__(self, ax, sheet, rock, depth):
        self.rock = rock
        self.sheet = sheet
        self.ax = ax
        self.rect = self.ax.get_position().bounds
        self.depth = depth
        self.vert = [Size.Scaled(1)]
        self.vp_plot = self.sheet.get_track_bytype(PropertyTemplates().PVelocity.type).plot(self.depth, rock.p_velocity, label=rock.type, color='red')
        self.vs_plot = self.sheet.get_track_bytype(PropertyTemplates().SVelocity.type).plot(self.depth, rock.s_velocity, label=rock.type, color='red')
        self.ro_plot = self.sheet.get_track_bytype(PropertyTemplates().Density.type).plot(self.depth, rock.density, label=rock.type, color='red')
        self.setup_settings()
        self.sheet.canvas.mpl_connect('close_event', self.on_close)

    def setup_settings(self):
        """Initialize all parameter sliders."""
        self.slider_list = []
        self.add_inclusion_sliders()
        self.add_mineral_sliders()
        self.add_fluid_sliders()

    def add_slider(self, slider, ny):
        """
        Add a slider to the layout.

        Parameters
        ----------
        slider : TuningSlider
            The slider widget to add.
        ny : int
            Vertical position index.
        """
        self.slider_list.append(slider)
        self.vert.insert(-2, Size.Fixed(0.38))
        self.setup_divider()
        self.slider_list[-1].ax.set_axes_locator(self.ax.divider.new_locator(nx=1, ny=ny))

    def setup_divider(self):
        """Configure layout divider for sliders."""
        horiz = [Size.Fixed(0.1), Size.Scaled(1), Size.Fixed(0.1)]
        self.ax.divider = Divider(self.sheet, self.rect, horiz, self.vert, aspect=False)

    def update_plot(self):
        """Update the plotted rock properties and redraw."""
        self.vp_plot.set_xdata(self.rock.p_velocity)
        self.vs_plot.set_xdata(self.rock.s_velocity)
        self.ro_plot.set_xdata(self.rock.density)
        self.sheet.canvas.draw_idle()

    def add_inclusion_sliders(self):
        """Add sliders for inclusion parameters."""
        for inclusion in self.rock.inclusions:
            self.add_slider(LogarithmicSlider(self, self.sheet, inclusion, 'aspect_ratio'), len(self.slider_list))

    def add_mineral_sliders(self):
        """Add sliders for mineral elastic properties."""
        for mineral in self.rock.minerals:
            self.add_slider(LinearSlider(self, self.sheet, mineral, 'bulk'), len(self.slider_list))
            self.add_slider(LinearSlider(self, self.sheet, mineral, 'shear'), len(self.slider_list))
            self.add_slider(LinearSlider(self, self.sheet, mineral, 'density', min=1.45, max=5), len(self.slider_list))
            self.add_porosity_weight_slider(mineral)

    def add_fluid_sliders(self):
        """Add sliders for fluid bulk modulus (not currently used in setup)."""
        for fluid in self.rock.fluids:
            self.add_slider(LinearSlider(self, self.sheet, fluid, 'bulk'), len(self.slider_list))
            self.add_slider(LinearSlider(self, self.sheet, fluid, 'density', min=0.01, max=1.5), len(self.slider_list))

    def add_porosity_weight_slider(self, mineral):
        """Add a logarithmic slider for porosity weight."""
        self.add_slider(LogarithmicSlider(self, self.sheet, mineral, 'porosity_weight', min=0.01, max=10), len(self.slider_list))

    def on_close(self, evt):
        for slider in self.slider_list:
            print(f"{slider.label.get_text():30s}:\t{slider.value:.3f}")

class TuningSlider(Slider):
    """
    Base class for parameter tuning sliders with custom appearance.

    Parameters
    ----------
    optimizer : RockOptimizer
        Parent optimizer instance.
    sheet : Sheet
        Host sheet.
    obj : object
        Target object whose attribute will be modified.
    attribute_name : str
        Name of the attribute to tune.
    min : float
        Minimum slider value.
    max : float
        Maximum slider value.
    valinit : float
        Initial value.
    label : str
        Label text.
    """

    def __init__(self, optimizer, sheet, obj, attribute_name, min, max, valinit, label):
        self.ax = sheet.add_axes([0, 0, 1, 1])
        self.obj = obj
        self.attribute = attribute_name
        super().__init__(ax=self.ax,
                         label=label,
                         valmin=min,
                         valmax=max,
                         valinit=valinit,
                         orientation='horizontal')
        self.optimizer = optimizer
        self.sheet = sheet
        self.valtext.set_text('{:.2f}'.format(valinit))
        self.on_changed(self.update)
        self.customize()

    @property
    def value(self):
        return self.val

    def customize(self):
        """Apply custom styling to the slider."""
        self.ax.set_box_aspect(0.03)
        self.poly.set_alpha(0)
        self.label.set_position((0.0, 1))
        self.label.set_ha('left')
        self.label.set_va('bottom')
        self.label.set_fontsize(10)
        self.label.set_fontfamily('monospace')
        self.valtext.set_position((1, 1))
        self.valtext.set_ha('right')
        self.valtext.set_va('bottom')
        self.valtext.set_fontsize(9)
        self.valtext.set_color('green')
        self.valtext.set_fontfamily('monospace')

    def update(self, val):
        """
        Update the object attribute and refresh plots.

        Parameters
        ----------
        val : float
            New slider value.
        """
        self.valtext.set_text('{:.2f}'.format(val))
        setattr(self.obj.properties, self.attribute, val)
        self.optimizer.update_plot()

class LinearSlider(TuningSlider):
    """
    Linear-scale tuning slider.

    Parameters
    ----------
    optimizer : RockOptimizer
        Parent optimizer.
    sheet : Sheet
        Host sheet.
    obj : object
        Target object.
    attribute : str
        Attribute name.
    min : float, optional
        Minimum value (default: 0).
    max : float, optional
        Maximum value (default: 100).
    """

    def __init__(self, optimizer, sheet, obj, attribute, min=0, max=100):
        super().__init__(optimizer, sheet, obj, attribute, min, max, valinit=float(getattr(obj, attribute)), label='{} ({})'.format(obj.type, attribute))

class LogarithmicSlider(TuningSlider):
    """
    Logarithmic-scale tuning slider.

    Parameters
    ----------
    optimizer : RockOptimizer
        Parent optimizer.
    sheet : Sheet
        Host sheet.
    obj : object
        Target object.
    attribute : str
        Attribute name.
    min : float, optional
        Minimum value (default: 0.01).
    max : float, optional
        Maximum value (default: 1).
    """

    def __init__(self, optimizer, sheet, obj, attribute, min=0.01, max=1):
        super().__init__(optimizer, sheet, obj, attribute, min=np.log10(min), max=np.log10(max), valinit=np.log10(getattr(obj, attribute)), label='{} ({})'.format(obj.type, attribute))
        self.valtext.set_text('{:.2f}'.format(np.power(10, self.valinit)))

    @property
    def value(self):
        return np.power(10, self.val)
    def update(self, val):
        """
        Update with logarithmic scaling.

        Parameters
        ----------
        val : float
            Log10 of the true value.
        """
        true_val = np.power(10, val)
        self.valtext.set_text('{:.2}'.format(true_val))
        setattr(self.obj, self.attribute, true_val)
        self.optimizer.update_plot()
 
def display_las(las_file: str, log_names: list[str]=None) -> Sheet:
    """
    Display well-log data from a LAS file in a structured sheet format.

    Parameters
    ----------
    las_file : str
        Path to the LAS file.

    Returns
    -------
    Sheet
        Configured sheet with plotted well logs.

    Raises
    ------
    ImportError
        If `lasio` is not installed.
    """
    import lasio
    import matplotlib.pyplot as plt
    las = lasio.read(las_file)
    sheet = plt.figure(FigureClass=Sheet, figsize=(9, 12))
    if  log_names is None:
        log_names=las.keys()
    depth_log = las.index
    sheet.set_depth_range(np.min(depth_log), np.max(depth_log))
    sheet.depth_type = las.keys()[0]
    sheet.depth_unit = las.index_unit
    for log_name in log_names:
        curve = las.curvesdict[log_name]
        track = sheet.add_track(curve.mnemonic)
        track.plot_xlim = (np.nanmin(curve.data), np.nanmax(curve.data))
        track.plot(depth_log, curve.data, label=curve.mnemonic)
    return sheet
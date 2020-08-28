import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.font_manager import FontProperties
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from matplotlib.colors import Normalize
from matplotlib import cm


""" 
Another ternary diagrams plotting wrapper. I created this because I needed a way to plot a 4D surface upon a 
a region of varying 3D parameters. A lot could be improved! I'd be happy to receive comments and/or feedback. 

Alexi Morin
"""


# Main plotting object. Could probably be improved by being a children of a matplotlib class.
class TernaryPlot:

    def __init__(self, resolution: int = 10, c: float = 1, plot_grid: bool = True, nbr_of_lines: int = None,
                 x_min: float = 0, y_min: float = 0, z_min: float = 0, fig_size: float = 5):

        """
        Parameters:
            resolution: The "pixel" resolution of the grid to be plotted. Higher resolution means a finer mesh.
            c: The constant which x + y + z = c. Defaults to 1. 100 is also a nice value.
            plot_grid: Boolean, plots a grid by default.
            nbr_of_lines: The number of lines in the grid.
            x_min, y_min, z_min: The minimum values of x, y and z. Their total must be less than the value of c.
            fig_size: Float, the width of the squared plot.
        """

        # Plotting parameters
        self.main_color = 'black'
        self.resolution = resolution
        self.fontname = 'serif'

        self.c = c
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min

        # The difference between every x_min and x_max
        self.delta = c - x_min - y_min - z_min

        if self.delta <= 0:
            raise ValueError('The value of c must be lower than the values of x_min, y_min and z_min combined. \n'
                             'i.e. c - x_min - y_min - z_min > 0')

        if nbr_of_lines is None:
            self.nbr_of_lines = int(self.delta / self.c * 10)
            if self.nbr_of_lines == 0:
                self.nbr_of_lines += 1
        else:
            self.nbr_of_lines = nbr_of_lines

        self.plot_grid = plot_grid

        # Initializes the triangle's 3D axes.
        self.x, self.y, self.z = self.init_xyz()

        # Initialising the matplotlib figure. We create a 3D subplot.
        self.fig = plt.figure()
        self.fig_size = fig_size
        self.fig.set_size_inches(fig_size, fig_size)
        self.ax = self.fig.add_subplot(111, projection='3d', azim=45, elev=45)

        self.ax.set_xlim(self.x_min, self.x_min + self.delta)
        self.ax.set_ylim(self.y_min, self.y_min + self.delta)
        self.ax.set_zlim(self.z_min, self.z_min + self.delta)

        # Disabling matplotlib 3D perks so it looks 2D.
        self.ax.disable_mouse_rotation()
        self.ax._axis3don = False
        self.plot_lines(x_min, y_min, z_min, self.nbr_of_lines)
        self.plot_ticks(x_min, y_min, z_min, self.nbr_of_lines)

    # Creates the xyz arrays which will contain the mesh.
    def init_xyz(self):

        x = np.linspace(self.x_min, self.x_min + self.delta, self.resolution)
        y = np.ones(shape=(1, self.resolution)) * self.y_min
        z = np.linspace(self.z_min + self.delta, self.z_min, self.resolution)

        for i in range(1, self.resolution):
            x = np.vstack((x, np.linspace(self.x_min, self.x_min + self.delta - i * self.delta / (self.resolution - 1),
                                          self.resolution)))
            y = np.vstack((y, np.linspace(self.y_min, self.y_min + i * self.delta / (self.resolution - 1),
                                          self.resolution)))
            z = np.vstack((z, np.linspace(self.z_min + self.delta, self.z_min,
                                          self.resolution)))
        return x, y, z

    # Plots the grid lines
    def plot_lines(self, x_min, y_min, z_min, n):

        t = np.array([0, 1])

        if not self.plot_grid:
            n = 1

        for i in range(n):
            linewidth = 0.5

            if i == 0:
                linewidth = 1

            # _
            self.ax.plot(xs=x_min + t * self.delta * (n - i) / n,
                         ys=y_min + self.delta * (n - i) / n - t * self.delta * (n - i) / n,
                         zs=t * (z_min + self.delta * i / n) + (1 - t) * (z_min + self.delta * i / n),
                         c=self.main_color, linewidth=linewidth)
            # \
            self.ax.plot(xs=t * (x_min + self.delta * i / n) + (1 - t) * (x_min + self.delta * i / n),
                         ys=y_min + t * self.delta * (n - i) / n,
                         zs=z_min + self.delta * (n - i) / n - t * self.delta * (n - i) / n,
                         c=self.main_color, linewidth=linewidth)
            # /
            self.ax.plot(xs=x_min + t * self.delta * (n - i) / n,
                         ys=t * (y_min + self.delta * i / n) + (1 - t) * (y_min + self.delta * i / n),
                         zs=z_min + self.delta * (n - i) / n - t * self.delta * (n - i) / n,
                         c=self.main_color, linewidth=linewidth)

    # Plots the grid ticks
    def plot_ticks(self, x_min, y_min, z_min, n):

        x_x = np.linspace(x_min, x_min + self.delta, n + 1)
        y_x = np.linspace(y_min + self.delta, y_min, n + 1)
        z_x = np.linspace(z_min, z_min, n + 1)

        x_y = np.linspace(x_min, x_min, n + 1)
        y_y = np.linspace(y_min, y_min + self.delta, n + 1)
        z_y = np.linspace(z_min + self.delta, z_min, n + 1)

        x_z = np.linspace(x_min, x_min + self.delta, n + 1)
        y_z = np.linspace(y_min, y_min, n + 1)
        z_z = np.linspace(z_min + self.delta, z_min, n + 1)

        pos = np.sqrt(self.fig_size) / np.sqrt(5) * self.delta / 10
        x_bar_adjust = 1.2
        divisor = self.delta / n

        if divisor >= 1:
            floating_format = f'1.{int(np.ceil(np.log10(divisor)) - 1)}'
        else:
            floating_format = f'1.{int(np.ceil(-np.log10(divisor)))}'

        for i in range(0, n + 1):
            self.ax.text(x_x[i] + pos * x_bar_adjust / 2, y_x[i] + pos * x_bar_adjust / 2, z_x[i] - pos * x_bar_adjust,
                         f'{y_min + self.delta - i * self.delta / n:{floating_format}f}'
                         , ha='center', fontname=self.fontname)  # _

            self.ax.text(x_y[i] - pos, y_y[i] + pos, z_y[i],
                         f'{z_min + self.delta - i * self.delta / n:{floating_format}f}'
                         , ha='center', fontname=self.fontname)  # /
            self.ax.text(x_z[i] + pos, y_z[i] - pos, z_z[i],
                         f'{x_min + i * self.delta / n:{floating_format}f}'
                         , ha='center', fontname=self.fontname)  # \
        dt = self.c / 50
        # Trouver comment arranger les markers
        self.ax.scatter(x_x, y_x, z_x, c=self.main_color, alpha=1)  # , marker = (2, 0, 0)) # _
        self.ax.scatter(x_y, y_y, z_y, c=self.main_color, alpha=1)  # , marker = (2, 0, 0)) # /
        self.ax.scatter(x_z, y_z, z_z, c=self.main_color, alpha=1)  # , marker = (2, 0,-0)) # \

    # The function I was first interested in. Plots a fourth dimension over a 3D surface.
    def plot_contours(self, func, cmap: str = 'jet', plot_colorbar: bool = True,
                      cbar_label: str = None, plot_contourlines: bool = False, **kwargs):

        X = np.array([self.x, self.y, self.z])
        w = func(X, **kwargs)
        norm = Normalize(w.min(), w.max())
        scamap = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fcolors = scamap.to_rgba(w)

        im = self.ax.plot_surface(self.x, self.y, self.z, facecolors=fcolors,
                                  cmap=cmap, alpha=0.6, linewidth=0, edgecolor=None)

        if plot_contourlines:
            extent = np.max(w) - np.min(w)
            if np.log10(extent) <= 0:
                extent = np.abs(np.log10(extent))
                decimals = int(np.floor(extent)) + 1
            else:
                decimals = int(np.floor(np.log10(extent))) + 1

            dw = np.floor(np.sqrt(extent))
            w0 = np.floor(np.min(w)) + dw
            while w0 < np.max(w):
                x = []
                y = []
                mins = np.abs(w - w0)
                mins = np.around(np.abs(w - w0), decimals=2)
                closest = np.where(mins == np.min(mins))
                for i in range(len(closest[0])):
                    x.append(self.x[closest[0][i], closest[1][i]])
                    y.append(self.y[closest[0][i], closest[1][i]])

                x = np.array(x)
                y = np.array(y)
                z = self.c - x - y
                self.ax.plot(x, y, z)
                w0 += dw

        if plot_colorbar:
            self.fig.add_subplot(212, aspect=0.5, visible=False)
            cbar = self.fig.colorbar(scamap, orientation='horizontal')
            cbar.set_label(cbar_label, fontname=self.fontname)
        return scamap

    # Plots scatter points
    def plot_points(self, x, y, z, **kwargs):
        maxs = [[x, self.x_min], [y, self.y_min], [z, self.z_min]]
        j = 0

        for i in maxs:
            if type(i[0]) is not pd.Series and type(i[0]) is not np.array:
                maxs[j][0] = np.array(i[0])
            if (i[0] < i[1]).sum() != 0 or (i[0] > i[1] + self.delta).sum() != 0:
                raise ValueError(f'Values are out of bounds. Look at values from x_min, y_min, z_min and c.')
            j += 1

        dt = self.c / 10
        self.ax.scatter(x + dt / np.sqrt(2), y + dt / np.sqrt(2), z + dt, alpha=1, linewidth=1, edgecolor='k', **kwargs)

    def label_names(self, x_name, y_name, z_name):  # Haven't found any other way

        self.ax.annotate(x_name, xy=(-0.06, 0.04), rotation=60, ha='center', va='center', fontname=self.fontname)
        self.ax.annotate(y_name, xy=(0, -0.06), ha='center', va='center', fontname=self.fontname)
        self.ax.annotate(z_name, xy=(0.06, 0.04), rotation=-60, ha='center', va='center', fontname=self.fontname)

    def set_title(self, title, pad=None, **kwargs):

        if pad is None:
            self.ax.set_title(title, pad=self.fig_size * 4.5, **kwargs, fontname=self.fontname)
        else:
            self.ax.set_title(title, **kwargs, fontname=self.fontname)

    def legend(self):
        handles, labels = self.ax.get_legend_handles_labels()

        w = len(labels)
        h = 1
        for i in range(1, w):
            if w % i == 0:
                h = i
        w = int(w/h)

        axbox = triplot.ax.get_position()
        leg_x_offset = -0.20
        leg_y_offset = -0.025
        triplot.ax.legend(loc=(axbox.x0 + leg_x_offset, axbox.y0 + leg_y_offset), ncol=w,
                          prop={'family': triplot.fontname})


if __name__ == '__main__':

    # Example figure
    def f(x):
        return np.exp(x[0]) + 1 / (1 + x[1]) + 1 * x[2]

    # Creating the triplot instance
    triplot = TernaryPlot(fig_size=5, c=1, plot_grid=True, x_min=0., y_min=0., z_min=0., resolution=100)

    # Plots the surface defined by f
    scamap = triplot.plot_contours(f, 'inferno', plot_colorbar=True, cbar_label='Some metric [Some unit]')

    # Plots an observation. It's 4D value can be plotted with the scamap returned by the plot_contour function
    triplot.plot_points(0.1, 0.4, 0.5, c=scamap.to_rgba([3]), label='Some points')

    # Plot calls, sensibly similar to matplotlib's
    triplot.label_names('A', 'B', 'C')
    triplot.set_title('Example', loc='right')
    triplot.legend()
    plt.show()

# import the required libraries
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
import textwrap

# Now need to convert out dataset into the same format
# Removed ImageNet dataset from the list
datasets = ["Eurosat", "Caltech", "DTD", "Pets", "FGVC", "Cars", "Flowers", "SUN397", "UCF", "FOOD"]
coop = [54.42, 91.78, 54.88, 94.10, 32.12, 69.23, 71.86, 72.15, 64.74, 89.25]
cocoop = [60.28, 95.09, 65.11, 96.69, 33.71, 71.91, 77.61, 77.99, 76.78, 91.20]
LFA = [69.56, 95.69, 67.43, 94.09, 34.27, 72.72, 84.38, 78.39, 82.71, 90.44]
Candle = [80.51, 95.89, 68.13, 95.99, 37.78, 74.30, 85.03, 79.26, 83.17, 90.80]
DroPLe = [80.26, 96.39, 67.39, 96.41, 39.39, 75.41, 86.85, 80.32, 82.63, 91.24]


all_data = [coop, cocoop, LFA, Candle, DroPLe]
df = pd.DataFrame(all_data, columns=datasets, index=['CoOp', 'CoCoOp', 'LFA', 'Candle', 'DroPLe'])
df.index.rename('Method', inplace=True)
data = df
result = df


# this class is taken from the the source https://towardsdatascience.com/how-to-create-and-visualize-complex-radar-charts-f7764d0f3652
class ComplexRadar():
    """
    Create a complex radar chart with different scales for each variable
    Parameters
    ----------
    fig : figure object
        A matplotlib figure object to add the axes on
    variables : list
        A list of variables
    ranges : list
        A list of tuples (min, max) for each variable
    n_ring_levels: int, defaults to 5
        Number of ordinate or ring levels to draw
    show_scales: bool, defaults to True
        Indicates if we the ranges for each variable are plotted
    """

    def __init__(self, fig, variables, ranges, n_ring_levels=5, show_scales=True):
        # Calculate angles and create for each variable an axes
        # Consider here the trick with having the first axes element twice (len+1)
        angles = np.arange(0, 360, 360. / len(variables))
        # Add more margin space around the radar chart
        axes = [fig.add_axes([0.2, 0.2, 0.6, 0.6], polar=True, label="axes{}".format(i)) for i in
                range(len(variables) + 1)]

        # Ensure clockwise rotation (first variable at the top N)
        for ax in axes:
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_axisbelow(True)

        # Writing the ranges on each axes
        for i, ax in enumerate(axes):

            # Here we do the trick by repeating the first iteration
            j = 0 if (i == 0 or i == 1) else i - 1
            ax.set_ylim(*ranges[j])
            # Set endpoint to True if you like to have values right before the last circle
            grid = np.linspace(*ranges[j], num=n_ring_levels,
                               endpoint=False)
            gridlabel = ["{}".format(round(x, 2)) for x in grid]
            gridlabel[0] = ""  # remove values from the center
            # Make grid labels smaller (was 44, now 30)
            lines, labels = ax.set_rgrids(grid, labels=gridlabel, angle=angles[j], fontsize=30)  # the values of scales

            ax.set_ylim(*ranges[j])
            ax.spines["polar"].set_visible(False)
            ax.grid(visible=False)

            if show_scales == False:
                ax.set_yticklabels([])

        # Set all axes except the first one unvisible
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.xaxis.set_visible(False)

        # Setting the attributes
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
        self.ax1 = axes[1]
        self.plot_counter = 0

        # Draw (inner) circles and lines
        self.ax.yaxis.grid(linewidth=2.5, color='#CCCCCC')
        self.ax.xaxis.grid(linewidth=2.5, color='#c6c6c6')

        # Draw outer circle
        self.ax.spines['polar'].set_visible(False)

        # ax1 is the duplicate of axes[0] (self.ax)
        # Remove everything from ax1 except the plot itself
        self.ax1.axis('off')
        self.ax1.set_zorder(9)

        # Create the outer labels for each variable
        # Adjust font size (was 200, now 120)
        l, text = self.ax.set_thetagrids(angles, labels=variables, fontsize=120, weight='bold')

        # Beautify them
        labels = [t.get_text() for t in self.ax.get_xticklabels()]
        labels = ['\n'.join(textwrap.wrap(l, 15, break_long_words=False)) for l in labels]
        # Adjust the font size (was 50, now 40)
        self.ax.set_xticklabels(labels, fontsize=40)

        for t, a in zip(self.ax.get_xticklabels(), angles):
            if a == 0:
                t.set_ha('center')
            elif a > 0 and a < 180:
                t.set_ha('left')
            elif a == 180:
                t.set_ha('center')
            else:
                t.set_ha('right')

        # Double the padding (was 15, now 30)
        self.ax.tick_params(axis='both', pad=30)

    def _scale_data(self, data, ranges):
        """Scales data[1:] to ranges[0]"""
        for d, (y1, y2) in zip(data[1:], ranges[1:]):
            assert (y1 <= d <= y2) or (y2 <= d <= y1)
        x1, x2 = ranges[0]
        d = data[0]
        sdata = [d]
        for d, (y1, y2) in zip(data[1:], ranges[1:]):
            sdata.append((d - y1) / (y2 - y1) * (x2 - x1) + x1)
        return sdata

    def plot(self, data, color, *args, **kwargs):
        """Plots a line"""
        sdata = self._scale_data(data, self.ranges)
        if color == 'violet':
            # Increased line width (was 3, now 6)
            self.ax1.plot(self.angle, np.r_[sdata, sdata[0]], color=color, linewidth=6, *args, **kwargs)
        else:
            # Increased line width (was 2, now 4)
            self.ax1.plot(self.angle, np.r_[sdata, sdata[0]], color=color, linewidth=4, *args, **kwargs)
        self.plot_counter = self.plot_counter + 1

    def fill(self, data, color, *args, **kwargs):
        """Plots an area"""
        sdata = self._scale_data(data, self.ranges)
        self.ax1.fill(self.angle, np.r_[sdata, sdata[0]], color=color, *args, **kwargs)

    def use_legend(self, *args, **kwargs):
        """Shows a legend"""
        # Adjust the font size (was 54, now 40)
        leg = self.ax1.legend(fontsize=40, *args, **kwargs)

        # Remove legend border
        leg.get_frame().set_linewidth(0.0)

        # change the line width for the legend
        # Make lines thinner (was 20, now 8)
        for line in leg.get_lines():
            line.set_linewidth(8.0)

    def set_title(self, title, pad=25, **kwargs):
        """Set a title"""
        self.ax.set_title(title, pad=pad, **kwargs)


min_max_per_variable = data.describe().T[['min', 'max']]
min_max_per_variable['min'] = min_max_per_variable['min'].apply(lambda x: int(x))
min_max_per_variable['max'] = min_max_per_variable['max'].apply(lambda x: math.ceil(x))

variables = result.columns
ranges = list(min_max_per_variable.itertuples(index=False, name=None))

# Increased figure size with more margin space
fig1 = plt.figure(figsize=(22, 22))
radar = ComplexRadar(fig1, variables, ranges, show_scales=True)
colors = ['#cf8780', '#e8ac2c', '#a07495', '#90b55f', '#83c8e4']
for g, color in zip(result.index, colors):
    radar.plot(result.loc[g].values, label=f"{g}", color=color)
    radar.fill(result.loc[g].values, alpha=0.2, color=color)

# radar.set_title("Radar chart solution with different scales")
# Changed to 2x2 layout (ncol=2)
# radar.use_legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)
# plt.savefig("radar_plot.svg", format='svg', bbox_inches='tight', dpi=600)
plt.show()
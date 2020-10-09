import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tikzplotlib

events = [_.replace(" ", "\n") for _ in [
    "infection ",
    "symptom onset",
    "testing ",
    "test confirmation",
    "test reporting",
    "public reporting",
]]

n = len(events)
dates  = list(range(n))
levels = np.array([1, 1, 1, 1, 1, 1])


palette = sns.cubehelix_palette(2 * (n - 1), start=.5, rot=-.75)[::2]


fig, ax = plt.subplots(figsize=(6, 2))

markerline, stemline, baseline = ax.stem(dates, levels,
                                         linefmt="kD-", basefmt=" ",
                                         use_line_collection=True)

plt.setp(markerline, mec="k", mfc="w", zorder=3)


# Shift the markers to the baseline by replacing the y-data by zeros.
markerline.set_ydata(np.zeros(len(dates)))

# annotate lines
vert = np.array(['top', 'bottom'])[(levels > 0).astype(int)]
for d, l, r, va in zip(dates, levels, events, vert):
    ax.annotate(r+"\n", xy=(d, l), xytext=(-3, np.sign(l)*3), textcoords="offset points", va=va, ha="center")

ax.get_yaxis().set_visible(False)
ax.get_xaxis().set_visible(False)
for spine in ["left", "top", "right", "bottom"]:
    ax.spines[spine].set_visible(False)

for (i, color) in enumerate(palette):
    plt.plot([i + 0.03, i + 0.97], [0, 0], color = "black", linewidth = 6, zorder = 1)
    plt.plot([i + 0.03, i + 0.97], [0, 0], color = color,   linewidth = 5, zorder = 2)

ax.margins(y=0.6)

print(tikzplotlib.get_tikz_code())
plt.show()


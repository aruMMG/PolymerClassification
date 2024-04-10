import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def plot_colored_line(y, colors, save_name):
    """
    Plots a line with variable color.

    Args:
    - x: array-like, x-coordinates of the points
    - colors: array-like, color values
    
    Returns:
    - None
    """
    # colors = 1-colors
    # Normalize the color values to range from 0 to 1
    normalized_colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors))
    # Create a colormap mapping the normalized colors to the color spectrum
    # colormap = plt.cm.seismic
    # colormap = plt.cm.Reds
    colormap = plt.cm.cool

    # Create a ScalarMappable to map normalized values to colors
    scalar_map = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=colormap)

    # Plot the line with variable color
    for i in range(len(y) - 1):
        if y[i] != y[i + 1]:  # Skip consecutive points with the same y-coordinate
            plt.plot([i, i + 1], [y[i], y[i + 1]], color=scalar_map.to_rgba(normalized_colors[i]))
        # plt.plot([x[i], x[i + 1]], [i, i + 1], color=scalar_map.to_rgba(normalized_colors[i]))

    # Add colorbar to show the mapping from colors to values
    # plt.colorbar(scalar_map, label='Color')

    # Show the plot
    plt.savefig(f"./logFile/EXP6/c/test/plots/incep/{save_name}")
    plt.close()
# Example usage:
heatmap0 = np.load("./logFile/EXP6/c/test/plots/incep/3/input_tensors0.npy")
colors0 = np.load("./logFile/EXP6/c/test/plots/incep/3/heatmap0.npy")
save_name = "check_heatmap_incep_30.png"
plot_colored_line(np.mean(heatmap0, axis=0), np.mean(colors0, axis=0), save_name)

heatmap1 = np.load("./logFile/EXP6/c/test/plots/incep/3/input_tensors1.npy")
colors1 = np.load("./logFile/EXP6/c/test/plots/incep/3/heatmap1.npy")
save_name = "heatmap_incep_31.png"
plot_colored_line(np.mean(heatmap1, axis=0), np.mean(colors1, axis=0), save_name)

heatmap2 = np.load("./logFile/EXP6/c/test/plots/incep/3/input_tensors2.npy")
colors2 = np.load("./logFile/EXP6/c/test/plots/incep/3/heatmap2.npy")
save_name = "heatmap_incep_32.png"
plot_colored_line(np.mean(heatmap2, axis=0), np.mean(colors2, axis=0), save_name)

heatmap3 = np.load("./logFile/EXP6/c/test/plots/incep/3/input_tensors3.npy")
colors3 = np.load("./logFile/EXP6/c/test/plots/incep/3/heatmap3.npy")
save_name = "heatmap_incep_33.png"
plot_colored_line(np.mean(heatmap3, axis=0), np.mean(colors3, axis=0), save_name)


heatmap0 = np.load("./logFile/EXP6/c/test/plots/incep/3/input_tensors0.npy")
colors0 = np.load("./logFile/EXP6/c/test/plots/incep/0/heatmap0.npy")
save_name = "check_heatmap_incep_00.png"
plot_colored_line(np.mean(heatmap0, axis=0), np.mean(colors0, axis=0), save_name)

heatmap1 = np.load("./logFile/EXP6/c/test/plots/incep/3/input_tensors1.npy")
colors1 = np.load("./logFile/EXP6/c/test/plots/incep/0/heatmap1.npy")
save_name = "heatmap_incep_01.png"
plot_colored_line(np.mean(heatmap1, axis=0), np.mean(colors1, axis=0), save_name)

heatmap2 = np.load("./logFile/EXP6/c/test/plots/incep/3/input_tensors2.npy")
colors2 = np.load("./logFile/EXP6/c/test/plots/incep/0/heatmap2.npy")
save_name = "heatmap_incep_02.png"
plot_colored_line(np.mean(heatmap2, axis=0), np.mean(colors2, axis=0), save_name)

heatmap3 = np.load("./logFile/EXP6/c/test/plots/incep/3/input_tensors3.npy")
colors3 = np.load("./logFile/EXP6/c/test/plots/incep/0/heatmap3.npy")
save_name = "heatmap_incep_03.png"
plot_colored_line(np.mean(heatmap3, axis=0), np.mean(colors3, axis=0), save_name)


heatmap0 = np.load("./logFile/EXP6/c/test/plots/incep/3/input_tensors0.npy")
# colors0 = np.load("./logFile/EXP6/c/test/plots/incep/5/heatmap0.npy")
# save_name = "check_heatmap_incep_50.png"
# plot_colored_line(np.mean(heatmap0, axis=0), np.mean(colors0, axis=0), save_name)

# heatmap1 = np.load("./logFile/EXP6/c/test/plots/incep/3/input_tensors1.npy")
# colors1 = np.load("./logFile/EXP6/c/test/plots/incep/5/heatmap1.npy")
# save_name = "heatmap_incep_51.png"
# plot_colored_line(np.mean(heatmap1, axis=0), np.mean(colors1, axis=0), save_name)

# heatmap2 = np.load("./logFile/EXP6/c/test/plots/incep/3/input_tensors2.npy")
# colors2 = np.load("./logFile/EXP6/c/test/plots/incep/5/heatmap2.npy")
# save_name = "heatmap_incep_52.png"
# plot_colored_line(np.mean(heatmap2, axis=0), np.mean(colors2, axis=0), save_name)

# heatmap3 = np.load("./logFile/EXP6/c/test/plots/incep/3/input_tensors3.npy")
# colors3 = np.load("./logFile/EXP6/c/test/plots/incep/5/heatmap3.npy")
# save_name = "heatmap_incep_53.png"
# plot_colored_line(np.mean(heatmap3, axis=0), np.mean(colors3, axis=0), save_name)
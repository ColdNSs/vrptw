import os
import sys
from pathlib import Path

# Get repo root (one level above src/)
root = Path(__file__).resolve().parent.parent

# Add vrptw root to path so 'src' is importable as a package
sys.path.insert(0, str(root))

import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

from src.parser import read_solomon_instance

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_coordinates(instance_path):
    """Reads the Solomon instance to extract X, Y coordinates for plotting."""
    instance = read_solomon_instance(instance_path)
    # Map node ID to its (X, Y) tuple
    coords = {node.id: (node.x, node.y) for node in instance.nodes}
    return coords


def plot_single_best_solution(instance_name, coords, sol, save_dir):
    """
    Plots the single best routing solution on a 2D map.
    """
    plt.figure(figsize=(10, 8))

    # Set a nice colormap for the vehicles
    cmap = plt.cm.get_cmap('tab20', max(20, len(sol['routes'])))

    # Coordinates for depot and all customers
    depot_x, depot_y = coords[0]
    all_x = [coords[i][0] for i in coords if i != 0]
    all_y = [coords[i][1] for i in coords if i != 0]

    # 1. Plot Background Nodes
    plt.scatter(all_x, all_y, c='grey', s=20, alpha=0.5, zorder=1)
    plt.scatter(depot_x, depot_y, c='red', marker='*', s=350, edgecolor='black', label='Depot', zorder=5)

    # 2. Plot Vehicle Routes
    for v_idx, route in enumerate(sol['routes']):
        # Ensure route starts and ends with Depot (0) for plotting
        if not route: continue
        full_route = route.copy()
        if full_route[0] != 0: full_route = [0] + full_route
        if full_route[-1] != 0: full_route = full_route + [0]

        route_x = [coords[n][0] for n in full_route]
        route_y = [coords[n][1] for n in full_route]

        # Plot the lines connecting the nodes
        plt.plot(route_x, route_y, color=cmap(v_idx % 20), linewidth=2, alpha=0.8,
                 marker='o', markersize=5, label=f'Vehicle {v_idx + 1}')

        # Add arrows to indicate direction (placed in the middle of the route)
        mid_idx = len(full_route) // 2
        if 0 < mid_idx < len(full_route):
            plt.annotate("", xy=(coords[full_route[mid_idx]][0], coords[full_route[mid_idx]][1]),
                         xytext=(coords[full_route[mid_idx - 1]][0], coords[full_route[mid_idx - 1]][1]),
                         arrowprops=dict(arrowstyle="->", color=cmap(v_idx % 20), lw=1.5))

    # 3. Formatting
    plt.title(f"{instance_name}\n"
              f"Total Distance: {sol['distance']:.2f} | Fleet Size: {len(sol['routes'])}",
              fontsize=16, pad=15)
    plt.xlabel("X Coord.", fontsize=12)
    plt.ylabel("Y Coord.", fontsize=12)

    # Put legend outside the plot to avoid covering nodes
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{instance_name}_best_route.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved route map to {save_path}")


def generate_visualizations():
    results_dir = root / "experiments" / "results_pop200_gen300"
    plots_dir = root / "experiments" / "plots_showcase"
    benchmark_dir = root / "data" / "benchmarks" / "solomon-100"

    os.makedirs(plots_dir, exist_ok=True)

    for filepath in glob.glob(os.path.join(results_dir, "*.json")):
        with open(filepath, 'r') as f:
            data = json.load(f)

        instance_name = data['instance']
        population = data['final_population']

        # 1. Filter perfectly feasible solutions
        feasible_inds = [ind for ind in population if ind['total_penalty'] == 0]
        if not feasible_inds:
            print(f"Skipping {instance_name}: No feasible solutions found.")
            continue

        # 2. Find the absolute minimum distance solution
        best_sol = min(feasible_inds, key=lambda x: x['distance'])

        # 3. Plot it
        instance_txt_path = os.path.join(benchmark_dir, f"{instance_name}.txt")
        if os.path.exists(instance_txt_path):
            coords = load_coordinates(instance_txt_path)
            plot_single_best_solution(instance_name, coords, best_sol, plots_dir)
        else:
            print(f"Warning: Could not find {instance_txt_path} to extract coordinates.")


if __name__ == "__main__":
    generate_visualizations()
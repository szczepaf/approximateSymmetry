import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Parameters
n = 100  # Number of nodes
k = 3  # Number of edges to attach from a new node to existing nodes

# Generate a Barabasi-Albert (BA) graph
ba_graph = nx.barabasi_albert_graph(n, k)

# Compute eccentricities
eccentricities = nx.eccentricity(ba_graph)

# Normalize eccentricities for color mapping
max_eccentricity = max(eccentricities.values())
min_eccentricity = min(eccentricities.values())

# Create a color map from blue (low) to red (high)
cmap = plt.cm.coolwarm
norm = mcolors.Normalize(vmin=min_eccentricity, vmax=max_eccentricity)

# Assign colors to nodes based on eccentricity
node_colors = [cmap(norm(eccentricities[node])) for node in ba_graph.nodes]

# Draw the graph
plt.figure(figsize=(10, 8))

# Use a spring layout for better visualization
pos = nx.spring_layout(ba_graph, seed=42)
nx.draw(
    ba_graph,
    pos,
    node_color=node_colors,
    with_labels=False,
    node_size=100,
    font_size=8,
    edge_color="gray",
)

# Add labels with eccentricity values
labels = {node: f"{eccentricities[node]}" for node in ba_graph.nodes}
nx.draw_networkx_labels(ba_graph, pos, labels, font_size=8)

# Add a color bar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
fig = plt.gcf()  # Get current figure
ax = plt.gca()  # Get current axis
cbar = fig.colorbar(sm, ax=ax, orientation="vertical", shrink=0.8)
cbar.set_label("Eccentricity", rotation=270, labelpad=15)

plt.title("BA Graph with Eccentricity-based Coloring")
plt.show()

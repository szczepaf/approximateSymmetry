import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 1. Create a BA graph with 100 vertices and parameter k=3
G = nx.barabasi_albert_graph(n=100, m=3)

# 2. Compute centralities
betweenness = nx.betweenness_centrality(G)
eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
degree = nx.degree_centrality(G)  # Already normalized between 0 and 1
clustering = nx.clustering(G)  # Clustering coefficients
eccentricity = nx.eccentricity(G)


# 3. Create a function to normalize values to [0, 1]
def normalize_dict_values(d):
    values = np.array(list(d.values()))
    vmin, vmax = values.min(), values.max()
    if vmax == vmin:  # Handle potential edge cases
        return {k: 0.0 for k in d}
    return {k: (val - vmin) / (vmax - vmin) for k, val in d.items()}


# 4. Normalize all except degree centrality (already between 0 and 1)
betweenness_norm = normalize_dict_values(betweenness)
eigenvector_norm = normalize_dict_values(eigenvector)
degree_norm = normalize_dict_values(degree)
clustering_norm = normalize_dict_values(clustering)
ecc_normed = normalize_dict_values(eccentricity)

# 5. Extract the normalized values for plotting
betweenness_vals = list(betweenness_norm.values())
eigenvector_vals = list(eigenvector_norm.values())
degree_vals = list(degree_norm.values())
clustering_vals = list(clustering_norm.values())
ecc_vals = list(ecc_normed.values())

# 6. Plot each distribution in a separate subplot
fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(6, 12), sharex=True)
bins = np.linspace(0, 1, 50)  # 50 bins from 0 to 1

axes[0].hist(betweenness_vals, bins=bins, alpha=0.7)
axes[0].set_title("Betweenness")
axes[0].set_ylabel("Frequency")

axes[1].hist(eigenvector_vals, bins=bins, alpha=0.7)
axes[1].set_title("Eigenvector")
axes[1].set_ylabel("Frequency")

axes[2].hist(degree_vals, bins=bins, alpha=0.7)
axes[2].set_title("Degree")
axes[2].set_ylabel("Frequency")

axes[3].hist(clustering_vals, bins=bins, alpha=0.7)
axes[3].set_title("Clustering")
axes[3].set_ylabel("Frequency")
axes[3].set_xlabel("Normalized Centrality")

axes[4].hist(ecc_vals, bins=bins, alpha=0.7)
axes[4].set_title("Eccentricity")
axes[4].set_ylabel("Frequency")
axes[4].set_xlabel("Normalized Centrality")


plt.tight_layout()
plt.show()

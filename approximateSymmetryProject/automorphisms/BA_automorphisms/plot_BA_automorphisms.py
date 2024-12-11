import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('BA_automorphisms.csv')
unique_ks = df['k'].unique()

# Create one plot for each k in separate windows
for k_val in unique_ks:
    # Filter data for this k
    sub_df = df[df['k'] == k_val].sort_values('n')
    
    # Create a new figure
    plt.figure(figsize=(8, 5))
    
    plt.bar(sub_df['n'].astype(str), sub_df['average_group_size'])

    
    # Labeling
    plt.title(f"k = {k_val}")
    plt.xlabel('n')
    plt.ylabel('Average Group Size.')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

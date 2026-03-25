import matplotlib.pyplot as plt
import numpy as np

# Data from Baselines
baselines = ['Random', 'CoreSet', 'Entropy', 'BALD', 'DIAL-style', 'AD-KUCS (Model A)', 'Wang-style', 'Model B (No Safe Control)']
miou_scores = [0.7075, 0.7020, 0.7026, 0.7102, 0.7114, 0.7104, 0.6828, 0.6287]

# Colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

plt.figure(figsize=(12, 7))
bars = plt.bar(baselines, miou_scores, color=colors, alpha=0.8)

# Add baseline line for Random
plt.axhline(y=0.7075, color='black', linestyle='--', alpha=0.5, label='Random Baseline')

# Customize
plt.title('Baseline Comparison under Extreme Imbalance (42:1)', fontsize=14, fontweight='bold')
plt.ylabel('Validation mIoU', fontsize=12)
plt.ylim(0.6, 0.73)
plt.xticks(rotation=45, ha='right')

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f'{yval:.4f}', ha='center', va='bottom', fontsize=10)

plt.legend()
plt.tight_layout()
plt.savefig('AAL-SD-Doc/figures/baseline_comparison.png', dpi=300)
print("Saved AAL-SD-Doc/figures/baseline_comparison.png")

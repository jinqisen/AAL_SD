import matplotlib.pyplot as plt
import numpy as np

# Data from A1 experiment and previous baseline runs
lambdas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

# Previous sweep (42:1 original imbalance)
original_42_1 = [0.6776, 0.6657, 0.6747, 0.6811, 0.6655, 0.6710]

# New sweep (10:1 imbalance)
imbalance_10_1 = [0.7128, 0.6913, 0.6918, 0.7087, 0.6829, 0.6929]

# New sweep (1:1 balanced)
balanced_1_1 = [0.6031, 0.6402, 0.6033, 0.5997, 0.6060, 0.6115]

plt.figure(figsize=(10, 6))

plt.plot(lambdas, original_42_1, marker='o', linewidth=2, label='Original (42:1) - Range: 1.56%')
plt.plot(lambdas, imbalance_10_1, marker='s', linewidth=2, label='Undersampled (10:1) - Range: 2.99%')
plt.plot(lambdas, balanced_1_1, marker='^', linewidth=2, label='Balanced (1:1) - Range: 4.05%')

plt.title('Lambda Optimization Landscape across Imbalance Ratios', fontsize=14, fontweight='bold')
plt.xlabel('Lambda ($\lambda$)', fontsize=12)
plt.ylabel('Validation mIoU', fontsize=12)
plt.xticks(lambdas)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)

# Add annotations for max/min difference
plt.annotate('Flat landscape\n(Isotropic Bottleneck)', xy=(0.5, 0.67), xytext=(0.3, 0.65),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
             fontsize=10, ha='center')

plt.annotate('Curvature emerges\nwhen balanced', xy=(0.2, 0.6402), xytext=(0.4, 0.62),
             arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=6),
             fontsize=10, ha='center', color='green')

plt.tight_layout()
plt.savefig('AAL-SD-Doc/figures/lambda_landscape_imbalance.png', dpi=300)
print("Saved AAL-SD-Doc/figures/lambda_landscape_imbalance.png")

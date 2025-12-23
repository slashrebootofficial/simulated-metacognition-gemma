import matplotlib.pyplot as plt
import numpy as np

# ==============================
# Data (adjust if your parsed numbers differ slightly)
# ==============================
metrics = [
    'Response Length\n(words)',
    'Sentiment Polarity',
    'Table/Bullet Usage (%)',
    'Reflectivity Notes (%)',
    'Lexical Diversity (TTR)'
]

vanilla_values = [450, 0.25, 45, 5, 0.55]
lumen_values   = [620, 0.40, 75, 60, 0.58]

# Compute percentage differences (for Figure 2 only)
differences = [(l - v) / v * 100 if v != 0 else 0 for v, l in zip(vanilla_values, lumen_values)]

# ==============================
# Figure 1: Side-by-Side Absolute Values (NO percentages)
# ==============================
x = np.arange(len(metrics))
width = 0.35

fig1, ax1 = plt.subplots(figsize=(12, 7))

bars1 = ax1.bar(x - width/2, vanilla_values, width, label='Vanilla', color='#1f77b4', edgecolor='black', alpha=0.9)
bars2 = ax1.bar(x + width/2, lumen_values,   width, label='Lumen',   color='#ff7f0e', edgecolor='black', alpha=0.9)

# Only raw value labels on top of bars
for bars in (bars1, bars2):
    for bar in bars:
        height = bar.get_height()
        if height < 1:  # Sentiment & TTR
            label = f'{height:.2f}'
        else:
            label = f'{height:.0f}' if height >= 10 else f'{height:.1f}'
        ax1.annotate(label,
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5),
                     textcoords="offset points",
                     ha='center', va='bottom',
                     fontsize=10, fontweight='bold')

ax1.set_ylabel('Average Value', fontsize=14)
ax1.set_title('Metric Comparison: Vanilla vs. Lumen (GPT-OSS:120b)', fontsize=16, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(metrics, fontsize=11)
ax1.legend(fontsize=12, loc='upper left')
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.set_axisbelow(True)

plt.tight_layout()
plt.savefig('figure1_absolute_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ==============================
# Figure 2: Percentage Improvement (extended x-axis for 1100%)
# ==============================
# Sort descending for visual impact
sorted_indices = np.argsort(differences)[::-1]
sorted_metrics = [metrics[i] for i in sorted_indices]
sorted_diffs   = [differences[i] for i in sorted_indices]

fig2, ax2 = plt.subplots(figsize=(11, 6))  # Slightly wider

bars = ax2.barh(sorted_metrics, sorted_diffs,
                color='green', edgecolor='black', alpha=0.9)

# Extend x-axis far enough for the largest bar (+1100%)
ax2.set_xlim(0, max(sorted_diffs) * 1.15)  # 15% extra headroom

# Percentage labels — inside the bar if space, otherwise just outside
for bar in bars:
    width = bar.get_width()
    # Place label inside if bar is long enough, else outside
    if width > 200:  # arbitrary threshold for readability
        label_x = width - (max(sorted_diffs) * 0.02)
        ha = 'right'
        color = 'white'
    else:
        label_x = width + (max(sorted_diffs) * 0.01)
        ha = 'left'
        color = 'black'
    ax2.text(label_x, bar.get_y() + bar.get_height()/2,
             f'{width:.1f}%',
             ha=ha, va='center',
             fontsize=11, fontweight='bold', color=color)

ax2.set_xlabel('Percentage Improvement (Lumen relative to Vanilla)', fontsize=14)
ax2.set_title('Impact of Vector-Based Prompting on Response Quality', fontsize=16, fontweight='bold', pad=20)
ax2.axvline(0, color='black', linewidth=1.2)
ax2.grid(axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('figure2_percentage_improvement.png', dpi=300, bbox_inches='tight')
plt.show()
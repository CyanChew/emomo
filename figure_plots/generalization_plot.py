import matplotlib.pyplot as plt
import numpy as np
font = "Times New Roman"

# Define tasks
tasks = ["Original\nCube", "Small/Large\nCube", "Cube with\nDistractors", "New Cube\nSpecified"]
x = np.arange(len(tasks))
width = 0.15  # Thinner bars

# Success counts (out of 20)
dp_wbc = [9, 6, 9, 6]
homer = [18, 16, 4, 2]
homer_cond = [18, 12, 2, 2]
homer_cond_aug = [16, 14, 15, 16]

# Colors
dp_wbc_color = '#ffd07b'
homer_color = '#fb8b24'
cond_color = '#dc602e'

# Function to draw shadow + actual bar
def draw_bar(xpos, height, width, color, hatch=None, label=None, zorder=2):
    if height > 0:
        plt.bar(xpos, height, width, color=color, hatch=hatch, label=label,
                zorder=zorder)

# Plot — target half-column width, so ~3.5 to 4 inches wide
plt.figure(figsize=(5.5, 2.75))

for i in range(len(x)):
    draw_bar(x[i] - 1.8 * width, dp_wbc[i], width, dp_wbc_color, label='DP (WBC)' if i == 0 else None)
    draw_bar(x[i] - 0.6 * width, homer[i], width, homer_color, label='HoMeR' if i == 0 else None)
    draw_bar(x[i] + 0.6 * width, homer_cond[i], width, cond_color, hatch=None, label='HoMeR-Cond-NoAugs' if i == 0 else None)
    draw_bar(x[i] + 1.8 * width, homer_cond_aug[i], width, cond_color, hatch='//', label='HoMeR-Cond' if i == 0 else None)

# Annotate bar tops
for i, values in enumerate(zip(dp_wbc, homer, homer_cond, homer_cond_aug)):
    for j, val in enumerate(values):
        if val > 0:
            xpos = x[i] + (j * 1.2 - 1.8) * width
            plt.text(xpos, val + 0.5, str(val), ha='center', va='bottom',
                     fontsize=14, fontname=font, weight='bold' if val == max(values) else 'normal')

# Formatting
plt.xticks(x, tasks, fontname=font, fontsize=14)
plt.ylabel('Success Rate / 20', fontname=font, fontsize=14)
plt.yticks(fontsize=14, fontname=font)
plt.ylim(0, 22)
#plt.title('Generalization Results', fontname=font, fontsize=14)

# Legend: smaller font, 2 rows × 2 columns, pushed further down
# Legend: single row again, nudged closer to the figure
plt.legend(
    prop={'family': font, 'size': 14},
    loc='upper center',
    bbox_to_anchor=(0.5, -0.3),
    ncol=2,
    frameon=False
)


plt.tight_layout()

# Clean up plot borders
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save figure
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
# plt.show()


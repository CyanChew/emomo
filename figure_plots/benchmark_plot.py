import matplotlib.pyplot as plt
import numpy as np

# Define tasks in desired order
tasks = ["Cube", "Dishwasher", "Cabinet", "Pillow", "TV Remote", "Sweep Trash"]
x = np.arange(len(tasks))
width = 0.2
dp_ba    = [5, 7, 8, 5, 7, 7]
dp_wbc   = [10, 9, 11, 8, 3, 5]
hybrid_ba= [13, 0, 15, 10, 10, 8]
homer    = [19, 11, 18, 16, 15, 16]

# Bar colors
colors = ['#7A4C92', '#D2691E', '#B07BA4', 'orange']
labels = ['DP (B+A)', 'DP (WBC)', 'HoMeR (B+A)', 'HoMeR']

# Function to draw a bar with shadow and black edge
def draw_bar(xpos, height, width, color, label=None, zorder=2):
    # Draw main bar with black edge
    plt.bar(xpos, height, width, color=color, edgecolor='black',
            label=label, zorder=zorder)

# Create plot
plt.figure(figsize=(10, 3))

# Draw all bars
for i in range(len(x)):
    draw_bar(x[i] - 1.5 * width, dp_ba[i], width, colors[0], label=labels[0] if i == 0 else None)
    draw_bar(x[i] - 0.5 * width, dp_wbc[i], width, colors[1], label=labels[1] if i == 0 else None)
    draw_bar(x[i] + 0.5 * width, hybrid_ba[i], width, colors[2], label=labels[2] if i == 0 else None)
    draw_bar(x[i] + 1.5 * width, homer[i], width, colors[3], label=labels[3] if i == 0 else None)

# Annotate values on top
for i, values in enumerate(zip(dp_ba, dp_wbc, hybrid_ba, homer)):
    for j, val in enumerate(values):
        if val > 0:
            xpos = x[i] + (j - 1.5) * width
            plt.text(xpos, val + 0.5, str(val), ha='center', va='bottom',
                     fontsize=14, fontname='Menlo', weight='bold' if j == 3 else 'normal')

# Formatting
plt.xticks(x, tasks, fontname='Menlo', fontsize=14)
plt.ylabel('Success Rate / 20', fontname='Menlo', fontsize=14)
plt.yticks(fontsize=12, fontname='Menlo')
plt.ylim(0, 22)
#plt.title('Benchmark Evaluation', fontname='Menlo', fontsize=14)

# Legend
plt.legend(prop={'family': 'Menlo', 'size': 14}, loc='upper center',
           bbox_to_anchor=(0.5, -0.25), ncol=4, frameon=False)

plt.tight_layout()

# Clean up borders
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save figure
plt.savefig('plot.png', dpi=300)
# plt.show()


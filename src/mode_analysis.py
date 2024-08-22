import os
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# 5 9 8 1 7 7 8 1 0 2 6 8 3 1 2 5 3 6 8 5 5 3 8 5 1 3 5 7 9 7 9 0 6 3 9 2 0 6 2 7 8 3 5 3 0 5 7 9 2 0 2 0 1 1 3 4 1 7 1 0 2 6 6 1 4 5 2 0 2 1 9 7 7 7 0 3 0 4 3 8 3 2 5 2 9 5 7 9 7 0 1 9 0 9 7 9 1 9 9 6 6 0 1 1 7 1 7 2 1 8 3 5 6 5 2 9 7 2 3 3 5 5 0 5 0 5 2 2 9 3 5 3 9 0 5 3 7 5 9 0 4 0 8 7 1 3 1 1 3 3 9 5 9 7 4 8 8 6 2 9 7 1 1 4 5 3 6 5 6 0 2 5 9 9 5 9 0 3 2 7 4 5 9 1 2 5 2 6 3 6 6 3 5 1 5 2 6 9 2 7 7 9 5 0 3 1 0 5 1 7 6 2 6 5 9 2 2 6 5 3 9 7 6 5 7 5 7 7 9 6 7 5 1 1 7 5 3 9 9 5 9 3 2 5 7 4 7 7 1 6 9 8 3 9 9 1 
predictions = [5, 9, 8, 1, 7, 7, 8, 1, 0, 2, 6, 8, 3, 1, 2, 5, 3, 6, 8, 5, 
5, 3, 8, 5, 1, 3, 5, 7, 9, 7, 9, 0, 6, 3, 9, 2, 0, 6, 2, 7, 8, 
3, 5, 3, 0, 5, 7, 9, 2, 0, 2, 0, 1, 1, 3, 4, 1, 7, 1, 0, 2, 6, 
6, 1, 4, 5, 2, 0, 2, 1, 9, 7, 7, 7, 0, 3, 0, 4, 3, 8, 3, 2, 5, 
2, 9, 5, 7, 9, 7, 0, 1, 9, 0, 9, 7, 9, 1, 9, 9, 6, 6, 0, 1, 1, 
7, 1, 7, 2, 1, 8, 3, 5, 6, 5, 2, 9, 7, 2, 3, 3, 5, 5, 0, 5, 0, 
5, 2, 2, 9, 3, 5, 3, 9, 0, 5, 3, 7, 5, 9, 0, 4, 0, 8, 7, 1, 3, 
1, 1, 3, 3, 9, 5, 9, 7, 4, 8, 8, 6, 2, 9, 7, 1, 1, 4, 5, 3, 6, 
5, 6, 0, 2, 5, 9, 9, 5, 9, 0, 3, 2, 7, 4, 5, 9, 1, 2, 5, 2, 6, 
3, 6, 6, 3, 5, 1, 5, 2, 6, 9, 2, 7, 7, 9, 5, 0, 3, 1, 0, 5, 1, 
7, 6, 2, 6, 5, 9, 2, 2, 6, 5, 3, 9, 7, 6, 5, 7, 5, 7, 7, 9, 6, 
7, 5, 1, 1, 7, 5, 3, 9, 9, 5, 9, 3, 2, 5, 7, 4, 7, 7, 1, 6, 9, 
8, 3, 9, 9, 1
]
image_dir = "manual_samples"
# Create a figure and axis
fig, ax = plt.subplots()

# Count predictions
prediction_counts = [predictions.count(i) for i in range(10)]

# Plot the bar chart
bars = ax.bar(range(10), prediction_counts, color='black')

# Function to add images as annotations
def add_image_annotation(ax, image_path, xy, image_size=0.4):
    image = plt.imread(image_path)
    imagebox = OffsetImage(image, zoom=image_size)
    ab = AnnotationBbox(imagebox, xy, frameon=False, pad=0)
    ax.add_artist(ab)

# Add images as annotations
for i in range (10):
    for j in range(len(predictions)):
        if predictions[j] == i:
            image_path = os.path.join(image_dir, "ms_{}.png".format(j))
            prediction_counts[i] = prediction_counts[i] - 1
            add_image_annotation(ax, image_path, (i, prediction_counts[i] + 0.4))

# Set labels and title
ax.set_xlabel('Predictions')
ax.set_ylabel('Count')
ax.set_title('Mode Analysis')

# Set x-axis ticks and labels
ax.set_xticks(range(10))
ax.set_xticklabels(range(10))

# Show the plot
plt.show()
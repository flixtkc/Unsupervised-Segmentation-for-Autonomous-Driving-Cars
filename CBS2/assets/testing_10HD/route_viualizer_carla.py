#%%
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Parse the XML file
tree = ET.parse('routes_10hd.xml')
root = tree.getroot()

# Extract waypoints
waypoints = []
for waypoint in root.find('route').findall('waypoint'):
    x = float(waypoint.get('x'))
    y = float(waypoint.get('y'))
    waypoints.append((x, y))

# Load the map image
map_img = mpimg.imread('../maps/Town10HD/map.png')

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(map_img, extent=[0, map_img.shape[1], 0, map_img.shape[0]])

# Adjust waypoints to match the image coordinates
# Assuming you know the transformation from CARLA coordinates to image coordinates
# You may need to adjust these values based on your specific map image and CARLA map
x_offset = 180  # Adjust this value
y_offset = 170  # Adjust this value
x_scale = 1.9  # Adjust this value
y_scale = 1.9  # Adjust this value

# Unpack waypoints into x and y coordinates
x_coords, y_coords = zip(*waypoints)

# Transform waypoints
x_coords_transformed = [(x * x_scale) + x_offset for x in x_coords]
y_coords_transformed = [(y * y_scale) + y_offset for y in y_coords]

# Plot the waypoints on the map
ax.plot(x_coords_transformed, y_coords_transformed, marker='o', linestyle='-', color='b')

# Display the plot
plt.title('Route Visualization on Map')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.grid(True)
plt.axis('equal')
plt.show()


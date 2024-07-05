import carla

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# List all available maps
available_maps = client.get_available_maps()
print("Available maps:", available_maps)

# Try to load Town10HD
try:
    world = client.load_world('Town10HD')
    print("Successfully loaded Town10HD")
except RuntimeError as e:
    print("Error loading Town10HD:", e)

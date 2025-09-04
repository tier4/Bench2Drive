import carla
client = carla.Client('localhost', 2000)
client.set_timeout(100.0)
list_map = client.get_available_maps()
client_version = client.get_client_version()
server_version = client.get_server_version()
world = client.get_world()
world = client.load_world('Town02')

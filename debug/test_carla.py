import carla
import sys

try:
    # Connect to CARLA
    print("Connecting to CARLA...")
    client = carla.Client("carla", 2000)
    client.set_timeout(10.0)

    # Get world info
    world = client.get_world()
    settings = world.get_settings()

    print(f"✅ CARLA connected successfully!")
    print(f"Map: {world.get_map().name}")
    print(f"Weather: {world.get_weather()}")

    # Test spawning a vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter("vehicle.*")[0]
    print(f"✅ Found vehicle blueprint: {vehicle_bp.id}")

    print("✅ All tests passed - CARLA is working!")

except Exception as e:
    print(f"❌ CARLA test failed: {e}")
    sys.exit(1)


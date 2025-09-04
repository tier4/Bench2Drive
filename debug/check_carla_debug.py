#!/usr/bin/env python3
import sys
import os

print("=== CARLA Debug Information ===")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path}")
print()

# Check environment variables
print("=== Environment Variables ===")
print(f"CARLA_ROOT: {os.environ.get('CARLA_ROOT', 'Not set')}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
print()

# Try to import carla
print("=== Attempting to import carla ===")
try:
    import carla
    print("SUCCESS: carla module imported")
    print(f"carla module location: {carla.__file__}")
    
    # Try to create client
    print("\n=== Attempting to create CARLA client ===")
    try:
        client = carla.Client('localhost', 2000)
        print("SUCCESS: Client created")
        
        # Try to set timeout
        print("Setting timeout...")
        client.set_timeout(10.0)
        print("SUCCESS: Timeout set")
        
        # Try to get version
        print("\nGetting version information...")
        try:
            client_version = client.get_client_version()
            print(f"Client version: {client_version}")
        except Exception as e:
            print(f"ERROR getting client version: {e}")
            
        try:
            server_version = client.get_server_version()
            print(f"Server version: {server_version}")
        except Exception as e:
            print(f"ERROR getting server version: {e}")
            
    except Exception as e:
        print(f"ERROR creating client: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError as e:
    print(f"ERROR: Failed to import carla module: {e}")
    print("\nPlease ensure:")
    print("1. CARLA Python API is properly installed")
    print("2. Python version matches CARLA requirements (3.7 or 3.8 for CARLA 0.9.15)")
    print("3. PYTHONPATH includes CARLA egg file")
except Exception as e:
    print(f"UNEXPECTED ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
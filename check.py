import pyopencl as cl

def list_devices():
    for platform in cl.get_platforms():
        print(f"Platform: {platform.name}")
        for device in platform.get_devices():
            print(f"  Device: {device.name}")

list_devices()
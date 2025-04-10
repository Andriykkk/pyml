class Device:
    def __init__(self, device_type ="cpu"):
        self.type = device_type.lower()

        if self not in ["cpu", "gpu", 'cuda', 'amd']:
            raise ValueError("Invalid device type")
        
    def __eq__(self, other):
        if isinstance(other, Device):
            return self.type == other.type
        elif isinstance(other, str):
            return self.type == other.lower()
        return False
    
    def __str__(self):
        return self.type

    def __repr__(self):
        return f'Device(type={self.type})'
    
    @staticmethod
    def is_available(device_type="cpu"):
        """Check if a device is available."""
        device_type = device_type.lower()
        if device_type not in ["cpu"]:
            return False
        return True
    

class DeviceManager:
    _devices = {}

    @classmethod
    def get_device(cls, device_type):
        device_type = device_type.lower()
        if device_type not in cls._devices:
            cls._devices[device_type] = Device(device_type)
        return cls._devices[device_type]

    @classmethod
    def reset(cls):
        """Reset all devices."""
        cls._devices.clear()

    @classmethod
    def is_available(cls, device_type):
        return Device.is_available(device_type)
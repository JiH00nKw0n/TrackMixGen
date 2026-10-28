#!/usr/bin/env python3
"""Enumerate actual Vulkan devices without installing a driver or starting a simulation."""
import ctypes as c
import json


class ApplicationInfo(c.Structure):
    _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p), ("pApplicationName", c.c_char_p),
                ("applicationVersion", c.c_uint32), ("pEngineName", c.c_char_p),
                ("engineVersion", c.c_uint32), ("apiVersion", c.c_uint32)]


class InstanceInfo(c.Structure):
    _fields_ = [("sType", c.c_uint32), ("pNext", c.c_void_p), ("flags", c.c_uint32),
                ("pApplicationInfo", c.POINTER(ApplicationInfo)), ("enabledLayerCount", c.c_uint32),
                ("ppEnabledLayerNames", c.c_void_p), ("enabledExtensionCount", c.c_uint32),
                ("ppEnabledExtensionNames", c.c_void_p)]


def main():
    c.CDLL("libGLX_nvidia.so.0")
    vk = c.CDLL("libvulkan.so.1")
    app = ApplicationInfo(0, None, b"tmg-probe", 1, b"none", 1, (1 << 22) | (2 << 12))
    info = InstanceInfo(1, None, 0, c.pointer(app), 0, None, 0, None)
    instance = c.c_void_p()
    vk.vkCreateInstance.argtypes = [c.POINTER(InstanceInfo), c.c_void_p, c.POINTER(c.c_void_p)]
    rc = vk.vkCreateInstance(c.byref(info), None, c.byref(instance))
    if rc != 0:
        raise RuntimeError(f"vkCreateInstance returned {rc}")
    vk.vkEnumeratePhysicalDevices.argtypes = [c.c_void_p, c.POINTER(c.c_uint32), c.c_void_p]
    vk.vkGetPhysicalDeviceProperties.argtypes = [c.c_void_p, c.c_void_p]
    vk.vkDestroyInstance.argtypes = [c.c_void_p, c.c_void_p]
    try:
        count = c.c_uint32()
        rc = vk.vkEnumeratePhysicalDevices(instance, c.byref(count), None)
        if rc != 0 or count.value == 0:
            raise RuntimeError(f"Vulkan found no usable GPU (rc={rc})")
        devices = (c.c_void_p * count.value)()
        rc = vk.vkEnumeratePhysicalDevices(instance, c.byref(count), devices)
        if rc != 0:
            raise RuntimeError(f"Vulkan enumeration returned {rc}")
        names = []
        for device in devices:
            properties = c.create_string_buffer(4096)
            vk.vkGetPhysicalDeviceProperties(device, properties)
            names.append(properties.raw[20:276].split(b"\0")[0].decode())
        if not any("NVIDIA" in name or "RTX" in name for name in names):
            raise RuntimeError(f"NVIDIA GPU not found: {names}")
        print(json.dumps({"vulkan_devices": names, "passed": True, "simulation_tested": False}))
    finally:
        vk.vkDestroyInstance(instance, None)


if __name__ == "__main__":
    main()

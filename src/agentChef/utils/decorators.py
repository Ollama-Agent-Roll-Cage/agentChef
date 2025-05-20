"""Utility decorators for AgentChef."""

def singleton(cls):
    """Decorator to create a singleton class."""
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

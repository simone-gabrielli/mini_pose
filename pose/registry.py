# pose/registry.py

MODEL_REGISTRY = {}
LOSS_REGISTRY = {}
DATASET_REGISTRY = {}


def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def register_loss(name):
    def decorator(cls):
        LOSS_REGISTRY[name] = cls
        return cls
    return decorator


def register_dataset(name):
    def decorator(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return decorator

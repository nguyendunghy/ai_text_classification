from pathlib import Path


def load_module(module_filename: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(__name__, str(module_filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

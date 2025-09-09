from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ANT")  # must match [project].name in pyproject.toml
except PackageNotFoundError:
    __version__ = "0.0.0"

from .realtime_nf import NFRealtime
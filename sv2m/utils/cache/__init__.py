import os


def get_cache_dir() -> str:
    default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "sv2m")
    cache_dir = os.getenv("SV2M_CACHE_DIR") or default_cache_dir

    return cache_dir

from pathlib import Path


def validate_beta9_credentials():
    """
    Validate that Beta9 credentials exist by checking for ~/.beta9/config.ini file.
    Raises an exception if credentials are not configured.
    """
    beta9_config_path = Path.home() / ".beta9/config.ini"
    if not beta9_config_path.exists():
        raise RuntimeError(
            "~/.beta9/config.ini not found - it looks like you haven't configured credentials for Beta9.\n"
            "Run 'beta9 login' in your terminal to configure credentials."
        )

def run_instances_beta9(*args, **kwargs):
    """Lazy import wrapper for run_instances_beta9."""
    from swebench.harness.beta9_eval.run_evaluation_beta9 import run_instances_beta9 as _run_instances_beta9
    return _run_instances_beta9(*args, **kwargs)


def validate_beta9_credentials():
    """Lazy import wrapper for validate_beta9_credentials."""
    from swebench.harness.beta9_eval.utils import validate_beta9_credentials as _validate_beta9_credentials
    return _validate_beta9_credentials()


__all__ = [
    "run_instances_beta9",
    "validate_beta9_credentials",
]

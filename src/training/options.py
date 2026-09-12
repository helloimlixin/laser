"""Translate compound-backend YAML options to its existing argument parser."""


def options_to_argv(options: dict) -> list[str]:
    argv = []
    for key, value in options.items():
        flag = "--" + key.replace("_", "-")
        if value is None:
            continue
        if isinstance(value, bool):
            argv.append(flag if value else "--no-" + flag[2:])
        elif isinstance(value, list):
            argv.extend([flag, *map(str, value)])
        elif isinstance(value, dict):
            raise ValueError(f"Compound option {key!r} must be a scalar or list.")
        else:
            argv.extend([flag, str(value)])
    return argv

"""SDV add-ons functionality."""
import warnings

from pkg_resources import iter_entry_points


def _find_addons(group, parent_globals):
    """Find and load add-ons based on the given group.

    Args:
        group (str):
            The name of the entry points group to load.
        parent_globals (dict):
            The caller's global scope. Modules will be added
            to the parent's global scope through their name.
    """
    for entry_point in iter_entry_points(group=group):
        try:
            module = entry_point.load()
        except Exception:
            msg = f'Failed to load "{entry_point.name}" from "{entry_point.module}".'
            warnings.warn(msg)
            continue

        if entry_point.name not in parent_globals:
            parent_globals[entry_point.name] = module

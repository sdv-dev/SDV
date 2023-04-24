"""SDV add-ons functionality."""
import warnings

from pkg_resources import iter_entry_points


def _find_addons(group, parent_globals, add_all=False):
    """Find and load add-ons based on the given group.

    Args:
        group (str):
            The name of the entry points group to load.
        parent_globals (dict):
            The caller's global scope. Modules will be added to the parent's global scope through
            their name.
        add_all (bool):
            Whether to also add everything in the add-on's ``module.__all__`` to the parent's
            global scope. Defaults to ``False``.
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

        if add_all:
            try:
                for entry in module.__all__:
                    if entry not in parent_globals:
                        parent_globals[entry] = getattr(module, entry)

                new_all = set(parent_globals.get('__all__', ())) | set(module.__all__)
                parent_globals['__all__'] = tuple(new_all)
            except AttributeError:
                continue

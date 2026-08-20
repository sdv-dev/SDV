"""Mixin classes for general use."""


class MissingModuleMixin:
    """Mixin for raising a custom error message when a module is not found."""

    @classmethod
    def raise_module_not_found_error(cls, error):
        """Takes in an existing ModuleNotFoundError and raises a new one with custom text."""
        raise ModuleNotFoundError(
            f"{error.msg}. Please install {error.name} in order to use the '{cls.__name__}'."
        )

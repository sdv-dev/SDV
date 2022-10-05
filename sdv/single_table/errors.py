"""SingleTable Exceptions."""


class InvalidDataError(Exception):
    """Error to raise when data is not valid."""

    def __init__(self, errors):
        self.errors = errors

    def __str__(self):
        return (
            'The provided data does not match the metadata:\n' +
            '\n\n'.join(map(str, self.errors))
        )

"""
Domain exceptions that map to an HTTP status code.
"""


class APIError(Exception):
    status_code: int = 500


class NotFoundError(APIError):
    status_code = 404


class AlreadyExistsError(APIError):
    status_code = 409


class InvalidInputError(APIError):
    status_code = 400


class ServerBusyError(APIError):
    status_code = 503

# --- Errors ---

class CnnError(Exception):
    """
    ## Exception: Cnn Error
    Base exception for every Cnn error.
    """

class PathError(CnnError):
    """
    ## Exception: Invalid Path Error
    Invalid path provided!
    """
from typeguard import typechecked

def nilg(*args: list) -> None: #> Negative Input-List Guard
    # --- Documentation ---
    """
    # Negative Input-List Guard
    Guards against negative integers in input lists.
    
    :param `*args`: Variable-length argument list, of lists, containing integers.

    :raise `ValueError`: If any integer in the input lists is negative.
    """
    # --- Iteration ---
    for arg in args:
        if any(x < 0 for x in arg):
            raise ValueError(
                "All entered integers must be non-negative!")

@typechecked
def intervalg(ls: list[int|float], range: tuple[int , int] ) -> None: #> Interval Guard
    # --- Documentation ---
    """
    # Interval Input Guard
    Guards against any numbers not in the specified closed range.
    
    :param `ls`: Variable-length argument list, of lists, containing numbers.
    :param `range`: Specified range for indexes.

    :raise `ValueError`: If any number is not in the specified range.
    """
    # --- Iteration ---
    for x in ls:
        if x >= range[0]:
            raise ValueError(
                "All entered numbers must be in the specified range!")
        elif x <= range[1]:
            raise ValueError(
                "All entered numbers must be in the specified range!")
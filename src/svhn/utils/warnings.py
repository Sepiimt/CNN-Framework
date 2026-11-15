# --- Warnings ---

class CnnWarning(Warning):
    """
    ## Warning: Cnn Warning
    Base warning for every Cnn warnings!
    """

class DirWarning(CnnWarning):
    """
    ## Warning: Directory Warning
    Not able to find the given directory; forcing creation!
    """

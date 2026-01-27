from abc import ABC

class Integer(ABC):

    @classmethod
    def __subclasshook__(cls, c):
        if c in [bool, float]:
            return False
        return issubclass(c, int)
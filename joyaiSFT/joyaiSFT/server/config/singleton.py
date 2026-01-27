import abc

class Singleton(abc.ABCMeta, type):
    """_summary_

    Args:
        abc.ABCMeta: Provide a mechanism for defining abstract methods and properties,
            enforcing subclasses to implement these methods and properties.
        type: Inherit from 'type' to make 'Singleton' a metaclass,
            enabling the implementation of the Singleton
    """
    _instances = {}

    def __call__(cls, *args, **kwds):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwds)
        return cls._instances[cls]

class AbstractSingleton(abc.ABC, metaclass=Singleton):
    """Provided an abstract Singleton base class, any class inheriting from
       this base class will automatically become a Singleton class.

    Args:
        abc.ABC: Abstract base class, it cannot be instantiated, only inherited. 
    """
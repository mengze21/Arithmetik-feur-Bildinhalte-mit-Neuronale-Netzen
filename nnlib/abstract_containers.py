# Abstract container base classes.
# Oursourced from nnlib.nnlib with mo_05 to make the files smaller.
__version__ = 'ac_01'

# Note: This submodule should not import contents of nnlib to avoid
# circular imports.
import six
from abc import ABCMeta, abstractmethod


def is_cont_obj(testobj):
    try:
        testobj.pull('type')
        testobj.pull('id')
        return True
    except AttributeError:
        return False


@six.add_metaclass(ABCMeta)
class AbstractContainer:

    @abstractmethod
    def provide_deduplicator(self):
        """This function needs to provide a deduplicator instance that
        fits the object that creates it."""
        pass

    def push(self, target, data):
        """Pushes data (provided via the 'data' argument) into the
        Container (to fields specified via the 'target' argument).  The
        precise action taken by this function is specified in
        loadstore_instr, which every implementation of
        AbstractContainer needs to provide for this 'push'
        implementation.
        """
        self.loadstore_instr[target][1](data)

    def pull(self, target, dpcopy=None):
        """Pulls information out of the container.  This function's
        actions are governed by the contents of loadstore_instr
        associated with the given 'target' key.  The 'dpcopy' argument
        is interpreted as a boolean.  If True, the functions must
        provide exact copies of the data they deliver instead of
        references to the data itself.  This is relevant in cases where
        the returned values are mutable.
        Both modes have their distinct advantages.  Copying provides
        data fields that are guaranteed to be independent of their
        source, while not copying them saves a lot of time and RAM.
        Each entry in loadstore_instr provides its own standard setting
        that makes the most sense in most use cases that is used if
        'copy' is not externally specified.
        """
        if target == 'id':
            return id(self)
        elif target == 'type':
            return type(self).__name__
        else:
            if dpcopy is None:
                dpcopy = self.loadstore_instr[target][2]
            return self.loadstore_instr[target][0](dpcopy)


@six.add_metaclass(ABCMeta)
class AbstractContainerGroup:

    @abstractmethod
    def register(self, source, dest):
        pass

    @abstractmethod
    def check(self, data):
        pass

    @abstractmethod
    def create_instance(self, cont):
        pass
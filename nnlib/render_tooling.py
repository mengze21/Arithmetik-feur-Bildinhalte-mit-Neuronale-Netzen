# Abstract render interface base class, machinery, and dummy implementation.
# Oursourced from nnlib.nnlib with mo_05 to make the files smaller.
__version__ = 'rt_01'

import six
from abc import ABCMeta, abstractmethod

from .support_lib import NNLibUsageError


@six.add_metaclass(ABCMeta)
class RenderInterface:
    """Outsourced from the old creator classes, classes of this type
    provide an interface to the actual renderers.  This became useful
    because, with nnlib_multiobject_v06, there are now two
    substantially different renderer solutions being supported, both of
    which can be addressed with a common interface through this class.
    """

    def __init__(self, array_access):
        self.dtarget = array_access

    @abstractmethod
    def check(array_access, thread_num, field_size, field_depth):
        pass

    def submit_obj(self, coords, shape, presence):
        pass

    def finalize(self):
        pass

    def close(self):
        pass


_render_ifs = []


def register_render_if(if_class):
    _render_ifs.append(if_class)


def select_render_if(array_access, thread_num, field_size, field_depth):
    for interface in _render_ifs:
        built_if = interface.check(array_access, thread_num,
                                   field_size, field_depth)
        if built_if:
            return built_if
    else:
        raise NNLibUsageError("Conditions unsatisfiable. No interface built.")


class DummyInterface(RenderInterface):
    """Just a dummy interface for bugfixing purposes."""

    def check(array_access, thread_num, field_scope, field_depth):
        if field_depth == 0:
            return DummyInterface()
        else:
            return None

    def __init__(self):
        pass


register_render_if(DummyInterface)
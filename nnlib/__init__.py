# Determine if we run interactively or not.  This is then exposed to all submodules.
import __main__ as main
_interactive = not hasattr(main, '__file__')
# Load the submodules
from nnlib import handlers, build_recipes, mo_renderer_new, talking_to_bpy
# Load the public interfaces
from nnlib.handlers import load, ModelHandler, SampleContainer
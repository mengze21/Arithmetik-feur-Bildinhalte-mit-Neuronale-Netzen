__version__ = 'ttbpy_02'

import os
import png
import time
import math
import tempfile
import numpy as np
from pathlib import Path

from .support_lib import nnlib_path, MOD_SETTINGS
from .support_lib import rad_index, ang_index
from .render_tooling import RenderInterface, register_render_if

# For now, set slice_depth as a constant in this submodule. Later to be
# integrated more globally in the render process.
SLICE_DEPTH = 3.0


class BlendContainer(RenderInterface):
    """This class contains all specific interactions with the blender
    save file 'ray_length_setup.blend' that was created to contain the
    necessary presets for our blender renderings.  As such, this class
    is very codependent on that file and one must be extremely careful
    when changing one or the other.
    """

    def check(array_access, thread_num, field_size, field_depth):
        if field_depth > 1:
            return BlendContainer(array_access, thread_num,
                                  field_size, field_depth)
        else:
            return None

    def __init__(self, array_access, thread_num, field_size, field_depth):
        # CHANGED in ttbpy_02: BlendContainer now inherits from
        # RenderInterface, which means the functionality of
        # BlenderInterface and BlendContainer is fused into one.
        super(BlendContainer, self).__init__(array_access)
        # We do a lazy import of bpy because we want to only load it in
        # the worker threads if multiprocessing is enabled.  This is
        # necessary because the import will fail with a strange error
        # if it is imported in both the main and the worker threads.
        # Also, this enables the use of most nnlib functionality
        # without bpy installed.
        import bpy
        self.bpy = bpy
        self.field_size = field_size
        self.field_depth = field_depth
        self.thread_num = thread_num
        # Open the previously prepared blender file.
        self.bpy.ops.wm.open_mainfile(
          filepath=str(nnlib_path / 'blender' / 'ray_length_setup.blend'))
        # Some basic blender setup.
        scn = bpy.data.scenes['Scene']
        scn.render.resolution_x = field_size
        scn.render.resolution_y = field_size
        scn.frame_end = field_depth
        scn["slice_depth"] = 3.0
        # In this module, we work with some temporary files
        self.td = tempfile.TemporaryDirectory()
        td_path = Path(self.td.name)
        if MOD_SETTINGS['DEBUG']:
            # If the debug mode is activated, we write to nnlib's own
            # log directory.
            self.logfile = (
              nnlib_path
              / "log/blender_render_th_{}.log".format(str(thread_num))
            )
        else:
            # If it isn't, we write to temp. In this case, the logs
            # will be deleted shortly after.
            self.logfile = (
              td_path / "blender_render_th_{}.log".format(str(thread_num))
            )
        self.imgbuffer = td_path / "temp_{}_".format(str(thread_num))
        self._clean()

    # Private methods (implementing Blender interactions)

    def _close(self):
        pass

    def _clean(self):
        """Cleans the scene by going through all objects, selecting
        only Meshes and Metaballs, and deleting them.
        """
        for obj in self.bpy.context.scene.objects:
            if obj.type in ('META', 'MESH'):
                obj.select_set(True)
            else:
                obj.select_set(False)
        self.bpy.ops.object.delete()

    def _render(self, array_access):
        """Performs the rendering itself by going through the frames of
        animation, rendering out each, and saving it.
        """
        scn = self.bpy.data.scenes['Scene']
        if MOD_SETTINGS['DEBUG']:
            print("Render started in thread {}: {}".format(self.thread_num,
                                                           time.asctime()))
        # CHANGED: Log to file during the rendering ONLY!
        # Note: File descriptor 1 always points to stdout
        stdout_descriptor = os.dup(1)
        os.close(1)
        os.open(self.logfile, os.O_RDWR | os.O_CREAT)
        scn.render.filepath = str(self.imgbuffer)
        # CHANGED: Render the whole animation at once to reduce render
        # times as much as possible.
        self.bpy.ops.render.render(animation=True)
        # Loop over the generated animation frames to read them back
        # in.  This is necessary because bpy does not (really) support
        # rendering to RAM.
        for frm in range(scn.frame_start, scn.frame_end+1, scn.frame_step):
            ani_file = self.imgbuffer.with_name(
              self.imgbuffer.name + str(frm).zfill(4) + ".png"
            )
            reader = png.Reader(filename=ani_file)
            _, _, values, _ = reader.read_flat()
            ordered_values = np.reshape(values,
                                        (self.field_size,self.field_size,1))
            normalized_values = ordered_values / 255
            array_access(normalized_values,
                         slice(0, None),
                         slice(0, None),
                         frm-scn.frame_start)
        # We had long render jobs sporadically fail here with an
        # OSError 9, indicating a bad file descriptor, meaning there is
        # no stdout opened that could be closed.  I don't currently
        # know how this can happen, but it seems safe to just ignore
        # this error since the next line will reopen stdout.
        try:
            os.close(1)
        except OSError:
            pass
        os.dup(stdout_descriptor)
        os.close(stdout_descriptor)
        if MOD_SETTINGS['DEBUG']:
            print("Render ended on thread {}: {}".format(self.thread_num,
                                                         time.asctime()))

    def _place_obj(self, coords, shape, presence):
        blender_coords = coords[:rad_index] * 10.0
        blender_resize = coords[rad_index:ang_index] * 10.0
        blender_rotate = coords[ang_index:] * math.pi / 2.0
        # Select the object to place.
        if shape == 'rectangle':
            # Variant cube.  Note that we start with a radius
            # (half-length) of 1 in agreement with the definitions in
            # the base parameter set.
            self.bpy.ops.mesh.primitive_cube_add(size=2.0,
                                                 location=blender_coords)
        elif shape == 'ellipse':
            # Variant ellipsoid.
            self.bpy.ops.object.metaball_add(radius=1.0,
                                             location=blender_coords)
        else:
            raise ValueError("Unknown shape specifier.")
        # Blender >=2.9 is unhappy when the area is not properly
        # defined.  To fix this, we prepare a context override.
        # To do so, we copy the native context, search for a suitable
        # area definition and inject it into the override context.
        # Also, it seems necessary to keep context definition as close
        # as possible, so we accept the overhead it causes and build
        # the context new whenever place_obj is run.
        ctxt = self.bpy.context.copy()
        for area in self.bpy.context.window.screen.areas:
            if area.type == 'VIEW_3D':
                ctxt['area'] = area
        self.bpy.ops.transform.resize(ctxt, value=blender_resize)
        self.bpy.ops.transform.rotate(ctxt, value=blender_rotate[0],
                                      orient_axis='Z')
        self.bpy.ops.transform.rotate(ctxt, value=blender_rotate[1],
                                      orient_axis='X')
        self.bpy.ops.transform.rotate(ctxt, value=blender_rotate[2],
                                      orient_axis='Z')
        if presence:
            material = self.bpy.data.materials['ray_len_texture']
        else:
            material = self.bpy.data.materials['neg_ray_len_texture']
        self.bpy.context.object.data.materials.append(material)

    # Public methods (interface implementation)

    def submit_obj(self, coords, shape, presence):
        self._place_obj(coords, shape, presence)

    def finalize(self):
        self._render(self.dtarget)
        self._clean()

    def close(self):
        self._close()


register_render_if(BlendContainer)

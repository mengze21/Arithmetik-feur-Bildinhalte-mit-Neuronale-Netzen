__version__ = 'ksl_05'

# Note: This submodule should not import contents of nnlib to avoid
# circular imports.
import warnings
import numpy as np
import math
import png
import tensorflow as tf
import tensorflow.keras.backend as K
from pathlib import Path
from tensorflow import keras
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import InputSpec, Dense, Conv2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.callbacks import Callback

from . import _interactive

# Make some Tensorflow settings.  They are only concerning GPU at this
# point, but if no GPU is present, they will simply be ignored, so no
# special cases are necessary.
# CHANGED in keras_support_lib v04: With the tensorflow 2 upgrade, GPU
# settings work very differently.
phys_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(phys_devices[0], True)
except IndexError:
    pass

# Module settings.  Mostly for debug purposes.
MOD_SETTINGS = {
  'DEBUG': False,
  'OVERRIDE_THREAD_COUNT': False,
  'MAX_MULTITHREAD_CORECOUNT': 64,
  'ESCALATE_WARNINGS': False,
  'HALT_ON_ERRORS': False,
  'REPORT_PROGRESS': False,
  'REPORT_FREQ': 16,
}

# Returns relatively reliably the nnlib directory.
nnlib_path = Path(__file__).resolve().parent


# Specifiers for the standard parameter set as used in base data.  To
# achieve higher portability, all base data needs to adhere to the
# layout in PARAM_SET and the labels used for training need to be
# a subset of it.
# Decoder for the strings:
# (x,y,z): Center point of an object.
# (r,s,t): Radii of the object along x,y,z axes.
# (a,b,g): (alpha,beta,gamma): Euler angles for specifying orientation
# in the 3D space.
# For convenience in changing specs in the future, the following list
# the entry specifiers that stand for center point coordinates, axis
# radii and angles.
cp_params = 'xyz'
rad_params = 'rst'
ang_params = 'abg'
param_set = cp_params + rad_params + ang_params
cp_index = 0
rad_index = cp_index + len(cp_params)
ang_index = rad_index + len(rad_params)

# Standard settings for the handler.
_STD_SETTINGS = {
  'NN_OPTIMIZER': SGD(lr=0.01, momentum=0.9, decay=5e-4),
  'FIELD_SIZE': 64,
  'FIELD_DEPTH': 16,
  'TS_LENGTH': 16,
  'MASK_SIZE': 8,
  'MASK_SAMPLES': 24,
  'IND_LENGTH': True,
  'ANGULAR_DOF': 1,
  'SUPPRESS_BIAS': False,
  'CROSS_VALIDATION': False,
  'DIST_FREE': False,
  'ADD_BACKGROUND': 0.5,
  'SEP_DISTURBANCES': False,
  'MASKING_LIST': ('train','val'),
  'MINOR_FOLD_SPLITS': (1,),
  'PARAM_SET': param_set,
}


def associate_lookup(alias_dict, nameset, meaning):
    """A little helper function to associate many names with the same
    meaning via a dict.  Be careful when using it: The originally
    passed 'alias_dict' itself gets altered and nothing is returned.
    """
    meaning_in_nameset = False
    for name in nameset:
        if name == meaning:
            meaning_in_nameset = True
        alias_dict[name] = meaning
    if not meaning_in_nameset:
        print("Warning: It is strongly advised that the set of aliases "
              "contains its associated meaning.")
        print("The problem arose for '{}'.".format(meaning))


# Sets of strings to enable interactivity and modesetting in certain
# methods. (Brought over from nnlib in nnlib_multiobject_v04.)
string_alts = {'yes': {'yes','y','Yes','Y','YES'},
               'no': {'no','n','No','N','NO'},
               'full': {'full','f','Full','F','FULL'},
               'compile_only': {'c','compile','compile_only','C','Compile',
                                'Compile_only','COMPILE','COMPILE_ONLY'},
               'both': {'both','b','Both','B','BOTH'},
               'internal': {'i','int','intern','internal','I','Int','Intern',
                            'Internal','INT','INTERN','INTERNAL'},
               'external': {'e','ext','extern','external','E','Ext','Extern',
                            'External','EXT','EXTERN','EXTERNAL'},
               'all': {'a', 'all', 'A', 'All','ALL'},
               'conservative': {'c','con','conservative','C','Con',
                                'Conservative','CON','CONSERVATIVE'},
               'former': {'f','former','F','Former','FORMER'},
               'latter': {'l','latter','L','Latter','LATTER'}}


# -- Custom error classes --

class NNLibUsageError(Exception):
    """A generic error class used wherever there exists a problem with
    nnlib functions that could not be fitted into one of the built in
    error classes.  The message given should explain specifically what
    went wrong in these cases.
    """

    def __init__(self, message):
        self.message = message


class Breakout(Exception):
    """This error is for the specific purpose of breaking out of nested
    loops as Python does not offer a multi-level break.  Additionally,
    this may be used to fall back to an except clause if a conditional
    was not satisfied.  It is encouraged to use this special exception
    for control flow so we do not bury serious errors.
    """

    def __init__(self):
        pass


# Tuple consisting of all shapes that are currently supported.
# Can be used by ModelHandler.prepare_full_fields as a set from which
# to randomly choose and will also be used to perform the One Hot
# encoding.
# CHANGED: Beginning with the fork of nnlib_multiobject, 'none' is its
# own object type as entry 0 in std_shapes_all.
std_shapes_all = ('none','ellipse','rectangle')

# Predefined shapes tuple for ellipses only
std_shapes_conservative = ('ellipse',)


# -- Drawing function --

class Selector:
    """This class provides interactivity for plots of the fields and
    masks.  It makes the previews clickable, upon which they are drawn
    again on a bigger canvas and with scales. The class is
    purpose-bulit for the multishow method below and is to be
    instantiated by that method only.
    """

    def __init__(self,
                 dview,
                 fig,
                 fields,
                 axdict):
        self.dview = dview
        self.fields = fields
        self.fig = fig
        self.axdict = axdict
        self.draw_field(0)

    def __call__(self, event):
        try:
            i = self.axdict[sum(event.inaxes.get_geometry())]
            self.draw_field(i)
        except (AttributeError, KeyError):
            pass

    def draw_field(self, index):
        self.dview.imshow(X=self.fields[index],
                          cmap='Greys',
                          origin='lower',
                          vmin=0.0,
                          vmax=1.0)
        self.fig.canvas.draw()


def multishow(fields,
              grid_adaption=True,
              nrows=8,
              ncols=8,
              last_channel_rgb=None,
              no_clipping=False):
    """Draws images of fields and masks, including a preview and
    detailed view.  It expects either the last two dimensions to be the
    image data or the last dimension to be RGB channels and the two
    dimensions before to be the image data.  All dimensions before the
    image data are considered listings and will be squashed into one
    before drawing.  Monochromatic data that still has a single RGB
    channel will get that automatically removed.  RGB-ness of the data
    is identified by considering a last dimension of exactly size one
    (monochromatic) or three to be RGB data.  Since this may lead to
    false positives, override functionality for this behaviour is
    provided by directly passing the info via the last_channel_rgb
    argument.
    """
    # If not interactive, we can stop right away as drawing is
    # unsupported within a batch job.
    if not _interactive:
        print("Plotting of any kind is not available outside of interactive "
              "contexts.")
        return
    # Import matplotlib globally if it has not been imported before.
    # This means that matplotlib only gets imported when needed and
    # conditions are satisfied, but once they are, it is imported such
    # that it and its components are then globally available.  This
    # makes sure we do not have unnecessary overhead through loading it
    # every time multishow is used in a session, as it has noticeable
    # loading times.
    elif 'matplotlib' not in globals():
        global matplotlib
        global plt
        global gridspec
        import matplotlib
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    # Test for sensible settings
    if not grid_adaption and nrows*ncols > 144:
        print("It is not recommended to use a preview matrix with more than "
              "144 entries as performance will be poor and crashes may occur.")
        resp = input("Continue anyway? [yes/NO] > ")
        if resp not in string_alts['yes']:
            return
    # Check for RGB
    if last_channel_rgb is None:
        if fields.shape[-1] == 3 or fields.shape[-1] == 1:
            last_channel_rgb = True
        else:
            last_channel_rgb = False
    # Reshape the field array such that all dimensions that do not
    # correspond to the image information are squashed into one.
    fields = np.reshape(fields, (-1,)+fields.shape[-2-last_channel_rgb:])
    # Eliminate singleton dimensions (implicitly takes care of
    # monochromatic input).
    fields = np.squeeze(fields)
    # Some heuristics to decide in what arrangement to draw the
    # thing.
    if grid_adaption:
        fields_to_draw = fields.shape[0]
        nrows = (fields_to_draw+5) // 6
        if nrows > 7:
            nrows = (fields_to_draw+7) // 8
            if nrows > 10:
                nrows = min((fields_to_draw+11) // 12, 12)
                ncols = 12
            else:
                ncols = 8
        else:
            ncols = 6
    # Give feedback upon which fields cannot be drawn.
    if nrows*ncols < fields.shape[0]:
        if grid_adaption:
            print("Warning: The last {} images do not fit on the finest grid "
                  "and will therefore not be "
                  "drawn.".format(fields.shape[0]-nrows*ncols))
        else:
            print("Warning: The last {} images cannot be shown on your "
                  "selected grid.".format(fields.shape[0]-nrows*ncols))
    # Due to occasional numerical problems in Python2, clip the array
    # before drawing.  This should not change anything if the data
    # given is usable in the first place.
    if no_clipping:
        clipfields = fields
        if fields.max() > 1.0:
            print("A max value of {} was encountered. It is drawn "
                  "as-is.".format(fields.max()))
    else:
        clipfields = np.clip(fields, 0.0, 1.0)
    # Set up the layout matrix
    fig = plt.figure()
    outer = gridspec.GridSpec(nrows=1, ncols=2,
                              width_ratios=(.6,.4), hspace=1)
    inner = gridspec.GridSpecFromSubplotSpec(nrows=nrows, ncols=ncols,
                                             subplot_spec=outer[0])
    # Draw the preview part of the plot.  Simultaneously prepare a dict
    # with all individual previews to later map clicks on them to the
    # right image to be magnified.
    axdict = {}
    try:
        for row in range(nrows):
            for col in range(ncols):
                f_index = ncols*row + col
                ax = plt.Subplot(fig, inner[row,col])
                # Note that cmap is ignored if a colour channel is
                # present.
                ax.imshow(X=clipfields[f_index],
                          cmap='Greys',
                          origin='lower',
                          vmin=0.0,
                          vmax=1.0)
                ax.tick_params(axis='both',
                               which='both',
                               bottom=False,
                               left=False,
                               labelbottom=False,
                               labelleft=False)
                fig.add_subplot(ax)
                axdict[sum(ax.get_geometry())] = f_index
    except IndexError:
        pass
    # Prepare the detailed view in the layout
    dview = plt.Subplot(fig, outer[1])
    # Attach a mouse click event listener, using the Selector class
    # defined above
    fig.canvas.mpl_connect('button_press_event',
                           Selector(dview, fig, clipfields, axdict))
    dview.axis('image')
    dview.set_title("detailed view")
    fig.add_subplot(dview)
    # Finally, present everything to the user
    plt.show()


def multisave(fields,
              path=Path.home(),
              name_prefix='Image_',
              scaling_factor=1,
              last_channel_rgb=None,
              last_channel_greyscale=None,
              no_clipping=False):
    """Saves images of fields and masks as .pngs.  The method works
    largely the same as multishow to simplify shared access
    infrastructure.  It expects either the last two dimensions to be
    the image data or the last dimension to be RGB channels and the two
    dimensions before to be the image data.  All dimensions before the
    image data are considered listings and will be squashed into one,
    each getting its own image file.  The fields can have a last
    dimension (constituting the colour channels) of length one
    (greyscale) or three (RGB) or no channel dimension at all.  The RGB
    and Greyscale properties of the data can be given or inferred by
    considering the last dimension, dimension one being interpreted as
    greyscale, three as RGB and everything else as having no channels
    dimension.
    """
    # CHANGED in ksl_10: nnlib switches over to using pathlib.  To
    # accomodate old scripts and allow simple strings, make sure to
    # create a Path object.
    path = Path(path)
    scaling_factor = int(scaling_factor)
    if scaling_factor < 1:
        raise ValueError("The scaling factor needs to be a positive integer "
                         "(only integer upscaling is currently supported.")
    # Check if the given path already exists.  In general, we assume
    # that if it doesn't, it should be created, so that's the default.
    if not path.exists():
        if _interactive:
            resp = input("The path you gave does not exist. Create? [Y/n] > ")
            if resp in string_alts['no']:
                return
        path.mkdir()
    # Check if the given path points to a file.  If it does, we stop to
    # not risk overwriting something important and raise a (hopefully)
    # helpful error message.
    elif path.is_file():
        raise ValueError("The path points to an existing file. Note that the "
                         "'path' argument is to be used for the directory "
                         "path only; file names will be created from "
                         "'name_prefix' and numerical values.")
    # If we find images with the same naming scheme this method would
    # use, we stop and ask to overwrite if we run interactively.
    # Defaults to not overwrite.
    matched = False
    for file in path.iterdir():
        # Note: Glob is too limited for exact matching in this case,
        # i.e. match an arbitrary number of numerals.  This is why we
        # resort to calling it a match if the first and last character
        # is a numeral. Should be good enough for us.
        if file.match('[0-9]*[0-9].png'):
            matched = True
            break
    if matched:
        if _interactive:
            resp = input("It seems there are other images saved in the "
                         "specified location already. Overwrite? [y/N] > ")
            if resp in string_alts['yes']:
                # NEW in ksl_05: Selectively delete the images.  Note
                # that this will delete every image fitting the
                # wildcard, no matter if it would actually be
                # overwritten or not, to avoid clutter.
                for file in path.iterdir():
                    if file.match('[0-9]*[0-9].png'):
                        file.unlink()
            else:
                return
        else:
            print("Found data from a previous run. No image has been saved. "
                  "Delete manually if the existing data is obsolete.")
            return
    # Check for RGB and Greyscale.
    if last_channel_greyscale and last_channel_rgb:
        raise ValueError("The last channel cannot be RGB and Greyscale at "
                         "the same time.")
    if last_channel_rgb is None:
        if fields.shape[-1] == 3:
            last_channel_rgb = True
        else:
            last_channel_rgb = False
    if last_channel_greyscale is None:
        if fields.shape[-1] == 1:
            last_channel_greyscale = True
        else:
            last_channel_greyscale = False
    # Reshape the field array such that all dimensions that do not
    # correspond to the image information are squashed into one.  Also,
    # pypng expects the rgb values of each row in the image to be given
    # as one long list.  This needs to be reflected as well.
    if last_channel_rgb:
        field_shape = (scaling_factor*fields.shape[-3],
                       3*scaling_factor*fields.shape[-2])
        image_shape = (scaling_factor*fields.shape[-3],
                       scaling_factor*fields.shape[-2])
        if scaling_factor > 1:
            fields = np.kron(fields, np.ones(2*(scaling_factor,)+(1,)))
    elif last_channel_greyscale:
        field_shape = (scaling_factor*fields.shape[-3],
                       scaling_factor*fields.shape[-2])
        image_shape = field_shape
        if scaling_factor > 1:
            fields = np.kron(fields, np.ones(2*(scaling_factor,)+(1,)))
    else:
        field_shape = (scaling_factor*fields.shape[-2],
                       scaling_factor*fields.shape[-1])
        image_shape = field_shape
        if scaling_factor > 1:
            fields = np.kron(fields, np.ones(2*(scaling_factor,)))
    reshaped_fields = np.reshape(fields, (-1,)+field_shape)
    # Status report
    print("There are {} individual images selected for "
          "saving.".format(reshaped_fields.shape[0]))
    # Sanity check.
    # CHANGED in ksl_05: There is no more upper limit on the amount of
    # images.  Still, we keep the following sanity check.
    if reshaped_fields.shape[0] > 10000 and _interactive:
        resp = input("This seems quite a lot. Continue? [Y/n] > ")
        if resp in string_alts['no']:
            print("OK, nothing was saved.")
            return
    # Set up the png writer.  Since the specs stay consistent within
    # one dataset, we can reuse this instance throughout the whole
    # process, saving some time.
    writer = png.Writer(*image_shape, greyscale=not last_channel_rgb)
    # Determine length for file name numbering
    num_len = math.ceil(math.log10(reshaped_fields.shape[0]))
    # Go through the images and save them.  Each gets their own file,
    # automatic combinations are currently not supported.
    for count, img in enumerate(reshaped_fields):
        # Compose the path to the file.
        file_path = path / (name_prefix + str(count).zfill(num_len) + '.png')
        # Since all internal representations are valued in a range
        # between zero and one, but pypng expects integer values, we
        # extract, multiply and cast the fields.
        fields_slice = (img*255.0).astype('uint8', copy=True)
        # Open the file...
        file = open(file_path, 'wb')
        # ...write to it...
        writer.write(file, fields_slice)
        # ...and close it again.
        file.close()
    print("OK")


# -- Quantized layers and supporting classes --
# Original source: Ke Ding
# [https://github.com/DingKe/nn_playground/tree/master/binarynet]

class Clip(Constraint):
    def __init__(self, min_value, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
        if not self.max_value:
            self.max_value = -self.min_value
        if self.min_value > self.max_value:
            self.min_value, self.max_value = self.max_value, self.min_value

    def __call__(self, p):
        return K.clip(p, self.min_value, self.max_value)

    def get_config(self):
        return {"min_value": self.min_value,
                "max_value": self.max_value}


class QuantizedDense(Dense):
    """Quantized Dense layer
    References:
    "BinaryNet: Training Deep Neural Networks with Weights and
    Activations Constrained to +1 or -1"
    [http://arxiv.org/abs/1602.02830]
    """

    def __init__(self,
                 units,
                 H=1.,
                 kernel_lr_multiplier='Glorot',
                 bias_lr_multiplier=None,
                 **kwargs):
        super(QuantizedDense, self).__init__(units, **kwargs)
        self.H = H
        self.kernel_lr_multiplier = kernel_lr_multiplier
        self.bias_lr_multiplier = bias_lr_multiplier
        super(QuantizedDense, self).__init__(units, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[1]

        # Initialize the weights tensor using Glorot's approach.
        if self.H == 'Glorot':
            self.H = np.float32(np.sqrt(1.5 / (input_dim + self.units)))
        # Same for the kernel lr multiplier.
        if self.kernel_lr_multiplier == 'Glorot':
            self.kernel_lr_multiplier = np.float32(
              1. / np.sqrt(1.5 / (input_dim + self.units)))
        # Set constraintas and the initializer itself.
        self.kernel_constraint = Clip(-self.H, self.H)
        self.kernel_initializer = RandomUniform(-self.H, self.H)
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        # Use or don't use bias.
        if self.use_bias:
            self.lr_multipliers = [self.kernel_lr_multiplier,
                                   self.bias_lr_multiplier]
            self.bias = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.lr_multipliers = [self.kernel_lr_multiplier]
            self.bias = None
        # Set the input spec and report that the layer was built.
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        binary_kernel = quantize(self.kernel, H=self.H)
        output = K.dot(inputs, binary_kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def get_config(self):
        config = {'H': self.H,
                  'kernel_lr_multiplier': self.kernel_lr_multiplier,
                  'bias_lr_multiplier': self.bias_lr_multiplier}
        base_config = super(QuantizedDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class QuantizedConv2D(Conv2D):
    """Quantized Convolution2D layer
    CHANGES by itobirk:
      - New Name
      - Calls quantize() instead of binarize()
      - To accomodate that, it allows for quantization level count Q to
        be passed at init
    References:
    "BinaryNet: Training Deep Neural Networks with Weights and
    Activations Constrained to +1 or -1"
    [http://arxiv.org/abs/1602.02830]
    """

    def __init__(self,
                 filters,
                 kernel_lr_multiplier='Glorot',
                 bias_lr_multiplier=None,
                 weight_range=[0.,1.],
                 weight_levels=2,
                 **kwargs):
        super(QuantizedConv2D, self).__init__(filters, **kwargs)
        try:
            self.hi = weight_range[1]
            self.lo = weight_range[0]
        except TypeError:
            self.hi = weight_range
            self.lo = 0
        except IndexError:
            self.hi = weight_range[0]
            self.lo = 0
        self.Q = weight_levels
        self.kernel_lr_multiplier = kernel_lr_multiplier
        self.bias_lr_multiplier = bias_lr_multiplier

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError("The channel dimension of the inputs should be "
                             "defined. Found 'None'.")
        # Set kernel shape
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        # Initialize the weights tensor using Glorot's approach.
        base = self.kernel_size[0] * self.kernel_size[1]
        if self.hi == 'Glorot':
            nb_input = int(input_dim * base)
            nb_output = int(self.filters * base)
            self.hi = np.float32(np.sqrt(1.5 / (nb_input + nb_output)))
        # Same for the kernel lr multiplier.
        if self.kernel_lr_multiplier == 'Glorot':
            nb_input = int(input_dim * base)
            nb_output = int(self.filters * base)
            self.kernel_lr_multiplier = np.float32(
              1. / np.sqrt(1.5 / (nb_input+nb_output)))
        # Set up constraints and the initializer itself.
        self.kernel_constraint = Clip(self.lo, self.hi)
        self.kernel_initializer = RandomUniform(self.lo, self.hi)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        # Use or don't use bias.
        if self.use_bias:
            self.lr_multipliers = [self.kernel_lr_multiplier,
                                   self.bias_lr_multiplier]
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.lr_multipliers = [self.kernel_lr_multiplier]
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        # Quantize the kernel before applying it in the forward pass.
        quantized_kernel = quantize(self.kernel,
                                    H=[self.lo, self.hi],
                                    Q=self.Q)
        # from here, we do what conv2d layers are always doing.
        outputs = K.conv2d(inputs,
                           quantized_kernel,
                           strides=self.strides,
                           padding=self.padding,
                           data_format=self.data_format,
                           dilation_rate=self.dilation_rate)
        if self.use_bias:
            outputs = K.bias_add(outputs,
                                 self.bias,
                                 data_format=self.data_format)
        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def get_config(self):
        config = {'weight_range': [self.hi,self.lo],
                  'kernel_lr_multiplier': self.kernel_lr_multiplier,
                  'bias_lr_multiplier': self.bias_lr_multiplier}
        base_config = super(QuantizedConv2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# -- additional supporting functions --
# Original source: Ke Ding
# [https://github.com/DingKe/nn_playground/tree/master/binarynet]

def round_through(x):
    """Element-wise rounding to the closest integer with full gradient
    propagation.  A trick from [Sergey Ioffe]
    (http://stackoverflow.com/a/36480182)
    """
    # Round to nearest int.
    rounded = K.round(x)
    # Return "rounded" in forward prop because stop_gradient lets its
    # input pass but returns real x in backprop because stop_gradient
    # returns zero for gradient calculations.
    return x + K.stop_gradient(rounded - x)


def quantize_through(x, Q):
    """Element-wise quantizing to the closest quantization level with
    full gradient propagation.  A trick from [Sergey Ioffe]
    (http://stackoverflow.com/a/36480182)
    """
    x = x * (Q-1)
    # Round to nearest int
    rounded = K.round(x)
    # Return "rounded" in forward prop because stop_gradient lets
    # its input pass but returns real x in backprop because
    # stop_gradient returns zero for gradient calculations.
    return (x + K.stop_gradient(rounded - x)) / (Q-1)


def _hard_sigmoid(x):
    """Hard sigmoid different from the more conventional form (see
    definition of K.hard_sigmoid).
    # Reference:
    "BinaryNet: Training Deep Neural Networks with Weights and
    Activations Constrained to +1 or -1, Courbariaux et al. 2016"
    [http://arxiv.org/abs/1602.02830]
    """
    x = (0.5 * x) + 0.5
    return K.clip(x, 0, 1)


def linear_saturation(x):
    """Based on _hard_sigmoid(). Basically a linear function saturating
    at 0 and 1.
    # Reference:
    "BinaryNet: Training Deep Neural Networks with Weights and
    Activations Constrained to +1 or -1, Courbariaux et al. 2016"
    [http://arxiv.org/abs/1602.02830]
    """
    return K.clip(x, 0, 1)


def binary_sigmoid(x):
    """Binary hard sigmoid for training binarized neural network.
    # Reference:
    "BinaryNet: Training Deep Neural Networks with Weights and
    Activations Constrained to +1 or -1, Courbariaux et al. 2016"
    [http://arxiv.org/abs/1602.02830]
    """
    return round_through(_hard_sigmoid(x))


def quantized_linear(x, Q):
    """Binary hard sigmoid for training binarized neural network.
    # Reference:
    "BinaryNet: Training Deep Neural Networks with Weights and
    Activations Constrained to +1 or -1, Courbariaux et al. 2016"
    [http://arxiv.org/abs/1602.02830]
    """
    return quantize_through(linear_saturation(x), Q)


def binary_tanh(x):
    """Binary hard sigmoid for training binarized neural network.
    The neurons' activations binarization function
    It behaves like the sign function during forward propagation
    And like:
      hard_tanh(x) = 2 * _hard_sigmoid(x) - 1
      clear gradient when |x| > 1 during back propagation
    # Reference:
    "BinaryNet: Training Deep Neural Networks with Weights and
    Activations Constrained to +1 or -1, Courbariaux et al. 2016"
    [http://arxiv.org/abs/1602.02830]
    """
    return 2 * round_through(_hard_sigmoid(x)) - 1


def quantize(W, H=[0.,1.], Q=2):
    """The weights' quantization function
    Derived from original binarize
    CHANGES by itobirk:
      - Name changed
      - Now uses quantized_sigmoid() instead of binary_tanh().
      - New input Q (allowed levels of output).
        This keeps the weights in [0, H] with Q levels between.
    # Reference:
    "BinaryNet: Training Deep Neural Networks with Weights and
    Activations Constrained to +1 or -1, Courbariaux et al. 2016"
    [http://arxiv.org/abs/1602.02830]
    """
    # [0, H] -> 0 or H
    span = H[1] - H[0]
    W = W - H[0]
    if Q == 'cont':
        Wb = span * linear_saturation(W / span)
    else:
        Wb = span * quantized_linear(W / span, Q)
    return Wb + H[0]


def _mean_abs(x, axis=None, keepdims=False):
    return K.stop_gradient(K.mean(K.abs(x), axis=axis, keepdims=keepdims))


def xnorize(W, H=1., axis=None, keepdims=False):
    Wb = quantize(W, H, 2)
    Wa = _mean_abs(W, axis, keepdims)
    return Wa, Wb


# -- Original supporting functions --
# (by itobirk)

class ModelCheckpointToInternal(Callback):
    """Write the model weights into a provided dictionary after every
    epoch. (The dict key will be the number of epochs trained.)
    This is a modified variant of the ModelCheckpoint Callback that
    comes with keras.  The main differences are that this callback does
    not directly save the weights, but rather writes them to a dict
    belonging to the ModelContainer instance that also owns the model
    itself.  It also does not support saving whole models as this
    would only cause excessive clutter with no added benefit.
    Arguments
      weights_hist: list to save the weights in.  Epoch count will be
        entry number.
      monitor: quantity to monitor.
      verbose: verbosity mode, 0 or 1.
      saving_mode: There are four modes
        0: all weights are kept
        1: only the weights that improved the model with respect
           to the monitored quantity are saved
        2: as above, but the weights that yielded the best previous
           results are actively deleted
        3: does not save weights at all and only increments epoch
           counter
      mode: one of {auto, min, max}.  For 'val_acc' as the monitored
        quantity, this should be 'max', for 'val_loss' this should be
        'min', etc. In 'auto' mode, the direction is automatically
        inferred from the name of the monitored quantity.
      period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, container, monitor='val_loss', verbose=0,
                 saving_mode=0, mode='auto', period=1):
        super(ModelCheckpointToInternal, self).__init__()
        self.container = container
        self.monitor = monitor
        self.verbose = verbose
        self.saving_mode = saving_mode
        self.period = period
        self.epochs_since_last_save = 0
        self.best_index = None

        if mode not in {'auto', 'min', 'max'}:
            warnings.warn("ModelCheckpoint mode {} is unknown, "
                          "fallback to auto mode.".format(mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        logs = logs or {}
        weights_to_save = None
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            # Saving mode 0 indicates that the weights should be saved
            # no matter what.
            if self.saving_mode == 0:
                weights_to_save = self.container.model.get_weights()
            # Saving modes 1 and 2 indicate that the weights should be
            # saved if and only if an improvement was achieved.
            elif self.saving_mode == 1 or self.saving_mode == 2:
                # Get the value of the monitored quantity
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn("Can save best model only with {} "
                                  "available, skipping.".format(self.monitor),
                                  RuntimeWarning)
                else:
                    # Compare to the previous best to decide whether or
                    # not to save the new weights.
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print("Epoch {}: {} improved from {:.5f} to "
                                  "{:.5f}, saving model".format(epoch+1,
                                                                self.monitor,
                                                                self.best,
                                                                current))
                        self.best = current
                        # Saving mode 2 additionally indicates that
                        # weights are to be deleted if further
                        # improvement was achieved.
                        if (self.best_index is not None
                              and self.saving_mode == 2):
                            self.container.weights_hist[self.best_index] = None
                        self.best_index = len(self.container.weights_hist)
                        weights_to_save = self.container.model.get_weights()
                    else:
                        if self.verbose > 0:
                            print("Epoch {}: {} did not improve from "
                                  "{:.5f}".format(epoch+1,
                                                  self.monitor,
                                                  self.best))
            # Saving mode 3 indicates that no weights should be saved
            # at all, only the counters increased, so it does not need
            # its own case here.
        # Append the weights.  Note that this is always done; if the
        # weights are not to be saved, this will append a None,
        # ensuring that the numbering in weights_hist is always
        # consistent with the total epochs trained.
        self.container.weights_lookup.append(len(self.container.weights_hist))
        self.container.weights_hist.append(weights_to_save)


class ComModelCheckpointToInternal(Callback):
    """Write the model weights into a provided dictionary after every
    epoch. (The dict key will be the number of epochs trained.)
    This is a modified variant of the ModelCheckpoint Callback that
    comes with keras.  The main differences are that this model does
    not directly save the weights, but rather writes them into an
    internal dictionary that can be internally dealt with further after
    training, and that it does not support saving whole models as this
    would only cause excessive clutter with no added benefit.

    Arguments
      container: The container that also owns the respective model.
        Note that this callback was made for convenience of use with
        nnlib and the naming and finding of places to store the weights
        is therefore specific to the ModelContainer setup.
      monitor: quantity to monitor.
      verbose: verbosity mode, 0 or 1.  1 prints some simple feedback
        to the console, 0 does not.
      saving_mode: There are four legal modes
         0: all weights are kept
         1: only the weights that improved the model with respect
            to the monitored quantity are saved
         2: as above, but the weights that yielded the best results
            before are actively deleted
         3: does not save weights at all and only increments epoch
            counter
      mode: one of {auto, min, max}.  If `save_best_only=True`, the
        decision to overwrite the current save file is made based
        on either the maximization or the minimization of the
        monitored quantity. For `val_acc`, this should be `max`,
        for `val_loss` this should be `min`, etc. In `auto` mode,
        the direction is automatically inferred from the name of
        the monitored quantity.
      period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, container, monitor='val_loss', verbose=0,
                 saving_mode=0, mode='auto', period=1):
        super(ComModelCheckpointToInternal, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.container = container
        self.saving_mode = saving_mode
        self.period = period
        self.epochs_since_last_save = 0
        self.best_index = None
        if mode not in {'auto', 'min', 'max'}:
            warnings.warn("ModelCheckpoint mode %s is unknown, "
                          "fallback to auto mode." % (mode),
                          RuntimeWarning)
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        self.epochs_since_last_save += 1
        logs = logs or {}
        save_command = False
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            # Saving mode 0 indicates that the weights should be saved
            # no matter what.
            if self.saving_mode == 0:
                save_command = True
            # Saving modes 1 and 2 indicate that the weights should be
            # saved if and only if an improvement was achieved.
            elif self.saving_mode == 1 or self.saving_mode == 2:
                # Get the value of the monitored quantity
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn("Can save best model only with %s "
                                  "available, skipping." % (self.monitor),
                                  RuntimeWarning)
                else:
                    # Compare to the previous best to decide whether or
                    # not to save the new weights.
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print("Epoch {}: {} improved from {:.5f} to "
                                  "{:.5f}, saving model".format(epoch+1,
                                                                self.monitor,
                                                                self.best,
                                                                current))
                        self.best = current
                        # Saving mode 2 additionally indicates that
                        # weights are to be deleted if further
                        # improvement was achieved.
                        if (self.best_index is not None
                              and self.saving_mode == 2):
                            # The deletion gets a bit complicated as we
                            # have to trace back through multiple
                            # objects.  First, we fetch the relevant
                            # indices from the container's weights
                            # lookup table.
                            indices = (
                              self.container.weights_lookup[self.best_index])
                            # Now we go through all the base containers
                            # and delete the entries indicated by these
                            # indices.
                            self.container._modify_base_models('delete',
                                                               indices)
                            self.container.weights_lookup[self.best_index] = (
                              None)
                        self.best_index = len(self.container.weights_lookup)
                        save_command = True
                    else:
                        if self.verbose > 0:
                            print("Epoch {}: {} did not improve from "
                                  "{:.4f}".format(epoch+1,
                                                  self.monitor,
                                                  self.best))
            # Saving mode 3 indicates that no weights should be saved
            # at all, only the counters increased, so it does not need
            # its own case here.
        # If 'save_command' has been set to True above, save the
        # weights in each base container's 'weights_hist' and attach
        # the corresponding indices to the combined container's
        # 'weights_lookup'.
        if save_command:
            indices = self.container._append_to_base_models('append')
        else:
            indices = self.container._append_to_base_models('append_none')
        self.container.weights_lookup.append(indices)


def _conditional_mse(onehot):
    """This allows conditional application of the mean squared error
    loss function by inserting this intermediate layer that either
    returns the real loss function (if there is a real object in the
    viewspace) or a dummy one that always returns zero, no matter the
    predictions.  This should keep the network from constantly being
    distracted in trying to learn the location of a non-existing object
    that it can obviously never guess accurately.
    """
    def _loss(y_true, y_pred):
        selector = 1 - onehot[...,0]
        squared_error = K.mean(K.square(y_pred - y_true), axis=-1)
        return selector * squared_error
    return _loss


def _overlap_error(y_true, y_pred):
    """This is an experimental new error function that uses total
    overlap as a performance metric.  The algorithm is based on the
    rasterizer implemented in mo_renderer_new.py.
    """
    # TODO


def _focus_point_error(y_true, y_pred):
    """A custom error function. Its minimum is reached when the real
    and the predicted focus points line up and the deviation in the
    first half-axis is zero (squared error).
    """
    sq_diff = (K.square(y_pred[...,2])
               - K.square(y_pred[...,3]))
    switch = K.cast(sq_diff < 0, tf.float32)
    sq_diff_abs_rt = K.sqrt(K.abs(sq_diff))
    sines = K.sin(y_pred[...,4] + switch*math.pi/2.0)
    cosines = K.cos(y_pred[...,4] + switch*math.pi/2.0)
    rad1 = (K.square(y_true[...,0] - y_pred[...,0]
                     - sq_diff_abs_rt*cosines)
            + K.square(y_true[...,1] - y_pred[...,1]
                       - sq_diff_abs_rt*sines))
    rad2 = (K.square(y_true[...,2] - y_pred[...,0]
                     + sq_diff_abs_rt*cosines)
            + K.square(y_true[...,3] - y_pred[...,1]
                       + sq_diff_abs_rt*sines))
    stack = K.stack((K.sqrt(rad1),
                     K.sqrt(rad2),
                     K.abs(y_true[...,4]-y_pred[...,2])),
                    axis=-1)
    return K.sum(stack, axis=-1)


def _fp_label_gen(labels):
    """Rewrites the labels such that they contain the coordinates of
    the focus points and the length of the first half axis rather than
    the parameters (x_center, y_center, rad_1, rad_2, angle). These new
    labels can be worked upon by the custom _focus_point_error loss
    function.
    """
    argument = labels[...,2]**2-labels[...,3]**2
    switch = np.array(argument < 0, dtype=np.float32)
    radix = np.sqrt(np.abs(argument))
    sines = np.sin(labels[...,4] + switch*math.pi/2.0)
    cosines = np.cos(labels[...,4] + switch*math.pi/2.0)
    fp_labels = np.stack((labels[...,0] + radix*cosines,
                          labels[...,1] + radix*sines,
                          labels[...,0] - radix*cosines,
                          labels[...,1] - radix*sines,
                          labels[...,2]),
                         axis=-1)
    return fp_labels


def _axis_end_point_error_alt1(y_true, y_pred):
    """A custom error function. Its minimum is reached when the real
    and the predicted focus points line up and the deviation in the
    first half-axis is zero (squared error).
    """
    sines = K.sin(y_pred[...,4])
    cosines = K.cos(y_pred[...,4])
    pred_labels = K.stack((y_pred[...,0],
                           y_pred[...,1],
                           y_pred[...,0] + cosines*y_pred[...,2],
                           y_pred[...,1] + sines*y_pred[...,2],
                           y_pred[...,0] - sines*y_pred[...,3],
                           y_pred[...,1] + cosines*y_pred[...,3]),
                          axis=-1)
    return K.mean(K.square(pred_labels - y_true), axis=-1)


def _aep_label_gen_alt1(labels):
    """Rewrites the labels such that they contain the coordinates of
    the focus points rather than the parameters (x_center, y_center,
    rad_1, rad_2, angle). These new labels can be worked upon by the
    custom _focus_point_error loss function.
    """
    aep_labels = np.empty(labels.shape[:-1] + (6,))
    sines = np.sin(labels[...,4])
    cosines = np.cos(labels[...,4])
    aep_labels[...,:2] = labels[...,:2]
    aep_labels[...,2] = labels[...,0] + cosines*labels[...,2]
    aep_labels[...,3] = labels[...,1] + sines*labels[...,2]
    aep_labels[...,4] = labels[...,0] - sines*labels[...,3]
    aep_labels[...,5] = labels[...,1] + cosines*labels[...,3]

    return aep_labels


def _axis_end_point_error_alt2(y_true, y_pred):
    """A custom error function. Its minimum is reached when the real
    and the predicted focus points line up and the deviation in the
    first half-axis is zero (squared error).
    """
    sines = K.sin(y_pred[...,4])
    cosines = K.cos(y_pred[...,4])
    pred_labels = K.stack((y_pred[...,0] + cosines*y_pred[...,2],
                           y_pred[...,1] + sines*y_pred[...,2],
                           y_pred[...,0] - sines*y_pred[...,3],
                           y_pred[...,1] + cosines*y_pred[...,3],
                           y_pred[...,0] - cosines*y_pred[...,2],
                           y_pred[...,1] - sines*y_pred[...,2],
                           y_pred[...,0] + sines*y_pred[...,3],
                           y_pred[...,1] - cosines*y_pred[...,3]),
                          axis=-1)
    return K.mean(K.square(pred_labels-y_true), axis=-1)


def _aep_label_gen_alt2(labels):
    """Rewrites the labels such that they contain the coordinates of
    the focus points rather than the parameters (x_center, y_center,
    rad_1, rad_2, angle). These new labels can be worked upon by the
    custom _focus_point_error loss function.
    """
    aep_labels = np.empty(labels.shape[:-1] + (8,))
    sines = np.sin(labels[...,4])
    cosines = np.cos(labels[...,4])
    aep_labels[...,0] = labels[...,0] + cosines*labels[...,2]
    aep_labels[...,1] = labels[...,1] + sines*labels[...,2]
    aep_labels[...,2] = labels[...,0] - sines*labels[...,3]
    aep_labels[...,3] = labels[...,1] + cosines*labels[...,3]
    aep_labels[...,4] = labels[...,0] - cosines*labels[...,2]
    aep_labels[...,5] = labels[...,1] - sines*labels[...,2]
    aep_labels[...,6] = labels[...,0] + sines*labels[...,3]
    aep_labels[...,7] = labels[...,1] - cosines*labels[...,3]
    return aep_labels


def _axis_end_point_error_alt3(y_true, y_pred):
    """A custom error function. Its minimum is reached when the real
    and the predicted focus points line up and the deviation in the
    first half-axis is zero (squared error).
    """
    sines = K.sin(y_pred[...,4])
    cosines = K.cos(y_pred[...,4])
    pred_labels = K.stack((y_pred[...,0],
                           y_pred[...,1],
                           y_pred[...,0] + cosines*y_pred[...,2],
                           y_pred[...,1] + sines*y_pred[...,2],
                           y_pred[...,0] - sines*y_pred[...,3],
                           y_pred[...,1] + cosines*y_pred[...,3]),
                          axis=-1)
    return K.mean(K.abs(pred_labels-y_true), axis=-1)


# Set up loss_dict, a convenient construct to connect error functions
# and label preparation functions that belong together to a string key
# for easier access.
loss_dict = {'cond_mse': (_conditional_mse,),
             'cat_cross': (categorical_crossentropy,),
             'fp_fit': (_focus_point_error,_fp_label_gen),
             'aep_fit_alt1': (_axis_end_point_error_alt1,_aep_label_gen_alt1),
             'aep_fit_alt2': (_axis_end_point_error_alt2,_aep_label_gen_alt2),
             'aep_fit_alt3': (_axis_end_point_error_alt3,_aep_label_gen_alt1)}


# Make the custom layers programmed herein globally visible for Keras.
# This is especially important when (de)serialization is performed on
# models containing these.
gco = keras.utils.get_custom_objects()
gco['QuantizedConv2D'] = QuantizedConv2D
gco['QuantizedDense'] = QuantizedDense
gco['Clip'] = Clip
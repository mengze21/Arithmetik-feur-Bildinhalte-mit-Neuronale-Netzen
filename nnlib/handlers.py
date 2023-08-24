__version__ = 'mo_08'

from math import pi, sin, cos, copysign
import six
from six.moves import input
from collections.abc import Iterable
from contextlib import contextmanager, ExitStack
import numpy as np
import numpy.random as rand
import warnings
import multiprocessing as mp
from copy import copy, deepcopy
from abc import ABCMeta, abstractmethod
from tensorflow.keras import layers as lrs
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.losses import categorical_crossentropy

# CHANGED (nnlib_multiobject_v0_3): All supporting constructs,
# especially those expanding on the capabilities of some vanilla Keras
# features, were outsourced to the new module keras_support_lib, whose
# functions are collectively loaded here.
from . import _interactive
from .support_lib import NNLibUsageError, MOD_SETTINGS, string_alts
from .support_lib import _STD_SETTINGS, param_set
from .support_lib import rad_params, ang_params, rad_index, ang_index
from .support_lib import std_shapes_all, std_shapes_conservative
from .support_lib import multishow, multisave
from .support_lib import associate_lookup
from .support_lib import ModelCheckpointToInternal
from .support_lib import ComModelCheckpointToInternal
from .build_recipes import build_dict
from .data_ops import Copier, H5PyHandler
from .data_ops import register_container, instantiate_container
from .abstract_containers import AbstractContainer, AbstractContainerGroup
from .abstract_containers import is_cont_obj
from .render_tooling import select_render_if

if MOD_SETTINGS['ESCALATE_WARNINGS']:
    warnings.simplefilter('error', category=UserWarning)
else:
    warnings.simplefilter('always', category=UserWarning)

# Version as integer for easy comparison.
_version = int(__version__[-2:])

# For readability, the _lookup dictionarys used to simplify addressing
# certain features in working with nnlib are built at runtime.  The
# performance cost should be negligable, but readability greatly
# improved.
_sample_alias_lookup = {}
_train_alts = {'training','train','t','Training','Train','T'}
associate_lookup(_sample_alias_lookup, _train_alts, 'train')
_val_alts = {'validation','val','v','Validation','Val','V'}
associate_lookup(_sample_alias_lookup, _val_alts, 'val')
_test_alts = {'test','Test'}
associate_lookup(_sample_alias_lookup, _test_alts, 'test')

_model_alias_lookup = {}
_mask_alts = {'m','mask','quantized_masks','M','Mask'}
associate_lookup(_model_alias_lookup, _mask_alts, 'quantized_masks')
_interpreter_alts = {'i','int','interpreter','I','Int','Interpreter'}
associate_lookup(_model_alias_lookup, _interpreter_alts, 'interpreter')
_classifier_alts = {'cl','class','classifier','Cl','Class','Classifier'}
associate_lookup(_model_alias_lookup, _classifier_alts, 'classifier')
_obj_counter_alts = {'co','count','counter','obj_counter','object_counter',
                     'Co','Count','Counter','Object Counter'}
associate_lookup(_model_alias_lookup, _obj_counter_alts, 'object_counter')
_encoder_alts = {'e','enc','encoder','E','Enc','Encoder'}
associate_lookup(_model_alias_lookup, _encoder_alts, 'encoder')
_encoder1d_alts = {'e1d','enc1d','encoder1d','encoder_1d','E1D','Enc1D',
                   'Encoder_1D'}
associate_lookup(_model_alias_lookup, _encoder1d_alts, 'encoder1d')
_adapter_alts = {'a','adapter','A','Adapter'}
associate_lookup(_model_alias_lookup, _adapter_alts, 'adapter')
_adapter1d_alts = {'a1d','adapter1d','adapter_1d','A1D','Adapter1D'}
associate_lookup(_model_alias_lookup, _adapter1d_alts, 'adapter1d')
_conv_adapter_alts = {'ca','conv_adapter','CA','ConvAdapter'}
associate_lookup(_model_alias_lookup, _conv_adapter_alts, 'conv_adapter')
_decoder_alts = {'d','dec','decoder','D','Dec','Decoder'}
associate_lookup(_model_alias_lookup, _decoder_alts, 'decoder')
_decoder1d_alts = {'d1d','dec1d','decoder1d','decoder_1d','D1D','Dec1D',
                   'Decoder_1D'}
associate_lookup(_model_alias_lookup, _decoder1d_alts, 'decoder1d')


# -- Private module methods --


def _mt_starter(thread_num, creator_list, param_pack, data_pack=None):
    """This function provides the new multiprocessing interfaces with
    the creators.
    Arguments:
      param_pack: A dict containing all general parameters required by
        the Creators.  Not all parameters are required by every Creator
        and the structure does not enforce that all parameters are
        provided, but the caller needs to make sure all necessary ones
        are present.  Otherwise, this function will error out.
        As of writing this, these parameters are:
        samples, mask_size, include_nones, min_distract_objects,
        max_distract_objects, shapes, nd_scale, field_size,
        field_depth, coords_as_floats, label_layout, full_layout,
        add_background, ts_length, ts_movement, ts_variation
      creator_list: A list or tuple of Creators to be called
        consecutively.  Creators in this list may depend on previous
        ones' outputs, but there is no reordering.  The caller has to
        make sure the order makes sense such that each Creator is
        usable when it comes up.
      data_pack: Can be None or a dict containing pre-known outputs
        from Creators (including loaded or imported ones).  The keys
        are expected to follow the standard naming as imposed by the
        Creators.  This dict will be continually updated and returned
        in the end.  Bear in mind that some Creators may or may not
        require certain data fields to be available when they are set
        up.  The dictionary does not have to contain all possible
        fields, but the caller has to make sure that the ones necessary
        are present when the loop comes around to it.
    """
    rand.seed()
    param_pack['thread_num'] = thread_num
    if not isinstance(param_pack, dict):
        raise TypeError("The parameters need to be provided in a dict with "
                        "standard string keys, otherwise we cannot know what "
                        "to do with them.")
    if data_pack is None:
        data_pack = {}
    elif not isinstance(data_pack, dict):
        raise TypeError("If pre-known data is provided to _mt_starter, it "
                        "needs to be in the form of a dictionary with "
                        "standard string keys, otherwise we cannot know what "
                        "to do with it.")
    for creator in creator_list:
        cr_instance = creator(**param_pack)
        cr_instance.input_data(**data_pack)
        cr_instance.generate()
        data_pack.update(cr_instance.output_data())
    return data_pack


# -- Public module methods --


def load(path):
    """Loads a previously saved handler from a .h5 file.
    CHANGED: From nnlib_v13 onwards, it uses the new Copier to do so.
    CHANGED: From nnlib_multiobject_v01 onwards, this function doubles
    as a deep copier of an existing handler instance.  Just pass an
    existing handler as the path argument.  'handler_copyall', the
    function previously used for this feature, has been deleted.
    """
    if isinstance(path, six.string_types):
        fhandler = H5PyHandler(path, True)
        access = fhandler.interactive_provide_access('ModelHandler')
    elif path.pull('type') == 'ModelHandler':
        access = path
    else:
        raise NNLibUsageError("The object you gave is neither a valid path "
                              "nor does it look like a ModelHandler object.")
    new_handler = ModelHandler()
    cp = Copier(access, new_handler)
    cp.copy('all')
    return new_handler


# -- Public module classes --

@six.add_metaclass(ABCMeta)
class Creator:
    """Abstract base class for all sample generation processors.
    This was established in nnlib_multiobject_v06 in an effort to
    refactor the simulation data creation.
    """

    def __init__(self, thread_num, **kwargs):
        self.thread_num = thread_num

    def _report_progress(self, current_num):
        """Provides the new progress reporting feature.  Enable via
        MOD_SETTINGS
        """
        if (MOD_SETTINGS['REPORT_PROGRESS']
              and current_num % MOD_SETTINGS['REPORT_FREQ'] == 0):
            print("{} in thread {}: Building sample {}".format(
                  type(self).__name__, self.thread_num, current_num))

    # array_access isn't required by all, but by some inheriting
    # classes to access the internal 'fields' variables for drawing.
    def array_access(self, value, x, y, z=0):
        """Provides an interface to access the arrays inside the
        various FieldCreator implementations.  The motivation is that
        only the FieldCreator instance owning a field matrix should be
        trusted to change said matrix.  Other methods or objects can
        then interact with the field by calling this method, whose
        handle can be passed around as needed instead of the field
        itself.  An error will occur if pointers to the current sample
        and time instance have not been set.
        CHANGED in nnlib_multiobject_v0_4: The arguments x, y can be
        ints or slices, but no longer tuples.  The slice creation has
        to be performed by the caller from now on.
        """
        if not 0 <= z < self.fields.shape[2]:
            raise IndexError("The requested depth is misaligned with the "
                             "render output.")
        try:
            if value is None:
                return self.fields[self.sam,self.ti,z,y,x]
            else:
                value = np.clip(value, 0.0, 1.0)
                self.fields[self.sam,self.ti,z,y,x] = self.intens * value
        except IndexError:
            return 0.0

    def input_data(self, **kwargs):
        """Used to input data produced by other creators in the chain.
        Needs to quietly ignore all data the specific creator does not
        need to keep usage generic.
        """
        pass

    @abstractmethod
    def output_data(self):
        """Used to output created data.  The element returned is always
        expected to be a dict with consistent naming in the form of
        strings.  This, again, is supposed to make it as generic as
        possible.
        """
        pass

    @abstractmethod
    def generate(self):
        """Triggers the generation process.  This is the public facing
        function triggering the functionality, but it can, of course,
        invoke further, private functions.
        'generate' does not accept arguments.  All data necessary for
        operating properly must be given to Creators via their __init__
        or input_data functions.
        """
        pass


class BaseDataCreator(Creator):
    """This class implements the processes necessary to create new base
    sample data from random numbers.

    The data fields this creator returns are always normed to the
    interval [0,1] and contain 9 parameters:
      [0:3] - positions x, y, z
      [3:6] - radii (half-lengths) a, b, c
      [6:9] - Euler angles alpha, beta, gamma

    The shape fields this creator returns are string identifiers for
    the object variants defined herein. (currently 'rectangle' and
    'ellipse')

    The intensity fields this creator returns are float values normed
    to the interval [0,1].
    """

    def __init__(self,
                 samples,
                 mask_size,
                 include_nones,
                 min_distract_objects,
                 max_distract_objects,
                 min_intensity,
                 max_intensity,
                 shapes,
                 nd_scale,
                 **kwargs):
        """To keep the creators general, they are all ready to accept
        the whole selection of input arguments and quietly ignore those
        that are not needed.  The ones not needed do not have to be
        supplied.
        """
        # Call super's __init__ function first.
        super(BaseDataCreator, self).__init__(**kwargs)
        self.samples = samples
        # The mask-field-ratio defines the minimal radius any object is
        # allowed to have such that they are never smaller than one
        # mask pixel as, if that was allowed, the results from the net
        # would be ambiguous and it would develop the tendency to
        # average out.
        self.m_f_ratio = 1.0 / (2.0*mask_size)
        self.include_nones = include_nones
        self.min_distract_objects = min_distract_objects
        self.max_distract_objects = max_distract_objects
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.shapes = tuple(shapes)
        self.nd_scale = nd_scale

    def _random_position_nd(self, m_f_ratio_modifier):
        """Create random sizes, positions and angles.  The sizes are
        drawn from a normal distribution to favour smaller objects (its
        effect can be scaled via nd_scale which is the standard
        deviation for the normal distribution from which the sizes are
        drawn) - positions and angles, however, stem from a uniform
        distribution.  Return them as a numpy matrix.  Outsourced from
        _prepare_full_fields_nd.
        """
        m_f_ratio = m_f_ratio_modifier * self.m_f_ratio
        while True:
            # First, generate the half-axis lengths.
            rad_1 = abs(rand.normal(scale=self.nd_scale)) + m_f_ratio
            rad_2 = abs(rand.normal(scale=self.nd_scale)) + m_f_ratio
            rad_3 = abs(rand.normal(scale=self.nd_scale)) + m_f_ratio
            # Next, generate the Euler angles alpha, beta and gamma
            # (where alpha represents the original turning angle as
            # used in versions prior to nnlib_multiobject_v06) and
            # calculate (an estimate of) the resulting object sizes
            # in the directions of the cartesian coordinates. To do
            # this, we assume a cuboid and check whether or not it is
            # bigger in diameter than the field is.  If it is, we
            # repeat the whole process to not introduce an artificial
            # bias.
            # CHANGED in mo_08: The angles are now represented by
            # normed values as well.  Functions using them need to
            # scale them to their geometric meaning (degree or radian).
            # They should be understood to scale from [0,1] to [0,pi/2]
            # (NOT [0,2*pi]) as we would get non-unique data points
            # otherwise.
            alpha = rand.random_sample()
            beta = rand.random_sample()
            gamma = rand.random_sample()
            # Calculate trigonometric functions beforehand as we need
            # them quite a lot in the following.
            sin_a, cos_a = sin(alpha*pi/2.0), cos(alpha*pi/2.0)
            sin_b, cos_b = sin(beta*pi/2.0), cos(beta*pi/2.0)
            sin_g, cos_g = sin(gamma*pi/2.0), cos(gamma*pi/2.0)
            # Based on the mapping matrix for the euler rotations, we
            # have the following lengths...
            len_x = (abs((cos_a*cos_g-sin_a*cos_b*sin_g)*rad_1)
                       + (cos_a*sin_g+sin_a*cos_b*cos_g)*rad_2
                       + sin_a*sin_b*rad_3)
            len_y = ((sin_a*cos_g+cos_a*cos_b*sin_g)*rad_1
                       + abs((-sin_a*sin_g+cos_a*cos_b*cos_g)*rad_2)
                       + cos_a*sin_b*rad_3)
            len_z = (sin_b*sin_g*rad_1
                       + sin_b*cos_g*rad_2
                       + cos_b*rad_3)
            # CHANGED in nnlib_multiobject_v0_5: To speed up the
            # process, reject parameter sets that yield objects too big
            # to fit the fields as early as possible, which is here.
            # Note for understanding: If the following if-clause is
            # TRUE, the objects are small enough to fit and the
            # overarching while loop is BROKEN.  Otherwise, all of the
            # above is REPEATED.
            if len_x < 0.5 and len_y < 0.5 and len_z < 0.5:
                break
        # Reuse the length calculation to a priori make sure that the
        # object lies within the field.
        px = (1.0-2.0*len_x) * rand.random_sample() + len_x
        py = (1.0-2.0*len_y) * rand.random_sample() + len_y
        pz = (1.0-2.0*len_z) * rand.random_sample() + len_z
        # CHANGED in nnlib_multiobject_v0_5: Due to the changes made
        # above, the dataset this function yields is bound to satisfy
        # all conditions previously checked by _check_base_data - a
        # function that is therefore obsolete and got removed.
        return np.array((px,py,pz,rad_1,rad_2,rad_3,alpha,beta,gamma),
                        dtype=np.float32)

    def _random_position_ed(self, m_f_ratio_modifier):
        """Creates random sizes, positions and angles using a uniform
        distribution and returns them as a numpy matrix.  Outsourced
        from _prepare_full_fields_nd.
        """
        m_f_ratio = m_f_ratio_modifier * self.m_f_ratio
        while True:
            # First, generate the half-axis lengths.
            rad_1 = (0.5-m_f_ratio)*rand.random_sample() + m_f_ratio
            rad_2 = (0.5-m_f_ratio)*rand.random_sample() + m_f_ratio
            rad_3 = (0.5-m_f_ratio)*rand.random_sample() + m_f_ratio
            # Next, generate the Euler angles alpha, beta and gamma
            # (where alpha represents the original turning angle as
            # used in versions prior to nnlib_multiobject_v06) and
            # calculate (an estimate of) the resulting object sizes
            # in the directions of the cartesian coordinates. To do
            # this, we assume a cuboid and check whether or not it is
            # bigger in diameter than the field is.  If it is, we
            # repeat the whole process to not introduce an artificial
            # bias.
            alpha = rand.random_sample()
            beta = rand.random_sample()
            gamma = rand.random_sample()
            # Calculate trigonometric functions beforehand as we need
            # them quite a lot in the following.
            sin_a, cos_a = sin(alpha*pi/2.0), cos(alpha*pi/2.0)
            sin_b, cos_b = sin(beta*pi/2.0), cos(beta*pi/2.0)
            sin_g, cos_g = sin(gamma*pi/2.0), cos(gamma*pi/2.0)
            # Based on the mapping matrix for the euler rotations, we
            # have the following lengths...
            len_x = (abs((cos_a*cos_g-sin_a*cos_b*sin_g)*rad_1)
                       + (cos_a*sin_g+sin_a*cos_b*cos_g)*rad_2
                       + sin_a*sin_b*rad_3)
            len_y = ((sin_a*cos_g+cos_a*cos_b*sin_g)*rad_1
                       + abs((-sin_a*sin_g+cos_a*cos_b*cos_g)*rad_2)
                       + cos_a*sin_b*rad_3)
            len_z = (sin_b*sin_g*rad_1
                       + sin_b*cos_g*rad_2
                       + cos_b*rad_3)
            # CHANGED in nnlib_multiobject_v0_5: To speed up the
            # process, reject parameter sets that yield objects too big
            # to fit the fields as early as possible, which is here.
            # Note for understanding: If the following if-clause is
            # TRUE, the objects are small enough to fit and the
            # overarching while loop is BROKEN.  Otherwise, all of the
            # above is REPEATED.
            if len_x < 0.5 and len_y < 0.5 and len_z < 0.5:
                break
        # Reuse the length calculation to a priori make sure that the
        # object lies within the field.
        px = (1.0-2.0*len_x) * rand.random_sample() + len_x
        py = (1.0-2.0*len_y) * rand.random_sample() + len_y
        pz = (1.0-2.0*len_z) * rand.random_sample() + len_z
        # CHANGED in nnlib_multiobject_v0_5: Due to the changes made
        # above, the dataset this function yields is sure to satisfy
        # all conditions previously checked by _check_base_data - a
        # function that is therefore rendered obsolete.
        return np.array((px,py,pz,rad_1,rad_2,rad_3,alpha,beta,gamma),
                        dtype=np.float32)

    def _gen_position(self):
        """Generates the individual positions (start and stop) for
        generate_base_data by taking random positions from either a
        normal or an equal distribution and checking their viability
        afterwards.
        """
        # Randomized size and placement.
        if self.nd_scale:
            return self._random_position_nd(1.0)
        else:
            return self._random_position_ed(1.0)

    def _gen_dst_position(self, main_obj):
        """Generates the individual positions (start and stop) for
        generate_base_data by taking random positions from either a
        normal or an equal distribution and checking their viability
        afterwards.  Generally, this does the same as _gen_position.
        However, the generation of disturbance objects requires some
        additional checking that this method implements.
        """
        while True:
            # Randomized size and placement
            if self.nd_scale:
                start_pos = self._random_position_nd(0.5)
            else:
                start_pos = self._random_position_ed(0.5)
            # Continue if the parameter set satisfies all conditions.
            # The checking function also returns the 'present' aspect
            # almost for free, so we save and use it as well.
            ok, present = self._check_distract_distance(start_pos, main_obj)
            if ok:
                break
        # Define the ending position.  No special requirements are
        # imposed for that.  This is subject to be checked.
        if self.nd_scale:
            end_pos = self._random_position_nd(0.5)
        else:
            end_pos = self._random_position_ed(0.5)
        pos = np.stack((start_pos,end_pos), axis=0)
        return pos, present

    def _check_distract_distance(self, datapiece, main_obj):
        """Makes sure distract objects aren't too similar in size to
        the actual object to detect.  Doubles as condition provider for
        whether a distract object should be drawn as-is or as having
        been cancelled out.
        """
        mo_area = main_obj[2] * main_obj[3]
        dist_area = datapiece[2] * datapiece[3]
        # Check for big enough difference in filled area.
        if 1.25 * dist_area > mo_area and dist_area < 1.25 * mo_area:
            # If this is true, the real and the distracting object are
            # considered too similar in size and therefore rejected.
            return False, None
        else:
            # Here, the check is passed and we are generating the value
            # for 'present'.  Considerably smaller object in comparison
            # to the main one are to be drawn as-is, while considerably
            # larger objects are to be imperfectly cancelled out in
            # drawing.
            if 1.25 * dist_area < mo_area:
                present = True
            else:
                present = False
        # Check for big enough distance.
        if ((datapiece[0]-main_obj[0])**2 + (datapiece[1]-main_obj[1])**2
              < 0.75 * max(mo_area, dist_area)):
            return False, None
        else:
            return True, present

    def output_data(self):
        """All creators return their generated data through the
        output_data method.  It builds a single dictionary out of the
        data to pass back, again aiming to keep interactions as generic
        as possible for the caller.  This requires consistency in
        naming of fields across creators and both inputs and outputs.
        """
        return {'basedata': self.base_data}

    def generate(self):
        """Beginning with NNlib_multiobject_v06, this Creator prepares
        the deterministic parts of the field data, out of which the
        full fields can be generated using the WorkingDataCreator.
        WorkingDataCreator will also take care of adding some noise
        to the data, if requested, and spreading the data out across
        the timeseries.  This also means that the data produced by this
        function is more portable and more easily reusable than the
        full fields.
        """
        # CHANGED: Beginning with the fork of nnlib_multiobject,
        # 'none'-objects are considered their own type.  To keep the
        # user-facing syntax compatible with earlier versions and easy
        # to understand while still reaping the benefits of not having
        # a special case for 'none'-objects, the 'shapes' argument gets
        # 'none' added as the first entry if it is not already present.
        if self.shapes[0] != 'none':
            self.shapes = ('none',) + self.shapes
        # CHANGED: From NNlib_v6a on, the data tensor is a complex
        # structure encompassing all information that was previously
        # split between data and meta vectors (that means that it
        # contains start points, end points and shapes for every
        # sample).
        # CHANGED: In all versions of nnlib_multiobject, multiple
        # objects per sample are supported.
        # CHANGED in nnlib_multiobject_v0_6: In contrast to previous
        # versions, the data for the additional objects is stored in
        # an additional field instead of at the base level.  The
        # training process is supposed to focus on one object per
        # sample, with the others only serving as distraction.  They
        # are therefore stored separately under 'dst'.
        # CHANGED in nnlib_multiobject_v06: The new 'intensity' field
        # saves the value that should be assigned to object pixels.
        # Before, we always assumed maximum contrast (which equals
        # intensity==1.0), but that is unrealistic.
        self.base_data = np.zeros(
          (self.samples,),
          dtype=[('data','f4',(2,len(param_set))),
                 ('shape','U12'),
                 ('intensity','f4'),
                 ('dst',[('data','f4',(2,len(param_set))),
                         ('shape','U12'),
                         ('presence','?')],(self.max_distract_objects,))]
        )
        for current_num, sample in enumerate(self.base_data):
            self._report_progress(current_num)
            # Generate the data for the object that is to be found.
            if self.include_nones:
                main_shape = self.shapes[rand.randint(len(self.shapes))]
            else:
                main_shape = self.shapes[rand.randint(1, len(self.shapes))]
            sample['shape'] = main_shape
            if main_shape != 'none':
                # Generate start and end positions for the main object.
                sample['data'][0] = self._gen_position()
                sample['data'][1] = self._gen_position()
            # Generate the data for the distractions.
            for dist in range(self.max_distract_objects):
                dist_sample = sample['dst'][dist]
                if dist < self.min_distract_objects:
                    shape = self.shapes[rand.randint(1, len(self.shapes))]
                else:
                    shape = self.shapes[rand.randint(len(self.shapes))]
                dist_sample['shape'] = shape
                if shape != 'none':
                    (dist_sample['data'],dist_sample['presence']) = (
                       self._gen_dst_position(sample['data'][0])
                    )
            # End of distraction for-loop.
            # Generate the intensity scale.
            sample['intensity'] = (
              self.min_intensity
              + rand.rand()*(self.max_intensity-self.min_intensity)
            )
        # End of outer for-loop.


class WorkingDataCreator(Creator):
    """Generates both fields (including distractions and randomization
    elements) and fitting labels.
    """

    def __init__(self,
                 thread_num,
                 field_size,
                 field_depth,
                 coords_as_floats,
                 label_layout,
                 full_layout,
                 ts_length,
                 ts_movement,
                 ts_variation,
                 **kwargs):
        """To keep the creators general, they are all ready to accept
        the whole selection of input arguments and quietly ignore those
        that are not needed.  The ones not needed do not have to be
        supplied.
        """
        # Call super's __init__ function first.
        super(WorkingDataCreator, self).__init__(thread_num=thread_num,
                                                 **kwargs)
        self.field_size = field_size
        self.field_depth = field_depth
        self.coords_as_floats = coords_as_floats
        self.full_layout = full_layout
        # For reordering the full parametrization to the reduced label
        # tensor, there are no special cases to be considered, so we
        # can do it in the following single line loop.
        self.reorder = [full_layout.index(lb) for lb in label_layout]
        # CHANGED in mo_08: Scaling is no longer supported.
        # The SampleContainers as well as the Creators work exclusively
        # with regularized values in all their parameters.
        # Replacement, however, still needs to be done.
        self.replace_mask = np.ones((len(full_layout),), dtype=np.float32)
        self.replace_values = np.zeros((len(full_layout),), dtype=np.float32)
        # We go through the specific full layout to find the places
        # where values need to be replaced with constants.
        for i, l in enumerate(full_layout):
            if l.isdigit():
                self.replace_mask[i] = 0.0
                self.replace_values[i] = float(l)
        # NOTE: Reordering from original full labels to modified full
        # labels is done on a case-by-case basis in the _modify
        # routine.
        self.ts_length = ts_length
        self.ts_movement = ts_movement
        self.ts_var = ts_variation
        # NOTE: As of nnlib_multiobject_v06, legacy can only deal with
        # 2D settings while Blender can only deal with 3D, so the
        # field_depth variable implicitly also changes the render
        # engine used and the label layout.  This may be changed in the
        # future.
        self.interface = select_render_if(self.array_access,
                                          thread_num,
                                          field_size,
                                          field_depth)

    def _randomize(self, paramset):
        """This function randomizes a given parameter set.  It includes
        a basic sanity check that makes sure no negative lengths are
        returned.
        """
        while True:
            randomized = (
              paramset + rand.normal(scale=self.ts_var,
                                     size=(len(param_set),))
            )
            if min(randomized[rad_index:ang_index]) > 0.0:
                return randomized

    def _modify(self, shape):
        """_modify produces a working parameter set for the rendering
        interfaces while taking into account the additional constraints
        represented via the layout strings.
        """
        # Get the deterministic current coordinate vector through
        # linear interpolation between start and end.
        try:
            c_det = (
              (self.ti/(self.ts_length-1.0)) * (shape[1]-shape[0]) + shape[0]
            )
        except ZeroDivisionError:
            # Timeseries lengths of 1 are supported as of nnlib_v12 to
            # emulate the behaviour of the scripts pre NNlib_v4.  This
            # requires zero divisions to be caught here.
            c_det = shape[0]
        # Add an additional element of randomness.
        current = self._randomize(c_det)
        # CHANGED in nnlib_multiobject_v06: This is where the
        # restrictions for simpler setups are processed from now on.
        # This will allow for less memory usage and relieve some of the
        # processing burden at runtime.
        # First, process 'c' indicator in full_layout.
        # Notice that the following for-loop does direct in-place swaps
        # on current.
        for ind, letter in enumerate(self.full_layout):
            # 'c' is a special character in the sense that it
            # signifies that the right entry needs to first be
            # chosen among the radii as the smallest entry.  This
            # is supposed to enable fail-safe operation by making
            # sure that no field boundary is touched by the object
            # to be drawn.  Note that this skews the algorithm
            # towards producing smaller objects.
            if letter == 'c':
                current[ind] = current[rad_index:ang_index].min()
        # Second, perform the scaling and replacing of values.
        current = (current*self.replace_mask + self.replace_values)
        # Third, process coords_as_floats.  This doesn't structurally
        # change the output.
        if not self.coords_as_floats:
            # We need to round everything except for the angles - it
            # wouldn't make a lot of sense to do it on them.
            current[:ang_index] = (
              np.around(self.field_size*current[:ang_index]) / self.field_size
            )
        # When everything is processed, return 'current' to the caller
        # which will handle the drawing and saving of labels itself.
        return current

    def _draw_timeseries(self):
        """Outsourced from 'generate', this function does most of the
        actual work on both the labels and the fields.  Works on a
        single sample (but does all timeseries and depth iterations);
        the iteration over samples is done in 'generate' itself.
        """
        # Get the base data slice. Note that self.sam gets set by this
        # function's caller, so we do indeed get a different slice
        # every time this runs.
        bd_slice = self.base_data[self.sam]
        lbl_slice = self.labels[self.sam]
        # NEW in nnlib_multiobject_v06: Set the intensity scale.
        self.intens = bd_slice['intensity']
        # Unpack the coordinate data for all elements to be drawn
        # within the current base data slice.
        additive_unpacked = []
        subtractive_unpacked = []
        if bd_slice['shape'] != 'none':
            if self.ts_movement:
                additive_unpacked.append((bd_slice['data'][0],
                                          bd_slice['data'][1],
                                          bd_slice['shape']))
            else:
                additive_unpacked.append((bd_slice['data'][0],
                                          bd_slice['data'][0],
                                          bd_slice['shape']))
        for dist in bd_slice['dst']:
            if dist['shape'] != 'none':
                if self.ts_movement:
                    additive_unpacked.append((dist['data'][0],
                                              dist['data'][1],
                                              dist['shape']))
                else:
                    additive_unpacked.append((dist['data'][0],
                                              dist['data'][0],
                                              dist['shape']))
                if not dist['presence']:
                    subtractive_unpacked.append(additive_unpacked[-1])
        # End of for-loop over disturbances.
        # Now, loop over the timeseries.
        for self.ti in range(self.ts_length):
            objlist = []
            try:
                for i, shape in enumerate(additive_unpacked):
                    # CHANGED in nnlib_multiobject_v06: The processing
                    # became more complex here because of the more
                    # generic nature of the new base data tensors.  The
                    # processing is therefore now handled in another
                    # function to make it reusable.
                    current = self._modify(shape)
                    # Save it first...
                    objlist.append(list(current) + [shape[2],True])
                    # ...then submit it.
                    self.interface.submit_obj(current, shape[2], True)
                    if i == 0:
                        # Write the parameters to the label tensor.
                        lbl_slice[self.ti] = current[self.reorder]
                # End of for-loop over additive shapes.
                for shape in subtractive_unpacked:
                    current = self._modify(shape)
                    # Save it first...
                    objlist.append(list(current) + [shape[2],False])
                    # then submit it.
                    self.interface.submit_obj(current, shape[2], False)
                # End of for-loop over subtractive shapes.
                self.interface.finalize()
            except (NNLibUsageError,ValueError,AttributeError,
                    IndexError,NotImplementedError) as e:
                # If anything goes wrong in rendering, we land here.
                # In order to not break execution, we inform the user
                # of where the problem happened, write zeros as the
                # label (because nothing has been drawn) and continue
                # with the outer for-loop.
                print("Draw failed for sample {}, timeseries step {}.".format(
                        self.sam, self.ti))
                if MOD_SETTINGS['HALT_ON_ERRORS']:
                    raise e
                else:
                    lbl_slice[self.ti] = np.zeros((len(self.reorder),),
                                                  dtype=np.float32)
                    self.skipped_combinations.append(objlist + [str(e)])
                    print("Skipped.")
        # End of outer for-loop over time instances.

    def input_data(self, basedata, **kwargs):
        """As with __init__, this function should accept all kinds of
        data and silently ignore those that it does not require.
        """
        self.base_data = basedata

    def output_data(self):
        """All creators return their generated data through the
        output_data method.  It builds a single dictionary out of the
        data to pass back, again aiming to keep interactions as generic
        as possible for the caller.  This requires consistency in
        naming of fields across creators and both inputs and outputs.
        """
        if MOD_SETTINGS['DEBUG']:
            return {'fields': self.fields, 'labels': self.labels,
                    'skipped_combinations': self.skipped_combinations}
        else:
            return {'fields': self.fields, 'labels': self.labels}

    def generate(self):
        """Prepares the fields and labels to train the network with.
        CHANGED: This method only compiles the raw data available in
        the FieldCreator instance. Interpolates linear motion over the
        two data points per sample and additionally adds some
        randomness in the form of background (random lines) and some
        additional stochastic variation on top of the deterministic
        linear motion.
        """
        # Get sample count and distraction object count from base data.
        samples, dist_objects = self.base_data['dst'].shape
        # Define field array. (6-dim -> compatible with Keras's Pooling
        # layer and including timeseries and depth variable)
        self.fields = np.zeros((samples,self.ts_length,self.field_depth,
                                self.field_size,self.field_size,1),
                               dtype=np.float32)
        # Define label array. (3-dim -> supporting multiple samples and
        # timeseries)
        self.labels = np.zeros((samples,self.ts_length,len(self.reorder)),
                               dtype=np.float32)
        if MOD_SETTINGS['DEBUG']:
            self.skipped_combinations = []
        # Main loop.  Note that the loop variable is an object
        # attribute, making sure the write operations always target the
        # right sample without requiring additional calls or passing
        # down attributes.
        for self.sam in range(samples):
            self._report_progress(self.sam)
            # CHANGED in nnlib_multiobject_v02: With the introduction
            # of the new multiobject drawing system, it does not make
            # sense any longer to split the drawing process between
            # background and foreground, so most is handled by the
            # _draw_timeseries method itself from now on.
            self._draw_timeseries()
        # End of for-loop
        self.interface.close()


class DisturbanceCreator(Creator):
    """Introduces noise to the fields (which need to have been built
    beforehand).
    """

    _min_line_intensity = 0.5
    _max_line_intensity = 1.0
    _min_white_noise_intensity = 0.0
    _max_white_noise_intensity = 0.1

    def __init__(self,
                 add_background,
                 separate_disturbances,
                 **kwargs):
        """To keep the creators general, they are all ready to accept
        the whole selection of input arguments and quietly ignore those
        that are not needed.  The ones not needed do not have to be
        supplied.

        Parameters
        ----------

        add_background : float
            Specifies the probability for a line representing a random
            disturbance to appear. Set to 0.0 to disable this Creator's
            functionality altogether.
        separate_disturbances : bool
            If True, we deepcopy the fields before modifying, therefore
            leaving the unmodified fields available.  If False, the
            disturbances will be applied to the original set.  The
            latter is mainly a memory saving option.
        """
        # Call super's __init__ function first.
        super(DisturbanceCreator, self).__init__(**kwargs)
        self.add_background = add_background
        self.separate_disturbances = separate_disturbances

    def _def_rand_lines(self):
        """Inserts a random amount of random lines into the given
        field, using a very simple rasterization implementation.
        Writes directly to the given field array via array_access;
        nothing is returned.
        """
        # CHANGED in mo_08: We now support disturbances on our 3D
        # images as well, which is why we must loop over depth slices
        # here.
        for z in range(self.fields.shape[2]):
            # The while loop continues with the probability given by
            # add_background.
            while rand.random_sample() < self.add_background:
                # Define random start and end points.
                xstart = rand.randint(0, self.field_size)
                xend = rand.randint(0, self.field_size)
                ystart = rand.randint(0, self.field_size)
                yend = rand.randint(0, self.field_size)
                # Also randomize the intensity.
                self.intens = (
                  self._min_line_intensity + rand.random()
                    * (self._max_line_intensity-self._min_line_intensity)
                )
                # Python's math library doesn't have a bog standard sign
                # function, so this awkward workaround is necessary...
                xsign = int(copysign(1.0, xend - xstart))
                ysign = int(copysign(1.0, yend - ystart))
                if abs(xend - xstart) > abs(yend - ystart):
                    # Remember to make this a true division even in
                    # Python2.
                    m = 1.0 * (yend - ystart) / (xend - xstart)
                    y = ystart
                    for x in range(xstart, xend+xsign, xsign):
                        self.array_access(1.0, x, int(round(y)), z)
                        y += m * xsign
                elif xend == xstart and yend == ystart:
                    self.array_access(1.0, xstart, ystart, z)
                else:
                    # Remember to make this a true division even in
                    # Python2.
                    m = 1.0 * (xend - xstart) / (yend - ystart)
                    x = xstart
                    for y in range(ystart, yend+ysign, ysign):
                        self.array_access(1.0, int(round(x)), y, z)
                        x += m * ysign
        # End of inner while/outer for loop.

    def input_data(self, fields, **kwargs):
        """As with __init__, this function accepts all kinds of data
        and silently ignores what it does not require.
        """
        if self.separate_disturbances:
            self.origfields = deepcopy(fields)
        self.fields = fields
        self.field_size = fields.shape[3]

    def output_data(self):
        """All creators return their generated data through the
        output_data method.  It builds a single dictionary out of the
        data to pass back, again aiming to keep interactions as generic
        as possible for the caller.  This requires consistency in
        naming of fields across creators and both inputs and outputs.
        """
        if self.separate_disturbances:
            return {'fields': self.fields, 'origfields': self.origfields}
        else:
            return {'fields': self.fields}

    def generate(self):
        """Prepares the fields and labels to train the network with.
        CHANGED: This method only compiles the raw data available in
        the FieldCreator instance. Interpolates linear motion over the
        two data points per sample and additionally adds some
        randomness in the form of background (random lines) and some
        additional stochastic variation on top of the deterministic
        linear motion.
        """
        # Main loop.  Note that the loop variable is an object
        # attribute, making sure the write operations always target the
        # right sample without requiring additional calls or passing
        # down attributes.
        if self.add_background > 0.0:
            for self.sam in range(self.fields.shape[0]):
                self._report_progress(self.sam)
                for self.ti in range(self.fields.shape[1]):
                    self._def_rand_lines()


class OnehotCreator(Creator):
    """Generates the Onehot encoding."""

    def __init__(self, ts_length, **kwargs):
        """To keep the creators general, they are all ready to accept
        the whole selection of input arguments and quietly ignore those
        that are not needed.  The ones not needed do not have to be
        supplied.
        """
        # Call super's __init__ function first.
        super(OnehotCreator, self).__init__(**kwargs)
        self.ts_length = ts_length

    def input_data(self, basedata, **kwargs):
        """As with __init__, this function should accept all kinds of
        data and silently ignore those that it does not require.
        """
        self.base_data = basedata

    def output_data(self):
        """All creators return their generated data through the
        output_data method.  It builds a single dictionary out of the
        data to pass back, again aiming to keep interactions as generic
        as possible for the caller.  This requires consistency in
        naming of fields across creators and both inputs and outputs.
        """
        return {'objtype': self.onehot}

    def generate(self):
        """Generates the One Hot encoding for use with the shape
        estimation.  It bases its outputs on the meta info given
        and the occurence of the different shapes in std_shapes_all.
        Writes directly to the onehot array; nothing is returned.
        """
        # CHANGED in nnlib_multiobject_v0_6: Although it contains
        # redundant information, it was decided to let the onehot
        # tensor have 3 dimensions: First for sample, second for
        # timeseries, third for shape encoding.  This is in contrast to
        # 2 dimensions as before, where timeseries was left out, to
        # enable straight forward comparisons with the model outputs.
        self.onehot = np.zeros((self.base_data.shape[0],
                                self.ts_length,
                                len(std_shapes_all)),
                               dtype=np.int8)
        for r in range(self.base_data.shape[0]):
            self._report_progress(r)
            onehot_index = std_shapes_all.index(self.base_data[r]['shape'])
            self.onehot[r,:,onehot_index] = 1


class ObjCountCreator(Creator):
    """Generates the object count tensor."""

    def __init__(self, ts_length, **kwargs):
        """To keep the creators general, they are all ready to accept
        the whole selection of input arguments and quietly ignore those
        that are not needed.  The ones not needed do not have to be
        supplied.
        """
        # Call super's __init__ function first.
        super(ObjCountCreator, self).__init__(**kwargs)
        self.ts_length = ts_length

    def input_data(self, basedata, **kwargs):
        """As with __init__, this function should accept all kinds of
        data and silently ignore those that it does not require.
        """
        self.base_data = basedata

    def output_data(self):
        """All creators return their generated data through the
        output_data method.  It builds a single dictionary out of the
        data to pass back, again aiming to keep interactions as generic
        as possible for the caller.  This requires consistency in
        naming of fields across creators and both inputs and outputs.
        """
        return {'objcount': self.obj_count}

    def generate(self):
        """Generates the obj_count vector for use with the proposed
        separate network for guessing the amount of object in the
        field.  Writes directly to the obj_count array; nothing is
        returned.
        """
        self.obj_count = np.zeros((self.base_data.shape[0],self.ts_length,1),
                                  dtype=np.int8)
        for r in range(self.base_data.shape[0]):
            self._report_progress(r)
            if self.base_data[r]['shape'] != 'none':
                counter = 1
                for dst_item in self.base_data[r]['dst']:
                    counter += int(dst_item['presence'])
            else:
                counter = 0
            self.obj_count[r,:,0] = counter
        # End of for-loop.


class DistFreeCreator(Creator):
    """Draws the distraction free fields.  Note that, in contrast to
    most creators, this one requires the true labels, and not the base
    data, to be given as inputs.
    """

    def __init__(self,
                 thread_num,
                 field_size,
                 field_depth,
                 label_layout,
                 full_layout,
                 **kwargs):
        """To keep the creators general, they are all ready to accept
        the whole selection of input arguments and quietly ignore those
        that are not needed.  The ones not needed do not have to be
        supplied.
        """
        # Call super's __init__ function first.
        super(DistFreeCreator, self).__init__(thread_num=thread_num, **kwargs)
        self.field_size = field_size
        self.field_depth = field_depth
        self.full_layout = full_layout
        # Define a reordering list.  This will be applied to the input
        # data to get a list of full values from the list of labels.
        # It is important to note that string.find(i) returns -1 if i
        # was not found, thus indicating that no mapping was found and,
        # by the structure we follow when developing these mappings,
        # therefore indicating a hard-coded constant in the full_labels
        # string.  This will be checked for and dealt with in
        # input_data().
        self.reorder = [label_layout.lower().find(l) for l in full_layout]
        # The labels may need to be scaled.  This is indicated by
        # having an uppercase letter in label_layout.  In order to
        # conveniently perform this scaling, we build a vector here
        # that simply needs to be multiplied element-wise with the
        # labels before reordering.
        self.scale = np.ones((len(label_layout),), dtype=np.float32)
        for i, l in enumerate(label_layout):
            if l.isupper():
                self.scale[i] = self.field_size
        self.interface = select_render_if(self.array_access,
                                          thread_num,
                                          field_size,
                                          field_depth)

    def input_data(self, labels, basedata, **kwargs):
        """As with __init__, this function should accept all kinds of
        data and silently ignore those that it does not require.
        """
        # We reconstruct a set of full labels from the (potentially)
        # reduced labels.  This involves three things:
        # 1. Upscaling of the values wherever regularization has been
        #    applied.
        # 2. Reordering of the values to suit the full layout.
        # 3. Entering hard-coded values from the specifier where
        #    necessary.  The places where hard-coded values were found
        #    are signified by a value of -1 in 'self.reorder list'.
        self.labels = (self.scale*labels)[...,self.reorder]
        for i, a in enumerate(self.reorder):
            if a == -1:
                self.labels[...,i] = float(self.full_layout[i])
        # Remember that self.shapes will only have one dimension as we
        # (reasonably) do not assume the shape to change within a
        # sample.
        self.shapes = basedata['shape']
        self.intensities = basedata['intensity']

    def output_data(self):
        """All creators return their generated data through the
        output_data method.  It builds a single dictionary out of the
        data to pass back, again aiming to keep interactions as generic
        as possible for the caller.  This requires consistency in
        naming of fields across creators and both inputs and outputs.
        """
        return {'nodistfields': self.fields}

    def generate(self):
        """Draws the fields specified in self.base_data.  No labels, no
        interpolation, no distractions.
        """
        self.fields = np.zeros(self.labels.shape[0:2]+(self.field_depth,
                                                       self.field_size,
                                                       self.field_size,1),
                               dtype=np.float32)
        # Define samples
        for self.sam in range(self.fields.shape[0]):
            self._report_progress(self.sam)
            # NEW in nnlib_multiobject_v06: Set the intensity scaling.
            self.intens = self.intensities[self.sam]
            for self.ti in range(self.fields.shape[1]):
                # To save some time, skip the whole process if the
                # object to be drawn is 'none'.
                # Note: As of now it makes sense to skip the whole
                # sample because the object shape type does not change
                # over time.  This may have to be reconsidered later.
                if self.shapes[self.sam] == 'none':
                    break
                # Write the field.
                self.interface.submit_obj(self.labels[self.sam,self.ti],
                                          self.shapes[self.sam], True)
                self.interface.finalize()
        # End of the for-loops
        self.interface.close()


class FunctionalContainer(AbstractContainer):
    """FunctionalContainer specifies common functionality for the
    working containers (in contrast to the Serialized Containers in
    data_ops submodule).
    """

    def _get_saving_hook(self, target):
        """Transforms the given 'target' into universally usable
        container objects.
        Additionally, it provides a fitting Deduplicator instance.
        _get_saving_hook can deal with various cases:
          - If 'target' is a pathlib Path or string, it opens/creates
            the file it points to by creating a H5PyHandler instance.
          - If 'target' is already a container instance, it passes it
            back.
        """
        if isinstance(target, six.string_types):
            # Create the handler for the specified h5 file.
            h5handler = H5PyHandler(target)
            target = h5handler.provide_access(self.pull('type'),
                                              self.pull('id'))
        if not is_cont_obj(target):
            raise NNLibUsageError("The target does not look like an "
                                  "AbstractContainer instance. Cannot "
                                  "continue.")
        deduplicator = target.provide_deduplicator()
        return target, deduplicator

    def provide_deduplicator(self):
        return ContainerGroup()


@six.add_metaclass(ABCMeta)
class SampleContainerInterface(FunctionalContainer):
    """Defines the interfaces for getting samples out of data pools.
    These interfaces can then be used by containers holding and
    creating real sample data as well as virtual containers that enable
    the new cross validation infrastructure.
    """
    @abstractmethod
    def checkout_field(self, field_identifier, sl):
        pass

    @abstractmethod
    def build_additional_fields(self, field_identifier):
        pass


class VirtualSampleContainer(SampleContainerInterface):
    """Serves as a virtual sample container on top of a group of other
    SampleContainer instances that provide the actual data.  The idea
    is that the virtual container masks the dataset folding, providing
    the same interfaces as a normal SampleContainer, while eliminating
    the need to copy unnecessary amounts of data around.
    """

    def __init__(self, sample_pools, total_folds, fold, start, end):
        """Builds the necessary information to efficiently access data
        through VirtualSampleContainer in the real SampleContainer
        instances.  It deals with the complexities therein such that it
        should be easy for the user to specify the links even manually.
        Arguments:
          sample_pools: Some kind of iterable containing the sample
            pools.  Specifically, tuples, lists and dicts (with the
            SampleContainer instances as values) were tested to work.
          total_folds: The total amount of folds the data is supposed
            to be split into.
          fold: The fold for which this instance is supposed to serve.
            Mathematically, this tells the container how much the
            interval will need to be shifted.
          start: The first chunk (inclusive) that should belong to the
            set.  Note: The function will perform the shifting, so this
            value is supposed to be the base one.
          end: The last chunk (exclusive) that should belong to the
            set.  Note: The function will perform the shifting, so this
            value is supposed to be the base one.
        """
        # For convenience, this function accepts (among others) a dict
        # as 'sample_pools'.  It can furthermore include empty pools,
        # that we silently drop right away to simplify further
        # processing.
        if isinstance(sample_pools, dict):
            sample_pools = sample_pools.values()
        self.sample_pools_list = []
        for pool in sample_pools:
            if pool.get_current_length() > 0:
                self.sample_pools_list.append(pool)
        # Next, we check whether all the pools line up in terms of
        # their data structure.  We do this by comparing their
        # initialization and the disturbances in each of their base
        # data tensors as these should catch all relevant
        # misalignments.  We won't try to recover from an error here
        # as it would be far too complicated to automatically fix.
        # Instead, we simply error out.
        try:
            pool_zero = self.sample_pools_list[0]
        except IndexError:
            NNLibUsageError("No datasets found. This most likely means that "
                            "no samples were created or loaded yet.")
        total_samples = 0
        # We create a list of integers identifying where we cross from
        # one pool to the next when concatenating the data later.
        split_points = [0]
        for pool in self.sample_pools_list:
            if ((pool_zero._init_call_dict != pool._init_call_dict)
                  or (pool_zero.checkout_field('basedata')['dst'].shape[1:]
                        != pool.checkout_field('basedata')['dst'].shape[1:])):
                raise NNLibUsageError("The data in all pools needs to be the "
                                      "same shape.")
            total_samples += pool.get_current_length()
            split_points.append(total_samples)
        chunk_length = total_samples / total_folds
        if chunk_length != int(chunk_length):
            print("Warning: Cannot perform an exact split due to the amount "
                  "of total folds ({}) and total samples ({}). Be aware that "
                  "this will lead to somewhat inexact comparisons between "
                  "folds.".format(total_folds, total_samples))
        chunk_length = int(chunk_length)
        # Define the ranges to pull, first in terms of total samples.
        start_shift = (start-fold) % total_folds
        end_shift = (start_shift+end) % total_folds
        if start_shift > end_shift:
            concat_ranges = [(0,end_shift*chunk_length),
                             (start_shift*chunk_length,total_samples)]
        else:
            concat_ranges = [(start_shift*chunk_length,end_shift*chunk_length)]
        # Now transform it into pull ranges in terms of the individual
        # SampleContainers.
        concat_list = []
        pool_pointer = 0
        for subrange in concat_ranges:
            # First, we need to find the starting pool.
            while subrange[0] >= split_points[pool_pointer+1]:
                pool_pointer += 1
            # Save the transformed starting index.
            lo = subrange[0] - split_points[pool_pointer]
            # Next, find out if the range's end point lies within the
            # same pool. If it doesn't, we need to define more
            # independent slices in the list.
            while subrange[1] > split_points[pool_pointer+1]:
                concat_list.append(
                  (self.sample_pools_list[pool_pointer],lo,
                   self.sample_pools_list[pool_pointer].get_current_length())
                )
                lo = 0
                pool_pointer += 1
            concat_list.append((self.sample_pools_list[pool_pointer],
                                lo,subrange[1]-split_points[pool_pointer]))
        self.concat_list = concat_list

    def checkout_field(self, field_identifier, sl=slice(0,None)):
        """Checking out fields on VirtualSampleContainer instances
        relies on checking out the respective fields in its linked
        SampleContainer instances, referring back to the concat_list
        for the information about what data to concatenate.
        """
        gathered = []
        for list_item in self.concat_list:
            gathered.append(list_item[0].checkout_field(field_identifier,
                                                        slice(list_item[1],
                                                              list_item[2])))
        return np.concatenate(gathered, axis=0)[sl]

    def build_additional_fields(self, field_identifier):
        """Instructions for building additional fields are simply
        passed down to the associated SampleContainer instances.
        """
        for cont in self.sample_pools_list:
            cont.build_additional_fields(field_identifier)


class SampleContainer(SampleContainerInterface):
    """Serves as a container for training and validation data.  This
    has been outsourced to make ModelHandler less convoluted as well as
    to enable easy data sharing between ModelHandler instances.
    """

    def __init__(self,
                 field_size=_STD_SETTINGS['FIELD_SIZE'],
                 field_depth=_STD_SETTINGS['FIELD_DEPTH'],
                 ts_length=_STD_SETTINGS['TS_LENGTH'],
                 separate_disturbances=_STD_SETTINGS['SEP_DISTURBANCES'],
                 label_layout=_STD_SETTINGS['PARAM_SET'],
                 full_layout=_STD_SETTINGS['PARAM_SET']):
        self.loadstore_instr = {
          'init': (self._retrieve_init_settings,
                     self._init_variable_settings,False),
          'fields': (self._retrieve_fields,self._set_up_fields,False),
        }
        # It is __init__'s job to build a dict of settings to be passed
        # to _init_variable_settings.  This aims to simplify loading
        # and saving since it allows the push operations to directly
        # shoot at the init functions independent of data type.
        settings = {'field_size': int(field_size),
                    'field_depth': int(field_depth),
                    'ts_length': int(ts_length),
                    'separate_disturbances': bool(separate_disturbances),
                    'label_layout': label_layout,
                    'full_layout': full_layout}
        self._init_variable_settings(settings)
        self._set_up_fields()
        if MOD_SETTINGS['DEBUG']:
            self.skipped_combinations = []

    def __deepcopy__(self, memo):
        new_container = SampleContainer()
        cp = Copier(self, new_container)
        cp.copy('all', True)
        return new_container

    def _retrieve_init_settings(self, dpcopy):
        if dpcopy:
            return deepcopy(self._init_call_dict)
        else:
            return self._init_call_dict

    def _init_variable_settings(self, settings):
        self._init_call_dict = settings
        self.ts_length = settings['ts_length']
        self.field_size = settings['field_size']
        self.field_depth = settings['field_depth']
        # The separate_disturbances setting is new, therefore we
        # provide a fallback.
        try:
            self.separate_disturbances = settings['separate_disturbances']
        except KeyError:
            self.separate_disturbances = _STD_SETTINGS['SEP_DISTURBANCES']
        self.label_layout = settings['label_layout']
        self.full_layout = settings['full_layout']

    def _retrieve_fields(self, dpcopy):
        if dpcopy:
            ret_list = []
            for to_copy in ('fields','nodistfields','origfields',
                            'labels','basedata','objtype','objcount'):
                try:
                    ret_list.append(self.contents[to_copy].copy())
                except AttributeError:
                    ret_list.append(None)
        else:
            ret_list = [self.contents['fields'],self.contents['nodistfields'],
                        self.contents['origfields'],
                        self.contents['labels'],self.contents['basedata'],
                        self.contents['objtype'],self.contents['objcount']]
        return ret_list

    def _set_up_fields(self, fields_list=None):
        """Sets up the internal data fields.
        CHANGED: If checks clear, this function will now attach data
        rather than replacing it.  To make sure data will be replaced,
        call _set_up_fields twice: First without arguments, then with
        new fields.
        """
        # If fields_list is None, we create a dict of Nones and return.
        if fields_list is None:
            self.contents = {'fields': None,
                             'nodistfields': None,
                             'origfields': None,
                             'labels': None,
                             'basedata': None,
                             'objtype': None,
                             'objcount': None}
            return

        # If the following test is True, we have a legal fields_list as input
        # which we expand to a preliminary content mapping.
        if all(((isinstance(item, np.ndarray) or item is None)
                  for item in fields_list)):
            if len(fields_list) == 7:
                new_content = {'fields': fields_list[0],
                               'nodistfields': fields_list[1],
                               'origfields': fields_list[2],
                               'labels': fields_list[3],
                               'basedata': fields_list[4],
                               'objtype': fields_list[5],
                               'objcount': fields_list[6]}
            elif len(fields_list) == 6:
                new_content = {'fields': fields_list[0],
                               'nodistfields': fields_list[1],
                               'origfields': None,
                               'labels': fields_list[2],
                               'basedata': fields_list[3],
                               'objtype': fields_list[4],
                               'objcount': fields_list[5]}
            else:
                raise TypeError("fields_list needs to be a list of seven "
                                "(six for fallback) numpy arrays to "
                                "constitute a full set of field "
                                "specifications.")
        else:
            raise TypeError("fields_list needs to be a list of numpy arrays "
                            "and/or NoneTypes to constitute a set of field "
                            "specifications.")

        # If we land here, we need to decide what to do with the data.
        # The rules are as follows:
        #   - If internal fields are empty (test by looking at
        #     basedata), use new_content as self.contents.
        #   - If internal fields are incompatible with the new content,
        #     replace old content.  This is not the safest way to do it
        #     per se, but it ensures backwards compatibility.
        #   - It internal fields and new content are compatible,
        #     append.
        if self.contents['basedata'] is None:
            self.contents = new_content
        elif not self._check_compat(new_content):
            print("Incompatilbe: New content replaces old content")
            self.contents = new_content
        else:
            print("Compatible: New content will be attached")
            for key in self.contents:
                if self.contents[key] is not None:
                    if new_content[key] is None:
                        # At this point we can safely assume that
                        # building the additional fields will work
                        # because _check_compat checks for that with a
                        # dryrun.
                        self.build_additional_fields(key, new_content)
                    self.contents[key] = np.append(self.contents[key],
                                                   new_content[key],
                                                   0)

    def _start_multithreaded(self, total_samples, creator_list,
                             param_pack, data_pack=None):
        """This function was outsourced in nnlib_multiobject_v06.  It
        contains all setup steps to initiate (multithreaded) data
        generation based on the newly introduced atomic Creators.  It
        is entirely data agnostic, so it depends on the caller to
        provide sensible lists of creators, parameters and base data
        and to make sense of the dict that will be passed back.
        The backend function corresponding to this is _mt_starter,
        which is called for every separate thread this function spawns.
        """
        # Some setup for the multiprocessing pool:
        # Setting the module constant OVERRIDE_THREAD_COUNT to 1
        # disables multithreading entirely. That makes sense if the
        # sample size is sufficiently small because starting the
        # workers produces non-negligible overhead.  Also useful for
        # debugging the backend functions.
        if MOD_SETTINGS['OVERRIDE_THREAD_COUNT'] != 1:
            # OVERRIDE_THREAD_COUNT==False (the standard setting)
            # spawns as many workers as there are logical CPU cores
            # (capped at MAX_MULTITHREAD_CORECOUNT).  A positive
            # integer spawns that amount of workers.
            if not MOD_SETTINGS['OVERRIDE_THREAD_COUNT']:
                corecount = min(mp.cpu_count(),
                                MOD_SETTINGS['MAX_MULTITHREAD_CORECOUNT'])
            else:
                corecount = int(MOD_SETTINGS['OVERRIDE_THREAD_COUNT'])
                if corecount < 1:
                    raise ValueError("The requested thread count needs to be "
                                     "a positive value.")
            # Status info.
            print("Checks completed. Starting child processes...")
            # Calculate an even split and the necessary rest of samples
            # to be processed.
            split = total_samples // corecount
            rest = total_samples % corecount
            # Initialize the multithreading pool.
            pool = mp.Pool(processes=corecount)
            reslist = []
            # Fill up the workers with the following for-loop.
            slice_start = 0
            for thread in range(corecount):
                param_pack_samples = param_pack.copy()
                samples = split + int(thread < rest)
                param_pack_samples['samples'] = samples
                # We need to split the contents of data_pack (if it is
                # not None) accordingly as well.  We depend on all data
                # tensors to be of fitting length and ordered in the
                # same way because only then this can be done in a
                # generic fashion.
                if data_pack is not None:
                    data_pack_slice = {}
                    for dataset in data_pack:
                        data_pack_slice[dataset] = (
                          data_pack[dataset][slice_start:slice_start+samples]
                        )
                else:
                    data_pack_slice = None
                # CHANGED in nnlib_multiobject_v04: The work is now
                # split more evenly by increasing the amount of
                # samples to process by one until all rest samples are
                # assigned instead of assigning all of them to the
                # first thread to start, potentially leading to a very
                # uneven work distribution on high thread count CPUs.
                reslist.append(pool.apply_async(_mt_starter,
                                                [thread,
                                                 creator_list,
                                                 param_pack_samples,
                                                 data_pack_slice]))
                slice_start += samples
            # Close the pool for other jobs and wait for the
            # calculations to finish.
            pool.close()
            # To initialize the respective fields, copy results from
            # worker zero...
            return_coll = reslist[0].get().copy()
            # Status info
            print("Child processes finished. Collecting data...")
            # ...and append the ones from the other workers.
            for element in reslist[1:]:
                for entry in return_coll:
                    return_coll[entry] = np.append(return_coll[entry],
                                                   element.get()[entry],
                                                   axis=0)
            # Stop the pool's workers, terminate their processes.
            pool.join()
        else:
            # Status info.
            print("Checks completed. Multithreading is disabled. Starting "
                  "calculation...")
            # CHANGED in nnlib_multiobject_v06: We go through
            # _mt_starter even if multithreading is disabled as this
            # function makes handling the returns easier without
            # introducing significant new error sources.
            param_pack['samples'] = total_samples
            return_coll = _mt_starter(0, creator_list, param_pack, data_pack)
        return return_coll

    def _check_compat(self, new_content):
        # Assume we are compatible and falsify.
        compat = True
        for key in self.contents:
            if self.contents[key] is not None:
                if new_content[key] is not None:
                    if (self.contents[key].shape[1:]
                          != new_content[key].shape[1:]):
                        compat = False
                        break
                else:
                    try:
                        build_additional_fields(key, new_content, True)
                    except NNLibUsageError:
                        compat = False
                        break
        return compat

    def get_current_length(self):
        """A little convenience function to tell the caller the length
        of the fields inside the Container.
        """
        try:
            return len(self.contents['basedata'])
        except TypeError:
            # TypeError is raised when basedata is None, indicating the
            # Container is unused.
            return 0

    def load_fields(self, path='Handler'):
        """Loads the fields from another handler or a save file into
        this handler to more easily reuse existing fields even with
        changed parameters.
        CHANGED in mo_08: The backend will now try to add the contents
        from the source to this handler.  Only if this is not possible
        for some reason, the existing data will be replaced.
        """
        # Little plausibility check.  If the path the user gave is to
        # another existing SampleContainer object, a warning is given
        # if its fields are empty as this may point to a misuse of this
        # function, potentially resulting in unwanted data loss.
        if isinstance(path, six.string_types):
            fhandler = H5PyHandler(path, True)
            path = fhandler.interactive_provide_access('SampleContainer')
        elif path.pull('type') == 'SampleContainer':
            if (_interactive and path.get_current_length() == 0):
                print("Warning: The fields in the SampleContainer instance "
                      "to copy from seem to not be assigned.")
                print("Continuing will convert the respective fields in this "
                      "instance (the one whose 'load_fields' method you "
                      "called) into NoneTypes, deleting all data that was "
                      "there before.")
                resp = input("Continue anyway? [yes/NO] > ")
                if resp not in string_alts['yes']:
                    return
        else:
            raise NNLibUsageError("Cannot load fields. They are either "
                                  "missing or the source does not look like "
                                  "a SampleContainer.")
        cp = Copier(path, self)
        cp.copy('fields')
        print('OK')

    def save_fields(self, path='Handler'):
        """Saves the fields contained in the SampleContainer instance
        to disk.  Does not accept anything other than a save path as an
        input.  Use load_fields on the new SampleContainer if you want
        to copy data over from one to another.
        """
        fhandler = H5PyHandler(path, False)
        access = fhandler.provide_access('SampleContainer', self.pull('id'))
        cp = Copier(self, access)
        cp.copy('all')
        print('OK')

    def checkout_field(self, field_identifier, sl=slice(0,None)):
        """This function handles the interaction of ModelHandler with
        SampleContainer from nnlib_multiobject_v06 onward.  The
        advantages over directly accessing the internal fields are that
        this function can mask internal name changes and trigger builds
        on the fly for certain fields that do not need external inputs.
        """
        try:
            if self.contents[field_identifier] is None:
                print("'{}' is empty.".format(field_identifier))
                if (field_identifier == 'origfields'
                      and not self.separate_disturbances):
                    raise NNLibUsageError("Original fields are unavailable "
                                          "(separate_disturbances was "
                                          "disabled for this container).")
                elif field_identifier in {'fields','origfields',
                                          'labels','basedata'}:
                    raise NNLibUsageError("Cannot build automatically. "
                                          "Import some data or run "
                                          "'prepare_full_fields' to get "
                                          "going.")
                else:
                    print("Trying to build it now...")
                    self.build_additional_fields(field_identifier)
            return self.contents[field_identifier][sl]
        except KeyError:
            raise ValueError("Unknown identifier '{}'"
                             ".".format(field_identifier))

    def compile_into_fields(self,
                            base_data=None,
                            ts_variation=1.0,
                            ts_movement=True,
                            coords_as_floats=True,
                            add_background=_STD_SETTINGS['ADD_BACKGROUND']):
        """Prepare the fields for training, validation, etc. based on
        existing base data.  Multithreaded execution is available and
        recommended (manual override via the module constant
        OVERRIDE_THREAD_COUNT=<thread count>) because this process is
        intense on the hardware and will take a while.
        Arguments:
          base_data: Supply base data here.  If left as None, internal
            data will be used.
            CHANGED in nnlib_multiobject_v04: Implicitly takes over
            for the old 'source' parameter which has been removed.
            CHANGED in nnlib_multiobject_v04: The base_data supplied
            hereby is also saved in the respective field of this
            object.  This is to ensure internal consistency of data.
            CHANGED in nnlib_multiobject_v06: There is no separate
            interface to add base_data to the existing tensor anymore.
            The rule from now on is simply that if you supply base_data
            and it checks out to combine it with the existing data, it
            is attached and the respective fields rendered and attached
            as well.  If the standard setting of None is used, the
            existing base data is taken and rendered again, overwriting
            any existing working data.
          ts_variation: Specifies how much the objects wiggle on their
            deterministic position for each piece of the timeseries.
            This was implemented in the hopes that it will stop the
            network from learning only fixed linear movement patterns
            but rather to abstract the concept.
          ts_movement: Specifies if the objects are supposed to be
            moving or stand still in a timeseries.
          coords_as_floats: Specifies if we allow for objects to be
            positioned exactly (down to machine accuracy) in the
            field.
          add_background: A number between 0 (inclusive) and 1
            (exclusive) specifying the probability for a line to be
            inserted in the field as an additional distraction.  The
            insertion algorithm stops when the rng first returns a
            float greater than add_background and continues adding if
            it is smaller.  This means that with 0, nothing will ever
            be added, and with 1, that very many lines are added.  Note
            that this argument gets checked for basic sensibility.
        """
        # The standard setting for base_data is None, meaning they
        # should be pulled from the Container's own internal fields.
        if base_data is None:
            # If the internal field is empty, too, we have no way of
            # continuing and therefore error out.
            if self.contents['basedata'] is None:
                raise NNLibUsageError("Both the SampleContainer instance and "
                                      "the base_data input are empty. This "
                                      "way, we cannot produce any output.")
            # Otherwise, assign the internal data to the base_data
            # variable so we do not need multiple cases further down
            # the line.
            base_data = self.contents['basedata']
            # overwrite = True will later tell us to rewrite the
            # internal base data field.
            overwrite = True
        else:
            # In case base_data was indeed provided externally, we need
            # to check it for compatibility.
            if ((base_data['data'].shape[1:]
                   != self.contents['basedata']['data'].shape[1:])
                  or (base_data['dst'].shape[1:]
                        != self.contents['basedata']['dst'].shape[1:])):
                # If base data already exists internally, but the
                # lengths don't fit, we error out.
                raise NNLibUsageError("Cannot add to existing fields because "
                                      "the shapes of existing and to-be-used "
                                      "base data do not fit. Expand or "
                                      "contract the existing data or "
                                      "overwrite it entirely by setting "
                                      "add_to_existing to False.")
            else:
                # If base data has been defined before and it does
                # indeed have the same length, we continue with no
                # changes.  Note that, due to the redesign of the
                # infrastructure, the new base data will implicitly be
                # appended to the existing when return_coll is
                # processed.
                # overwrite = False will later tell us to append the
                # results.
                overwrite = False
        # Check the add_background argument.
        try:
            if add_background < 0.0 or add_background >= 1.0:
                raise ValueError("Illegal choice of 'add_background': A "
                                 "float between 0.0 (incl., no background "
                                 "objects) and 1.0 (excl., very many "
                                 "background objects) is required. (False "
                                 "works, too, but not True!)")
        except TypeError:
            raise TypeError("Invalid data type for 'add_background': Must be "
                            "a float between 0.0 (incl., no background "
                            "objects) and 1.0 (excl., very many background "
                            "objects). (False works, too, but not True!)")
        # For legacy compatibility and a better user experience,
        # ts_variation is provided in pixels.  Since all Creators work
        # on normalized values only, we need to scale it.
        ts_var_scaled = 1.0 * ts_variation / self.field_size
        # Prepare the list of Creators to call.  Since this function
        # supports introducing additional base data, we need to pay
        # extra attention to keep objtype, objcount and nodistfields in
        # check, i.e. their additions or changes need to be created as
        # well when this function runs.
        creator_list = [WorkingDataCreator,DisturbanceCreator]
        if self.contents['objtype']:
            creator_list.append(OnehotCreator)
        if self.contents['objcount']:
            creator_list.append(ObjCountCreator)
        if self.contents['nodistfields']:
            creator_list.append(DistFreeCreator)
        # Prepare the parameter pack for the Creators.
        param_pack = {'field_size': self.field_size,
                      'field_depth': self.field_depth,
                      'ts_length': self.ts_length,
                      'separate_disturbances': self.separate_disturbances,
                      'label_layout': self.label_layout,
                      'full_layout': self.full_layout,
                      'ts_variation': ts_var_scaled,
                      'ts_movement': ts_movement,
                      'coords_as_floats': coords_as_floats,
                      'add_background': add_background}
        data_pack = {'basedata': base_data}
        # Start the multithreaded creation process.
        total_samples = len(base_data)
        return_coll = self._start_multithreaded(total_samples, creator_list,
                                                param_pack, data_pack)
        # The following puts the results where they belong.  Since
        # return_coll and self.contents are designed to look the same
        # in terms of legal keys and value structure, this is very easy
        # to do.
        if overwrite:
            self.contents.update(return_coll)
        else:
            for key in return_coll:
                self.contents[key] = np.append(self.contents[key],
                                               return_coll[key],
                                               axis=0)
        # Indicate that everything (seemingly) went alright.
        print('OK')

    def build_additional_fields(self,
                                field_identifier,
                                dataset=None,
                                dryrun=False):
        """This function triggers builds of additional fields if it can
        do so without further input.
        CHANGED: Now supports working on datasets apart from the
          internal ones.  Still uses internal settings, however.
          Useful for importing datasets into the handler.
        CHANGED: Added a dryrun option.  Checks requirements and issues
          errors, but does not perform the actual operation.  Useful
          for checking compatibility.
        """
        if dataset is None:
            dataset = self.contents
        # In the following, we go through the legal choices for
        # field_identifier to raise useful errors where appropriate or
        # arm the Creators with the data they will need.
        if field_identifier == 'basedata':
            raise NNLibUsageError("'basedata' cannot be built with generic "
                                  "settings. Run 'prepare_full_fields' "
                                  "instead.")
        elif field_identifier in {'fields', 'labels', 'origfields'}:
            raise NNLibUsageError("'{}' cannot be built with generic "
                                  "settings. Run 'prepare_full_fields' or "
                                  "'compile_into_fields' "
                                  "instead.".format(field_identifier))
        elif field_identifier == 'nodistfields':
            if (dataset['labels'] is None
                  or dataset['basedata'] is None):
                raise NNLibUsageError("'nodistfields' cannot be built "
                                      "because the necessary data ('labels' "
                                      "and 'basedata') is missing.")
            creator_list = [DistFreeCreator]
            param_pack = {'field_size': self.field_size,
                          'field_depth': self.field_depth,
                          'full_layout': self.full_layout,
                          'label_layout': self.label_layout}
            data_pack = {e: dataset[e] for e in ('basedata','labels')}
        elif field_identifier in {'objtype','objcount'}:
            if self.contents['basedata'] is None:
                raise NNLibUsageError("'{}' is empty and cannot be built "
                                      "because the necessary data "
                                      "('basedata') is "
                                      "missing.".format(field_identifier))
            creator_list = [OnehotCreator]
            param_pack = {'ts_length': self.ts_length}
            data_pack = {'basedata': dataset['basedata']}
        else:
            # Since this method may be called by a user, we provide
            # some actual feedback on what went wrong if we land here.
            raise NNLibUsageError("The key '{}' is "
                                  "unknown.".format(field_identifier))
        # Triggering the actual creation and inserting the results is
        # for all above cases.
        if not dryrun:
            total_samples = self.get_current_length()
            return_coll = self._start_multithreaded(total_samples,
                                                    creator_list,
                                                    param_pack,
                                                    data_pack)
            dataset[field_identifier] = return_coll[field_identifier]

    def prepare_full_fields(self,
                            samples=4096,
                            add_to_existing=True,
                            mask_size=_STD_SETTINGS['MASK_SIZE'],
                            shapes=std_shapes_all,
                            nd_scale=0.25,
                            ts_movement=True,
                            ts_variation=1.0,
                            include_nones=True,
                            min_distract_objects=1,
                            max_distract_objects=1,
                            min_intensity=0.5,
                            max_intensity=1.0,
                            coords_as_floats=True,
                            add_background=_STD_SETTINGS['ADD_BACKGROUND']):
        """Prepare the fields for training, validation, etc. by first
        creating new base data and then directly compiling that into
        new fields. Multithreaded execution is available and
        recommended because this process is intense on the hardware and
        will take a while.
        Arguments:
          samples: The amount of dataset samples to be created.  Note
            that the amount of actual images drawn is
            (samples * self.ts_length).  Only has an effect in 'full'
            mode.  Otherwise, the result's length depends on the length
            of the base data supplied.
          add_to_existing: True means the newly created data will be
            attached to the existing (if it exists), False overwrites
            it.  If add_to_existing is True, the new data needs to
            have the same form as the existing.  An error will be
            raised if that is not the case.  Provisions have been made
            to raise the error before any calculations.
          mask_size: Specify the mask size for which to build the
            objects (creating a fitting lower limit on object size).
          shapes: Tuple of shapes (indicated by their name strings)
            that the samples should consist of.  Has no effect in
            'compile_only' mode.
          nd_scale: Scales the normal distribution from which the
            objects' size parameters are drawn.  Smaller values
            increasingly favor smaller objects.  Set to 0 or False for
            an equal distribution.
          ts_movement: Specifies if the objects move within a
            timeseries.
          ts_variation: Specifies how much the objects wiggle on their
            deterministic position for each piece of the timeseries.
            This was implemented in the hopes that it will stop the
            network from learning only fixed linear movement patterns
            but rather to abstract the concept.
          include_nones: Specifies if 'no object' is an accepted
            outcome of the object type randomization.  This has no
            effect in 'compile_only' mode.
          min_distract_objects: The amount of distracting objects
            minimally inserted into the fields.  Distraction objects
            are either smaller than the main object or bigger than it,
            but in the latter case they get subsequently imperfectly
            eliminated from the fields so a halo-like rest of the
            object may be visible.  The neural network is supposed to
            learn to ignore both kinds of distraction.
          max_distract_objects: The amount of distraction objects
            inserted at most into each field.
          coords_as_floats: Specifies if we allow for objects to be
            positioned exactly (down to machine accuracy) in the
            field.
          add_background: A number between 0 (inclusive) and 1
            (exclusive) specifying the probability for a line to be
            inserted in the field as an additional distraction.  The
            insertion algorithm stops when the rng first returns a
            float greater than add_background and continues adding if
            it is smaller.  This means that with 0, nothing will ever
            be added, and a number close to 1 generally means that
            very many lines are added.  It gets checked whether this
            argument received a sensible value.
        """

        # -- Private functions --

        def _get_shapelist_from_str(string):
            """Turn a string given as the shape argument into a tuple
            of strings that can be used by the respective functions
            with regards to _shape_dict. Makes prepare_full_fields
            better usable by functions that are external to this module
            and via manual user input.
            """
            if string in string_alts['all']:
                return std_shapes_all
            elif string in string_alts['conservative']:
                return std_shapes_conservative
            elif string in std_shapes_all:
                return (string,)
            else:
                raise ValueError("I do not know the key you gave for "
                                 "'shapes'.")

        # -- Function body --

        # Check the shapes argument
        if isinstance(shapes, six.string_types):
            shapes = _get_shapelist_from_str(shapes)

        # Check the add_background argument.
        try:
            if add_background < 0.0 or add_background >= 1.0:
                raise ValueError("Illegal choice of 'add_background': A "
                                 "float between 0.0 (incl., no background "
                                 "objects) and 1.0 (excl., very many "
                                 "background objects) is required. (False "
                                 "works, too, but not True!)")
        except TypeError:
            raise TypeError("Invalid data type for 'add_background': Must be "
                            "a float between 0.0 (incl., no background "
                            "objects) and 1.0 (excl., very many background "
                            "objects). (False works, too, but not True!)")

        # For legacy compatibility and a better user experience,
        # ts_variation is provided in pixels.  Since all Creators work
        # on normalized values only, we need to scale it.
        ts_var_scaled = 1.0 * ts_variation / self.field_size

        # Keep track of the creators we spawn and the data that the
        # results need to go to.
        # Standard operation for prepare_full_fields includes creating
        # base data as well as labels and fields.
        creator_list = [BaseDataCreator,WorkingDataCreator,DisturbanceCreator]
        # add_to_existing makes keeping data in sync more complicated.
        # Specifically, lengths need to line up and fields that have
        # already been used need to be updated with the new data.
        if add_to_existing and self.get_current_length() > 0:
            # Check if add_to_existing can work.
            if (max_distract_objects
                  != self.contents['basedata']['dst'].shape[1]):
                raise NNLibUsageError("Cannot add to existing fields because "
                                      "the value you gave for "
                                      "'max_distract_objects' does not fit "
                                      "the existing data. Expand or contract "
                                      "the existing data or overwrite it "
                                      "entirely by setting add_to_existing "
                                      "to False.")
            # For each field that isn't empty at this point, the
            # respective Creator needs to be attached.  Otherwise, some
            # fields will be shorter than the base ones, which would be
            # a lot more difficult to deal with than if we just extend
            # them right away.
            if self.contents['nodistfields'] is not None:
                creator_list.append(DistFreeCreator)
            if self.contents['objtype'] is not None:
                creator_list.append(OnehotCreator)
            if self.contents['objcount'] is not None:
                creator_list.append(ObjCountCreator)
        else:
            # Just for convenience, we set add_to_existing to False.
            # This only changes something if it was True, but no data
            # was present, in which case we save checking again later.
            add_to_existing = False
        # CHANGED in nnlib_multiobject_v06: The backend functions are
        # now implemented as atomic creators, each of which performs
        # exactly one part of the data generation process.  This
        # function prepares a pack of parameters as well as a list of
        # Creators that are to be run consecutively.
        param_pack = {'field_size': self.field_size,
                      'field_depth': self.field_depth,
                      'ts_length': self.ts_length,
                      'label_layout': self.label_layout,
                      'full_layout': self.full_layout,
                      'separate_disturbances': self.separate_disturbances,
                      'mask_size': mask_size,
                      'shapes': shapes,
                      'nd_scale': nd_scale,
                      'ts_movement': ts_movement,
                      'ts_variation': ts_var_scaled,
                      'include_nones': include_nones,
                      'min_distract_objects': min_distract_objects,
                      'max_distract_objects': max_distract_objects,
                      'min_intensity': min_intensity,
                      'max_intensity': max_intensity,
                      'coords_as_floats': coords_as_floats,
                      'add_background': add_background}
        # CHANGED in nnlib_multiobject_v06: Call _start_multithreaded
        # to trigger the calculations themselves.
        return_coll = (
          self._start_multithreaded(samples, creator_list, param_pack)
        )
        # The following handles where to put the results. This function
        # can write to both internal datasets (validation and training)
        # or return the generated fields to the caller.
        if add_to_existing:
            for key in return_coll:
                self.contents[key] = np.append(self.contents[key],
                                               return_coll[key],
                                               axis=0)
        else:
            self.contents.update(return_coll)
        # Indicate that everything (seemingly) went alright.
        print('OK')

    def visualize_fields(self,
                         distraction_free=False,
                         disturbance_free=False,
                         total_images=64,
                         start_sample=0,
                         ts_show_every=4,
                         dp_show_every=4,
                         mode='show',
                         save_draw_options={}):
        """Plots the fields (with or without disturbances) using
        multishow.
        Arguments
          show_dist_free: If True, the disturbance free fields will be
            drawn.  If False, the fields will be drawn the way they are
            passed to a neural net.
          total_images: The total number of fields to extract and
            draw/save.
          start_sample: The sample from which to start with the
            drawing.
          ts_show_every: The period for the timeseries dimension.  (For
            example, the standard setting of 4 means that only every
            fourth time instance is drawn at all.)
          dp_show_every: The period for the depth dimension.  (For
            example, the standard setting of 4 means that only every
            fourth depth is drawn.)
          mode: Selects what to do with the resulting output.
            Currently supported options are 'show' (draw the results
            using 'multishow' in keras_support_lib) and 'save' (save
            the results as png images using 'multisave' in
            keras_support_lib).
          save_draw_options: A dict containing further options for
            multishow and multisave, respectively.  Allows to easily
            override their standard arguments.
        """
        samples_to_draw = ((total_images*ts_show_every*dp_show_every)
                             // (self.ts_length*self.field_depth))
        stop_sample = start_sample + samples_to_draw + 1
        # Build the slice tuple that will be passed to the checkout
        # function to not move unnecessary amounts of data.
        sam_sl = slice(start_sample, stop_sample)
        ts_sl = slice(0, None, ts_show_every)
        dp_sl = slice(0, None, dp_show_every)
        tot_sl = (sam_sl,ts_sl,dp_sl)
        if distraction_free:
            fields_slice = self.checkout_field('nodistfields', tot_sl)
        elif disturbance_free:
            fields_slice = self.checkout_field('origfields', tot_sl)
        else:
            fields_slice = self.checkout_field('fields', tot_sl)
        new_shape = ((np.prod(fields_slice.shape[0:3]),)
                       + fields_slice.shape[3:])
        flattened_slice = np.reshape(fields_slice, new_shape)
        if mode == 'show':
            # Hand the whole thing over to multishow.
            multishow(flattened_slice, **save_draw_options)
        elif mode == 'save':
            # Hand the slice over to multisave.
            multisave(flattened_slice, **save_draw_options)
        else:
            raise NNLibUsageError("I do not understand the key '{}' for "
                                  "input 'mode'.".format(mode))


register_container(SampleContainer)


@six.add_metaclass(ABCMeta)
class ModelContainer(FunctionalContainer):

    def __init__(self, folds=1, silent=False):
        settings = {'silent': bool(silent),
                    'total_folds': int(folds)}
        self._init_variable_settings(settings)
        self.changeable = True
        self.fold = 0
        self.compile_args = None
        self.model_holder = self.total_folds * [None]
        self.model = None
        # Create all holders...
        self.add_funcs_holder = []
        self.weights_hist_holder = []
        self.weights_lookup_holder = []
        self.training_hist_holder = []
        # ...and populate them in a for loop.
        for i in range(self.total_folds):
            # NOTE: Doing this in a for loop makes sure each element
            # of the holders is a different mutable instance and not
            # just different references to the same instance.
            self.weights_hist_holder.append([])
            self.weights_lookup_holder.append([])
            self.training_hist_holder.append({})
            self.add_funcs_holder.append([])
        # Select the first element of each holder.
        self.weights_hist = self.weights_hist_holder[0]
        self.weights_lookup = self.weights_lookup_holder[0]
        self.training_hist = self.training_hist_holder[0]
        self.add_funcs = self.add_funcs_holder[0]
        self.loadstore_instr = {
          'init': (self._retrieve_init_settings,
                     self._init_variable_settings,True),
          'struct': (self._retrieve_build_specs,self._rebuild,False),
          'hist': (self._retrieve_hist,self._insert_hist,True),
          'compile': (self._retrieve_compile_specs,self._recompile,False),
        }

    def _init_variable_settings(self, settings):
        self.total_folds = settings['total_folds']
        self.silent = settings['silent']
        self._init_call_dict = settings

    def _retrieve_init_settings(self, dpcopy):
        if dpcopy:
            return deepcopy(self._init_call_dict)
        else:
            return self._init_call_dict

    def _retrieve_compile_specs(self, dpcopy):
        if dpcopy:
            try:
                return self.compile_args.copy()
            except AttributeError:
                return None
        else:
            return self.compile_args

    def _find_epoch(self, index):
        """Finds the epoch number for which we saw the best results on
        'index' during training and returns it.

        Parameters
        ----------
        index : {'loss', 'val_loss'}
          The index to look for in the training history.
        """
        epoch = np.argmin(self.training_hist[index])
        value = self.training_hist[index][epoch]
        print("Best result on {} found in epoch {} with a value of "
              "{}.".format(index, epoch, value))
        # We need to add one to the obtained argmin value since the
        # internal weight history is always one element longer than the
        # training history.  This is because the weight history
        # contains the weights the network was initialized with as its
        # first entry.
        return epoch + 1

    def _recompile(self, cdict):
        if cdict is None:
            # If cdict was saved as None, it means that the model
            # wasn't compiled in the first place.  In this case, we
            # silently pass and do nothing.
            pass
        else:
            # Since we do not want to impose an order in which the
            # individual parts of the container must be restored, we
            # either save the dictionary or compile directly.  If it is
            # only saved, it is the job of '_rebuild' to start the
            # recompilation.
            if self.changeable:
                self.compile_args = cdict
            else:
                self.compile(**cdict)

    @abstractmethod
    def _retrieve_hist(self, dpcopy):
        pass

    @abstractmethod
    def _insert_hist(self, hist):
        pass

    @abstractmethod
    def _retrieve_build_specs(self, dpcopy):
        pass

    @abstractmethod
    def _rebuild(self, data):
        pass

    @contextmanager
    def access_fold(self, fold):
        """Provides access to folds through the old object variables
        via a 'with' structure.  Also takes care of resetting them to
        fold zero (which exists even if folds are entirely disabled)
        when leaving it.
        """
        self.fold = fold
        self.model = self.model_holder[fold]
        self.weights_hist = self.weights_hist_holder[fold]
        self.weights_lookup = self.weights_lookup_holder[fold]
        self.training_hist = self.training_hist_holder[fold]
        self.add_funcs = self.add_funcs_holder[fold]
        yield
        self.fold = 0
        self.model = self.model_holder[0]
        self.weights_hist = self.weights_hist_holder[0]
        self.weights_lookup = self.weights_lookup_holder[0]
        self.training_hist = self.training_hist_holder[0]
        self.add_funcs = self.add_funcs_holder[0]

    def _attach_hist(self, hist):
        """The appending of training history is performed by this
        special function.
        CHANGED in mo_08: The training_hist dicts are initialized at
        init time now, so the NoneType case is no longer necessary.
        """
        # Try to add to all the fields in the preexisting dict.  If the
        # field isn't already present (KeyError), save the one from
        # hist directly.  Note that a simple dict update is not
        # suitable here as that would overwrite existing values in the
        # dict while we want to append them.
        for reg in hist.history:
            try:
                self.training_hist[reg] += hist.history[reg]
            except KeyError:
                self.training_hist[reg] = hist.history[reg]

    def compile(self,
                optimizer='adam',
                lr=None,
                loss=None,
                loss_weights=None):
        """Compiles the model in the container.  Optimizer and
        loss can be given.
        NOTE: All fold-specific modules will be compiled by this
          function using the same settings as everything else would not
          make a lot of sense to compare after training.
        """
        # Save a dictionary of the received arguments to reconstruct
        # after saving/loading.
        self.compile_args = {'optimizer': optimizer,
                             'lr': lr,
                             'loss': loss,
                             'loss_weights': loss_weights}
        # There are multiple legal ways to assign an optimizer.
        #  1. Directly pass a keras optimizer instance.
        #  2. Pass None (The module standard optimizer, assigned in the
        #     file header, is then used.)
        #  3. Pass a string that can be associated with a Keras
        #     standard optimizer (like 'adam' or 'SGD')
        #      3a. The 'lr' parameter can be additionally set to
        #          a custom learning rate for this case.
        #      3b. If 'lr' is left blank, the optimizer's standard
        #          learning rate setting is used.
        if optimizer is None:
            # The former standard setting: If no optimizer is given (in
            # the form of a function or string), a custom preset for
            # SGD, to be specified in the module header, is used.
            optimizer = _STD_SETTINGS['NN_OPTIMIZER']
        elif isinstance(optimizer, six.string_types) and lr is not None:
            # NEW in nnlib_multiobject_v01: Setting the learning rate
            # in the 'compile' call has been reenabled if 'optimizer'
            # is given as a string.  To keep it simple, we use the
            # undocumented capability of Keras (to be found in the
            # 'optimizers.get' function) to pass down arguments for the
            # optimizer by using dictionaries of a specific form.
            # (Note: In all other cases, the 'lr' parameter is ignored)
            optimizer = {'class_name': optimizer,
                         'config': {'lr': lr}}
        # There are multiple legal ways to assign a loss function.
        #  1. Directly pass a Keras loss function (or a list of them).
        #  2. Pass None (The function will try to guess a suitable (set
        #     of) loss functions.  The output types must be known to
        #     this function for this to work; it is basically only a
        #     lookup table.)
        #  3. Pass a string alias (or a list of them) as used by Keras
        #     itself.
        if loss is None:
            loss = []
            print("Automatically assigned losses:")
            for kind in self.output_config:
                # The standard setting for the losses: Mean squared
                # error for the coordinates and sizes; categorical
                # crossentropy for the shapes.
                type = kind.split('_')[0]
                if type == 'objtype':
                    loss.append(categorical_crossentropy)
                    print("categorical crossentropy")
                elif type in {'labels','objcount','fields'}:
                    loss.append(mean_squared_error)
                    print("mean squared error")
                else:
                    raise NNLibUsageError("There is no standard setting "
                                          "associated with model outputs of "
                                          "type {}. Choose another model or "
                                          "provide a loss function "
                                          "explicitly.".format(kind))
        elif not (isinstance(loss, list) or isinstance(loss, tuple)):
            # Turn a single loss into a list of fitting length.
            loss = [loss] * len(self.output_config)
        elif len(loss) != len(self.output_config):
            # Check if the loss argument, if given as a list of
            # multiple losses, is compatible with what the model
            # exposes as outputs.  Raise an error if it is not.
            raise IndexError("The length of the list of losses does not "
                             "match the list of outputs.")
        # Check the length of loss_weights as well.
        if loss_weights and len(loss_weights) != len(self.output_config):
            raise IndexError("The length of the list of loss weights does "
                             "not match the list of model outputs.")
        # Compile the models in the holder with given parameters
        # (ready-to-fit).
        for model in self.model_holder:
            model.compile(loss=loss, optimizer=optimizer,
                          loss_weights=loss_weights)
        print('OK')

    def fit_and_eval(self, x, y, validation_data, batch_size, epochs,
                     early_stopping, saving_mode, more_callbacks):
        """Partially outsourced from the ModelHandler's fit_and_eval
        function.  The ModelContainers are now responsible for training
        the models within them by themselves, but need to get the
        necessary data from the ModelHandler.
        """
        # Get the epoch counter for the chosen model and
        # calculate its target state.
        iep = self.epoch_counter()
        tep = iep + epochs
        # Initialize the callback list.
        # CHANGED in nnlib_multiobject_v04: The ModelContainer
        # is now responsible for building a callback that
        # guarantees the epoch counters are incremented and the
        # weights are saved as specified via the
        # 'weights_saving_mode' argument, replacing the
        # LambdaCallback and the old-style
        # ModelCheckpointToInternal Callback.
        cb = [TerminateOnNaN(), self.build_callback(saving_mode)]
        # Add the EarlyStopping callback if requested.
        if early_stopping is not None:
            cb.append(EarlyStopping(patience=early_stopping))
        # If any, import the additional callbacks into the
        # list.
        if more_callbacks is not None:
            try:
                cb = cb + list(more_callbacks)
            except TypeError:
                cb = cb + [more_callbacks]
        # New in nnlib_v12: Use a less sophisticated reporting
        # setting (no progress bar) when running in batch mode
        # to produce much more compact logs.
        if _interactive:
            verb = 1
        else:
            verb = 2
        # Train the model.
        self._attach_hist(
          self.model.fit(x, y, batch_size, tep, verb, cb,
                         validation_data=validation_data, initial_epoch=iep)
        )

    def get_training_hist(self):
        # TODO!
        return {}

    def epoch_counter(self):
        """Returns the number of epochs the contained model has been
        trained for.
        """
        return len(self.weights_lookup)-1

    @abstractmethod
    def build_callback(self):
        """Builds and returns a callback instance.  This being a
        function of the ModelContainer allows for masking the actual
        callback instance returned by each container.
        """
        pass


class BaseModelContainer(ModelContainer):
    """This class stores models as well as their associated data.  It
    supersedes the previous EpochCounter class.  It should simplify
    passing around models while retaining important internals.
    CHANGED in nnlib_multiobject_v06: To make this module compatible
    with folding functionality for cross validation, it now gets an
    input to tell it how many folds there need to be.  The respective
    models and their individual support functions are saved in lists
    and the old variable names are simple references to the objects in
    these lists that are set by the new context manager 'access_fold'.
    This way, we need minimal rewrites in other areas.
    """

    def __init__(self, folds=1, silent=False):
        super(BaseModelContainer, self).__init__(folds, silent)
        self._recipe_name = None

    def __deepcopy__(self, memo):
        new_container = BaseModelContainer(self.total_folds, self.silent)
        cp = Copier(self, new_container)
        cp.copy('all', True)
        return new_container

    def _retrieve_build_specs(self, dpcopy):
        """Gets the build specifications.  It can either copy them or
        only pass the reference back, to be signaled by the truth value
        of dpcopy.  Inverse of _rebuild.
        """
        if self._recipe_name == '@injected':
            print("Sorry, injected models cannot be reliably saved. When "
                  "reloaded, the container will be empty.")
            return self._recipe_name
        ret_list = [self._recipe_name,self._build_args,
                    self.model.get_weights()]
        if dpcopy:
            ret_list = deepcopy(ret_list)
        return ret_list

    def _rebuild(self, data):
        """Inverse of _retrieve_build_specs.  Rebuilds a model
        according to the specification passed within the 'data'
        argument.
        """
        if data == '@injected':
            print("Loaded a model container that has been injected with a "
                  "model. Since it could not be serialized, this container "
                  "remains unconfigured.")
            return
        # To avoid useless output to console, set the module to silent,
        # but save the current setting to restore it later.
        silent_bak = self.silent
        self.silent = True
        # Go through the build process as indicated by the data
        # received.
        self.load_build_recipe(data[0])
        self.modify_build_recipe(data[1])
        self.bake()
        # Set the model weights to the last value they had before
        # saving.
        self.model.set_weights(data[2])
        # Since we do not want to impose an ordering on the allowed
        # load operations (but can obviously only compile a model that
        # was already constructed), it is the job of '_rebuild' to
        # trigger the compile process if 'compile_args' has already
        # been saved to the container.
        if self.compile_args is not None:
            self.compile(**self.compile_args)
        self.silent = silent_bak

    def _retrieve_hist(self, dpcopy):
        """Get the weights history.  Inverse of _insert_hist."""
        ret_list = [self.weights_hist,self.weights_lookup]
        if dpcopy:
            ret_list = deepcopy(ret_list)
        return ret_list

    def _insert_hist(self, data):
        """Set the weights history (but not the weights actually loaded
        into the model -- that is _rebuild's job).  Inverse of
        _retrieve_hist.
        """
        self.weights_hist = data[0]
        self.weights_lookup = data[1]

    def inject_prebuilt_model(self,
                              model,
                              input_config,
                              output_config,
                              add_funcs={}):
        """NEW in nnlib_multiobject_v04: Since a ModelContainer object
        is now able to kick off building known net types by itself,
        providing a model and its metadata to '__init__' makes no sense
        anymore.  This function does the same the old version of
        '__init__' did in letting you inject a prebuilt model,
        circumventing the baking procedure while allowing for the same
        usability.
        """
        if not self.changeable:
            print("This ModelContainer instance has been locked by another "
                  "function. It is highly discouraged to modify already "
                  "locked ModelContainers. Consider creating another one.")
            print("No changes have been made.")
            return
        self._recipe_name = '@injected'
        self.model = model
        self.input_config = input_config
        self.output_config = output_config
        self.add_funcs = add_funcs
        self.weights_hist = [self.model.get_weights()]
        self.weights_lookup = [0]
        # Check the given data for internal consistency.
        if (len(output_config) != len(model.outputs)
              or len(input_config) != len(model.inputs)):
            raise ValueError("There is something wrong. Model in-/outputs do "
                             "not align with in-/output config data.")
        self.changeable = False

    def load_build_recipe(self, recipe_name):
        """NEW in nnlib_multiobject_v04: It is now the ModelContainer
        instances that are building the models themselves.  This is
        supposed to make saving, loading and working with the build
        functions a lot easier.  Calling 'load_build_recipe' with a
        known 'recipe_name' loads a dict of arguments containing
        standard settings and the function to build into the
        ModelContainer instance.  Standard settings are shown to the
        user (in interactive mode) and can be altered by calling
        'modify_build_recipe'.  Once the settings are adjusted
        accordingly, the model can be baked (by calling 'bake()'.
        """
        if not self.changeable:
            print("This ModelContainer instance has been locked by another "
                  "function. It is highly discouraged to modify already "
                  "locked ModelContainers. Consider creating another one.")
            print("No changes have been made.")
            return
        try:
            self._build_function = build_dict[recipe_name][0]
            self._build_args = build_dict[recipe_name][1].copy()
        except KeyError:
            raise KeyError("The recipe name you gave is unknown.")
        if _interactive and not self.silent:
            print("The preset arguments for this recipe look like this:")
            print(self._build_args)
        self._recipe_name = recipe_name

    def modify_build_recipe(self, mod_args):
        """Takes a dictionary of arguments to change from their
        standard settings for the selected build recipe.  Gives some
        useful feedback on what it does if there is anything out of the
        ordinary.
        """
        if not self.changeable:
            print("This ModelContainer instance has been locked by another "
                  "function. It is highly discouraged to modify already "
                  "locked ModelContainers. Consider creating another one.")
            print("No changes have been made.")
            return
        for arg in mod_args:
            if arg in self._build_args:
                self._build_args[arg] = mod_args[arg]
            else:
                print("The key '{}' in the list of modifications was ignored "
                      "as it does not seem to be an accepted argument of the "
                      "selected recipe.".format(arg))
        if _interactive and not self.silent:
            print("The list of arguments now looks like this:")
            print(self._build_args)

    def bake(self):
        """Triggers the actual building process if the model is
        supposed to be built from a recipe.  Make sure that you set all
        build parameters before calling this.
        """
        if not self.changeable:
            print("This ModelContainer instance has been locked by another "
                  "function. It is highly discouraged to modify already "
                  "locked ModelContainers. Consider creating another one.")
            print("No changes have been made.")
            return
        # CHANGED in nnlib_multiobject_v06: Go through all folds and
        # construct a model for each.  All contents are saved in
        # corresponding 'holder' lists.  To keep code changes to a
        # minimum, existing object attributes like self.model are now
        # just references to elements of these holder lists that can be
        # reassigned by functions.  This should keep ModelContainer
        # instances looking mostly consistent to the outside world
        # while allowing full fold functionality.
        for fold in range(self.total_folds):
            (self.model_holder[fold],self.input_config,
             self.output_config,self.add_funcs_holder[fold]) = (
               self._build_function(**self._build_args)
            )
            self.weights_hist_holder[fold] = [
              self.model_holder[fold].get_weights()
            ]
            self.weights_lookup_holder[fold] = [0]
        self.model = self.model_holder[0]
        self.add_funcs = self.add_funcs_holder[0]
        self.weights_hist = self.weights_hist_holder[0]
        self.weights_lookup = self.weights_lookup_holder[0]
        if _interactive and not self.silent and self.add_funcs[0]:
            print("The model you baked provides the following additional "
                  "functionality:")
            print(list(self.add_funcs[0].keys()))
            print("To use it, call "
                  "<this_container>.add_funcs[<function_key>](<arguments>)")
        self.changeable = False

    def restore_weights(self, epoch, consider_all_changes=False):
        """Restores weights to the ones at the end of the selected
        epoch.  Depending on the setting for saving progress, the
        selected epoch may not be present.  To make it easier to use,
        this function offers reasonable alternatives if called in an
        interactive context or chooses the closest if called in a
        non-interactive context.
        WARNING: As with most functions, this only deals with the
        currently active fold!
        """
        # Check whether the given epoch number makes basic sense.
        if not isinstance(epoch, int) or epoch < 0:
            raise NNLibUsageError("Your epoch selection does not make sense "
                                  "(no integer or negative).")
        elif consider_all_changes and self.total_epoch_counter() < epoch:
            raise NNLibUsageError("There have not been {} weight updates in "
                                  "total yet. (Currently {}.)".format(
                                    epoch, self.total_epoch_counter()))
        elif self.epoch_counter() < epoch:
            raise NNLibUsageError("The model has not been trained for {} "
                                  "epochs. (Currently {} epochs "
                                  "trained.)".format(
                                    epoch, self.epoch_counter()))
        # The checking and potential search for an alternative to load
        # can be performed with the same rules applying for both
        # weights_hist and weights_lookup.
        if consider_all_changes:
            list_to_search = self.weights_hist
        else:
            list_to_search = self.weights_lookup
        if list_to_search[epoch] is None:
            # We land here if the requested data is not available.  All
            # that follows is to make sure a reasonable selection of an
            # alternative epoch to load is made.
            print("No weights were saved for update {}.".format(epoch))
            print("Scanning for closest entries...")
            sel = ''
            for ep in range(epoch-1, 0, -1):
                if list_to_search[ep] is not None:
                    former = ep
                    sel += 'former/'
                    if _interactive:
                        print("Found closest available previous update at "
                              "position {}.".format(ep))
                    break
            else:
                former = None
                if _interactive:
                    print("Found no previous data available. (Initialization "
                          "values are not considered. Run 'reset()' to use "
                          "those.)")
            for ep in range(epoch+1, len(list_to_search)):
                if list_to_search[ep] is not None:
                    latter = ep
                    sel += 'latter/'
                    if _interactive:
                        print("Found closest available following update at "
                              "position {}.".format(ep))
                    break
            else:
                latter = None
                if _interactive:
                    print("Found no following entries.")
            if _interactive:
                resp = input('Your selection? [' + sel + 'ABORT] > ')
                if resp in string_alts['former'] and former is not None:
                    epoch = former
                elif resp in string_alts['latter'] and latter is not None:
                    epoch = latter
                else:
                    print("Aborted with no changes.")
                    return
            else:
                if bool(former) and bool(latter):
                    if epoch-former <= latter-epoch:
                        epoch = former
                    else:
                        epoch = latter
                elif not (bool(former) or bool(latter)):
                    print("No valid alternatives.")
                    return
                elif former is None:
                    epoch = latter
                else:
                    epoch = former
                print("Chose to import update {} as closest available "
                      "entry.".format(epoch))
        # Finally, the requested or chosen weights are set.
        if not consider_all_changes:
            epoch = self.weights_lookup[epoch]
        self.model.set_weights(self.weights_hist[epoch])

    def reset(self):
        """Resets the model by infusing its originally assigned weights
        back into it.  This deliberately foregoes another run of the
        initializer that is usually at least partly based on
        randomization.
        WARNING: As with most functions, this only deals with the
        currently active fold!
        """
        self.model.set_weights(self.weights_hist[0])

    def build_callback(self, saving_mode):
        """Builds and returns a callback instance.  This being a
        function of the ModelContainer allows for masking the actual
        callback instance returned by each container.
        """
        return ModelCheckpointToInternal(container=self,
                                         saving_mode=saving_mode)

    def total_epoch_counter(self):
        """Returns the total number of weight updates the contained
        base model has seen.
        WARNING: As with most functions, this only deals with the
        currently active fold!
        """
        return len(self.weights_hist) - 1


register_container(BaseModelContainer)


class CombinedModelContainer(ModelContainer):

    def __init__(self, model_collection, folds=1, silent=False):
        super(CombinedModelContainer, self).__init__(folds, silent)
        self._build_combined_model(model_collection)

    def _retrieve_build_specs(self, dpcopy):
        if dpcopy:
            return deepcopy(self.model_collection)
        else:
            return self.model_collection

    def _rebuild(self, model_collection):
        self._build_combined_model(model_collection)

    def _retrieve_hist(self, dpcopy):
        if dpcopy:
            return deepcopy(self.weights_lookup)
        else:
            return self.weights_lookup

    def _insert_hist(self, weights_lookup):
        self.weights_lookup = weights_lookup

    def _build_combined_model(self, model_collection):
        # Initialize some variables that will allow for efficient
        # signal routing once all connections have been figured out.
        # input_list and output_list will list metadata on outward
        # facing inputs and outputs.
        input_list = []
        output_list = []
        # signal_board will contain information about the types of
        # signals passed around in the combined system.  This includes
        # every type of signal: Intermediate ones, outward facing
        # inputs and outward facing outputs.  The idea is that this
        # collection can serve as a kind of switch board to which
        # everything else can connect itself.
        signal_board = []
        # signal_source will register from which model the respective
        # data on the signal_board stems.  This is necessary to later
        # resolve the order under which each model needs to be
        # processed.
        signal_source = []
        # First, go through all outputs of all models and link them to
        # the board of signals.
        for model in model_collection:
            # Each model gets its own output routing instruction set.
            # It isn't the nicest solution to just attach a list to
            # another object, but it seems like the most efficient
            # approach because this way, the later reordering of models
            # does not concern us.  We will want to make sure to delete
            # this list at the end of the process.
            model.mdl_output_route = len(model.output_config) * [None]
            for port_index, output in enumerate(model.output_config):
                # Outputs defined here are added to output_list, the
                # model's own mdl_output_route info, to signal_board
                # (output spec only) and to signal_source (model only).
                # The latter one is used to process the sequence to
                # follow.
                output_list.append((len(signal_board),output))
                model.mdl_output_route[port_index] = len(signal_board)
                signal_board.append(output)
                signal_source.append(model)
        # Second, go through all inputs of all models and look if the
        # necessary input is available.  If not, an outward-facing
        # input will have to be defined.  That outward-facing input
        # will also go through the signal board, so there will be no
        # duplicates.  We can also use this section to define the
        # sorted list of models to go through.  The principle here is
        # that if a signal is generated by another model, that model
        # should take priority and we slot in the currently considered
        # one directly behind the last model it depends upon.  With
        # this setup, we do not need to consider any model that has not
        # been processed up to this point.
        sorted_model_list = []
        for model in model_collection:
            slot_in_index = 0
            model.mdl_input_route = len(model.input_config) * [None]
            for port_index, inp in enumerate(model.input_config):
                # First, check if the input is available on the board.
                if inp in signal_board:
                    # If it is, we save the index.
                    source_index = signal_board.index(inp)
                    try:
                        # Try to get the index of the source model for
                        # the current input tensor.
                        source_model_index = sorted_model_list.index(
                          signal_source[source_index]
                        )
                    except (ValueError, IndexError):
                        # This except clause is hit both if the source
                        # model isn't in the sorted list yet, and if
                        # the input is an additional outward facing
                        # input that has been constructed for a
                        # previously processed model.  In both cases,
                        # we want the current model to not be pushed
                        # back, so we assign an artificial value of -1
                        # to the source model index.
                        source_model_index = -1
                    # Save the source index in the model's own
                    # mdl_input_route specification.
                    model.mdl_input_route[port_index] = source_index
                    slot_in_index = max(slot_in_index, source_model_index+1)
                    try:
                        output_list.remove((signal_board.index(inp),inp))
                    except ValueError:
                        pass
                else:
                    # get_input_at seems to be unstable. Let's try with
                    # model.input
                    if len(model.input_config) > 1:
                        tf_shape = model.model.input[port_index].get_shape()
                    else:
                        tf_shape = model.model.input.get_shape()
                    shape = tf_shape.as_list()[1:]
                    input_list.append((len(signal_board),inp,shape))
                    model.mdl_input_route[port_index] = len(signal_board)
                    signal_board.append(inp)
            sorted_model_list.insert(slot_in_index, model)
        # We take the sorted list of models as the model_collection to
        # save.  Notice how this has nothing much to do with this
        # method's input variable of the same name, being an ordered
        # list.
        self.model_collection = sorted_model_list
        # Third, connect the actual models.  We can safely reset the
        # signal_board list because the identifiers have all been
        # processed at this point.  It will be refilled in the
        # following with the actual output tensors themselves.
        for fold in range(self.total_folds):
            signal_board = len(signal_board) * [None]
            input_layers = []
            output_layers = []
            for inp in input_list:
                signal_board[inp[0]] = lrs.Input(shape=inp[2])
                input_layers.append(signal_board[inp[0]])
            for model in sorted_model_list:
                with model.access_fold(fold):
                    i_tensors = [
                      signal_board[i] for i in model.mdl_input_route]
                    if len(model.output_config) == 1:
                        o_tensors = [model.model(i_tensors)]
                    else:
                        o_tensors = model.model(i_tensors)
                    for tensor, address in zip(o_tensors,
                                               model.mdl_output_route):
                        signal_board[address] = tensor
            for output in output_list:
                output_layers.append(signal_board[output[0]])
            self.model_holder[fold] = Model(inputs=input_layers,
                                            outputs=output_layers)
            # Use the new _append_to_base_models utility function to make
            # sure the weights of the new model at init time are saved.
            with self.access_fold(fold):
                start_weights = (
                  self._append_to_base_models('replace_latest_none'))
                self.weights_lookup_holder[fold] = [start_weights]
        # Finally, do some cleaning up in the base model containers.
        for model in sorted_model_list:
            del model.mdl_input_route
            del model.mdl_output_route
        self.input_config = [i[1] for i in input_list]
        self.output_config = [o[1] for o in output_list]
        self.changeable = False

    def _append_to_base_models(self, op_mode):
        """Simplifies and standardizes communication with the combined
        model's associated base models for appending entries to their
        weight histories and returning the indices of the new entries.
        Notice that 'append' and 'replace_latest_none' both make sure
        the current weights are saved, but 'append' assumes there has
        been an update, while 'replace_latest_none' assumes there has
        been none.  It is the caller's responsibility to ensure each
        one is called only when appropriate as this function does not
        check.
        CHANGED in nnlib_multiobject_v06: Made this function aware of
          folds.  It now only modifies the currently selected fold.
          Needs to be iterated if all folds should do the same.
        Arguments
          op_mode: Allowed arguments are 'append', 'append_none' and
            'replace_latest_none'.
            'append' saves the models' current weights to their
            respective history.
            'append_none' saves Nones in the weights histories. (This
            is done to keep track of the weight updates even if the
            weights are not to be saved.)
            'replace_latest_none' makes sure that the latest weights
            are saved in the base model containers (saves them if they
            haven't been saved yet).  This will NOT consume another
            entry in the base handlers' weight histories as we presume
            there has been no actual update in the weights.
        """
        if op_mode not in {'append','append_none','replace_latest_none'}:
            raise ValueError("Unknown operations mode: {}. Allowed ones are "
                             "'append', 'append_none' and "
                             "'replace_latest_none'.".format(op_mode))
        indices = []
        for container in self.model_collection:
            # We expect this function to be called in this handler's
            # own 'with access_fold' context which sets the object
            # variable 'fold'.  This is why there are no provisions to
            # directly tell this function which fold to process.
            with container.access_fold(self.fold):
                if op_mode == 'replace_latest_none':
                    if container.weights_hist[-1] is None:
                        container.weights_hist.pop()
                    else:
                        indices.append(container.total_epoch_counter())
                        continue
                if op_mode == 'append_none':
                    new_entry = None
                else:
                    new_entry = container.model.get_weights()
                container.weights_hist.append(new_entry)
                indices.append(container.total_epoch_counter())
        return indices

    def _modify_base_models(self, op_mode, indices):
        """Simplifies and standardizes communication with the combined
        model's associated base models for modifying (specifically,
        deleting and restoring) saved weight history elements.
        CHANGED in nnlib_multiobject_v06: Made this function aware of
          folds.  It now only modifies the currently selected fold.
          Needs to be iterated if all folds should do the same.

        Arguments:
          op_mode: Allowed arguments are 'set' and 'delete'.
            'set' takes the indices provided and recovers the weights
            in the respective weights histories.
            'delete' replaces entries in the histories with Nones,
            effectively deleting them (while keeping the amount of
            entries the same to still keep track of total updates).
          indices: List of indices with the length of model_collection
            to address the values in the base containers.
        """
        if op_mode not in {'set', 'delete'}:
            raise ValueError("Unknown operations mode: {}. Allowed ones are "
                             "'set' and 'delete'.".format(op_mode))
        if len(indices) != len(self.model_collection):
            raise IndexError("Indices list provided has the wrong length. It "
                             "needs to be the same as the list of associated "
                             "base models.")
        for container, index in zip(self.model_collection, indices):
            with container.access_fold(self.fold):
                if op_mode == 'set':
                    container.model.set_weights(container.weights_hist[index])
                elif op_mode == 'delete':
                    container.weights_hist[index] = None

    def list_epochs_with_weights(self):
        """Returns a list of all epochs for which weights are saved in
        and by this handler.
        """
        return [ep[0] for ep in enumerate(self.weights_lookup) if ep[1] is not None]

    def restore_weights(self, epoch):
        """Restores weights to the ones at the end of the selected
        epoch.
        CHANGED in mo_08: Removed searching for alternatives around the
          selected epoch as it is rarely important. Instead, we just
          warn if the weights were not present.  Instead, we allow
          passing the strings 'best_train' and 'best_val' that will
          autoselect the epochs where the network achieved best results
          on training/validation data.

        Parameters
        ----------
        epoch : int/str
          The epoch number or one of 'best_train', 'best_val'.  If an
          int is given, the weights for the given epoch number are
          loaded.  If one of the strings is given, a search for the
          best result in the history is triggered.
          We expect epochs based on one, the counting that TensorFlow
          uses, such that it aligns with the CLI output in training.
        """
        # Check whether the given epoch number makes basic sense.
        if epoch == 'best_train':
            epoch = self._find_epoch('loss')
        elif epoch == 'best_val':
            epoch = self._find_epoch('val_loss')
        elif not isinstance(epoch, int) or epoch < 0:
            raise NNLibUsageError("Your epoch selection does not make sense "
                                  "(no integer or negative).")
        elif self.epoch_counter() < epoch:
            raise NNLibUsageError("The model has not been trained for {} "
                                  "epochs. (Currently {} epochs "
                                  "trained.)".format(epoch,
                                                     self.epoch_counter()))
        if self.weights_lookup[epoch] is None:
            # We land here if the requested data is not available.
            # CHANGED in mo_08: We do not select an alternative, but
            # simply fail with an appropriate warning.
            warnings.warn("There is no data for the selected epoch. "
                          "Select another. Available epochs are "
                          "{}".format(self.list_epochs_with_weights()))
            return
        # Finally, the requested or chosen weights are set.
        self._modify_base_models('set', self.weights_lookup[epoch])

    def build_callback(self, saving_mode):
        """Builds and returns a callback instance.  This being a
        function of the ModelContainer allows for masking the actual
        callback instance returned by each container.
        """
        return ComModelCheckpointToInternal(container=self,
                                            saving_mode=saving_mode)


register_container(CombinedModelContainer)


class ModelHandler(FunctionalContainer):
    """Handles all functions for preparation and use of the Keras
    models.  The methods within this class used to be base-level calls
    in nnlib, but to increase usability and stability, it was
    transformed into a more object-oriented approach with v2.  Now,
    most base variables are given at init and only such variables that
    can safely be changed independently are exposed at the creation of
    the models etc.
    """

    def __init__(self,
                 field_size=_STD_SETTINGS['FIELD_SIZE'],
                 field_depth=_STD_SETTINGS['FIELD_DEPTH'],
                 ts_length=_STD_SETTINGS['TS_LENGTH'],
                 mask_size=_STD_SETTINGS['MASK_SIZE'],
                 mask_samples=_STD_SETTINGS['MASK_SAMPLES'],
                 ind_lengths=_STD_SETTINGS['IND_LENGTH'],
                 angular_dof=_STD_SETTINGS['ANGULAR_DOF'],
                 suppress_bias=_STD_SETTINGS['SUPPRESS_BIAS'],
                 train_on_no_dist=_STD_SETTINGS['DIST_FREE'],
                 use_cross_validation=_STD_SETTINGS['CROSS_VALIDATION'],
                 sep_disturbances=_STD_SETTINGS['SEP_DISTURBANCES'],
                 masking_list=_STD_SETTINGS['MASKING_LIST'],
                 minor_fold_splits=_STD_SETTINGS['MINOR_FOLD_SPLITS']):
        """ModelHandler's __init__ method accepts and sets all model
        parameters that should not be changed during the object's
        lifetime.  Additionally, it initializes some fields to None to
        avoid certain AttributeErrors.

        Arguments:
          field_size: The size of the field in real pixels.  This is
            used both in creating the datasets and in building the mask
            model.
          field_depth: The depth of the field in distinct ToF splits.
            This is used in creating the datasets, choosing the
            renderer and building the mask model.
          ts_length: The length of the timeseries.  This is used
            throughout the handler as it determines a dimension in both
            input and label data.
          mask_size: The size of masks that the mask model assumes.
            Also relevant for determining the smallest radius allowed
            for creating the samples.
          mask_samples: The amount of different masks.  Determines the
            size and output of the mask model and therefore all model
            inputs that depend on it.
          ind_lengths: If True, all sample radii are independent.  If
            False, objects are effectively limited to cubes/balls.
            Also influences the output of the interpreter as redundant
            description dimensions are eliminated.
          free_rot: If True, angled objects are enabled, if False,
            disabled.  Also influences the output of the interpreter as
            redundatn dimensions are eliminated.
            CHANGED in mo_08: This option is deprecated. Replace by
            using `angular_dof` argument.
          angular_dof: Integer between 0 and 3, inclusive.  Set how
            many angular degrees of freedom there are.  If
            `field_depth` is 1, setting more than 1 is quietly ignored.
            This setting will reflect in the data as well as in the
            models.
            NOTE: The first degree of freedom is always a rotation
            around the z axis, so 2D and 3D are consistent for a single
            rotation.
          use_reg: If True, the expected interpreter output is
            normalized to [0,1] in all outputs.
            This is also directly reflected by the labels the creators
            generate for this handler because the normalization is
            built into them directly.
            CHANGED in mo_08: Working on non-regularized data is no
            longer supported as it increases maintainance burden
            considerably and has been unused for some time now.  For
            now, the argument is still accepted to not break loading,
            but ignored, and if set to False, a warning is issued.
          suppress_bias: If True, no biases are allowed for any dense
            layer in any model.  This is mainly for testing purposes
            and will significantly hamper network performance.
          train_on_no_dist: If True, all field outputs are expected to
            not contain any objects or disturbances aside from the
            biggest object.  This is reflected in the decoder
            implementations and the samples that will be built.
          use_cross_validation: If set to an integer, a cross
            validation training approach is used.  Cross validation is
            handled automatically by all models.  To properly align all
            submodels with the ones trained on the same dataset, you
            must supply the number of folds directly here.
          masking_list: Tuple of strings listing the internal fields
            that should be masked in cross validation.  These names
            represent both the data to use and which data to mask the
            folds as when a masking context is entered, thereby making
            sure no data is unused or used twice.  The first element
            represents what is regarded as the major dataset ('train'
            in the standard setting), while all others are viewed as
            minor datasets ('val' in the standard settings). <only
            relevant if cross validation is used>
          minor_fold_splits: Tuple of integers defining the relative
            size of all minor splits.  This tuple must be one element
            smaller than 'masking_list' because the size of the first
            dataset is automatically asserted to fit the number of
            folds and other sizes. <only relevant if cross validation
            is used>
        """
        # The loadstore_instr dict is the central place to store all
        # saving and loading processes for a specific BaseContainer
        # instance.
        self.loadstore_instr = {
          'init': (self._retrieve_init_settings,
                     self._init_variable_settings,False),
          'models': (self._retrieve_models,self._set_up_models,True),
          'data': (self._retrieve_sample_pools,
                     self._set_up_sample_pools,False),
        }
        # Initialize internal variables to their given standard
        # settings.
        # CHANGED in nnlib_multiobject_v04: In an effort to streamline
        # the initialization process from a save file, the settings
        # that used to be set in __init__ itself are now set in
        # _init_variable_settings to make them accessible in other
        # contexts.  This specifically allows the pushing and
        # therefore the setting of these options in an already existing
        # ModelHandler instance.  If this new feature is used, care
        # should be taken to re-set the options as soon as possible.
        # Otherwise, strange errors due to internally created
        # incompatibilities may insue.
        variable_settings = {
          'field_size': int(field_size),
          'field_depth': int(field_depth),
          'ts_length': int(ts_length),
          'mask_size': int(mask_size),
          'mask_samples': int(mask_samples),
          'ind_lengths': bool(ind_lengths),
          'angular_dof': int(angular_dof),
          'suppress_bias': bool(suppress_bias),
          'train_on_no_dist': bool(train_on_no_dist),
          'use_cross_validation': int(use_cross_validation),
          'sep_disturbances': bool(sep_disturbances),
          'masking_list': masking_list,
          'minor_fold_splits': minor_fold_splits,
        }
        self._samples = {}
        # CHANGED in nnlib_multiobject_v06: To shadow the cross
        # validation functionality, we keep two references to the
        # sample pools: self._samples ALWAYS points at the dict of
        # SampleContainer instances and self.pools points to either the
        # same dict or, in cross validation contexts, to a pool of
        # VirtualSampleContainer instances representing the current
        # folds, thus every function accessing self.pools potentially
        # finds the masked sample sets, therefore shadowing the cross
        # validation functionality for them.
        # self.pools will be set and reset when entering or exiting a
        # fold context using init_fold_shadow.  Standard setting is
        # for it to point to self._samples.
        self.pools = self._samples
        self._init_variable_settings(variable_settings)
        self._set_up_models()
        # Initialize other internal variables (that are not settable at
        # the point of initialization) to their standard values.
        self.shape_est = False
        if MOD_SETTINGS['DEBUG']:
            self.skipped_combinations = []

    # --Private Functions--
    # Private Functions generally have no standard arguments because
    # they are supposed to only be called from inside this module
    # by functions that know what they are doing.

    def _retrieve_init_settings(self, dpcopy):
        if dpcopy:
            return deepcopy(self._init_call_dict)
        else:
            return self._init_call_dict

    def _init_variable_settings(self, settings):
        """The setting of the internal information has been outsourced
        from __init__ to here to allow for them to be reset later.  This
        serves to enable a much cleaner approach to saving and loading,
        where containers can be created generically with the saved data
        getting set afterwards.
        """
        # Check for basic compatibility.
        if settings['field_size'] % settings['mask_size'] != 0:
            raise ValueError("field_size needs to be a multiple of mask_size "
                             "for the pooling to work as intended.")
        self.field_size = settings['field_size']
        self.field_depth = settings['field_depth']
        self.ts_length = settings['ts_length']
        self.mask_size = settings['mask_size']
        self.mask_samples = settings['mask_samples']
        # free_rot is deprecated. Modify the settings dict according
        # to the setting and warn if it has a value.
        if 'free_rot' in settings:
            if settings['free_rot']:
                settings['angular_dof'] = 3
            else:
                settings['angular_dof'] = 0
            warnings.warn("`free_rot` setting is deprecated and superseded "
                          "by `angular_dof`, which has been set accordingly.",
                          DeprecationWarning)
            # Delete the free_rot key so it isn't saved. angular_dof
            # will be saved instead.
            del settings['free_rot']
        # Using a regularizer is unsupported going forward. We ignore
        # the setting, but warn if it conflicts with the new behaviour.
        if 'use_reg' in settings:
            if not settings['use_reg']:
                warnings.warn("Running without regularization is no longer "
                              "supported. The setting will be ignored. If "
                              "the handler contains fields that were created "
                              "without regularization, you need to re-run "
                              "sample creation or expect errors and "
                              "unsatisfactory results.", DeprecationWarning)
            # Delete the use_reg key so it isn't saved.
            del settings['use_reg']
        self.suppress_bias = settings['suppress_bias']
        self.train_on_no_dist = settings['train_on_no_dist']
        # separate_disturbances is a new option.  For backwards
        # compatibility, use the standard setting if none is provided.
        try:
            self.sep_disturbances = settings['sep_disturbances']
        except KeyError:
            self.sep_disturbances = _STD_SETTINGS['SEP_DISTURBANCES']
            settings['sep_disturbances'] = _STD_SETTINGS['SEP_DISTURBANCES']
        self._build_fold_mapping(settings['use_cross_validation'],
                                 settings['masking_list'],
                                 settings['minor_fold_splits'])
        self._build_label_mapping(settings['ind_lengths'],
                                  settings['angular_dof'])
        self._set_up_sample_pools()
        # Finally, save the settings to the handler.
        self._init_call_dict = settings

    def _build_fold_mapping(self, total_folds, masking_list,
                            minor_fold_splits):
        if len(masking_list) != len(minor_fold_splits) + 1:
            raise NNLibUsageError("'minor_fold_splits' needs to be exactly "
                                  "one entry shorter than 'masking_list'.")
        self.masking_list = masking_list
        if total_folds <= 1:
            self.fold_splits = None
            self.data_folds = 1
            return
        minor_sum = sum(minor_fold_splits)
        if minor_sum >= total_folds:
            raise NNLibUsageError("The total number of folds must be greater "
                                  "than the sum of minor folds.")
        self.fold_splits = (total_folds-minor_sum,) + tuple(minor_fold_splits)
        self.data_folds = total_folds

    def _build_label_mapping(self, ind_lengths, angular_dof):
        """This function builds label maps that serve three purposes:
        Saving the layout of the labels this container holds, saving
        the mask for going from full labels to the set of reduced
        labels and saving the mask for going from the reduced labels to
        the full labels.  All three masks are saved here and can be
        passed down to the creator objects as needed.
        """
        # First, copy the full label layout.  This will be modified in
        # the following.
        self.label_layout = deepcopy(param_set)
        self.full_layout = deepcopy(param_set)
        if self.field_depth <= 1:
            # If field_depth is exactly 1, so no depth rendering, we
            # set the z coordinate as well as the second and third turn
            # to zero and the depth to 1. This should ensure that the
            # objects still render correctly in a 3D scenario.
            for letter, subst in zip(('z','t','b','g'),('0','1','0','0')):
                self.label_layout = self.label_layout.replace(letter, '')
                self.full_layout = self.full_layout.replace(letter, subst)
        if not ind_lengths:
            # If ind_lengths (individual lengths) are disabled, we
            # replace all radii with a special indicator, 'c', because
            # there is no static mapping from the full labels to the
            # reduced ones.  This option needs to be appropriately
            # handled by all methods using this info.
            # In the label_layout, we replace 'r' with 'c' to mark that
            # this will need to be chosen from the full_layout.
            self.label_layout = self.label_layout.replace('r','c')
            # Now we can go over all radius parameters and delete them
            # in label_layout and replace with 'c' in full_layout. Note
            # that this will NOT delete the 'c' we just put in as it is
            # not in rad_params.
            for letter in rad_params:
                self.label_layout = self.label_layout.replace(letter, '')
                self.full_layout = self.full_layout.replace(letter, 'c')
        # Disable angular parameters as requested via angular_dof.
        for letter in ang_params[angular_dof:]:
            self.label_layout = self.label_layout.replace(letter, '')
            self.full_layout = self.full_layout.replace(letter, '0')

    def _retrieve_sample_pools(self, dpcopy):
        if dpcopy:
            return deepcopy(self._samples)
        else:
            # CHANGED in nnlib_multiobject_v06: The _samples dict
            # itself is always copied, just its contents are the same
            # if deepcopy isn't selected.
            return copy(self._samples)

    def _set_up_sample_pools(self, sample_pool=None):
        """Creates a new sample pool if called with no argument or
        inserts complete sample pools if they are given as the
        sample_pool argument.
        Arguments:
          sample_pool: If None (default), the internal '_samples' dict
            is filled up with empty SampleContainer for the standard
            keys.  If True, an entirely new '_samples' dict is created
            and filled up as well.  If it is a dict, it is assumed it
            contains SampleContainer instances that are then loaded
            into the ModelHandler.
        """
        if sample_pool is None:
            # Standard settings means that the sample dict should be
            # filled with empty containers.  This if clause only makes
            # sure that no error is triggered, but the work is done by
            # the for-loop below.
            pass
        elif sample_pool is True:
            # CHANGED in nnlib_multiobject_v06: It should not make a
            # difference if we load fields or init settings for the
            # ModelHandler first.  This means that this function must
            # not overwrite an existing _samples dict in its standard
            # setting (which it did earlier) because it is implicitly
            # called in both situations.  However, it may still be
            # useful to allow this function to reset it if explicitly
            # requested, so from now on we allow sample_pool to be
            # True, triggering that reset, while the standard setting
            # of None only adds standard fields if they don't exist,
            # but never overwrites anything.
            self._samples = {}
            self.pools = self._samples
        elif isinstance(sample_pool, dict):
            # Go through the items in 'sample_pool' and insert them.
            # Raise errors if the entries are incompatible in any way.
            for key, value in sample_pool.items():
                try:
                    value.checkout_field('fields')
                except NNLibUsageError:
                    # NNLibUsageError indicates that the field is
                    # empty, but we assume the user knows that that may
                    # be the case, so we just continue.
                    pass
                except AttributeError:
                    raise ValueError("The value linked to the key '{}' in "
                                     "the dict you passed does behave like "
                                     "a sample container.".format(key))
                if key not in {'train','val','test'}:
                    # CHANGED in nnlib_multiobject_v06: Since from now
                    # on we allow custom keys, we raise no error here
                    # anymore, but simply inform the user of what's
                    # happening.
                    print("A sample pool with the custom key '{}' was "
                          "added.".format(key))
                self._samples[key] = value
        else:
            raise TypeError("'sample_pool' is supposed to be a dict with "
                            "valid strings as keys and SampleContainers as "
                            "values.")
        # The following for loop creates all missing containers.
        for container_string in ('train','val','test'):
            if container_string not in self._samples.keys():
                self._samples[container_string] = (
                  SampleContainer(label_layout=self.label_layout,
                                  full_layout=self.full_layout,
                                  field_size=self.field_size,
                                  field_depth=self.field_depth,
                                  separate_disturbances=self.sep_disturbances,
                                  ts_length=self.ts_length)
                )

    def _get_sample_pool(self, source):
        """This function implements access to the internal
        SampleContainers.
        CHANGED in nnlib_multiobject_v06: This function accesses the
          dict that the 'pools' variable points to, which means it gets
          shadowed when a fold is activated by 'init_fold_shadow'.
          This means that it neither needs to know the current fold,
          nor does it need any special cases.
          Also, 'source' is now allowed to be an iterable of string
          identifiers.  We return a list of SampleContainerInterface
          objects in that case and continue the object directly
          otherwise.
        CHANGED in nnlib_multiobject_v06: From now on, this function
          returns two items: The SampleContainerInterface object or
          list of objects itself and its string identifier or list of
          string identifiers, in the same order. Therefore, to achieve
          the same functionality as before, throw away the second
          returned object.
        """
        # Private function that performs the task itself
        def _getter(source):
            try:
                source = _sample_alias_lookup[source]
            except KeyError:
                pass
            try:
                return self.pools[source], source
            except KeyError:
                raise KeyError("The string '{}' you gave for 'source' could "
                               "not be found.".format(source))

        # Outer function supporting both simple strings and iterables
        # of strings.
        if (not isinstance(source, six.string_types)
              and isinstance(source, Iterable)):
            pool_list = []
            id_list = []
            for str in source:
                pool, id = _getter(str)
                pool_list.append(pool)
                id_list.append(id)
            return pool_list, id_list
        else:
            return _getter(source)

    def _retrieve_models(self, dpcopy):
        # Note: Combined models, when copied from the handler level,
        # are not copied as objects (it was deemed too difficult
        # regarding getting the references right), but by saving the
        # weights lookup table and compile arguments.
        if dpcopy:
            ret_list = [deepcopy(self.base_models)]
        else:
            # CHANGED in nnlib_multiobject_v06: The _samples dict
            # itself is always copied, just its contents are the same
            # if deepcopy isn't selected.
            ret_list = [copy(self.base_models)]
        cm_proto = {}
        for key, value in self.combined_models.items():
            if dpcopy:
                cm_proto[key] = [value.weights_lookup[:],
                                 value.compile_args.copy()]
            else:
                cm_proto[key] = [value.weights_lookup,value.compile_args]
        ret_list.append(cm_proto)
        return ret_list

    def _set_up_models(self, packed_models=None):
        if packed_models is None:
            self.base_models = {}
            self.combined_models = {}
        elif (isinstance(packed_models, list)
                and len(packed_models) == 2):
            self.base_models = packed_models[0]
            for key, value in packed_models[1].items():
                model = self._model_picker(key)
                model._insert_hist(value[0])
                model._recompile(value[1])
        else:
            raise TypeError("packed_models is supposed to be a tuple of two "
                            "dicts (first for base models, second for "
                            "combined models) containing ModelContainers as "
                            "values.")

    def _calculate_errors(self,
                          target,
                          source,
                          ext_data,
                          real_out,
                          acceptable_deviation,
                          return_value):
        """Performs some advanced error calculations on the model.
        Arguments:
          target: Specifies the target model.  As in all other
            functions, base models can be addressed with a single
            string and combined models with a list or tuple of strings.
          source: Specifies the data source to use.  If 'external' is
            given, ext_data and real_out are used, otherwise those are
            ignored and the internal data source specified here is
            used.
          ext_data: If 'source' is set to 'external', ext_data must be
            a compatibly formatted set of inputs for model 'target'.
            Otherwise, this input is ignored.
          real_out: If 'source' is set to 'external', real_out must be
            a conpatible formatted set of labels for model 'target' to
            compare its output to.  Otherwise, this input is ignored.
          acceptable_deviation: Specifies the maximum difference of a
            network output from the real label (in actual field pixels)
            under which the prediction is still considered a hit.  This
            is necessary because the chance of the net yielding exact
            results down to floating point accuracy are practically
            zero.
          return_value: If True, the real network output is returned
            alongside the error dict.
        """
        # Make the NN predict on given data.
        model = self._model_picker(target)
        output = self._model_evaluator(model, source, ext_data)
        try:
            real_out = self._label_picker(model, source)
        except NNLibUsageError:
            pass
        # Calculate errors by going over all output fields. This way,
        # only applicable errors are actually returned.
        error_dict = {}
        for index, string in enumerate(model.output_config):
            specifier = string.split('_')
            if specifier[0] == 'labels':
                f_error = output[index] - real_out[index]
                abs_f_error = abs(f_error)
                # To find and count all misses, take the maximum error
                # over all predicted parameters along the last axis...
                maxerror = abs_f_error.max(axis=-1)
                # ...call it a miss if said maximum is above acceptable
                # error levels (as spec'd in function call)...
                boolerror = (maxerror >= acceptable_deviation)
                # ...extract all indices for which misses were found...
                miss_indices = np.array(boolerror.nonzero())
                # ...and count them.
                misses = boolerror.sum()
                error_dict['absolute parameter error'] = abs_f_error
                error_dict['parameter misses'] = misses
                error_dict['indices of parameter misses'] = miss_indices
            elif specifier[0] == 'objtype':
                s_error = output[index] - real_out[index]
                # Due to the nature of the onehot-encoded output, the
                # absolute error will always lie between 0 and 2.
                # (Basically, a wrong guess gets penalized twice: Once
                # for the false positive, once for the false negative.)
                # this is why divide by two, to somewhat norm the
                # error.
                abs_s_error = abs(s_error) // 2
                shape_errors = np.sum(abs_s_error, axis=-1)
                shape_miss_indices = np.array(shape_errors > 0.5)
                error_dict['indices of shape misses'] = shape_miss_indices
        if return_value:
            # Return prediction, absolute error, miss indices and total
            # misses
            return output, error_dict
        else:
            # Return absolute error, miss indices and total misses
            return error_dict

    def _model_picker(self, target, autobuild_if_missing=True):
        """Returns an internally saved ModelContainer instance.

        Centralizes picking models in a ModelHandler instance based on
        an externally providable `target` string.

        Parameters
        ----------
        target : string or Iterable of strings or ModelContainer
                 instance.
            String needs to be contained in the `_model_alias_lookup`
            dict.  In case of other Iterables, combined models are
            assumed and either returned from the list of existing ones
            or built automatically. (See `autobuild_if_missing`!)
            In case of ModelContainer instances, the same instance is
            returned with no changes and/or lookup step.
        autobuild_if_missing : bool, default: True
            If True, `_model_picker` will build the requested combined
            model.
            If False, `_model_picker` will error out according to the
            descriptions below.
            This only applies for combined models as building base
            models automatically is dangerous.
            Defaults to True to keep backwards compatibility. (True
            equals the behaviour before this parameter existed.)

        Returns
        -------
        model : ModelContainer
            The requested model's container instance.

        Raises
        ------
        ValueError
            Whenever (content of) `target` is somehow wrong (wrong
            kind, unknown key, etc.)
        KeyError
            When (content of) `target` is known, but the respective
            model has not been initialized.
        """
        def _lookup(string):
            """_lookup first searches for the unified label of the given
            model, then it gets the model itself from the handler's
            database.  It returns both model and label to the caller.
            """
            try:
                label = _model_alias_lookup[string]
            except KeyError:
                if string in self.base_models:
                    print("The target '{}' was found, however its namespace "
                          "is not registered correctly. Registering it for "
                          "you... This implies incorrect behaviour, "
                          "though.".format(string))
                    # To avoid this warning popping up repeatedly, we
                    # add an automated entry in _model_alias_lookup for
                    # now.
                    _model_alias_lookup[string] = string
                    label = string
                else:
                    raise ValueError("I do not know the key {} as 'target'. "
                                     "Please check the "
                                     "spelling.".format(string))
            try:
                model = self.base_models[label]
            except KeyError:
                raise KeyError("No model has not been registered for target "
                               "'{}' yet. Have you run its "
                               "initializer?".format(label))
            return model, label

        # --Function Body--

        # If 'target' is a string, have '_lookup' provide the model
        # from the internal dataset and we are done.
        if isinstance(target, six.string_types):
            model, _ = _lookup(target)
        # New: All combinations can be addressed by simply passing some
        # iterable as 'target'.  The following process first tries to
        # get the base models, then finds out if the model combining
        # all of them exists and is up-to-date and then either passes
        # it back or builds and registers it by instantiating a new
        # CombinedModelContainer.
        elif isinstance(target, Iterable):
            # Stop early if the list of targets is empty as the
            # procedure will otherwise work, simply producing an empty
            # model, potentially causing weird errors down the line.
            if len(target) == 0:
                raise ValueError("The 'target' argument cannot be empty.")
            # Initialize collection sets for the model container and
            # the labels.
            model_container_coll = set()
            label_coll = set()
            # Go through all keys to get the base models necessary for
            # the combined model.
            for item in target:
                basemodel, baselabel = _lookup(item)
                # Reminder: Due to label_coll and model_container_coll
                # being sets, duplicate or redundant entries in
                # 'target' are quietly discarded without notice.
                model_container_coll.add(basemodel)
                label_coll.add(baselabel)
            # If it turns out that the iterable given as 'target' was
            # singleton, we can stop early and return the corresponding
            # base model.  Additionally, this makes sure that
            # 'combined_models' will not be filled up by recompiled
            # versions of the base models.
            if len(model_container_coll) == 1:
                return model_container_coll.pop()
            # Freeze the collection of labels so we can easily index
            # the combined models.
            frozen_set = frozenset(label_coll)
            # We only want this function to build a new model if
            #  a) it does not yet exist or
            #  b) at least one of its base models has been replaced
            #     since it was originally built.
            # Otherwise, the previously built model should be reused.
            model_coll = [mdl.model for mdl in model_container_coll]
            if frozen_set in self.combined_models:
                # Assume that if there is a model with the index
                # provided in the collection of combined models, the
                # model should be reused.
                model_exists = True
                for bmdl in self.combined_models[frozen_set].model.layers:
                    # Falsify this assumption by comparing the old
                    # combined model's content with the currently
                    # registered ones.
                    if (isinstance(bmdl, Model)
                          and bmdl not in model_coll):
                        model_exists = False
                        break
            else:
                model_exists = False
            # If, with all the above considered, the model exists in
            # the desired form, take it.
            if model_exists:
                model = self.combined_models[frozen_set]
            elif autobuild_if_missing:
                # Otherwise, build it if requested...
                model = CombinedModelContainer(model_container_coll,
                                               self.data_folds)
                self.combined_models[frozen_set] = model
            else:
                # ...or raise an error if not.
                raise KeyError("Combined model could be built, but wasn't "
                               "yet.")
        elif is_cont_obj(target):
            # NEW in nnlib_multiobject_v04: To minimize the overhead
            #   and make the data picking infrastructure more flexible,
            #   we allow 'target' to already be a ModelContainer
            #   instance.  To support this, we catch it here by simply
            #   returning the ModelHandler as-is.
            # CHANGED in nnlib_multiobject_v06: To better support
            #   effective debugging, we do not perform an instance
            #   check anymore because that will generate a false
            #   negative if the module has been reloaded.  Instead, we
            #   check the type.  Since we thereby rely on the pull/push
            #   infrastructure present in AbstractContainer instances,
            #   we need to be prepared to deal with an AttributeError
            #   if 'target' does not point to an AbstractContainer at
            #   all.
            if target.pull('type')[-14:] == 'ModelContainer':
                model = target
            else:
                raise ValueError("You passed a container object, but it is "
                                 "no ModelContainer instance.")
        else:
            raise ValueError("Your input cannot be interpreted. Please pass "
                             "a ModelContainer instance, a string or an "
                             "iterable of strings as 'target'.")
        return model

    def _model_exists(self, target):
        """Checks if a model exists inside the ModelHandler.

        It uses _model_picker's machinery for checking to achieve
        robust results, so the same specs and limitations for the
        `target` parameter apply.

        Parameters
        ----------
        target : string or Iterable of strings or ModelContainer
                 instance.
            String needs to be contained in the `_model_alias_lookup`
            dict.  In case of other Iterables, combined models are
            assumed and either returned from the list of existing ones
            or built automatically. (See `autobuild_if_missing`!)
            In case of ModelContainer instances, the same instance is
            returned with no changes and/or lookup step.

        Returns
        -------
        bool
            True if model specified by target exists in the handler,
            False otherwise.
        """
        try:
            self._model_picker(target, autobuild_if_missing=False)
            return True
        except KeyError:
            return False

    def _data_picker(self, target, source, sl=slice(0,None)):
        """Since many methods need to pick data from inside the
        ModelHandler instance based on an externally providable
        'target' (specifying the model) and 'source' (specifying either
        training or validation data) string, it is becoming tedious to
        fix errors arising from changes in the internal setup.
        Therefore, this function is taking care of providing the
        fitting model input data from version nnlib_multiobject_v0_6
        going forward as a centralized instance.
        CHANGED in nnlib_multiobject_v01: _data_picker now supports
          returning only slices of the dataset it collects via its new
          argument 'sl'.  Its standard setting generates a slice that
          contains the whole tensor, so there is no change in
          usability for methods that do not take advantage of this
          feature.
        """
        # Set source tensors to the apporpriate ones, depending on the
        # selection of source.  Raise an error in case of invalid
        # source key.
        try:
            container, _ = self._get_sample_pool(source)
        except KeyError:
            raise ValueError("I do not know {} as a key for "
                             "'source'.".format(source))
        # Fetch the model container.  It contains the metadata to
        # perform the necessary processing on the source data.
        model_cont = self._model_picker(target)
        # CHANGED in nnlib_multiobject_v0_6: The model container is now
        # the one to know everything about the expected inputs of its
        # model, so we do no longer hardcode a process, but rather read
        # what the model needs and provide data accordingly.
        data = []
        # Go through the input configuration of the model...
        for string in model_cont.input_config:
            specifier = string.split('_')
            # ...and act according to what comes up.
            # First, there are the cases where the data is directly
            # available.  In those cases, we simply ask the selected
            # container to pass them over.
            try:
                data.append(container.checkout_field(specifier[0])[sl])
            except ValueError:
                # If we reach this 'except' clause, it means the
                # container does not provide the data because it does
                # not know the key (as opposed to the data simply not
                # being present, in which case the container builds it
                # for us or user input is required, making it necessary
                # to escalate the error).  We can still try to get the
                # data from existing models.  To achieve this, we go
                # through all available base models and stop if we find
                # one that provides the data needed.
                for mdl in self.base_models.values():
                    if specifier[0] in mdl.output_config:
                        # Reminder: Calling _model_evaluator will cause
                        # an indirect recursion on this method. It is
                        # therefore critical that all in-/outputs have
                        # unique names, otherwise there is the
                        # potential for an infinite recursion.
                        intermediate = self._model_evaluator(mdl, source)
                        if isinstance(intermediate, Iterable):
                            index = mdl.output_config.index(specifier[0])
                            data.append(intermediate[index][sl])
                        else:
                            data.append(intermediate[sl])
                        break
                else:
                    # We only end up here if no break was hit above, so
                    # this is the error case of the data being neither
                    # available nor constructable.
                    raise NNLibUsageError("The necessary dataset could not "
                                          "be constructed because an input "
                                          "of your specified model is not "
                                          "available to this ModelHandler.")
        return data

    def _label_picker(self, target, source, sl=slice(0,None)):
        """Since many methods need to pick data from inside the
        ModelHandler instance based on an externally providable
        'target' (specifying the model) and 'source' (specifying either
        training or validation data) string, it is becoming tedious to
        fix errors arising from changes in the internal setup.
        Therefore, this function is taking care of providing the
        fitting labels from version nnlib_multiobject_v0_6 going
        forward as a centralized instance.
        CHANGED in nnlib_multiobject_v06: Introduced fold functionality
        on top of the classic train/val/test datasets.  To support
        this, we introduce the new VirtualSampleContainer structure.
        If 'use_cross_validation' is True for the ModelHandler object,
        a valid integer for 'fold' MUST be passed, otherwise we error
        out.  On the other hand, if cross validation is disabled,
        'fold' will be ignored in all cases.
        """
        # Set source tensors to the appropriate ones, depending on the
        # selection of 'source'.  Raise an error in case of invalid
        # source key.
        try:
            container, _ = self._get_sample_pool(source)
        except KeyError:
            raise ValueError("I do not know '{}' as a key for "
                             "'source'.".format(source))
        # Fetch the model container.  It contains the metadata to
        # perform the necessary processing on the source data.
        model_cont = self._model_picker(target)
        labels = []
        # Go through the output configuration of the model...
        for string in model_cont.output_config:
            specifier = string.split('_')
            if specifier[0] == 'hidden':
                raise NNLibUsageError("There is no sensible way to train or "
                                      "validate on a hidden output. Please "
                                      "choose another model.")
            else:
                # CHANGED in nnlib_multiobject_v06: Due to the API
                # changes, we now access the data in the containers via
                # a function, no longer directly.  The correct field is
                # selected by passing the identifier string.
                # Note: These identifier strings must manually be kept
                # in sync over the various nnlib modules.
                # Furthermore, special transformations, for example on
                # the labels, are no longer necessary as they are
                # purpose-built by the container to suit ModelHandler's
                # internal setup.
                labels.append(container.checkout_field(specifier[0]))
        return labels

    def _model_evaluator(self, target, source, data=None):
        """Since many methods need to evaluate some model based on
        externally providable 'target' and/or 'source' strings, it is
        becoming tedious to fix errors arising from changes in the
        internal setup.  Therefore, this function is taking care of
        these evaluations from version nnlib_multiobject_v0_6 going
        forward as a centralized instance.
        """
        # Fetch the model container.
        model_cont = self._model_picker(target)
        # Set source tensors to the appropriate ones, depending on the
        # selection of source.  Raise an error in case of invalid
        # source key.
        if source not in string_alts['external']:
            data = self._data_picker(model_cont, source)
        # Evaluate the model.
        output = model_cont.model.predict(data)
        # Bring the output into a standardized form: It is expected to
        # be a (possibly singleton) list.
        if not isinstance(output, list):
            output = [output]
        return output

    def _find_likely_target(self, compile_required):
        """With increasing flexibility in model definitions, this method
        implements a very simple search algorithm that enables user
        facing functions to deal with cases where no explicit target key
        was provided to them.
        Its method is that it first goes through the combined models,
        then through the base models searching for models that were
        either compiled (if 'compile_required' was set to True) or not
        (if it was set to False).  The first model found that fits the
        definition will be proposed to the user (in interactive
        contexts) so the user may take the suggestion or break the
        execution.
        """
        print("No target was given, so we try to guess...")
        for name, model in self.combined_models.items():
            if compile_required == bool(model.compile_args):
                name = list(name)
                break
        else:
            for name, model in self.base_models.items():
                if compile_required == bool(model.compile_args):
                    break
            else:
                raise NNLibUsageError("There was no suitable guess for "
                                      "'target' in the given context.")
        print(str(name) + " was found.")
        if _interactive:
            resp = input("Continue? [YES/no] > ")
            if resp in string_alts['no']:
                raise NNLibUsageError("Stop execution for safety.")
        return name

    def _execute_recipe_build(self, name, recipe_mod):
        """Builds and registers Models built via recipes.

        Instantiates BaseModelContainers and uses their bake interfaces
        to create Keras Neural Network models.

        Parameters
        ----------
        name : string
            The name to select the build recipe from the
            `build_recipes` submodule and to register the finished
            model under.
        recipe_mod : dict
            A dictionary containing the necessary build instructions.
            These will override the respective standard settings from
            `build_recipes`.
        """
        cont = BaseModelContainer(folds=self.data_folds, silent=True)
        cont.load_build_recipe(name)
        cont.modify_build_recipe(recipe_mod)
        cont.bake()
        self.base_models[name] = cont

    # -- Public Functions --
    # Public Functions are designed to be called by the user.  Save for
    # a few exceptions, these can be called without any explicit
    # argument.  Sensible standard settings, defined in the function
    # headers, will then be used.

    def full_standard_init(self):
        """Initializes every part of the handler necessary for
        training, using standard values only.  Therefore, arguments
        are not necessary and/or can be given.  This is useful
        primarily for testing changes in existing functions.
        """
        print("Preparing training fields")
        self.prepare_full_fields(target="t", samples=64)
        print("Preparing validation fields")
        self.prepare_full_fields(target="v", samples=16)
        print("Preparing mask-modelling network")
        self.prepare_qmask_net()
        print("Preparing interpreting network")
        self.prepare_interpreter_net()
        print("Preparing/compiling the combined network")
        self.compile_model(target=('m','i'))
        print("")
        print("Handler ready to train.")

    def save(self,
             path='Handler',
             save_fields=False,
             save_models=True,
             save_model_weight_history=True):
        """Saves all data that the handler object contains such that it
        can be fully reloaded.  This becomes essential as nnlib gets
        ready to be used in batch jobs.  H5py is used to perform all
        save/load operations to files via the Copier/H5PyHandler
        classes.  save_fields controls if the data used for training is
        included in the save file (this can make it rather big and slow
        to save/load).  save_models controls if the models, including
        their current weights, are stored.  save_model_weight_history
        controls if, additionally, all weight updates generated and
        written back while training will be saved.  Note that this
        option only takes effect if save_models is True.
        """
        target, deduplicator = self._get_saving_hook(path)
        print('Saving...')
        loadstore_keys = {'init'}
        if save_models:
            loadstore_keys.add('models')
            for cont in self.base_models.values():
                model_target = deduplicator.create_instance(cont)
                cp = Copier(cont, model_target, deduplicator)
                if save_model_weight_history:
                    cp.copy(('struct','hist'))
                else:
                    cp.copy('struct')
        if save_fields:
            loadstore_keys.add('data')
        cp = Copier(self, target, deduplicator)
        cp.copy(loadstore_keys, dpcopy=False)
        print('OK')

    def evaluate_cnet(self, fields=None):
        """Evaluates the combined model.  With the streamlined processes
        introduced with nnlib_multiobject, this is mostly a convenience
        function masking a call to _model_evaluator for the user.
        """
        # If no fields are given, the function will choose the internal
        # validation dataset.
        if fields is None:
            source = 'val'
        else:
            source = 'ext'
        # Let the full model predict and return its output(s).
        return self._model_evaluator(target=('m','i'),
                                     source=source,
                                     data=fields)

    def get_model(self, target):
        """This is a front-end for '_model_picker'."""
        return self._model_picker(target)

    def get_samples(self, source):
        """This is a front-end for '_get_sample_pool'."""
        return self._get_sample_pool(source)[0]

    @contextmanager
    def init_fold_shadow(self, fold):
        """This context manager sets up all associated models and
        sample pools such that all fits, error calculations and so on
        are performed on the given fold.
        For programming convenience, the context manager can also be
        started in settings where cross validation is completely
        disabled.  In that case, it will do nothing.
        """
        # Check if there is more than one fold.  If there isn't, cross
        # validation is inactive and we do nothing, meaning that
        # self.pools continues to link to self._samples.
        if self.data_folds == 1:
            yield
        else:
            print("Entering shadow for fold {} of "
                  "{}.".format(fold, self.data_folds))
            pool_list, id_list = self._get_sample_pool(self.masking_list)
            self.pools = copy(self._samples)
            chunk_counter = 0
            for pool, id, end in zip(pool_list, id_list, self.fold_splits):
                self.pools[id] = VirtualSampleContainer(pool_list,
                                                        self.data_folds,
                                                        fold,
                                                        chunk_counter,
                                                        end)
                chunk_counter += end
            with ExitStack() as stack:
                for model in self.base_models.values():
                    stack.enter_context(model.access_fold(fold))
                for model in self.combined_models.values():
                    stack.enter_context(model.access_fold(fold))
                # We need to yield INSIDE this 'with' environment. Once
                # we leave, the models get reset (as they should be).
                yield
            self.pools = self._samples

    # The following functions used to be base level functions of
    # ModelHandler, but have been outsourced to SampleContainer.
    # However, for convenience, they remain here as stubs, calling
    # their respective equivalent in _samples, the handler's standard
    # path for its sample data.  For explanations concerning their
    # arguments, see their implementation in SampleContainer.

    def load_samples(self, obj, copy=False):
        """load_samples, if called on a ModelHandler level, can do
        different things depending on what 'object' is.  In particular,
        it can deal with paths to hdf5 files containing ModelHandlers
        and other ModelHandler instances themselves.  We copy the
        content if 'copy' is set to True; otherwise, we simply link the
        existing containers to this ModelHandler.  The function is not
        designed to deal with single SampleContainers as 'object' or
        hdf5 files not containing a full ModelHandler.  For these
        cases, refer to a SampleContainer's load_fields function.
        Arguments
          object: The object from which to copy from.  Other
            ModelHandler instances and paths to save files containing a
            ModelHandler instance are supported.
          copy: If True, a true copy of the data will be created; if
            False, only the references are set accordingly.
        """
        if isinstance(obj, six.string_types):
            fhandler = H5PyHandler(obj, True)
            source_cont = fhandler.interactive_provide_access('ModelHandler')
        elif obj.pull('type') == 'ModelHandler':
            source_cont = obj
        cp = Copier(source_cont, self)
        cp.copy('data', dpcopy=copy)

    def input_custom_sample_set(self, container, key):
        if not isinstance(key, six.string_types):
            raise ValueError("Sample sets are addressed using a string. "
                             "Provide a valid key.")
        try:
            if not container.pull('type') == 'SampleContainer':
                raise AttributeError()
        except AttributeError:
            raise ValueError("The custom data needs to be provided in a "
                             "SampleContainer.")
        self._samples[key] = container

    def prepare_full_fields(self,
                            target=None,
                            samples=256,
                            add_to_existing=True,
                            shapes=std_shapes_all,
                            nd_scale=0.25,
                            ts_movement=True,
                            ts_variation=1.0,
                            include_nones=True,
                            min_distract_objects=1,
                            max_distract_objects=1,
                            min_intensity=0.5,
                            max_intensity=1.0,
                            coords_as_floats=True,
                            add_background=0.5):
        """The meat of the function has been outsourced to the
        SampleContainer class.  This merely serves to choose the right
        SampleContainer to redirect to based on the 'target' key, and
        then passes on the arguments it receives.
        """
        def _auto_target_selector():
            """Sets the target (where to put the results) to the
            internal fields, starting with 'training'.  Throws an error
            if none is free.  In the latter case, the user must supply a
            valid 'target' key.
            """
            print("Auto target selection:")
            if self._get_sample_pool('train')[0].get_current_length() == 0:
                print("Training data chosen")
                target = 'train'
            elif self._get_sample_pool('val')[0].get_current_length() == 0:
                print("Validation data chosen")
                target = 'val'
            elif self._get_sample_pool('test')[0].get_current_length() == 0:
                print("Test data chosen")
                target = 'test'
            else:
                raise NNLibUsageError("Auto target selector could not find "
                                      "an empty field. Please specify the "
                                      "target explicitly.")
            return target

        # Check target and initiate _auto_target_selector if necessary.
        if target is None:
            target = _auto_target_selector()
        try:
            sample_pool, _ = self._get_sample_pool(target)
        except KeyError:
            raise KeyError("The key '{}' you gave for 'target' is "
                           "unknown.".format(target))
        # Most options are just taken and passed down.  Exceptions are
        # all those for which an instance of ModelHandler contains its
        # own settings.  To circumvent taking those from the
        # ModelHandler itself, please call prepare_full_fields on the
        # SampleContainer instance directly.
        sample_pool.prepare_full_fields(samples,
                                        add_to_existing,
                                        self.mask_size,
                                        shapes,
                                        nd_scale,
                                        ts_movement,
                                        ts_variation,
                                        include_nones,
                                        min_distract_objects,
                                        max_distract_objects,
                                        min_intensity,
                                        max_intensity,
                                        coords_as_floats,
                                        add_background)

    def prepare_qmask_net(self,
                          dropout=None,
                          quantization_levels=None,
                          weight_range=None):
        """Prepares the mask generation net and saves it in the
        handler.
        CHANGED in nnlib_multiobject_v04: The creation process has been
        ported to the new mechanism of build recipes, making this
        method considerably shorter.
        """
        # Create the modified build recipe according to this mothod's
        # arguments.
        recipe_mod = {'field_size': self.field_size,
                      'field_depth': self.field_depth,
                      'mask_samples': self.mask_samples,
                      'mask_size': self.mask_size}
        if dropout is not None:
            recipe_mod['dropout'] = dropout
        if quantization_levels is not None:
            recipe_mod['quantization_levels'] = quantization_levels
        if weight_range is not None:
            recipe_mod['weight_range'] = weight_range
        self._execute_recipe_build('quantized_masks', recipe_mod)

    def prepare_counter_net(self,
                            hidden_layers=None,
                            hidden_nodes=(64,16),
                            hidden_layertype='Dense',
                            hidden_activation='relu',
                            output_layertype='Dense',
                            output_activation='linear',
                            dropout=None,
                            batch_norm=None):
        """Prepare the classifier net and save it in the handler.
        Node count, layer type and activations can be given as lists,
        being interpreted as the layers from left to right in the
        resulting network topology.  As an alternative, there exists a
        string-based shorthand notation that can be given as the
        hidden_nodes argument containing node count and layer type.
        See below for further description.
        """
        recipe_mod = {'mask_samples': self.mask_samples*self.field_depth,
                      'shorthand_hidden_layers': hidden_layers,
                      'hidden_nodes': hidden_nodes,
                      'hidden_layertype': hidden_layertype,
                      'hidden_activation': hidden_activation,
                      'output_layertype': output_layertype,
                      'output_activation': output_activation}
        if dropout is not None:
            recipe_mod['dropout'] = dropout
        if batch_norm is not None:
            recipe_mod['batch_norm'] = batch_norm
        self._execute_recipe_build('counter', recipe_mod)

    def prepare_interpreter_net(self,
                                hidden_layers=None,
                                hidden_nodes=(64,256,64,16),
                                hidden_layertype=('Dense','LSTM',
                                                  'Dense','Dense'),
                                hidden_activation=('relu','tanh',
                                                   'relu','relu'),
                                hidden_output=False,
                                output_layertype='Dense',
                                output_activation='linear',
                                override_objtype=False,
                                override_objcount=False,
                                dropout=None,
                                batch_norm=None):
        """Prepare the interpreter net and save it in the handler.
        Node count, layer type and activations can be given as lists,
        being interpreted as the layers from left to right in the
        resulting network topology.

        Parameters
        ----------

        """
        recipe_mod = {'mask_samples': self.mask_samples*self.field_depth,
                      'shorthand_hidden_layers': hidden_layers,
                      'hidden_nodes': hidden_nodes,
                      'hidden_layertype': hidden_layertype,
                      'hidden_activation': hidden_activation,
                      'output_layertype': output_layertype,
                      'output_activation': output_activation,
                      'label_layout': self.label_layout}
        recipe_mod['objtype_present'] = (self._model_exists('classifier')
                                           or override_objtype)
        recipe_mod['counter_present'] = (self._model_exists('count')
                                            or override_objcount)
        if dropout is not None:
            recipe_mod['dropout'] = dropout
        if batch_norm is not None:
            recipe_mod['batch_norm'] = batch_norm
        self._execute_recipe_build('interpreter', recipe_mod)

    def prepare_classifier_net(self,
                               hidden_layers=None,
                               hidden_nodes=(64,16),
                               hidden_layertype='Dense',
                               hidden_activation='relu',
                               output_layertype='Dense',
                               output_activation='softmax',
                               dropout=None,
                               batch_norm=None):
        """Prepare the classifier net and save it in the handler.
        Node count, layer type and activations can be given as lists,
        being interpreted as the layers from left to right in the
        resulting network topology.  As an alternative, there exists a
        string-based shorthand notation that can be given as the
        hidden_nodes argument containing node count and layer type.
        See below for further description.
        """
        recipe_mod = {'shorthand_hidden_layers': hidden_layers,
                      'hidden_nodes': hidden_nodes,
                      'hidden_layertype': hidden_layertype,
                      'hidden_activation': hidden_activation,
                      'output_layertype': output_layertype,
                      'output_activation': output_activation}
        try:
            intcontainer = self.base_models['interpreter']
            if intcontainer.output_config[1] == 'hidden':
                recipe_mod['input_config'] = 'hidden'
                recipe_mod['input_nodes'] = (
                  intcontainer.model.outputs.shape.as_list()[-1]
                )
            else:
                raise KeyError()
        except (KeyError,IndexError):
            recipe_mod['input_config'] = 'intensities'
            recipe_mod['input_nodes'] = self.mask_samples*self.field_depth
        if dropout is not None:
            recipe_mod['dropout'] = dropout
        if batch_norm is not None:
            recipe_mod['batch_norm'] = batch_norm
        self._execute_recipe_build('classifier', recipe_mod)

    def prepare_encoder_net(self,
                            residual_blocks=2,
                            no_add=False,
                            use_lstm=False):
        """Prepare the decoder net and save it in the handler.
        This method implements the decoder part from the autoencoder
        model presented by Theis et al. in "Lossy Image Compression with
        Compressiver Autoencoders".  It is based heavily off of the
        pytorch implementation of said network from
        https://github.com/alexandru-dinu/cae/blob/master/src/models/cae_16x8x8_zero_pad_bin.py
        by Alexandru Dinu.

        Arguments:
          residual_blocks: Specifies how many residual (meaning input
            shape=output shape) blocks are used in the decoder line.
            There can theoretially be as many as one likes of these.
            Three of them are used in the paper.
          no_add: The reference implementation includes bypasses for
            each residual block.  This feature can be disabled by
            setting this option to True.
          use_lstm: As it turns out that the overall object detection
            quality can be improved by taking the continuous nature of
            the objects moving/turning/scaling over time into account,
            we can harness this by using convolutional LSTMs instead of
            non-time-aware convolutional layers.  Note that this is an
            extension on the autoencoder presented in the paper as that
            one is not designed to work on moving images.
        """
        recipe_mod = {'field_size': self.field_size,
                      'residual_blocks': residual_blocks,
                      'no_add': no_add,
                      'use_lstm': use_lstm}
        self._execute_recipe_build('encoder', recipe_mod)

    def prepare_1d_encoder_net(self,
                               residual_blocks=3,
                               no_add=False,
                               use_lstm=False):
        """Prepare the decoder net and save it in the handler.
        This method implements the decoder part from the autoencoder
        model presented by Theis et al. in "Lossy Image Compression with
        Compressiver Autoencoders".  It is based heavily off of the
        pytorch implementation of said network from
        https://github.com/alexandru-dinu/cae/blob/master/src/models/cae_16x8x8_zero_pad_bin.py
        by Alexandru Dinu.

        Arguments:
          residual_blocks: Specifies how many residual (meaning input
            shape=output shape) blocks are used in the decoder line.
            There can theoretially be as many as one likes of these.
            Three of them are used in the paper.
          no_add: The reference implementation includes bypasses for
            each residual block.  This feature can be disabled by
            setting this option to True.
          use_lstm: As it turns out that the overall object detection
            quality can be improved by taking the continuous nature of
            the objects moving/turning/scaling over time into account,
            we can harness this by using convolutional LSTMs instead of
            non-time-aware convolutional layers.  Note that this is an
            extension on the autoencoder presented in the paper as that
            one is not designed to work on moving images.
        """
        recipe_mod = {'field_size': self.field_size,
                      'residual_blocks': residual_blocks,
                      'no_add': no_add,
                      'use_lstm': use_lstm}
        self._execute_recipe_build('1d_encoder', recipe_mod)

    def prepare_adapter_net(self,
                            add_layers=1,
                            use_activations=True,
                            use_lstm=False):
        """Build a little adapter net to enable the interconnection
        between the intensity mask network (prepare_qmask_net) and the
        decoder network (prepare_decoder_net).
        """
        recipe_mod = {'field_size': self.field_size,
                      'mask_samples': self.mask_samples,
                      'add_layers': add_layers,
                      'use_activations': use_activations,
                      'use_lstm': use_lstm}
        self._execute_recipe_build('adapter', recipe_mod)

    def prepare_1d_adapter_net(self,
                               add_layers=2,
                               use_activations=True,
                               use_lstm=False):
        """Build a little adapter net to enable the interconnection
        between the intensity mask network (prepare_qmask_net) and the
        decoder network (prepare_decoder_net).
        """
        recipe_mod = {'field_size': self.field_size,
                      'mask_samples': self.mask_samples,
                      'add_layers': add_layers,
                      'use_activations': use_activations,
                      'use_lstm': use_lstm}
        self._execute_recipe_build('1d_adapter', recipe_mod)

    def prepare_conv_adapter_net(self,
                                 use_activations=True,
                                 use_lstm=False):
        """Build a little adapter net to enable the interconnection
        between the intensity mask network (prepare_qmask_net) and the
        decoder network (prepare_decoder_net).

        Arguments:
          use_activations: Enables or disables the use of tanh as
            the final activation function, analogous to the standard
            encoder net.  If False, no extra activation layer will be
            added, implicitly yielding linear activation.
          use_lstm: As it turns out that the overall object detection
            quality can be improved by taking the continuous nature of
            the objects moving/turning/scaling over time into account,
            we can harness this by using convolutional LSTMs instead of
            non-time-aware convolutional layers.  Note that this is an
            extension on the autoencoder presented in the paper as that
            one is not designed to work on moving images.
        """
        recipe_mod = {'field_size': self.field_size,
                      'mask_samples': self.mask_samples,
                      'use_activations': use_activations,
                      'use_lstm': use_lstm}
        self._execute_recipe_build('conv_adapter', recipe_mod)

    def prepare_decoder_net(self,
                            residual_blocks=2,
                            no_add=False,
                            use_lstm=False,
                            dist_free=False):
        """Prepare the decoder net and save it in the handler.
        This method implements the decoder part from the autoencoder
        model presented by Theis et al. in "Lossy Image Compression with
        Compressiver Autoencoders".  It is based heavily off of the
        pytorch implementation of said network from
        https://github.com/alexandru-dinu/cae/blob/master/src/models/cae_16x8x8_zero_pad_bin.py
        by Alexandru Dinu.

        Arguments:
          residual_blocks: Specifies how many residual (meaning input
            shape=output shape) convolutional blocks are used in the
            decoder line.  There can theoretially be as many as one
            likes of these.  Three of them are used in the paper.
          no_add: The reference implementation includes bypasses for
            each residual block.  This feature can be disabled by
            setting this option to True.
          use_lstm: As it turns out that the overall object detection
            quality can be improved by taking the continuous nature of
            the objects moving/turning/scaling over time into account,
            we can harness this by using convolutional LSTMs instead of
            non-time-aware convolutional layers.  Note that this is an
            extension on the autoencoder presented in the paper as that
            one is not designed to work on moving images.
          dist_free: Determines whether the decoder is supposed to
            generate disturbance free fields as output.  This does not
            influence the topology, only the output_config, but
            nevertheless it makes sense to specify this here.
        """
        recipe_mod = {'field_size': 64,
                      'residual_blocks': residual_blocks,
                      'no_add': no_add,
                      'use_lstm': use_lstm,
                      'dist_free': dist_free}
        self._execute_recipe_build('decoder', recipe_mod)

    def prepare_1d_decoder_net(self,
                               residual_blocks=3,
                               no_add=False,
                               use_lstm=False,
                               dist_free=False):
        """Prepare the decoder net and save it in the handler.
        This method implements the decoder part from the autoencoder
        model presented by Theis et al. in "Lossy Image Compression with
        Compressiver Autoencoders".  It is based heavily off of the
        pytorch implementation of said network from
        https://github.com/alexandru-dinu/cae/blob/master/src/models/cae_16x8x8_zero_pad_bin.py
        by Alexandru Dinu.

        Arguments:
          residual_blocks: Specifies how many residual (meaning input
            shape=output shape) convolutional blocks are used in the
            decoder line.  There can theoretially be as many as one
            likes of these.  Three of them are used in the paper.
          no_add: The reference implementation includes bypasses for
            each residual block.  This feature can be disabled by
            setting this option to True.
          use_lstm: As it turns out that the overall object detection
            quality can be improved by taking the continuous nature of
            the objects moving/turning/scaling over time into account,
            we can harness this by using convolutional LSTMs instead of
            non-time-aware convolutional layers.  Note that this is an
            extension on the autoencoder presented in the paper as that
            one is not designed to work on moving images.
          dist_free: Determines whether the decoder is supposed to
            generate disturbance free fields as output.  This does not
            influence the topology, only the output_config, but
            nevertheless it makes sense to specify this here.
        """
        recipe_mod = {'field_size': self.field_size,
                      'residual_blocks': residual_blocks,
                      'no_add': no_add,
                      'use_lstm': use_lstm,
                      'dist_free': dist_free}
        self._execute_recipe_build('1d_decoder', recipe_mod)

    def compile_model(self,
                      target=None,
                      optimizer='adam',
                      lr=None,
                      loss=None,
                      loss_weights=None):
        """Compiles the models of the handler.  Target, optimizer and
        loss can be given.
        """
        if target is None:
            target = self._find_likely_target(compile_required=False)
        # CHANGED in nnlib_multiobject_v04: The internals of this
        # function have been outsourced to the identically named
        # function 'compile' in the ModelContainer class, to which this
        # now simply redirects.
        model = self._model_picker(target)
        model.compile(optimizer=optimizer,
                      lr=lr,
                      loss=loss,
                      loss_weights=loss_weights)

    def reset_model(self, target='interpreter'):
        """New implementation with nnlib_multiobject_v01: Simply call
        reset on the given model.  This method is basically just a
        front-end for that with some additional warnings attached.
        Also makes accessing the desired model easier by using the
        string-based user friendly interface provided by _model_picker.
        """
        if not isinstance(target, six.string_types):
            print("Warning: Calling reset_model on a combined model resets "
                  "all of its base models along with it.")
            if _interactive:
                resp = input("Continue? [YES/no] > ")
                if resp in string_alts['no']:
                    return
        model = self._model_picker(target=target)
        model.reset()
        print("OK")

    def fit_and_eval(self,
                     target=None,
                     train_source='train',
                     val_source='val',
                     epochs=10,
                     batch_size=32,
                     weights_saving_mode=1,
                     returns=False,
                     early_stopping=None,
                     more_callbacks=None):
        """Largely automates the process of training the network by
        using the internal datasets of ModelHandler.
        This method also calculates some performance metrics of the
        trained network and returns them to the caller.
        Arguments
          target: Specifies which model to train.  This is done by
            providing a string or a collection of strings identifying
            models this handler holds.  Models are gathered by the
            backend function '_model_picker', so for further info,
            refer to its documentation.  If left blank, the backend
            function '_find_likely_target' will try to guess a model.
          train_source: Specifies the dataset from which to take the
            training data.  Standard is obviously 'train', but it
            can be overridden.
          val_source: Specifies the dataset from which to take the
            validation data.  Standard is obviously 'val', but it
            can be overridden.
          epochs: Number of epochs to train.
          batch_size: Size of training batches in which the data will
            be presented to the neural net.
          weights_saving_mode: There are four legal modes
             0: all weights are kept
             1: only the weights that improved the model with respect
                to the monitored quantity are saved
             2: as above, but the weights that yielded the best results
                before are actively deleted
             3: does not save weights at all and only increments epoch
                counter
          returns: If True, the classic set of training history,
            specialized training data errors and, if applicable,
            validation data errors is returned.  If it is False
            (default), nothing is returned.  Note, however, that the
            training history is attached to the fields in the
            ModelContainers regardless.
          early_stopping: If an integer is given, Keras's EarlyStopping
            callback is used with the given number as patience (meaning
            training stops if no improvement was made for the given
            number of epochs.)
          more_callbacks: Use this to provide additional callbacks for
            the training process directly.  (Can be a callback instance
            or a list of multiple ones.)
        """
        # Engage the auto target finder if none is specified. It will
        # ask for user confirmation in interactive mode.
        if target is None:
            target = self._find_likely_target(compile_required=True)
        # Choose the model
        model = self._model_picker(target)
        # Test if the model is compiled.  If not, offer the option to
        # compile using standard settings.
        if model.model.optimizer is None and _interactive:
            print("The model you selected has not been compiled.")
            resp = input("Do you want to compile it now (using "
                         "standard settings)? [yes/NO] > ")
            if resp in string_alts['yes']:
                model.compile()
            else:
                raise RuntimeError("You must compile a model before "
                                   "training/testing.")

        if self.data_folds > 1:
            print("Cross validation is active for {} "
                  "folds.".format(self.data_folds))
        if returns:
            t_error_dicts = self.data_folds * [None]
            v_error_dicts = self.data_folds * [None]
        # CHANGED in nnlib_multiobject_v06: Most of the function is now
        #   contained in the following for-loop.  Note that when cross
        #   validation is disabled, it will only be executed once and,
        #   wherever necessary, we fall back on the old code paths to
        #   make sure the traditional behaviour stays intact.
        for fold in range(self.data_folds):
            # CHANGED in nnlib_multiobject_v06: Cross validation
            #   context is established by init_fold_shadow.  For
            #   convenience, it can also be called if 'self.data_folds'
            #   is 1, in which case it will not do anything.
            with self.init_fold_shadow(fold):
                ctd = self._data_picker(model, source=train_source)
                ctl = self._label_picker(model, source=train_source)
                try:
                    cvd = self._data_picker(model, source=val_source)
                    cvl = self._label_picker(model, source=val_source)
                    val_dataset = (cvd,cvl)
                except NNLibUsageError:
                    val_dataset = None
                    print("No validation data present. We will train without "
                          "it.")
                # CHANGED in mo_08:  Control over the actual training
                # process now lives inside the ModelContainer instance
                # to increase granularity and flexibility.
                model.fit_and_eval(x=ctd,
                                   y=ctl,
                                   validation_data=val_dataset,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   early_stopping=early_stopping,
                                   saving_mode=weights_saving_mode,
                                   more_callbacks=more_callbacks)
                # CHANGED in nnlib_multiobject_v04: It turns out that
                # in most instances, the error calculation is
                # redundant.  Therefore, if the new argument 'returns'
                # is False (default), no further error metrics are
                # calculated and nothing is returned.
                if returns:
                    hist = model.get_training_hist()
                    # Let the model predict on training data...
                    t_error_dicts[fold] = (
                      self._calculate_errors(target,
                                             'train',
                                             None,
                                             None,
                                             acceptable_deviation=1.0,
                                             return_value=False)
                    )
                    # ...and on validation data, if present.
                    if val_dataset is not None:
                        v_error_dicts[fold] = (
                          self._calculate_errors(target,
                                                 'train',
                                                 None,
                                                 None,
                                                 acceptable_deviation=1.0,
                                                 return_value=False)
                        )
        print("OK")
        if not returns:
            return
        elif val_dataset is None:
            # If no validation data is present, return the new-style
            # dictionaries of errors for the training data only.
            return hist, t_error_dicts
        else:
            return hist, t_error_dicts, v_error_dicts

    def calculate_errors(self,
                         target=None,
                         source='validation',
                         fields=None,
                         labels=None,
                         acceptable_deviation=0.5,
                         return_value=True):
        """Calculates the model errors with respect to the internal
        model specified in the 'model' key against internally or
        externally provided fields.  If 'source' is set to 'external',
        the fields and labels given in the respective arguments are
        used.  (Remember, however, that you yourself are responsible
        for providing usable data.  There is no further preprocessing
        in this case.)  If it is set to any other valid key, the data
        will be generated accordingly, externally provided fields and
        labels data discarded.
        """
        if target is None:
            target = self._find_likely_target(compile_required=True)
        # The private function _calculate_errors returns one of the
        # following, depending upon return_value.
        # > return_value == True:
        # >     return f_y, abserror, miss_indices, misses
        # > return_value == False:
        # >     return abserror, miss_indices, misses
        return self._calculate_errors(target,
                                      source,
                                      fields,
                                      labels,
                                      acceptable_deviation,
                                      return_value)

    def compare_visually(self,
                         range_start=4,
                         range_stop=0,
                         target=None,
                         source='val',
                         data=None,
                         timesteps=None,
                         show_separate=False,
                         mode='show',
                         save_draw_options={}):
        """Creates a visualization of the network performance by
        drawing the real and predicted objects for a selectable part
        of the data tensors to the same canvas, but in different
        colours.
        CHANGED in nnlib_multiobject_v01: Made the function much more
          flexible.  Target models to evaluate on can be selected and a
          sensible way of comparing the specific combination of outputs
          with the reference is used automatically.
        CHANGED in nnlib_multiobject_v06: The function was renamed to
          compare_visually (formerly compare_validation) and once again
          extended in flexibility.  It now allows selecting the source
          instead of assuming validation.  Also, the functionality to
          have it draw directly without another call was reintroduced.
        """
        # If range_start is bigger than range_stop, switch range_start
        # and range_stop around.  Allows both arguments to be given in
        # arbitrary order.  Specifically: If only one argument is
        # given, that one will be the upper limit.
        if range_stop < range_start:
            range_start, range_stop = range_stop, range_start
        # Initiate the guessing function if no target was provided.
        if target is None:
            target = self._find_likely_target(compile_required=True)
        # Get the model.
        model = self._model_picker(target=target)
        # Get the requested data container.
        container_obj = self._get_sample_pool(source)[0]
        # Get the first piece for our slicing operations by converting
        # range_start and range_stop into a usable slice.
        sample_slice = slice(range_start, range_stop)
        # There are two ways to legally provide the timesteps to be
        # drawn.  It can be a slice (in which case the slice is
        # attached to the sample specification) or a single int (in
        # which case, for compatibility reasons, that int is turned
        # into a slice which is then attached to the sample
        # specifications).  In all other cases, 'timesteps' is quietly
        # ignored.  This includes it being a NoneType as in the
        # standard setting.
        if isinstance(timesteps, slice):
            time_slice = timesteps
        elif isinstance(timesteps, int):
            time_slice = slice(timesteps, timesteps+1)
        else:
            time_slice = slice(None)
        extract_slice = (sample_slice,time_slice)
        # If 'source' points to external data sources, we do a very
        # basic sanity check for it being a NoneType (because it's the
        # standard setting), but if it is not, we just assume the best.
        # No further checks are done in this case, the user needs to
        # provide the data in a usable format.
        if source in string_alts['external']:
            if data is None:
                raise NNLibUsageError("Data must be provided through the "
                                      "'data' argument if source is selected "
                                      "to be external.")
        else:
            data = self._data_picker(target, source, extract_slice)
        # Evaluate the network.  Note that we inject the data as
        # external even if it was taken from the internal fields before
        # because that way we can apply the desired slicing beforehand.
        # The resulting data flow is otherwise the same, though.
        compare = self._model_evaluator(target=model,
                                        source='ext',
                                        data=data)
        # CHANGED in nnlib_multiobject_v06: We now fully depend on
        # SampleContainer to build the reconstructed fields as it
        # contains every tool we need.
        # Set up fields for the return data this method can deal with.
        reconst_labels = None
        reconst_obj_type = None
        reconst_fields = None
        for out in model.output_config:
            specifier = out.split('_')
            if specifier[0] == 'labels':
                reconst_labels = compare[model.output_config.index(out)]
                reconst_labels = reconst_labels[:,time_slice]
            elif specifier[0] == 'fields':
                reconst_fields = compare[model.output_config.index(out)]
                reconst_fields = reconst_fields[:,time_slice]
            elif specifier[0] == 'objtype':
                reconst_obj_type = compare[model.output_config.index(out)]
                reconst_obj_type = reconst_obj_type[:,time_slice]
        if reconst_labels is None and reconst_fields is None:
            raise ValueError("There is a problem with the specified model: "
                             "Cannot find an output based on which fields can "
                             "be compared.")
        if reconst_labels is not None and reconst_obj_type is None:
            reconst_obj_type = (
              container_obj.checkout_field('objtype')[extract_slice]
            )
        if reconst_fields is None:
            # If no fields were reconstructed directly by the model,
            # draw the predictions that were returned in the form of
            # labels.
            # CHANGED: Beginning with NNlib_multiobject_v06, this is
            # dealt with by instantiating a SampleContainer object as
            # that comes with all the tooling necessary to draw the
            # field.
            temp_container = SampleContainer(self.field_size,
                                             self.field_depth,
                                             reconst_labels.shape[1],
                                             self.label_layout,
                                             self.full_layout)
            # We define a kind of dummy base data tensor that only
            # contains shapes.  We can do that because the
            # DistFreeCreator that will ultimately be tasked with
            # drawing does not care about anything else and it is the
            # only Creator that will be called.
            bdata = np.empty(reconst_labels.shape[:1],
                               dtype=[('shape','U12'),('intensity','f2')])
            # To fix the missing 'intensity' value in base data, set
            # it to 1 for all entries.  This may be suboptimal, though,
            # and require a change in the future.
            bdata['intensity'] = np.ones((reconst_labels.shape[0],))
            for i, otype in enumerate(reconst_obj_type):
                bdata[i]['shape'] = std_shapes_all[otype.argmax()]
            # Submit the collected data to our SampleContainer instance.
            temp_container._set_up_fields((None,None,reconst_labels,
                                             bdata,None,None))
            reconst_fields = temp_container.checkout_field('nodistfields')
        orig_fields = container_obj.checkout_field('fields')[extract_slice]
        # Concatenate predicted and real fields to a valid RGB space.
        if self.train_on_no_dist:
            red_fields = (
              container_obj.checkout_field('nodistfields')[extract_slice])
            if show_separate:
                combined = np.concatenate((orig_fields,
                                           red_fields,
                                           reconst_fields),
                                          axis=0)
            else:
                combined = np.concatenate((1.0-0.8*orig_fields,
                                           1.0-0.8*red_fields,
                                           1.0-0.8*reconst_fields),
                                          axis=-1)
        else:
            if show_separate:
                combined = np.concatenate((orig_fields,
                                           reconst_fields),
                                          axis=0)
            else:
                combined = np.concatenate((1.0-0.8*orig_fields,
                                           1.0-0.5*orig_fields
                                             - 0.5*reconst_fields,
                                           1.0-0.8*reconst_fields),
                                          axis=-1)
        # CHANGED in nnlib_multiobject_v06: A new 'mode' switch is
        # introduced.  If 'show' is chosen, the 'combined' tensor will
        # be drawn by multishow.  If 'save' is chosen, the 'combined'
        # tensor will be saved as a series of images using multisave.
        # The behaviour of both can be adjusted by passing a dict of
        # options as the new 'save_draw_options' argument.  If 'mode'
        # says anything else, the 'combined' tensor is simply returned
        # as was the standard behaviour in
        # nnlib_multiobject_v04...nnlib_multiobject_v05.
        if mode == 'show':
            multishow(combined, **save_draw_options)
        elif mode == 'save':
            multisave(combined, **save_draw_options)
        else:
            return combined

    def visualize_fields(self, sample_pool='train', **kwargs):
        """Provides the functionality of
        SampleContainer.visualize_fields through a ModelHandler
        instance.

        Arguments:
          sample_pool: Select the sample pool from which to draw the
            data by passing an appropriate string.
          kwargs: The rest of the arguments are passed through to
            SampleContainer.visualize_fields.  See its documentation
            for allowed options.
        """
        if not isinstance(sample_pool, six.string_types):
            raise NNLibUsageError("visualize_fields needs a string identifier "
                                  "as its sample_pool argument.")
        self._get_sample_pool(sample_pool)[0].visualize_fields(**kwargs)

    def ind_test(self,
                 param_set,
                 shape='rectangle',
                 add_background=0.0,
                 show_results=True):
        """Allows the user to manually enter an object specification
        and compare what the network makes of it.

        Arguments
          param_set: Numpy array (or convertible) with 1, 2 or 3
            dimensions.  Last dimension contains 5 entries (x_pos,
            y_pos,width,height,alpha) constituting the object
            parameters.  2nd-to-last dimension (if present) contains an
            arbitrary number of poses the object(s) will go through
            within the timeseries.  3rd-to-last dimension (if present)
            contains multiple objects, the first of which is considered
            the main one for the nets to predict.
          shape: String or list of strings describing the shape(s) of
            the object(s).  If it is a list, it needs to contain as
            many entries as there are objects (see above).
          add_background: A value between 0.0 (incl.) and 1.0 (excl.)
            describing the probability for the inclusion of random
            errors (constituted by random lines within the field to
            predict on).
          show_results: If True, the results are drawn using multishow.
            If False, the numeric results are returned to the caller.
        """
        if not _interactive and show_results:
            print("Plotting of any kind is not available outside of "
                  "interactive contexts. Falling back to returning the "
                  "error.")
            show_results = False

        if add_background < 0.0 or add_background >= 1.0:
            raise ValueError("Illegal choice of add_background: A float "
                             "between 0.0 (incl., no background artifacts) "
                             "and 1.0 (excl., very many background "
                             "artifacts) is required. (False works, too.)")

        # Check the input for basic plausibility
        try:
            param_set_np = np.array(param_set)
            # Check the size of param_set's last dimension.
            if not param_set_np.shape[-1] == 5:
                raise IndexError("param_set does not have the right shape. "
                                 "It needs to be of size 5 in the last "
                                 "dimension.")
        except IndexError:
            raise TypeError("'param_set' needs to be array-like.")

        if param_set_np.ndim == 1:
            param_set_np = np.expand_dims(param_set_np, 0)
        if param_set_np.ndim == 2:
            param_set_np = np.expand_dims(param_set_np, 0)

        # Test if param_set_np is a singleton wrt its states dimension.
        if param_set_np.shape[1] == 1:
            # If so, repeat it so we get two entries for the
            # interpolator.  This saves us from having a special case
            # in the following.
            param_set_np = np.repeat(param_set_np, 2, 1)
            ts_movement = False
        else:
            ts_movement = True

        if isinstance(shape, six.string_types):
            shape = (shape,)
        if len(shape) == 1:
            shape = shape * param_set_np.shape[0]
        elif len(shape) != param_set_np.shape[0]:
            raise ValueError("The shape argument does not fit the amount of "
                             "objects suggested by the data argument.")
        dist_objs = param_set_np.shape[0] - 1
        phases = param_set_np.shape[1]
        base_data = np.zeros(
          (1,),
          dtype=[('data','f4',(phases,5)),
                 ('shape','U12'),
                 ('dst',[('data','f4',(phases,5)),
                         ('shape','U12'),
                         ('presence','?')],(dist_objs,))]
        )
        # Now that the parameters are transformed into a generic form
        # and the empty base data tensor is built, we can fill the
        # latter.  Start by looping through all phases to input the
        # pose data.
        for ph in range(phases):
            base_data[0]['data'][ph] = param_set_np[0,ph]
            # At the same time, also loop over the distraction objects.
            for dst in range(dist_objs):
                base_data[0]['dst'][dst]['data'][ph] = param_set_np[dst+1,ph]
        # Fill in the requested shape.
        base_data[0]['shape'] = shape[0]
        # Do the same for the distraction objects.
        for dst in range(dist_objs):
            base_data[0]['dst'][dst]['shape'] = shape[dst+1]
            # For now, consider all distraction objects to be present.
            # Further logic may be implemented to adapt and/or to give
            # the user a choice.
            base_data[0]['dst'][dst]['presence'] = True
        # CHANGED in nnlib_multiobject_v04: Use the more feature-rich
        # NewFieldCreator to draw the images that will be fed into the
        # network.
        # Create a NewFieldCreator instance, feed the created base_data
        # to it and let it do its thing.
        creator = NewFieldCreator(self.field_size, True)
        creator.set_predefined_data(base_data)
        creator.compile_into_fields(ts_length=self.ts_length,
                                    ts_variation=0.0,
                                    ts_movement=ts_movement,
                                    add_background=add_background)
        # Let the network predict.
        if MOD_SETTINGS['DEBUG']:
            _, fields, labels, onehot, _, _ = creator.pass_back()
        else:
            _, fields, labels, onehot, _ = creator.pass_back()
        predicted = self.evaluate_cnet([fields,onehot])
        # Remember that in non-interactive environments, show_results
        # has been reset above, so no need to check again.
        if show_results:
            # Get a LabelFieldCreator instance to draw the labels
            # returned by the net.
            creator = LabelFieldCreator(self.field_size)
            creator.set_predefined_data(predicted[0], shape)
            creator.compile_into_fields()
            comp_fields = creator.pass_back()
            # Create an RGB-like image matrix to have multishow draw it
            combined = (
              np.concatenate((1-fields,np.ones(fields.shape),1-comp_fields),
                             axis=-1)
            )
            multishow(combined)
        else:
            # Calculate the error
            error = labels - predicted[0]
            return error

    def summary(self):
        """Provides a human-readable summary about some basic settings
        as well as the models that are present in the handler.  To be
        expanded in the future as more functionality is added.
        """
        # First, print info about the basic setup.
        print("Basic setup:")
        print("  Field size: {}x{} px".format(self.field_size,
                                              self.field_size))
        print("  Depth slices: {}".format(self.field_depth))
        print("  Mask samples: {}".format(self.mask_samples))
        print("  Mask size: {}x{} px".format(self.mask_size, self.mask_size))
        print("  Timeseries length: {}".format(self.ts_length))
        try:
            train_pool = self._get_sample_pool('train')[0]
            print("  Current size of registered training dataset: {}".format(
                    train_pool.get_current_length()))
        except TypeError:
            print("  No training dataset registered.")
        try:
            val_pool = self._get_sample_pool('val')[0]
            print("  Current size of registered validation dataset: "
                  "{}".format(val_pool.get_current_length()))
        except TypeError:
            print("  No validation dataset registered.")
        try:
            test_pool = self._get_sample_pool('test')[0]
            print("  Current size of registered test dataset: {}".format(
                    test_pool.get_current_length()))
        except TypeError:
            print("  No test dataset registered.")
        # Second, print some info about all the models currently
        # registered.
        print("Listing currently registered base models...")
        for mdl_name, model in self.base_models.items():
            print("  Found " + mdl_name)
            for i_name, inp in zip(model.input_config, model.model.inputs):
                print("    Input '{}' is of shape {}".format(i_name,
                                                             inp.shape))
            for o_name, outp in zip(model.output_config, model.model.outputs):
                print("    Output '{}' is of shape {}".format(o_name,
                                                              outp.shape))
            if model.compile_args is None:
                print("    Currently not compiled for training.")
            else:
                print("    Trained for {} epochs by itself.".format(
                        model.epoch_counter))
                print("    Trained for {} epochs total.".format(
                        model.total_epoch_counter()))
        print("Listing currently registered combined models...")
        for mdl_name, model in self.combined_models.items():
            print("  Found {}".format(list(mdl_name)))
            for i_name, inp in zip(model.input_config, model.model.inputs):
                print("    Input '{}' is of shape {}".format(i_name,
                                                             inp.shape))
            for o_name, outp in zip(model.output_config, model.model.outputs):
                print("    Output '{}' is of shape {}".format(o_name,
                                                              outp.shape))
            if model.compile_args is None:
                print("    Currently not compiled for training.")
            else:
                print("    Trained for {} epochs.".format(
                        model.epoch_counter()))


register_container(ModelHandler)


class ContainerGroup(AbstractContainerGroup):

    def __init__(self):
        self.replace_map = {}

    def _check(self, data):
        """Recursive function looking at the contents to be passed to
        new objects.  It checks for Container instances that are known
        to have been saved before and replaces the references to them
        with references to the new object.
        """
        if isinstance(data, dict):
            for key, element in data.items():
                result = self._check(element)
                if result is not None:
                    data[key] = result
        elif isinstance(data, list):
            for index, element in enumerate(data):
                result = self._check(element)
                if result is not None:
                    data[index] = result
        elif isinstance(data, tuple):
            # Note to self: If this triggers, change the respective
            # object to use lists.  Properly supporting tuples here
            # would be hard.
            for index, element in enumerate(data):
                result = self._check(element)
                if result is not None:
                    raise NotImplementedError("Traversed over a tuple "
                                              "containing a Container "
                                              "instance. Cannot continue.")
        elif is_cont_obj(data):
            try:
                # This here is the only situation where _check returns
                # an object that is not None!  The calling _check
                # function simply replaces its currently traversed
                # entry with what the child _check function returns
                # here.
                return self.replace_map[(data.pull('type'),data.pull('id'))]
            except KeyError:
                pass
        # Note that, for most data types, _check simply falls off here,
        # returning None.

    def register(self, source, dest):
        src_type = source.pull('type')
        src_id = source.pull('id')
        self.replace_map[(src_type,src_id)] = dest

    def check(self, data):
        self._check(data)

    def create_instance(self, obj):
        """Creates a vanilla object of the same type as obj."""
        new_obj = instantiate_container(obj)
        return new_obj

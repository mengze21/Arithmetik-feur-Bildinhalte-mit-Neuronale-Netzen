__version__ = 'br_02'

import six
import warnings
import numpy as np
from collections.abc import Iterable
from math import log2
from tensorflow.keras import layers as lrs
from tensorflow.keras.models import Model

from .support_lib import QuantizedConv2D
from .support_lib import NNLibUsageError
from .support_lib import std_shapes_all


def _decode_shorthand(string, dol):
    """Decodes the shorthand syntax for layer generation introduced
    with nnlib_v14.  With nnlib_multiobject, the code for creating the
    networks themselves has been refactored to genericize the process.
    This resulted in making _decode_shorthand a general private method
    within the 'build_recipes'-module instead of a private function of
    prepare_interpreter_net or a private method of 'ModelHandler'.

    Arguments:
      string: The string that will be converted into layer specs (see
        'Syntax' below)
      dol: Short for 'dictionary of layers'.  Needs to be supplied by
        the calling recipe.  Specifies what layers are accepted for the
        network definition (the dict's keys) and provides the
        nonlinearity to be used with them (the dict's values).

    Syntax: Layer specifiers in the string start with a single letter,
      specifying the layer type.  These letters are the first ones of
      the layers' full names - case insensitive.  This specifier is
      followed by a number of arbitrary length, defining the node count
      for this layer.  Repeat this structure to get as many layers as
      you need, in the order you defined them in.  To make the notation
      more human readable, it is allowed to insert spaces and commas in
      the string.  Bear in mind, however, that both of these will
      simply be removed for processing, with no further consideration
      of common sense, meaning that e.g. the string ' d  ,14,89L, ,3,'
      is perfectly valid, giving you a dense layer with 1,489 nodes and
      an LSTM layer with three nodes.
      Additionally, if numbers are followed by a '>' symbol, they will
      be made available as an output.  This output comes in addition to
      the normal output at the end of the net.
      Examples of strings as they are intended to be used:
      string = "d24,d96>,r48"
        (-> Dense layer with 24 nodes followed by a dense layer with 96
            nodes that will be turned into an output followed by a
            simple RNN layer with 48 nodes.)
      string = "L42 D21 d12> r24>"
        (-> LSTM layer with 42 nodes followed by a dense layer with
            21 nodes followed by a dense layer with 12 nodes that will
            be turned into an output followed by a simple RNN layer
            with 24 nodes that will also be turned into an output.)
    """

    # The list of supported layers is now defined within the calling build
    # recipe and the correspondence is defined ad-hoc in the following lines.
    correspondence = {}
    for layer in dol:
        correspondence[layer[0].lower()] = layer

    # Before parsing, some unifying operations are applied to the
    # string: Eliminate whitespaces and commas, turn everything into
    # lowercase.
    string = string.replace(' ','')
    string = string.replace(',','')
    string = string.lower()
    hidden_nodes = []
    hidden_layertype = []
    hidden_activation = []
    hidden_output = []
    while string:
        # Check if the first symbol is a letter.  This is required
        # to specify the layer type, so an error is raised if none
        # is found.
        if not string[0].isalpha():
            raise ValueError("Missing layer specifier inside the "
                             "shorthand notation. Please stick to the "
                             "documented syntax!")
        # Get and save the requested layer type.
        # 'correspondence' is freshly prepared with every call of this
        # function, so it stays up-to-date with extensions of the
        # calling layer recipe.
        try:
            hidden_layertype.append(correspondence[string[0]])
            hidden_activation.append(dol[correspondence[string[0]]])
        except KeyError:
            raise ValueError("I do not know the key '%s' you gave in "
                             "the shorthand notation." % string[0])
        # Check if the second symbol is a digit.  This is required
        # to specify the amount of nodes in the respective layer,
        # so an error is raised if none is found.
        if not string[1].isdigit():
            raise ValueError("Missing number of nodes inside the "
                             "shorthand notation. Please stick to the "
                             "documented syntax!")
        # To keep the string's syntax easy and scalable, no
        # requirements concerning the length of the number of nodes are
        # imposed.  That is why we need to find it out first.
        length = 1
        try:
            while string[1+length].isdigit():
                length += 1
            # NEW in nnlib_multiobject: Shorthand notation now
            # supports auxiliary outputs.  This is processed here.
            if string[1+length] == '>':
                hidden_output.append(True)
            else:
                hidden_output.append(False)
        except IndexError:
            hidden_output.append(False)
        # Now that we know the number's length, we can append its
        # value to the hidden_nodes list.
        hidden_nodes.append(int(string[1:1+length]))
        # Last, cut off the piece of the string that has been dealt
        # with in this pass of the loop.
        if hidden_output[-1]:
            string = string[2+length:]
        else:
            string = string[1+length:]

    return hidden_nodes, hidden_layertype, hidden_activation, hidden_output


def _layer_from_string(string, units, activation):
    """This function builds the layers that are supported as string
    descriptors by the recipes.  It errors out if the string argument
    is unknown or not a string at all.  This follows the same idea as
    the string lookups in Keras for activations, losses etc. but allows
    setting units and activations to give enough flexibility for most
    use cases.

    Arguments:
      string: The string that specifies the layer.  This function
        statically implements building certain layers in a way that is
        most useful for nnlib use cases.  If further customization is
        needed, use the standard way of declaring layers directly.
      units: The amount of nodes used in the layer (must be int)
      activation: Keras activation function instance or string
        specifier.
    """
    if string == 'Dense':
        dense = lrs.Dense(units=units, activation=activation)
        layer = lrs.TimeDistributed(layer=dense)
    elif string == 'LSTM':
        layer = lrs.LSTM(units=units,
                         activation=activation,
                         return_sequences=True,
                         implementation=1)
    elif string == 'RNN':
        layer = lrs.SimpleRNN(units=units,
                              activation=activation,
                              return_sequences=True)
    else:
        raise NNLibUsageError("The layer type '{}' is not supported "
                              "here.".format(string))
    return layer


def _check_layer_specs(nodes, layer_type, activation, output=None):
    """This function checks the nodes, layer types, activations and
    optionally outputs for cross-compatibility or, if another set of
    conditions is satisfied, enforces it.
    The length of the 'nodes' argument implicitly specifies the target
    length for all fields.  It is turned into a singleton or empty
    tuple if it is not iterable.
    'layer_type' and 'activation' may be iterables of the same length
    as nodes, or strings.  If they are strings, they are repeated to
    create a tuple of fitting length.
    'output' may be None (meaning the functionality is not invoked),
    a boolean, or an iterable of fitting length.  If it is a boolean,
    it is repeated to create a tuple of fitting length.  Note that
    'output' is not returned if set to None.
    All cases not described here are probably illegal.  Some of them
    will raise an error here, but others might pass this check and
    yield errors later.  Specifically, if the fields are iterables of
    fitting lengths, their content is generally not checked any further
    and it is left to the caller to error out where appropriate.
    """
    # Check if nodes is iterable.  If it is not, check whether it is
    # None (or zero or False), which means no hidden layers.
    # Otherwise, assume it is a normal integer and turn it into a
    # singleton tuple, such that the caller can iterate on it.
    if not isinstance(nodes, Iterable):
        if not nodes:
            nodes = ()
        else:
            nodes = (nodes,)
    # Check the 'layer_type' argument: Function accepts a list of the
    # same length as 'hidden_nodes' or a single string which is then
    # extended into a tuple of fitting length.  Non-fitting lengths are
    # caught here because they could affect the network topology
    # without notice.
    if isinstance(layer_type, six.string_types):
        layer_type = (layer_type,) * len(nodes)
    try:
        if len(layer_type) != len(nodes):
            raise IndexError("Length of 'hidden_nodes' does not match the "
                             "length of 'hidden_activation'.")
    except TypeError:
        raise TypeError("'hidden_layertype' must be a string or a list "
                        "of strings and/or Keras-compatible neural "
                        "network layers.")
    # Check the 'hidden_activation' argument: Function accepts a list
    # of the same length as 'hidden_nodes' or a single string which
    # is then extended into a list of fitting length.
    if isinstance(activation, six.string_types):
        activation = (activation,) * len(nodes)
    try:
        if len(activation) != len(nodes):
            raise IndexError("Length of 'hidden_nodes' does not match the "
                             "length of 'hidden_activation'.")
    except TypeError:
        raise TypeError("'hidden_activation' must be a string or a list of "
                        "strings and/or Keras activation functions.")
    # Check the 'output' argument (if present): Function accepts a list
    # of the same length as 'nodes' or a single boolean (or
    # bool-convertible) value that is then extended into a list of
    # fitting length.
    if output is None:
        return nodes, layer_type, activation
    else:
        if isinstance(output, bool):
            output = (output,) * len(nodes)
        try:
            if len(output) != len(nodes):
                raise IndexError("Length of 'hidden_nodes' does not match the "
                                 "length of 'hidden_output'.")
        except TypeError:
            raise TypeError("'hidden_output' must be a boolean or a list of "
                            "booleans.")
        return nodes, layer_type, activation, output


def prepare_qmask_net(field_size,
                      field_depth,
                      mask_samples,
                      mask_size,
                      dropout,
                      quantization_levels,
                      weight_range):
    """Prepare the mask generation net and save it in the handler.
    CHANGED: convlayer is also saved seperately to simplify
    accessing the masks after training.
    """
    # CHANGED in nnlib_v14a: Scale the mask's intensity outputs
    # such that they are approximately zero-centered to help
    # convergence of the interpreter part.
    def _scale_intensities(args, mask_size):
        return args / (mask_size**2)
    # Define input layer
    # CHANGED in nnlib_multiobject_v06: With the introduction of 3D
    # sampling, there is another axis in the field data to the right of
    # the timeseries and to the left of the field dimensions.  It is of
    # size 'field_depth'.  For the pooling and convolution, we treat it
    # as yet another time distributed axis and reshape the whole output
    # afterwards so that the filter results from all 3D samples are
    # flattened into one dimension.  This yields minimal changes for
    # the rest of the setup and implicitly ensures direct backwards
    # compatibility.
    image_input = lrs.Input(shape=(None,field_depth,field_size,field_size,1))
    input_config = ('fields_orig',)
    # Define pooling layer
    pool_factor = int(field_size // mask_size)
    pool = lrs.AveragePooling2D(pool_size=pool_factor)
    tdpool = lrs.TimeDistributed(layer=pool)
    td3dpool = lrs.TimeDistributed(layer=tdpool)(image_input)
    # Define the convolutional layer
    conv = QuantizedConv2D(filters=mask_samples,
                           kernel_size=mask_size,
                           weight_levels=quantization_levels,
                           weight_range=weight_range)
    tdconv = lrs.TimeDistributed(layer=conv)
    td3dconv = lrs.TimeDistributed(layer=tdconv)(td3dpool)
    # Add dropout layer if desired
    if dropout:
        td3dconv = lrs.Dropout(rate=dropout)(td3dconv)
    # CHANGED in nnlib_multiobject_v06: The simple Flatten layer got
    # replaced by a Reshape layer because we want to not only eliminate
    # singleton dimensions anymore, but wrap all four output dimensions
    # into one, independent of the length of each.
    # Flatten the output (6dim -> 3dim)
    flatten = lrs.Reshape((mask_samples*field_depth,))
    tdflatten = lrs.TimeDistributed(layer=flatten)(td3dconv)
    # Scale output
    lam = lrs.Lambda(function=_scale_intensities,
                     output_shape=(mask_samples*field_depth,),
                     arguments={'mask_size': mask_size})
    tdscaled = lrs.TimeDistributed(layer=lam)(tdflatten)
    # Define the model and assign it to the handler's maskmodel
    # attribute.
    model = Model(inputs=image_input, outputs=tdscaled)
    output_config = ('intensities',)
    lkms = load_known_mask_set_generator(conv, mask_size)
    gm = get_masks_generator(conv)
    cmo = calc_mask_orthogonality_generator(conv)
    add_fcns = {'load_known_mask_set': lkms,
                'get_masks': gm,
                'calc_mask_orthogonality': cmo}
    # Returns on all build recipes need to follow a unified structure,
    # namely to provide a list of four elements, in order,
    # representing:
    #  1. The created model itself
    #  2. The input_config of the model (tuple listing the model inputs
    #     in order)
    #  3. The output_config of the model (same as input_config, but for
    #     output)
    #  4. A dictionary of additional functions the recipe intends to
    #     register for the model it has created.  It is the build
    #     recipe's responsibility to make sure these functions are
    #     correctly linked to the created model.
    return (model,input_config,output_config,add_fcns)


def load_known_mask_set_generator(convlayer, mask_size):
    """This function produces another function to insert known sets of
    pixel masks into a quantized mask layer.  This generator function
    will be called by 'prepare_qmask_net'.  It registers the given
    convolutional layer as the target for mask changes.
    """
    # The following is the function that this generator spawns.
    def load_known_mask_set(set_name):
        """Loads a known mask set such as the Hadamard-Walsh
        transformation basis into the maskmodel.  This is to compare
        the performance of the self-tuning maskmodel.
        """

        # --Private functions-- (to generate the mask sets themselves)

        def _hadamard():
            """The construction of the hadamard mask set is based upon
            the definition via recursive Kronecker product of the
            Hadamard matrix with a matrix A with starting value of 1.
            """
            cycles = log2(mask_size)
            if not cycles.is_integer():
                raise ValueError("Hadamard matrices are only defined for "
                                 "mask sizes of 2^n.")
            else:
                cycles = int(cycles)
            a = 1
            hada = np.array(((1,1),(1,-1)))
            for i in range(cycles):
                a = np.kron(a, hada)
            b = np.empty((mask_size,mask_size,1,mask_size**2))
            i = 0
            for row in a:
                for col in a:
                    b[...,0,i] = np.outer(row, col)
                    i += 1
            return b

        def _stripes():
            """This constructor method returns simple vertical and
            horizontal stripes as masks.
            """
            b = np.zeros((mask_size,mask_size,1,2*mask_size))
            for i in range(mask_size):
                b[i,:,0,i] = 1.0
            for i in range(mask_size):
                b[:,i,0,i+mask_size] = 1.0
            return b

        # --Function body--

        # Switch based on the given set type. Currently, the basis of
        # the Hadamard-Walsh transformation is the only one supported
        # but this may easily be extended in the future.
        if set_name == 'hadamard':
            base_matrices = _hadamard()
        elif set_name == 'stripes':
            base_matrices = _stripes()
        else:
            # Raise an error if an unknown type key appears
            raise ValueError("I do not know the key you gave for 'type'.")

        # Get total mask samples from the layer
        mask_samples = convlayer.filters
        # Warn if the produced mask set is too big or too small for the
        # network
        if base_matrices.shape[-1] > mask_samples:
            print("Warning: Not all masks from your chosen set can be "
                  "assigned to the convolutional stage of the network.")
        if base_matrices.shape[-1] < mask_samples:
            print("Warning: There are not enough unique masks in the set to "
                  "fill all convolution kernels in the network. Some will "
                  "therefore be repeated, providing no additional info for "
                  "the interpretation.")
            # To keep the algorithm relatively simple, just attach the
            # set of matrices to itself repeatedly until it is
            # guaranteed to be long enough
            while base_matrices.shape[-1] < mask_samples:
                base_matrices = np.concatenate((base_matrices,base_matrices),
                                               axis=-1)

        # Finally, assign the mask set
        convlayer.set_weights([base_matrices[...,:mask_samples]])
    return load_known_mask_set


def get_masks_generator(convlayer):
    """This function produces another function that allows the user to
    easily view the masks via the interface provided by the container.
    """
    def get_masks(quantize=True):
        """Plots the masks trained in the maskmodel using multishow."""
        # Extract masks from convolutional layer
        masks = convlayer.get_weights()[0]
        # Perform the quantization if requested (necessary because
        # convlayer saves the real value, not the quantized one)
        if quantize:
            masks = masks * ((convlayer.Q-1.0) / (convlayer.hi-convlayer.lo))
            masks = masks.round()
            masks = masks * ((convlayer.hi-convlayer.lo) / (convlayer.Q-1.0))
        # Move the last axis around so the array becomes compatible
        # with multishow
        masks = np.moveaxis(masks,-1,0)
        return masks
    return get_masks


def calc_mask_orthogonality_generator(convlayer):
    """This function produces another function that generates a metric
    for the orthogonality within the current set of masks.
    """
    def calc_mask_orthogonality(quantize=True):
        # Extract masks from convolutional layer
        masks = convlayer.get_weights()[0]
        # Perform the quantization if requested (necessary because
        # convlayer saves the real value, not the quantized one)
        if quantize:
            masks = masks * ((convlayer.Q-1.0) / (convlayer.hi-convlayer.lo))
            masks = masks.round()
            masks = masks * ((convlayer.hi-convlayer.lo) / (convlayer.Q-1.0))
        comb_sum = 0
        max_corr = 0
        combs = 0
        for ind1 in range(masks.shape[3]):
            for ind2 in range(ind1+1, masks.shape[3]):
                curr_sum = np.sum(np.multiply(masks[...,ind1],
                                              masks[...,ind2]))
                if abs(curr_sum) > max_corr:
                    max_corr = abs(curr_sum)
                comb_sum += curr_sum
                combs += 1
        print("Mean scalar product: ", comb_sum / combs)
        print("Maximum scalar product: ", max_corr)
        return (comb_sum/combs,max_corr)
    return calc_mask_orthogonality


def prepare_counter_net(mask_samples,
                        shorthand_hidden_layers,
                        hidden_nodes,
                        hidden_layertype,
                        hidden_activation,
                        output_layertype,
                        output_activation,
                        dropout,
                        batch_norm):
    """Prepare the counting net.  It interfaces with the quantized mask
    layer and has a single neuron in the output, specifying the amount
    of objects in the scene.
    Node count, layer type and activations for the hidden layers can be
    given as lists, being interpreted as the layers from left to right
    in the resulting network topology.  As an alternative, there exists
    a string-based shorthand notation that can be given as the
    hidden_nodes argument containing node count and layer type.
    See the definition of _decode_shorthand for further description.
    """
    # New in nnlib_v14a: A shorthand notation for hidden layers is
    # now available to enable more compact calls.  This shorthand
    # notation can be used by passing a single string as the
    # 'shorthand_hidden_layers'-argument.  If such a case is
    # encountered, the information contained will override the classic
    # hidden_nodes, hidden_layertype and hidden_activation argument.
    if shorthand_hidden_layers:
        hidden_nodes, hidden_layertype, hidden_activation, _ = (
          _decode_shorthand(string=shorthand_hidden_layers,
                            dol={'Dense': 'relu',
                                 'RNN': 'relu',
                                 'LSTM': 'tanh'}))
    # For backwards compatibility reasons and to continue offering
    # a consistent interface to use custom activation functions,
    # we branch into the old code if no shorthand notation is
    # detected.
    # CHANGED in build_recipes v1: Since the following code path is
    # shared between multiple build recipes, the checking is now
    # performed by a separate function in order to reuse the code.
    else:
        hidden_nodes, hidden_layertype, hidden_activation = (
          _check_layer_specs(hidden_nodes,
                             hidden_layertype,
                             hidden_activation))

    # Define the input.
    count_input = lrs.Input(shape=(None,mask_samples))
    input_config = ('intensities',)

    # Attach hidden layers as specified.
    hid_tensor = count_input
    for t, u, a in zip(hidden_layertype, hidden_nodes, hidden_activation):
        if isinstance(t, six.string_types):
            hid_layer = _layer_from_string(t, u, a)
        else:
            hid_layer = t
        try:
            hid_tensor = hid_layer(hid_tensor)
        except TypeError:
            raise NNLibUsageError("This build recipe only accepts layer "
                                  "types given by their string aliases or an "
                                  "already built, Keras compatible, neural "
                                  "network layer.")
        if dropout:
            hid_tensor = lrs.Dropout(rate=dropout)(hid_tensor)
        if batch_norm:
            hid_tensor = lrs.BatchNormalization()(hid_tensor)

    if isinstance(output_layertype, six.string_types):
        out_layer = _layer_from_string(output_layertype, 1, output_activation)
    else:
        out_layer = output_layertype
    try:
        out_tensor = out_layer(hid_tensor)
    except TypeError:
        raise NNLibUsageError("This build recipe only accepts layer types "
                              "given by their string aliases or an already "
                              "built, Keras compatible neural network "
                              "layer.")
    output_config = ('objcount',)

    # Build the model over the layers.
    model = Model(inputs=count_input, outputs=out_tensor)

    # Returns on all build recipes need to follow a unified structure,
    # namely to provide a list of four elements, in order,
    # representing:
    #  1. The created model itself
    #  2. The input_config of the model (tuple listing the model inputs
    #     in order)
    #  3. The output_config of the model (same as input_config, but for
    #     output)
    #  4. A dictionary of additional functions the recipe intends to
    #     register for the model it has created.  It is the build
    #     recipe's responsibility to make sure these functions are
    #     correctly linked to the created model.
    return model, input_config, output_config, {}


def prepare_interpreter_net(mask_samples,
                            shorthand_hidden_layers,
                            hidden_nodes,
                            hidden_layertype,
                            hidden_activation,
                            hidden_output,
                            output_layertype,
                            output_activation,
                            label_layout,
                            objtype_present,
                            counter_present,
                            dropout,
                            batch_norm):
    """Prepare the interpreter net.
    Node count, layer type and activations can be given as lists,
    being interpreted as the layers from left to right (top to bottom)
    in the resulting network topology.
    NEW in build_recipes v1: The layertype arguments may now also
    contain ready-built layers instead of string keys.  However, there
    are limitations:  First, they cannot be saved to disc.  Second,
    they will not be expanded to multiple layers; therefore, they
    must be given as list elements.
    If this new option is taken, the corresponding entries for nodes
    and activations are ignored.
    They still need to be present to satisfy the equal length
    condition, though.
    """

    # New in nnlib_v14a: A shorthand notation for hidden layers is
    # now available to enable more compact calls.  This shorthand
    # notation can be used by passing a single string as the
    # 'hidden_layers'-argument.  If such a case is encountered, the
    # information contained will override the classic hidden_nodes,
    # hidden_layertype and hidden_output argument.
    if shorthand_hidden_layers:
        hidden_nodes, hidden_layertype, hidden_activation, hidden_output = (
          _decode_shorthand(string=shorthand_hidden_layers,
                            dol={'Dense': 'relu',
                                 'RNN': 'relu',
                                 'LSTM': 'tanh'})
        )
    # For backwards compatibility reasons and to continue offering a
    # consistent interface to use custom activation and layer
    # functions, we branch into the old code if no shorthand notation
    # is given.
    else:
        # CHANGED in nnlib_multiobject_v04: _check_layer_specs does the
        # checking and trivial transformation stuff for all recipes
        # accepting the classic (nodes,layertype,activation,output)
        # quadruple.
        hidden_nodes, hidden_layertype, hidden_activation, hidden_output = (
          _check_layer_specs(hidden_nodes,
                             hidden_layertype,
                             hidden_activation,
                             hidden_output)
        )

    # Define the inputs. Intensities input is obligatory.
    mask = lrs.Input(shape=(None,mask_samples))
    input_obj_list = [mask]
    input_config = ['intensities']
    # Objtype input (onehot encoded) is optional and can be provided
    # either directly from the data set or through a neural network.
    if objtype_present:
        objtype_supp = lrs.Input(shape=(None,len(std_shapes_all)))
        input_obj_list.append(objtype_supp)
        input_config.append('objtype_supp')
    # Counter input (as a float) is optional and can be provided either
    # directly from the data set of through a neural network.
    if counter_present:
        count = lrs.Input(shape=(None,1))
        input_obj_list.append(count)
        input_config.append('objcount')
    # To keep it consistent, turn input_config into a tuple.
    input_config = tuple(input_config)

    # If there is more than one input, concatenate them.
    if len(input_obj_list) > 1:
        hid_tensor = lrs.Concatenate()(input_obj_list)
    else:
        hid_tensor = input_obj_list[0]

    output_list = []
    # Attach hidden layers as specified.
    for t, u, a, o in zip(hidden_layertype, hidden_nodes,
                          hidden_activation, hidden_output):
        # All layer specifications herein follow the same rule: If
        # specified by a string, _layer_from_string must know how to
        # deal with it and create the layer.  If it is not a string,
        # we don't worry and assume it is a ready-built layer.  If this
        # turns out not to be the case, some error will be raised when
        # trying to connect.
        if isinstance(t, six.string_types):
            # If t is a string, _layer_to_string will build the neural
            # network layer it specifies.
            hid_layer = _layer_from_string(t, u, a)
        else:
            # Else, we assume t is itself a ready-built neural network
            # layer.  If this turns out not to be the case, we will get
            # an error when it gets connected.
            hid_layer = t
        try:
            # Connect the old hidden tensor with the current layer to
            # get a new hidden tensor.
            hid_tensor = hid_layer(hid_tensor)
        except TypeError:
            raise NNLibUsageError("This build recipe only accepts layer "
                                  "types given by their string aliases "
                                  "or an already built, Keras "
                                  "compatible, neural network layer.")
        if o:
            # This recipe supports hidden output layers.  If o is True
            # (or something to that effect), we attach the new hidden
            # tensor to the output list.
            output_list.append(hid_tensor)
        if dropout:
            # The redesign of the network creation in
            # nnlib_multiobject_v04 somewhat de-automatizes the use of
            # supporting layers, which is why e check dropout at every
            # for-loop and attach a fitting dropout layer if requested.
            # While this is a less elegant solution compared to the
            # builder simply knowing to attach supporting layers after
            # them having been specified once, it reduces code overhead
            # and should make the process more maintainable overall.
            hid_tensor = lrs.Dropout(rate=dropout)(hid_tensor)
        if batch_norm:
            # The same considerations go for batch_norm.
            hid_tensor = lrs.BatchNormalization()(hid_tensor)

    if any(output_list):
        print("Warning: Since you enabled additional outputs for this "
              "network, you need to make sure that every output is "
              "connected to something when training, either by directly "
              "giving multiple fitting label sets or by connecting "
              "subsequent networks.")

    # The output layer follows the same rules as the hidden layer(s).
    if isinstance(output_layertype, six.string_types):
        out_layer = _layer_from_string(output_layertype, len(label_layout),
                                       output_activation)
    else:
        out_layer = output_layertype
    try:
        out_tensor = out_layer(hid_tensor)
        output_list.append(out_tensor)
    except TypeError:
        raise NNLibUsageError("This build recipe only accepts layer types "
                              "given by their string aliases or an already "
                              "built, Keras compatible, neural network "
                              "layer.")

    # Build the model over the layers.
    model = Model(inputs=input_obj_list, outputs=output_list)

    # CHANGED in nnlib_multiobject_v06: The label layout is passed down
    # to the recipe, so we assume that it is sensible and build it into
    # our output spec.  No further logic necessary.
    label_specifier = 'labels_' + label_layout

    # Build the full output_config tuple to be returned.
    output_config = ('hidden',)*(len(output_list)-1) + (label_specifier,)

    # Returns on all build recipes need to follow a unified structure,
    # namely to provide a list of four elements, in order,
    # representing:
    #  1. The created model itself
    #  2. The input_config of the model (tuple listing the model inputs
    #     in order)
    #  3. The output_config of the model (same as input_config, but for
    #     output)
    #  4. A dictionary of additional functions the recipe intends to
    #     register for the model it has created.  It is the build
    #     recipe's responsibility to make sure these functions are
    #     correctly linked to the created model.
    return model, input_config, output_config, {}


def prepare_classifier_net(input_nodes,
                           input_config,
                           shorthand_hidden_layers,
                           hidden_nodes,
                           hidden_layertype,
                           hidden_activation,
                           output_layertype,
                           output_activation,
                           dropout,
                           batch_norm):
    """Prepare the classifier net and save it in the handler.
    Node count, layer type and activations can be given as lists,
    being interpreted as the layers from left to right in the
    resulting network topology.  As an alternative, there exists a
    string-based shorthand notation that can be given as the
    hidden_nodes argument containing node count and layer type.
    See below for further description.
    """
    # New in nnlib_v14a: A shorthand notation for hidden layers is
    # now available to enable more compact calls.  This shorthand
    # notation can be used by passing a single string as the
    # 'hidden_layers'-argument.  If such a case is encountered, the
    # information contained will override the classic hidden_nodes,
    # hidden_layertype and hidden_output argument.
    if shorthand_hidden_layers:
        hidden_nodes, hidden_layertype, hidden_activation, hidden_output = (
          _decode_shorthand(string=shorthand_hidden_layers,
                            dol={'Dense': 'relu',
                                 'RNN': 'relu',
                                 'LSTM': 'tanh'}))
    # For backwards compatibility reasons and to continue offering
    # a consistent interface to use custom activation functions,
    # we branch into the old code if no shorthand notation is
    # detected.
    else:
        hidden_nodes, hidden_layertype, hidden_activation = (
          _check_layer_specs(hidden_nodes,
                             hidden_layertype,
                             hidden_activation))

    # Define output node count.
    ocount = len(std_shapes_all)

    # Define the inputs.
    in_tensor = lrs.Input(shape=(None,input_nodes))
    if not (input_config == 'hidden' or input_config == 'intensities'):
        print("Warning: 'input_config' is used to configure connections for "
              "this model, so they should be standardized in some way. The "
              "key '" + input_config + "'you gave is unknown, however. This "
              "may lead to problems or unexpected behaviour down the line.")
    input_config = (input_config,)
    hid_tensor = in_tensor
    # Attach hidden layers as specified.
    for t, u, a in zip(hidden_layertype, hidden_nodes, hidden_activation):
        if isinstance(t, six.string_types):
            hid_layer = _layer_from_string(t, u, a)
        else:
            hid_layer = t
        try:
            hid_tensor = hid_layer(hid_tensor)
        except TypeError:
            raise NNLibUsageError("This build recipe only accepts layer "
                                  "types given by their string aliases or a "
                                  "prebuilt, Keras compatible, neural "
                                  "network layer.")
        if dropout:
            hid_tensor = lrs.Dropout(rate=dropout)(hid_tensor)
        if batch_norm:
            hid_tensor = lrs.BatchNormalization()(hid_tensor)

    if isinstance(output_layertype, six.string_types):
        out_layer = _layer_from_string(output_layertype,
                                       ocount,
                                       output_activation)
    else:
        out_layer = output_layertype
    try:
        out_tensor = out_layer(hid_tensor)
    except TypeError:
        raise NNLibUsageError("This build recipe only accepts layer types "
                              "given by their string aliases or a prebuilt, "
                              "Keras compatible, neural network layer.")
    output_config = ('objtype_out',)

    # Build the model over the layers.
    model = Model(inputs=in_tensor, outputs=out_tensor)

    # Returns on all build recipes need to follow a unified structure,
    # namely to provide a list of four elements, in order,
    # representing:
    #  1. The created model itself
    #  2. The input_config of the model (tuple listing the model inputs
    #     in order)
    #  3. The output_config of the model (same as input_config, but for
    #     output)
    #  4. A dictionary of additional functions the recipe intends to
    #     register for the model it has created.  It is the build
    #     recipe's responsibility to make sure these functions are
    #     correctly linked to the created model.
    return model, input_config, output_config, {}


def prepare_encoder_net(field_size=64,
                        residual_blocks=2,
                        no_add=False,
                        use_lstm=False,
                        use_convlstm=None):
    """Prepare the decoder net and save it in the handler.
    This method implements the decoder part from the autoencoder
    model presented by Theis et al. in "Lossy Image Compression with
    Compressiver Autoencoders".  It is based heavily off of the
    pytorch implementation of said network from
    https://github.com/alexandru-dinu/cae/blob/master/src/models/cae_16x8x8_zero_pad_bin.py
    by Alexandru Dinu.

    Arguments:
      field_size: Specifies the size of the image patch that is to be
        reconstructed.  Note that this needs to align with other
        autoencoder parts.
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
      use_convlstm: DEPRECATED.  The old name of use_lstm.
        Overwrites use_lstm if set, but spawns a FutureWarning.
    """
    def _create_encoder_block(itensor):
        """To make this function more compact, we specify the
        building of the generic decoder block once and reuse it as
        often as necessary.
        """
        # The residual encoder blocks consist of two convolutional
        # layers and a leaky ReLU-function between them. Note that
        # this way and by performing zero padding in the
        # convolutional layers, the dimensionality does not get
        # changed anywhere.
        if use_lstm:
            btensor = lrs.ConvLSTM2D(filters=128,
                                     kernel_size=3,
                                     padding='same',
                                     return_sequences=True)(itensor)
            btensor = lrs.LeakyReLU(alpha=0.01)(btensor)
            otensor = lrs.ConvLSTM2D(filters=128,
                                     kernel_size=3,
                                     padding='same',
                                     return_sequences=True)(btensor)
        else:
            conv_1 = lrs.Conv2D(filters=128, kernel_size=3, padding='same')
            btensor = lrs.TimeDistributed(layer=conv_1)(itensor)
            btensor = lrs.LeakyReLU(alpha=0.01)(btensor)
            conv_2 = lrs.Conv2D(filters=128, kernel_size=3, padding='same')
            otensor = lrs.TimeDistributed(layer=conv_2)(btensor)
        # Final step: There is supposed to be a direct data path
        # from the input to the output (sidestepping the
        # convolutional layers). This is implemented by simply
        # adding the input to the last result.
        if not no_add:
            otensor = lrs.Add()([otensor,itensor])
        return otensor

    def _create_downconv_block(itensor, target_channels):
        """To make this function more compact, we specify the
        building of the downconvolving block once and reuse it as often
        as necessary.
        """
        if use_lstm:
            btensor = lrs.ConvLSTM2D(filters=target_channels,
                                     kernel_size=5,
                                     strides=2,
                                     padding='same',
                                     return_sequences=True)(itensor)
        else:
            # Modified!
            convlayer = lrs.Conv2D(filters=target_channels,
                                   kernel_size=5,
                                   strides=2,
                                   padding='same')
            btensor = lrs.TimeDistributed(layer=convlayer)(itensor)
        otensor = lrs.LeakyReLU(alpha=0.01)(btensor)
        return otensor

    # --Function Body--

    if use_convlstm is not None:
        warnings.warn("'use_convlstm' as an option is deprecated.  It "
                      "has been globally substituted by the more general "
                      "'use_lstm' argument.", FutureWarning)
        use_lstm = use_convlstm
    # Check if this function is actually usable with this handler's
    # setup.
    # This obscure condition is false if and only if field_size is a
    # power of 2.
    if field_size & (field_size-1) != 0:
        raise NNLibUsageError("This algorithm is based off of one that "
                              "features picture patches of 128x128 pixels. "
                              "This is only up- or downscalable by powers of "
                              "2, which does not fit the field size of "
                              + field_size + " that was given.")
    # Define input layer
    input_tensor = lrs.Input(shape=(None,field_size,field_size,1))
    input_config = ('fields_orig',)
    # Current dimensionality: 128x128x1 (channels last)
    # Perform initial downsampling (2 steps)
    current_tensor = _create_downconv_block(input_tensor, target_channels=64)
    # 64x64x64
    current_tensor = _create_downconv_block(current_tensor,
                                            target_channels=128)
    # 32x32x128
    # First standard encoder block
    current_tensor = _create_encoder_block(current_tensor)
    # 32x32x128
    pool_layer = lrs.MaxPooling2D()
    current_tensor = lrs.TimeDistributed(layer=pool_layer)(current_tensor)
    # 16x16x128
    # Since all decoder blocks are structurally identical, we can
    # just attach them to one another in a loop.
    for i in range(residual_blocks):
        current_tensor = _create_encoder_block(current_tensor)
    # 8x8x16
    current_tensor = _create_downconv_block(current_tensor,
                                            target_channels=16)
    # 128x128x1 (target size)
    # Finally, employ tanh activation
    output_tensor = lrs.Activation('tanh')(current_tensor)
    output_config = ('statespace',)

    # Complete definition of model.
    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model, input_config, output_config, {}


def prepare_1d_encoder_net(field_size=64,
                           residual_blocks=3,
                           no_add=False,
                           use_lstm=False,
                           use_convlstm=None):
    """Prepare the decoder net and save it in the handler.
    This method implements the decoder part from the autoencoder model
    presented by Theis et al. in "Lossy Image Compression with
    Compressive Autoencoders".  It is based heavily off of the pytorch
    implementation of said network from
    https://github.com/alexandru-dinu/cae/blob/master/models/cae_16x8x8_zero_pad_bin.py
    by Alexandru Dinu.

    Arguments:
      field_size: Specifies the size of the image patch that is to be
        reconstructed.  Note that this needs to align with other
        autoencoder parts.
      residual_blocks: Specifies how many residual (meaning input shape
        = output shape) blocks are used in the decoder line. There can
        be as many of these as one likes.  Three of them are used in
        the paper.
      no_add: The reference implementation includes bypasses for each
        residual block.  This feature can be disabled by setting this
        option to True.
      use_lstm: As it turns out that the overall object detection
        quality can be improved by taking the continuous nature of the
        objects moving/turning/scaling over time into account, we can
        harness this by using convolutional LSTMs instead of
        non-time-aware convolutional layers.  Note that this is an
        extension on the autoencoder presented in the paper as that one
        is not designed to work on moving images.
      use_convlstm: DEPRECATED.  The old name of use_lstm.
        Overwrites use_lstm if set, but spawns a FutureWarning.
    """
    def _create_encoder_block(itensor):
        """To make this function more compact, we specify the building
        of the generic decoder block once and reuse it as often as
        necessary.
        """
        # The residual decoder blocks consist of two convolutional
        # layers and a leaky ReLU-function between them. Note that this
        # way and by performing zero padding in the convolutional
        # layers, the dimensionality does not get changed anywhere.
        if use_lstm:
            btensor = lrs.ConvLSTM2D(filters=128,
                                     kernel_size=3,
                                     padding='same',
                                     return_sequences=True)(itensor)
            btensor = lrs.LeakyReLU(alpha=0.01)(btensor)
            otensor = lrs.ConvLSTM2D(filters=128,
                                     kernel_size=3,
                                     padding='same',
                                     return_sequences=True)(btensor)
        else:
            conv_1 = lrs.Conv2D(filters=128, kernel_size=3, padding='same')
            btensor = lrs.TimeDistributed(layer=conv_1)(itensor)
            btensor = lrs.LeakyReLU(alpha=0.01)(btensor)
            conv_2 = lrs.Conv2D(filters=128, kernel_size=3, padding='same')
            otensor = lrs.TimeDistributed(layer=conv_2)(btensor)
        # Final step: There is supposed to be a direct data path
        # from the input to the output (sidestepping the
        # convolutional layers). This is implemented by simply
        # adding the input to the last result.
        if not no_add:
            otensor = lrs.Add()([otensor,itensor])
        return otensor

    def _create_downconv_block(itensor, target_channels):
        """To make this function more compact, we specify the building
        of the upsampling block once and reuse it as often as
        necessary.
        """
        if use_lstm:
            btensor = lrs.ConvLSTM2D(filters=target_channels,
                                     kernel_size=5,
                                     strides=2,
                                     padding='same',
                                     return_sequences=True)(itensor)
        else:
            # Modified!
            convlayer = lrs.Conv2D(filters=target_channels,
                                   kernel_size=5,
                                   strides=2,
                                   padding='same')
            btensor = lrs.TimeDistributed(layer=convlayer)(itensor)
        otensor = lrs.LeakyReLU(alpha=0.01)(btensor)
        return otensor

    # --Function Body--

    if use_convlstm is not None:
        warnings.warn("'use_convlstm' as an option is deprecated.  It "
                      "has been globally substituted by the more general "
                      "'use_lstm' argument.", FutureWarning)
        use_lstm = use_convlstm
    # Check if this function is actually usable with this handler's
    # setup.
    # This obscure condition is true if and only if field_size is a
    # power of 2.
    if field_size & (field_size-1) != 0:
        raise NNLibUsageError("This algorithm is based off of one that "
                              "features picture patches of 128x128 pixels. "
                              "This is only up- or downscalable by powers of "
                              "2, which does not fit the field size of "
                              + field_size + " that was given.")
    # Define input layer
    input_tensor = lrs.Input(shape=(None,field_size,field_size,1))
    input_config = ('fields_orig',)
    # Current dimensionality: 128x128x1 (channels last)
    # Perform initial downsampling (2 steps)
    current_tensor = _create_downconv_block(input_tensor, target_channels=64)
    # 64x64x64
    current_tensor = _create_downconv_block(current_tensor,
                                            target_channels=128)
    # 32x32x128
    # First standard encoder block
    current_tensor = _create_encoder_block(current_tensor)
    # 32x32x128
    pool_layer = lrs.MaxPooling2D()
    current_tensor = lrs.TimeDistributed(layer=pool_layer)(current_tensor)
    # 16x16x128
    # Since all decoder blocks are structurally identical, we can just
    # attach them to one another in a loop.
    for i in range(residual_blocks):
        current_tensor = _create_encoder_block(current_tensor)
    # 16x16x128
    current_tensor = _create_downconv_block(current_tensor,
                                            target_channels=192)
    # 8x8x192
    for ds in range(int(log2(field_size))-4):
        current_tensor = _create_downconv_block(current_tensor,
                                                target_channels=256)
    # 1x1x256
    reshape_layer = lrs.Reshape(target_shape=(256,))
    current_tensor = lrs.TimeDistributed(layer=reshape_layer)(current_tensor)
    # 256 [1D]
    # Finally, employ tanh activation
    output_tensor = lrs.Activation('tanh')(current_tensor)
    output_config = ('statespace_1d',)

    # Complete definition of model.
    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model, input_config, output_config, {}


def prepare_adapter_net(field_size=64,
                        mask_samples=24,
                        add_layers=1,
                        use_activations=True,
                        use_lstm=False):
    """Build a little adapter net to enable the interconnection
    between the intensity mask network (prepare_qmask_net) and the
    decoder network (prepare_decoder_net).

    Arguments:
      field_size: Specifies the size of the image patch that is to be
        reconstructed.  Note that this needs to align with other
        autoencoder parts.
      mask_samples: Specifies the amount of intensities the adapter
        expects.  Note that this needs to align with the settings for
        qmask_net.
      add_layers: Specifies the amount of additional dense layers.
        These may be helpful to better align pretrained decoders with
        qmask_net outputs.
      use_activations: Enables or disables the use of tanh as the final
        activation function, analogous to the standard encoder net.  If
        False, no extra activation layer will be added, implicitly
        yielding linear activation.
      use_lstm: As it turns out that the overall object detection
        quality can be improved by taking the continuous nature of the
        objects moving/turning/scaling over time into account, we can
        harness this by using convolutional LSTMs instead of
        non-time-aware convolutional layers.  Note that this is an
        extension on the autoencoder presented in the paper as that one
        is not designed to work on moving images.
    """
    def _create_dense(units, activation, itensor):
        if use_lstm:
            return lrs.LSTM(units=units,
                            return_sequences=True,
                            activation=activation)(itensor)
        else:
            dense_layer = lrs.Dense(units=units, activation=activation)
            return lrs.TimeDistributed(layer=dense_layer)(itensor)

    sc_factor = field_size / 128
    input_tensor = lrs.Input((None,mask_samples))
    input_config = ('intensities',)
    # Since in comparison to the paper by Theis et al. our intensities
    # from the masks are few, we attach a normal Dense layer and a
    # reshape to generate a dataset that is similar to the 16x8x8
    # encoding used by the code sample on github.
    current_tensor = input_tensor
    for i in range(add_layers):
        if not use_activations:
            current_tensor = _create_dense(units=int(sc_factor**2*1024),
                                           activation=None,
                                           itensor=current_tensor)
        elif i < add_layers-1:
            current_tensor = _create_dense(units=int(sc_factor**2*1024),
                                           activation='relu',
                                           itensor=current_tensor)
        else:
            current_tensor = _create_dense(units=int(sc_factor**2*1024),
                                           activation='tanh',
                                           itensor=current_tensor)
    reshape_layer = (
      lrs.Reshape(target_shape=(int(sc_factor*8),int(sc_factor*8),16))
    )
    output_tensor = lrs.TimeDistributed(layer=reshape_layer)(current_tensor)
    output_config = ('statespace',)
    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model, input_config, output_config, {}


def prepare_1d_adapter_net(field_size=64,
                           mask_samples=24,
                           add_layers=1,
                           use_activations=True,
                           use_lstm=False):
    """Build a little adapter net to enable the interconnection between
    the intensity mask network (prepare_qmask_net) and the decoder
    network (prepare_decoder_net).

    Arguments:
      field_size: Specifies the size of the image patch that is to be
        reconstructed.  Note that this needs to align with other
        autoencoder parts.
      mask_samples: Specifies the amount of intensities the adapter
        expects.  Note that this needs to align with the settings for
        qmask_net.
      add_layers: Specifies the amount of additional dense layers.
        These may be helpful to better align pretrained decoders with
        qmask_net outputs.
      use_activations: Enables or disables the use of tanh as the final
        activation function, analogous to the standard encoder net.  If
        False, no extra activation layer will be added, implicitly
        yielding linear activation.
      use_lstm: As it turns out that the overall object detection
        quality can be improved by taking the continuous nature of the
        objects moving/turning/scaling over time into account, we can
        harness this by using convolutional LSTMs instead of
        non-time-aware convolutional layers.  Note that this is an
        extension on the autoencoder presented in the paper as that one
        is not designed to work on moving images.
    """
    def _create_dense(units, activation, itensor):
        if use_lstm:
            return lrs.LSTM(units=units,
                            return_sequences=True,
                            activation=activation)(itensor)
        else:
            dense_layer = lrs.Dense(units=units, activation=activation)
            return lrs.TimeDistributed(layer=dense_layer)(itensor)

    sc_factor = field_size / 128
    input_tensor = lrs.Input((None,mask_samples))
    input_config = ('intensities',)
    # Since in comparison to the paper by Theis et al. our intensities
    # from the masks are few, we attach a normal Dense layer and a
    # reshape to generate a dataset that is similar to the 16x8x8
    # encoding used by the code sample on github.
    current_tensor = input_tensor
    for i in range(add_layers):
        if not use_activations:
            current_tensor = _create_dense(units=int(sc_factor**2*1024),
                                           activation=None,
                                           itensor=current_tensor)
        elif i < add_layers-1:
            current_tensor = _create_dense(units=int(sc_factor**2*1024),
                                           activation='relu',
                                           itensor=current_tensor)
        else:
            current_tensor = _create_dense(units=int(sc_factor**2*1024),
                                           activation='tanh',
                                           itensor=current_tensor)
    dense_layer = lrs.Dense(units=16*int(sc_factor*8)**2)
    output_tensor = lrs.TimeDistributed(layer=dense_layer)(current_tensor)
    output_config = ('statespace_1d',)
    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model, input_config, output_config, {}


def prepare_conv_adapter_net(field_size=64,
                             mask_samples=24,
                             use_activations=True,
                             use_lstm=False,
                             use_convlstm=None):
    """Build a little adapter net to enable the interconnection between
    the intensity mask network (prepare_qmask_net) and the decoder
    network (prepare_decoder_net).

    Arguments:
      field_size: Specifies the size of the image patch that is to be
        reconstructed.  Note that this needs to align with other
        autoencoder parts.
      mask_samples: Specifies the amount of intensities the adapter expects.
        Note that this needs to align with the settings for qconv_net.
      use_activations: Enables or disables the use of tanh as the final
        activation function, analogous to the standard encoder net.  If
        False, no extra activation layer will be added, implicitly
        yielding linear activation.
      use_lstm: As it turns out that the overall object detection
        quality can be improved by taking the continuous nature of the
        objects moving/turning/scaling over time into account, we can
        harness this by using convolutional LSTMs instead of
        non-time-aware convolutional layers.  Note that this is an
        extension on the autoencoder presented in the paper as that one
        is not designed to work on moving images.
      use_convlstm: DEPRECATED.  The old name of use_lstm.
        Overwrites use_lstm if set, but spawns a FutureWarning.
    """
    def _create_upconv_block(itensor, target_channels):
        """To make this function more compact, we specify the building
        of the upsampling block once and reuse it as often as
        necessary.
        """
        if use_lstm:
            btensor = lrs.ConvLSTM2D(filters=target_channels[0],
                                     kernel_size=3,
                                     padding='same',
                                     return_sequences=True)(itensor)
        else:
            conv_layer = lrs.Conv2D(filters=target_channels[0],
                                    kernel_size=3,
                                    padding='same')
            btensor = lrs.TimeDistributed(layer=conv_layer)(itensor)
        btensor = lrs.LeakyReLU(alpha=0.01)(btensor)
        inv_conv_layer = lrs.Conv2DTranspose(filters=target_channels[1],
                                             kernel_size=2,
                                             strides=2)
        otensor = lrs.TimeDistributed(layer=inv_conv_layer)(btensor)
        return otensor

    # --Function body--

    if use_convlstm is not None:
        warnings.warn("'use_convlstm' as an option is deprecated.  It "
                      "has been globally substituted by the more general "
                      "'use_lstm' argument.", FutureWarning)
        use_lstm = use_convlstm

    sc_factor = field_size / 128
    input_tensor = lrs.Input((None,mask_samples))
    input_config = ('intensities',)
    # Since in comparison to the paper by Theis et al. our intensities
    # from the masks are few, we attach a normal Dense layer and a
    # reshape to generate a dataset that is similar to the 16x8x8
    # encoding used by the code sample on github.
    reshape_layer = lrs.Reshape(target_shape=(1,1,mask_samples))
    current_tensor = lrs.TimeDistributed(layer=reshape_layer)(input_tensor)
    current_tensor = _create_upconv_block(itensor=current_tensor,
                                          target_channels=(mask_samples,32))
    current_tensor = _create_upconv_block(itensor=current_tensor,
                                          target_channels=(32,16))
    while current_tensor.shape[1] < sc_factor*8:
        current_tensor = _create_upconv_block(itensor=current_tensor,
                                              target_channels=(16,16))
    if use_activations:
        current_tensor = lrs.Activation('tanh')(current_tensor)
    output_config = ('statespace',)
    model = Model(inputs=input_tensor, outputs=current_tensor)

    return model, input_config, output_config, {}


def prepare_decoder_net(field_size=64,
                        residual_blocks=2,
                        no_add=False,
                        use_lstm=False,
                        dist_free=False,
                        use_convlstm=None):
    """Prepare the decoder net and save it in the handler.  This method
    implements the decoder part from the autoencoder model presented by
    Theis et al. in "Lossy Image Compression with Compressive
    Autoencoders".  It is based heavily off of the pytorch
    implementation of said network from
    https://github.com/alexandru-dinu/cae/blob/master/models/cae_16x8x8_zero_pad_bin.py
    by Alexandru Dinu.
    Arguments:
      field_size: Specifies the size of the image patch that is to be
        reconstructed.  Note that this needs to align with other
        autoencoder parts.
      residual_blocks: Specifies how many residual (meaning input_shape
        = output_shape) convolutional blocks are used in the decoder
        line.  There can theoretially be as many as one likes of these.
        Three of them are used in the paper.
      no_add: The reference implementation includes bypasses for each
        residual block.  This feature can be disabled by setting this
        option to True.
      use_lstm: As it turns out that the overall object detection
        quality can be improved by taking the continuous nature of the
        objects moving/turning/scaling over time into account, we can
        harness this by using convolutional LSTMs instead of
        non-time-aware convolutional layers.  Note that this is an
        extension on the autoencoder presented in the paper as that one
        is not designed to work on moving images.
      dist_free: Determines whether the decoder is supposed to generate
        disturbance free fields as output.  This does not influence the
        topology, only the output_config, but nevertheless it makes
        sense to specify this here.
      use_convlstm: DEPRECATED.  The old name of use_lstm.
        Overwrites use_lstm if set, but spawns a FutureWarning.
    """
    def _create_decoder_block(itensor):
        """To make this function more compact, we specify the building
        of the generic decoder block once and reuse it as often as
        necessary.
        """
        # The residual decoder blocks consist of two convolutional
        # layers and a leaky ReLU-function between them. Note that this
        # way and by performing zero padding in the convolutional
        # layers, the dimensionality does not get changed anywhere.
        if use_lstm:
            btensor = lrs.ConvLSTM2D(filters=128,
                                     kernel_size=3,
                                     padding='same',
                                     return_sequences=True)(itensor)
            btensor = lrs.LeakyReLU(alpha=0.01)(btensor)
            otensor = lrs.ConvLSTM2D(filters=128,
                                     kernel_size=3,
                                     padding='same',
                                     return_sequences=True)(btensor)
        else:
            conv_1 = lrs.Conv2D(filters=128, kernel_size=3, padding='same')
            btensor = lrs.TimeDistributed(layer=conv_1)(itensor)
            btensor = lrs.LeakyReLU(alpha=0.01)(btensor)
            conv_2 = lrs.Conv2D(filters=128, kernel_size=3, padding='same')
            otensor = lrs.TimeDistributed(layer=conv_2)(btensor)
        # Final step: There is supposed to be a direct data path
        # from the input to the output (sidestepping the
        # convolutional layers). This is implemented by simply
        # adding the input to the last result.
        if not no_add:
            otensor = lrs.Add()([otensor,itensor])
        return otensor

    def _create_upconv_block(itensor, target_channels):
        """To make this function more compact, we specify the building
        of the upsampling block once and reuse it as often as
        necessary.
        """
        if use_lstm:
            btensor = lrs.ConvLSTM2D(filters=target_channels[0],
                                     kernel_size=3,
                                     padding='same',
                                     return_sequences=True)(itensor)
        else:
            # Modified!
            conv_layer = lrs.Conv2D(filters=target_channels[0],
                                    kernel_size=3,
                                    padding='same')
            btensor = lrs.TimeDistributed(layer=conv_layer)(itensor)
        btensor = lrs.LeakyReLU(alpha=0.01)(btensor)
        inv_conv_layer = lrs.Conv2DTranspose(filters=target_channels[1],
                                             kernel_size=2,
                                             strides=2)
        otensor = lrs.TimeDistributed(layer=inv_conv_layer)(btensor)
        return otensor

    # --Function Body--

    if use_convlstm is not None:
        warnings.warn("'use_convlstm' as an option is deprecated.  It "
                      "has been globally substituted by the more general "
                      "'use_lstm' argument.", FutureWarning)
        use_lstm = use_convlstm
    # Check if this function is actually usable with this handler's
    # setup.
    # This obscure condition is true if and only if field_size is a
    # power of 2.
    if field_size & (field_size-1) != 0:
        raise NNLibUsageError("This algorithm is based off of one that "
                              "features picture patches of 128x128 pixels. "
                              "This is only up- or downscalable by powers of "
                              "2, which does not fit the field size of "
                              + field_size + " that was given.")
    # Define input layer
    sc_factor = field_size / 128
    input_tensor = lrs.Input(shape=(None,int(sc_factor*8),
                                    int(sc_factor*8),16))
    input_config = ('statespace',)
    # Current dimensionality: 8x8x16 (channels last)
    # Perform initial upsampling
    current_tensor = _create_upconv_block(itensor=input_tensor,
                                          target_channels=(64,128))
    # 16x16x128
    # First standard decoder block
    current_tensor = _create_decoder_block(itensor=current_tensor)
    # 16x16x128
    upsamp_layer = lrs.UpSampling2D()
    current_tensor = lrs.TimeDistributed(layer=upsamp_layer)(current_tensor)
    # 32x32x128
    # Since all decoder blocks are structurally identical, we can just
    # attach them to one another in a loop.
    for i in range(residual_blocks):
        current_tensor = _create_decoder_block(current_tensor)
    # 32x32x128
    current_tensor = _create_upconv_block(itensor=current_tensor,
                                          target_channels=(64,64))
    # 64x64x64 (Modified)
    current_tensor = _create_upconv_block(itensor=current_tensor,
                                          target_channels=(16,1))
    # 128x128x1 (target size)
    # Finally, employ tanh activation
    output_tensor = lrs.Activation('tanh')(current_tensor)
    if dist_free:
        output_config = ('nodistfields_reconst',)
    else:
        output_config = ('fields_reconst',)
    # Create model
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model, input_config, output_config, {}


def prepare_1d_decoder_net(field_size=64,
                           residual_blocks=3,
                           no_add=False,
                           use_lstm=False,
                           dist_free=False,
                           use_convlstm=None):
    """Prepare the decoder net and save it in the handler.  This method
    implements the decoder part from the autoencoder model presented by
    Theis et al. in "Lossy Image Compression with Compressiver
    Autoencoders".  It is based heavily off of the pytorch
    implementation of said network from
    https://github.com/alexandru-dinu/cae/blob/master/models/cae_16x8x8_zero_pad_bin.py
    by Alexandru Dinu.

    Arguments:
      field_size: Specifies the size of the image patch that is to be
        reconstructed.  Note that this needs to align with other
        autoencoder parts.
      residual_blocks: Specifies how many residual (meaning input shape
        = output shape) convolutional blocks are used in the decoder
        line.  There can theoretially be as many as one likes of these.
        Three of them are used in the paper.
      no_add: The reference implementation includes bypasses for each
        residual block.  This feature can be disabled by setting this
        option to True.
      use_lstm: As it turns out that the overall object detection
        quality can be improved by taking the continuous nature of the
        objects moving/turning/scaling over time into account, we can
        harness this by using convolutional LSTMs instead of
        non-time-aware convolutional layers.  Note that this is an
        extension on the autoencoder presented in the paper as that one
        is not designed to work on moving images.
      dist_free: Determines whether the decoder is supposed to generate
        disturbance free fields as output.  This does not influence the
        topology, only the output_config, but nevertheless it makes
        sense to specify this here.
      use_convlstm: DEPRECATED.  The old name of use_lstm.
        Overwrites use_lstm if set, but spawns a FutureWarning.
    """
    def _create_decoder_block(itensor):
        """To make this function more compact, we specify the building
        of the generic decoder block once and reuse it as often as
        necessary.
        """
        # The residual decoder blocks consist of two convolutional
        # layers and a leaky ReLU-function between them. Note that this
        # way and by performing zero padding in the convolutional
        # layers, the dimensionality does not get changed anywhere.
        if use_lstm:
            btensor = lrs.ConvLSTM2D(filters=128,
                                     kernel_size=3,
                                     padding='same',
                                     return_sequences=True)(itensor)
            btensor = lrs.LeakyReLU(alpha=0.01)(btensor)
            otensor = lrs.ConvLSTM2D(filters=128,
                                     kernel_size=3,
                                     padding='same',
                                     return_sequences=True)(btensor)
        else:
            conv_1 = lrs.Conv2D(filters=128, kernel_size=3, padding='same')
            btensor = lrs.TimeDistributed(layer=conv_1)(itensor)
            btensor = lrs.LeakyReLU(alpha=0.01)(btensor)
            conv_2 = lrs.Conv2D(filters=128, kernel_size=3, padding='same')
            otensor = lrs.TimeDistributed(layer=conv_2)(btensor)
        # Final step: There is supposed to be a direct data path from
        # the input to the output (sidestepping the convolutional
        # layers).  This is implemented by simply adding the input to
        # the last result.
        if not no_add:
            otensor = lrs.Add()([otensor,itensor])
        return otensor

    def _create_upconv_block(itensor, target_channels):
        """To make this function more compact, we specify the building
        of the upsampling block once and reuse it as often as
        necessary.
        """
        if use_lstm:
            btensor = lrs.ConvLSTM2D(filters=target_channels[0],
                                     kernel_size=3,
                                     padding='same',
                                     return_sequences=True)(itensor)
        else:
            # Modified!
            conv_layer = lrs.Conv2D(filters=target_channels[0],
                                    kernel_size=3,
                                    padding='same')
            btensor = lrs.TimeDistributed(layer=conv_layer)(itensor)
        btensor = lrs.LeakyReLU(alpha=0.01)(btensor)
        inv_conv_layer = lrs.Conv2DTranspose(filters=target_channels[1],
                                             kernel_size=2,
                                             strides=2)
        otensor = lrs.TimeDistributed(layer=inv_conv_layer)(btensor)
        return otensor

    # --Function Body--

    if use_convlstm is not None:
        warnings.warn("'use_convlstm' as an option is deprecated.  It "
                      "has been globally substituted by the more general "
                      "'use_lstm' argument.", FutureWarning)
        use_lstm = use_convlstm
    # Check if this function is actually usable with this handler's
    # setup.
    # This obscure condition is true if and only if field_size is a
    # power of 2.
    if field_size & (field_size-1) != 0:
        raise NNLibUsageError("This algorithm is based off of one that "
                              "features picture patches of 128x128 pixels. "
                              "This is only up- or downscalable by powers of "
                              "2, which does not fit the field size of "
                              + field_size + " that was given.")

    input_tensor = lrs.Input(shape=(None,256))
    input_config = ('statespace_1d',)
    # Current dimensionality: 256 [1D]
    reshape_layer = lrs.Reshape(target_shape=(1,1,256))
    current_tensor = lrs.TimeDistributed(layer=reshape_layer)(input_tensor)
    # 1x1x256
    for ds in range(int(log2(field_size))-5):
        current_tensor = _create_upconv_block(itensor=current_tensor,
                                              target_channels=(256,256))
    # 4x4x256
    current_tensor = _create_upconv_block(itensor=current_tensor,
                                          target_channels=(256,192))
    # 8x8x192
    current_tensor = _create_upconv_block(itensor=current_tensor,
                                          target_channels=(192,128))
    # 16x16x128
    # First standard decoder block
    current_tensor = _create_decoder_block(current_tensor)
    # 16x16x128
    upsamp_layer = lrs.UpSampling2D()
    current_tensor = lrs.TimeDistributed(layer=upsamp_layer)(current_tensor)
    # 32x32x128
    # Since all decoder blocks are structurally identical, we can
    # just attach them to one another in a loop.
    for i in range(residual_blocks):
        current_tensor = _create_decoder_block(current_tensor)
    # 32x32x128
    current_tensor = _create_upconv_block(itensor=current_tensor,
                                          target_channels=(64,64))
    # 64x64x64 (Modified)
    current_tensor = _create_upconv_block(itensor=current_tensor,
                                          target_channels=(16,1))
    # 128x128x1 (target size)
    # Finally, employ tanh activation
    output_tensor = lrs.Activation('tanh')(current_tensor)
    if dist_free:
        output_config = ('nodistfields_reconst',)
    else:
        output_config = ('fields_reconst',)
    # Create model
    model = Model(inputs=input_tensor, outputs=output_tensor)
    return model, input_config, output_config, {}


# Build dictionary to easily export this submodule's functionality.
build_dict = {
  'quantized_masks': [
    prepare_qmask_net,
    {'field_size': 64,
     'field_depth': 16,
     'mask_samples': 24,
     'mask_size': 8,
     'dropout': False,
     'quantization_levels': 2,
     'weight_range': [0.0,1.0],
     },
  ],
  'interpreter': [
    prepare_interpreter_net,
    {'mask_samples': 24,
     'shorthand_hidden_layers': None,
     'hidden_nodes': (64,256,64,16),
     'hidden_layertype': ('Dense','LSTM','Dense','Dense'),
     'hidden_activation': ('relu','tanh','relu','relu'),
     'hidden_output': False,
     'output_layertype': 'Dense',
     'output_activation': 'linear',
     'label_layout': std_shapes_all,
     'objtype_present': False,
     'counter_present': False,
     'dropout': False,
     'batch_norm': True,
     },
  ],
  'counter': [
    prepare_counter_net,
    {'mask_samples': 24,
     'shorthand_hidden_layers': None,
     'hidden_nodes': (64,16),
     'hidden_layertype': ('Dense','Dense'),
     'hidden_activation': ('relu','relu'),
     'output_layertype': 'Dense',
     'output_activation': 'linear',
     'dropout': 0.3,
     'batch_norm': True,
     },
  ],
  'classifier': [
    prepare_classifier_net,
    {'input_nodes': 24,
     'input_config': 'intensities',
     'shorthand_hidden_layers': None,
     'hidden_nodes': (64,16),
     'hidden_layertype': ('Dense','Dense'),
     'hidden_activation': ('relu','relu'),
     'output_layertype': 'Dense',
     'output_activation': 'softmax',
     'dropout': 0.3,
     'batch_norm': True,
     },
  ],
  'encoder': [
    prepare_encoder_net,
    {'field_size': 64,
     'residual_blocks': 2,
     'no_add': False,
     'use_lstm': False,
     'use_convlstm': None,
     },
  ],
  '1d_encoder': [
    prepare_1d_encoder_net,
    {'field_size': 64,
     'residual_blocks': 3,
     'no_add': False,
     'use_lstm': False,
     'use_convlstm': None,
     },
  ],
  'adapter': [
    prepare_adapter_net,
    {'field_size': 64,
     'mask_samples': 24,
     'add_layers': 1,
     'use_activations': True,
     'use_lstm': False,
     },
  ],
  '1d_adapter': [
    prepare_1d_adapter_net,
    {'field_size': 64,
     'mask_samples': 24,
     'add_layers': 2,
     'use_activations': True,
     'use_lstm': False,
     },
  ],
  'conv_adapter': [
    prepare_conv_adapter_net,
    {'field_size': 64,
     'mask_samples': 24,
     'use_activations': True,
     'use_lstm': False,
     'use_convlstm': None,
     },
  ],
  'decoder': [
    prepare_decoder_net,
    {'field_size': 64,
     'residual_blocks': 2,
     'no_add': False,
     'use_lstm': False,
     'use_convlstm': None,
     },
  ],
  '1d_decoder': [
    prepare_1d_decoder_net,
    {'field_size': 64,
     'residual_blocks': 3,
     'no_add': False,
     'use_lstm': False,
     'use_convlstm': None,
     },
  ],
}
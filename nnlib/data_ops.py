# All functions and classes that involve file interactions.
# Oursourced from nnlib.nnlib with mo_05 to make the files smaller.
__version__ = 'do_01'

import h5py
import numpy as np
from pathlib import Path
from collections.abc import Iterable
from six import string_types

from . import _interactive
from .support_lib import MOD_SETTINGS, string_alts, NNLibUsageError
from .abstract_containers import AbstractContainer, AbstractContainerGroup
from .abstract_containers import is_cont_obj


_func_containers = {}


def register_container(cont):
    _func_containers[cont.__name__] = cont


def instantiate_container(cont):
    """Accepts either the name of a container in string form or a
    container object and builds an empty version of the container."""
    if isinstance(cont, string_types):
        cont_name = cont
    elif is_cont_obj(cont) and cont.__name__ in _func_containers:
        cont_name = cont.__name__
    else:
        raise NNLibUsageError("Can only instantiate based on another "
                              "object or string.")
    try:
        return _func_containers[cont_name]()
    except KeyError:
        raise NNLibUsageError("Unknown container type: %s", cont_name)


# New in nnlib_v13: Unified design for all copy, save and load
# operations that have been introduced up until now, handled by the
# Copier class.

class Copier:
    """Copiers can be instantiated directly or used by the functions
    implementing copy, load and save operations throughout the module.
    This is meant to streamline the process and make it easily
    expandable and flexible.
    """

    def __init__(self, source=None, dest=None, deduplicator=None):
        """Copier supports reading from and writing to
        AbstractContainer objects (source/dest need to be references to
        the requested objects).  Note that this includes
        SerializedObjectHandler instances which provide the interface
        to data saved on disk.
        """
        # A little sanity check (that can be overridden by setting the
        # DEBUG flag): If source and dest are able to tell their type
        # and they are equal, we should be safe to continue.
        # (Reminder: Them being equal does not necessarily mean they
        # are of the same type, but they contain the same kind of data.
        # This distinction is especially pertinent looking at
        # SerializedObjectHandler instances.)  Otherwise we fail (or
        # warn).
        if source is None and dest is None:
            raise NNLibUsageError("Need to attach to at least one object.")
        container_error = False
        if source is None:
            if is_cont_obj(dest):
                print("No source. Only 'set' operation is supported.")
            else:
                container_error = True
        elif dest is None:
            if is_cont_obj(source):
                print("No destination. Only 'get' operation is supported.")
            else:
                container_error = True
        else:
            if is_cont_obj(source) and is_cont_obj(dest):
                if not source.pull('type') == dest.pull('type'):
                    if MOD_SETTINGS['DEBUG']:
                        print("The types do not seem to match.  Continue "
                              "at your own risk.")
                    else:
                        raise TypeError("The types do not match.")
            else:
                container_error = True
            if deduplicator is not None:
                deduplicator.register(source, dest)
        if container_error:
            if MOD_SETTINGS['DEBUG']:
                print("The copier can only function on Container class "
                      "instances, but at least one of source and dest "
                      "does not look like one. Continue at your own "
                      "risk.")
            else:
                raise TypeError("The copier will only work on Container "
                                "class instances.")
        self.source = source
        self.dest = dest
        self.deduplicator = deduplicator

    def copy(self, target, dpcopy=None):
        """Copys whatever is specified by 'target' (see the source's
        'loadstore_keys' dict for valid keys) from source to
        destination.  Note: For simplicity and security reasons, each
        Copier object can only work in one direction.
        Arguments
          target: Specifies what should be transferred from source to
            dest.  Valid 'target' keys are either (iterables of)
            strings defined in the source's 'loadstore_instr' dict or
            'all'.  'all' will transfer all targets that are currently
            enabled, i.e. contained in the source's loadstore_keys
            list.
            CHANGED in do_01: Target can now be an iterable, so more
            refined copy instructions are possible without rewriting
            'loadstore_keys' of 'source'.
          dpcopy: Passing True forces the source to create carbon copys
            of the fields to be copied.  False makes it only pass back
            references where applicable.  When choosing None (default),
            the best guess (encoded in the source's 'loadstore_instr'
            dict) for each target key will be used.
        """
        def _do_copy(key):
            """Actually does the copying and the querying of the
            deduplicator.
            """
            data = self.source.pull(key, dpcopy)
            if self.deduplicator:
                self.deduplicator.check(data)
            self.dest.push(key, data)
        # --Main function block--
        # Check if copying is available at all.
        if self.source is None or self.dest is None:
            raise NNLibUsageError("Copying is not supported in this context.")
        # Choose applicable mode of operation.
        if target in string_alts['all']:
            # Shorthand notation for when everything is to be copied.
            for key in self.source.loadstore_instr.keys():
                _do_copy(key)
        elif (not isinstance(target, string_types)
                and isinstance(target, Iterable)):
            for key in target:
                _do_copy(key)
        else:
            _do_copy(target)

    def get(self, target, dpcopy=None):
        """Gets whatever is specified by 'target' (see the source's
        loadstore_keys dict for valid keys) from source and returns it.
        Arguments
          target: Specifies what should be passed back.  Valid 'target'
            keys are strings defined in the source's 'loadstore_instr'
            dict.
          dpcopy: Passing True forces the source to create carbon copys
            of the fields to be copied.  False makes it only pass back
            references where applicable.  When choosing the default
            None, the best guess (encoded in the source's
            'loadstore_instr' dict) for each target key will be used.
        """
        if self.source is None:
            raise NNLibUsageError("Getting is not supported in this context.")
        return self.source.pull(target, dpcopy)

    def set(self, target, data):
        """Sets the fields specified by 'target' (see the target's
        loadstore_keys dict for valid keys) in the destination to the
        list of data.
        """
        if self.dest is None:
            raise NNLibUsageError("Setting is not supported in this context.")
        self.dest.push(target, data)


class H5PyHandler(AbstractContainerGroup):
    """This class contains everything necessary to interact with a .h5
    file that is used for loading and saving data to disk.
    """

    def __init__(self, path, read_only=False):
        """Accepts the path to a .h5 file.  If read_only is True, the
        file must exist.  Also, no operations that include a write
        operation will work.  This is to guarantee that an existing
        save file cannot be changed accidentally.  If read_only is
        False and the specified file does not exist, it will be
        created.
        """
        self.read_only = read_only
        # H5PyHandler tracks the objects it already rebuilt.
        # This way, we avoid recreating multiple copies of the same
        # object.
        self.autobuilt_objs = {}
        self.tracked_objs = []
        # Make sure path is a pathlib path.
        path = Path(path)
        # Check the given path for plausible name extensions; add one
        # if none or a non-fitting one is present.
        if path.suffix not in {'.h5', '.hdf5'}:
            path = path.with_suffix('.h5')
        if path.exists() and not read_only:
            if _interactive:
                # In an interactive environment, have the user confirm
                # the overwrite.
                print("The given path already exists.")
                resp = input("Overwrite completely? (If not, entries will "
                             "only be selectively replaced) [yes/NO] > ")
            if not _interactive or resp in string_alts['yes']:
                try:
                    path.unlink()
                except PermissionError:
                    raise PermissionError("There was a problem overwriting "
                                          "the existing file. A likely cause "
                                          "is that it was not properly "
                                          "closed by another instance. "
                                          "(Path: %s)" % str(path))
        # Open the file.  These will throw errors if problems of any
        # sort are encountered.  They are not caught as it makes the
        # most sense to fail and present them to the user.
        if read_only:
            self.file = h5py.File(str(path), mode='r')
        else:
            self.file = h5py.File(str(path), mode='a')

    def __del__(self):
        """It is critical to properly close the hdf5 file once the
        Copier using the instance has expired, so we do just that when
        the current object gets garbage collected.
        """
        self.file.close()
        del self.file

    def _write_dispatch(self, supergroup, name, data):
        """There are several data structures that are not natively
        supported in a standard hdf5 file, so this function provides
        the necessary switching functionality to deal with all of them.
        As a general rule, all data is stored in attributes of the
        current group and that is also what is scanned when loading a
        file.  If, however, there is anything more complex to be
        stored, the data will not be assigned directly; instead, a flag
        in the form of a string lead by two '@' symbols is assigned.
        Depending on the specific data type, either a dataset or a
        subgroup of the same name gets written into the file structure
        that then contains the data itself.
        """
        # Check for Python strings.
        if isinstance(data, string_types):
            # These must be encoded into byte arrays before saving.
            supergroup.attrs[name] = data.encode('utf-8')
        # Check for Nonetypes.
        elif data is None:
            supergroup.attrs[name] = '@@None'
        # Check for numpy arrays.
        elif isinstance(data, np.ndarray):
            try:
                # Find arrays with multiple named fields by getting an
                # iterator over them.  This will provoke an error if
                # there is only one field, serving as an implicit case
                # switch for simple datasets.
                names = iter(data.dtype.names)
                supergroup.attrs[name] = '@@multifieldds'
                subgroup = supergroup.create_group(name)
                self._write_dispatch(subgroup,
                                     '@shape',
                                     data.shape)
                for counter, fieldname in enumerate(names):
                    self._write_dispatch(subgroup,
                                         str(counter).zfill(2) + fieldname,
                                         data[fieldname])
            except TypeError:
                # If no iterator could be obtained, we have a simple
                # dataset.
                supergroup.attrs[name] = '@@ds'
                self._write_dataset(supergroup, name, data)
        # Check for dictionaries.
        elif isinstance(data, dict):
            supergroup.attrs[name] = '@@dict'
            subgroup = supergroup.create_group(name)
            # CHANGED in nnlib_multiobject_v04: Not every immutable
            # data type can be used as a key in hdf5 files.  To ensure
            # the fail-safeness and compatibility of the saving
            # process, the keys and values are now indexed (the
            # ordering doesn't matter) and both are saved individually.
            index = 0
            for key, value in data.items():
                # Call this function recursively for every key and
                # value in the dictionary.
                self._write_dispatch(subgroup, 'key_' + str(index), key)
                self._write_dispatch(subgroup, 'value_' + str(index), value)
                index += 1
        # Check for lists and tuples.
        elif isinstance(data, list) or isinstance(data, tuple):
            # Both can be dealt with in the same way as no functions in
            # this module care about the differences between them in
            # their inputs.
            supergroup.attrs[name] = '@@list'
            subgroup = supergroup.create_group(name)
            for i, d in enumerate(data):
                # Call this function recursively for every entry in the
                # list or tuple.  Note that the attributes will be
                # numbered in ascending order to save the exact
                # sequence.
                self._write_dispatch(subgroup, str(i), d)
        # Check for sets.
        elif isinstance(data, set):
            # These are basically handled the same way as lists and
            # tuples. However, they are unordered and not interoperable
            # with lists and tuples in every context (e.g. they support
            # set operations like union), so we deal with them
            # separately.
            supergroup.attrs[name] = '@@set'
            subgroup = supergroup.create_group(name)
            for i, d in enumerate(data):
                # Call this function recursively for every entry in the
                # set.  Note that sets are not ordered, but the items do
                # nevertheless receive a number (i) to simplify loading.
                self._write_dispatch(subgroup, str(i), d)
        # Check for frozen sets.
        elif isinstance(data, frozenset):
            # Frozen sets are mostly the same as sets, but they are
            # immutable which is important in certain contexts.  (For
            # example, frozen sets can be used as dictionary keys.)
            # This is why they get special treatment.
            supergroup.attrs[name] = '@@frozenset'
            subgroup = supergroup.create_group(name)
            for i, d in enumerate(data):
                # Call this function recursively for every entry in the
                # set.  Note that sets are not ordered, but the items do
                # nevertheless receive a number (i) to simplify loading.
                self._write_dispatch(subgroup, str(i), d)
        # Check if the data is an AbstractContainer instance (or looks
        # reasonably similar to one).
        elif is_cont_obj(data):
            # First, pull the identifying info from the container.
            cont_type = data.pull('type')
            cont_id = data.pull('id')
            # Create the magic key that will trigger instantiation of a
            # container object when the data is loaded and save
            # corresponding meta information.
            supergroup.attrs[name] = '@@contobject'
            subgroup = supergroup.create_group(name)
            subgroup.attrs['type'] = cont_type
            subgroup.attrs['id'] = cont_id
            # Test if the object was registered as having been saved
            # before. In that case, we do not want to overwrite.
            if not (cont_type,cont_id) in self.tracked_objs:
                # If it was not saved by this handler instance before,
                # we trigger an autosave.
                # Note that, during the copying process, the data will
                # be registered as a tracked object.  This means that
                # a handler will not do implicit saving twice using the
                # same deduplication mechanism.
                subobj_handler = self.provide_access(cont_type, cont_id)
                subobj_copier = Copier(data, subobj_handler, self)
                subobj_copier.copy('all', dpcopy=False)
        # Check for function handles.
        elif callable(data):
            try:
                # Try out whether or not the function is accessible
                # from the base level of the nnlib module.  If it is
                # not, a NameError will be thrown and caught below and
                # the function will be marked as having failed to be
                # saved.  This is to make the saving and loading
                # process more reliable.
                exec(data.__name__)
                supergroup.attrs[name] = '@@func'
            except NameError:
                print("Function handles whose targets do not live in the "
                      "namespace of the NNLib module cannot be recovered "
                      "correctly as of version " + __version__ + ". The "
                      "function " + data.__name__ + " will therefore be "
                      "skipped.")
                supergroup.attrs[name] = '@@failed_func'
            supergroup[name] = data.__name__
        else:
            # The standard case for primitive data types: Save the data
            # as-is.
            supergroup.attrs[name] = data
            # Note that all errors raised here mean we have an
            # unhandled case, so there is no further catching
            # because the user should see the error.

    def _write_dataset(self, group, name, data):
        """Takes care of writing datasets.  Since H5Py does not like
        Unicode, this function will try to convert the data to fixed
        length binary strings if the assignment fails at first.
        """
        try:
            group.create_dataset(name, data=data, compression='gzip')
        except TypeError:
            group.create_dataset(name, data=data.astype('a12'),
                                 compression='gzip')

    def _read_dispatch(self, supergroup, name):
        """The sister function to _write_dispatch, designed to restore
        data that was saved using that.  Returns the data in the
        original form to the caller.
        """
        try:
            # Fetch the data from the file.
            data_in = supergroup.attrs[name]
        except KeyError:
            print("The key '" + name + "' is not present in the file. The "
                  "reason may be that this file was saved with a previous "
                  "version of the program.")
            if _interactive:
                resp = input("Continue and replace with None? [yes/NO] > ")
                if resp in string_alts['yes']:
                    return None
                else:
                    raise
            else:
                raise
        # For compatibility and stability reasons, it is better to turn
        # the read data back into basic python types if they are numpy
        # ones.  Note that this will not convert datasets because they
        # are handled separately, but only basic data types that get
        # stored directly as attributes in the file.
        if hasattr(data_in, 'dtype'):
            data_in = data_in.item()
        # The following are all the necessary switch cases to support
        # more complex data structures that h5py does not support
        # natively.
        if isinstance(data_in, bytes):
            data_out = data_in.decode('utf-8')
        elif data_in == '@@multifieldds':
            data_out = self._read_multifieldds(supergroup[name])
        elif data_in == '@@ds':
            data_out = self._read_dataset(supergroup[name])
        elif data_in == '@@dict':
            data_out = {}
            subgroup = supergroup[name]
            index = 0
            while ('key_' + str(index)) in subgroup.attrs.keys():
                key = self._read_dispatch(subgroup, 'key_' + str(index))
                value = self._read_dispatch(subgroup, 'value_' + str(index))
                data_out[key] = value
                index += 1
        elif data_in == '@@list':
            data_out = []
            subgroup = supergroup[name]
            for key in range(len(subgroup.attrs)):
                data_out.append(self._read_dispatch(subgroup, str(key)))
        elif data_in in ('@@set','@@frozenset'):
            data_out = set()
            subgroup = supergroup[name]
            for key in range(len(subgroup.attrs)):
                data_out.add(self._read_dispatch(subgroup, str(key)))
            if data_in == '@@frozenset':
                data_out = frozenset(data_out)
        elif data_in == '@@contobject':
            # Container objects are to be understood as internal links
            # within the save file.  They reference another base item
            # in the same file and are identified via a unique ID.  It
            # is common to see multiple objects referencing the same
            # subobjects.  These cross connections are all saved and
            # restored as well, in the sense that no automatically
            # built subobject is reconstructed twice, basically
            # creating multiple copys, but instead providing references
            # when the subobject comes up again.
            cont_type = supergroup[name].attrs['type']
            cont_id = supergroup[name].attrs['id']
            if (type,id) in self.autobuilt_objs:
                # If the type and ID are in autobuilt_subobjs, simply
                # pull the reference and return that.
                data_out = self.autobuilt_objs[(cont_type,cont_id)]
            else:
                # If we land here, the object needs to be built.
                # Instantiate the object...
                data_out = instantiate_container(cont_type)
                # ...and fill it with the saved data by creating a new
                # copier and have it copy everything it finds.
                subobj_handler = self.provide_access(cont_type, cont_id)
                subobj_copier = Copier(subobj_handler, data_out)
                subobj_copier.copy('all')
                self.autobuilt_objs[(cont_type,cont_id)] = data_out
        elif data_in == '@@func':
            # Currently untested and likely broken. TODO: Test and fix!
            data_out = eval(supergroup[name][()])
        elif data_in == '@@failed_func':
            print("A function handle that could not be recovered was found: "
                  + supergroup[name][()])
            print("It will be substituted with a NoneType and we will try to "
                  "run anyway. Expect errors!")
            data_out = None
        elif data_in == '@@None':
            data_out = None
        else:
            data_out = data_in
        return data_out

    def _read_multifieldds(self, group):
        """Reconstructs multi-field datasets like the ones FieldCreator
        objects create for base_data.
        """
        shape = self._read_dispatch(group, '@shape')
        base_ndim = len(shape)
        daughter_ds = []
        daughter_dtypes = []
        for key in group.attrs.keys():
            if group.attrs[key] == '@@ds':
                daughter_ds.append(self._read_dataset(group[key]))
            elif group.attrs[key] == '@@multifieldds':
                daughter_ds.append(self._read_multifieldds(group[key]))
            elif key == '@shape':
                continue
            else:
                raise ValueError("Multifield datasets can only contain "
                                 "elementary datasets or other multifield "
                                 "datasets. The key '" + key + "' that was "
                                 "found points to neither of them.")
            # It turns out that with multi-field datasets, it is
            # helpful to save the order of the internal fields as they
            # are indeed stored as a list internally.  While this is
            # not noticeable in most situations, it does cause errors
            # when concatenating datasets of the same shape, but a
            # different internal ordering.  We conserve the order by
            # putting a number in front of the field name at save time.
            # The ordering imposed this way is automatically followed
            # by the 'for' loop, so here we just have to eliminate the
            # numbers again.
            # The case switch should make sure that we can at least
            # adequately load the old saved data even though the
            # original order is lost.
            daughter_dtypes.append((key[2:],
                                    daughter_ds[-1].dtype,
                                    daughter_ds[-1].shape[base_ndim:]))
        data = np.empty(shape, dtype=daughter_dtypes)
        for field, name in zip(daughter_ds, daughter_dtypes):
            data[name[0]] = field
        return data

    def _read_dataset(self, dataset):
        """Reads datasets.  This is the inverse function of
        _write_dataset.
        """
        data = np.empty(dataset.shape, dtype=dataset.dtype)
        # UPDATE: This read_direct instruction will fail rather
        #   ungracefully if the dataset is of size zero, meaning there
        #   is a zero dimension.  By far the easiest solution in this
        #   case is to simply not try to read anything.  Since the
        #   array created above cannot contain data anyway, there is
        #   nothing further to consider.
        if min(dataset.shape) > 0:
            dataset.read_direct(data)
        # FIX: Data type checking for byte type must be done even if
        # the dataset is empty to avoid errors.
        if np.issubdtype(data.dtype, np.bytes_):
            data = data.astype('U12')
        return data

    # --Public functions--

    def register(self, source, dest):
        if dest.outer is not self:
            raise TypeError("Error using the deduplication engine: "
                            "H5PyHandler instances can only track their own "
                            "contents.")
        self.tracked_objs.append((source.pull('type'),source.pull('id')))

    def check(self, data):
        """Checking for duplications is redundant for H5PyHandler as
        the data dict is traversed when saving, anyway, so it is
        enough to deduplicate then."""
        return data

    def create_instance(self, obj):
        obj_type = obj.pull('type')
        obj_id = obj.pull('id')
        return self.provide_access(obj_type, obj_id)

    def compile_entries_list(self):
        """This method provides a nested dict with informations on the
        h5 file's contents.  It is designed to enable a (user-driven or
        automated) selection of datasets to access.
        """
        type_dict = {}
        for group_name in self.file.keys():
            base_name, base_id = group_name.split('_')
            if base_name not in type_dict.keys():
                type_dict[base_name] = {}
            contained_data = list(self.file[group_name].keys())
            type_dict[base_name][base_id] = contained_data
        return type_dict

    def interactive_provide_access(self, obj_type):
        """A more user friendly variant of 'provide_access', this
        function takes a type and simplifies dealing with the internal
        IDs.  If there is only one object fitting 'type' in the file,
        its ID is completely shadowed.  If there is more than one and
        we are running interactively, a selection is presented to the
        user.  In the same case but not running interactively, the
        first instance on the list is selected.  If there is no fitting
        entry, an error is raised.
        """
        type_dict = self.compile_entries_list()
        if obj_type in type_dict:
            mhandler_dict = type_dict[obj_type]
            if _interactive and len(mhandler_dict) > 1:
                print("Multiple " + obj_type + " instances found.")
                selection = []
                for counter, variant in enumerate(mhandler_dict):
                    selection.append(variant)
                    print("  " + str(counter) + ": ID " + str(variant)
                          + " containing " + str(mhandler_dict[variant]))
                resp = input("Please select [<entry_number>/ABORT] > ")
                try:
                    obj_id = selection[int(resp)]
                except (KeyError,ValueError):
                    print("Aborted.")
                    return
            else:
                obj_id = list(mhandler_dict.keys())[0]
        else:
            raise NNLibUsageError("No " + obj_type + " instance found.")
        return self.provide_access(obj_type, obj_id)

    def provide_access(self, obj_type, obj_id):
        """Provides access to data groups within the file.  Up-to-date
        infos about the file's contents can be gathered by calling
        'compile_entries_list'.
        CHANGED in nnlib_multiobject_v06:  This function got a more
        feature-rich partner in interactive_provide_access that
        automates the selection process and takes only the 'type'
        string directly.
        Arguments
          type: A string that specifies the data group type.  This
            should indicate what its contents are used for or where
            they came from.  Must be provided.
          id: Since we allow multiple data groups of the same type,
            some kind of counting number is always attached to the type
            string.  All AbstractContainer instances offer such an ID
            through their pull function.
        """
        if not (obj_type + '_' + str(obj_id)) in self.file.keys():
            self.file.create_group(obj_type + '_' + str(obj_id))
        return SerializedObjectHandler(self, obj_type, obj_id)


class SerializedObjectHandler(AbstractContainer):
    """CHANGED in nnlib_multiobject_v04: H5PyHandler has become a lot
    dumber in the sense that it does not rely on keys provided from the
    outside for reading and writing to files anymore.  Instead, it (or
    more to the point, the SerializedObjectHandler instances it creates
    for file access) reads and writes whatever you throw at it.  It is
    the partner methods' responsibility to deal with complications
    arising from this.  The idea behind this is that the Hdf5 access
    can be generic while each container and handler that wants to get
    or save data knows for itself how to deal with its contents.
    """

    def __init__(self, outer, cont_type, cont_id):
        """SerializedObjectHandler needs the following arguments at
        init time.
        Arguments:
          outer: The H5PyHandler instance that created this handler
            instance.
          type: The type of object stored, i.e. the name of a nnlib
            container class.
          id: The unique ID of object stored as provided by said
            object.
        """
        self.outer = outer
        self.type = cont_type
        self.id = cont_id
        self.file_access_point = outer.file[cont_type + '_' + str(cont_id)]
        # CHANGED: For compatibility reasons, loadstore_instr needs to
        # be a dict.  SerializedObjectHandler instances don't need
        # instructions, though, so we fill it with Nones.
        self.loadstore_instr = {}
        for key in self.file_access_point.keys():
            self.loadstore_instr[key] = None

    def provide_deduplicator(self):
        return self.outer

    def push(self, target, data):
        """Receives data that it then writes to the file."""
        # Since the file may have been written to before, try to delete
        # the respective group.  If that doesn't work, it seems safe to
        # assume that it was not present in the first place and we can
        # continue.
        try:
            del self.file_access_point.attrs[target]
            del self.file_access_point[target]
        except KeyError:
            pass
        # Dispatch.  Note that this handler relies on the H5PyHandler
        # it is attached to for the actual saving.
        self.outer._write_dispatch(self.file_access_point, target, data)

    def pull(self, target, dpcopy=None):
        """Gets and returns data from the h5 file.
        To keep the pull infrastructure consistent with other Container
        classes, the 'dpcopy' argument is accepted but quietly ignored
        because copying (into RAM) is the only mode of operation that
        makes sense when loading from disk.
        """
        if target == 'id':
            return self.id
        elif target == 'type':
            return self.type
        else:
            # Dispatch.
            return self.outer._read_dispatch(self.file_access_point, target)
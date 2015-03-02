from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class HandlerBase(object):
    """
    Base-class for Handlers to provide the boiler plate to
    make them usable in context managers by provding stubs of
    ``__enter__``, ``__exit__`` and ``close``
    """
    specs = set()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        pass


class HandlerRegistry(dict):
    """
    Sub-class of dict to serve as a registry of available handlers.

    Doing this as a sub-class to give more readable API and to allow
    for more sophisticated validation on the way in later
    """
    def register_handler(self, key, handler, overwrite=False):
        """
        Register a new handler

        Parameters
        ----------
        key : str
            Name of the spec as it will appear in the FS documents

        handler : callable
            This needs to be a callable which when called with the
            free parameters from the FS documents

        overwrite : bool, optional
            If False, raise an exception when re-registering an
            existing key.  Default is False
        """
        if (not overwrite) and (key in self):
            if self[key] is handler:
                return
            raise RuntimeError("You are trying to register a second handler "
                               "for spec {}, {}".format(key, self))

        self[key] = handler

    def deregister_handler(self, key):
        """
        Remove a handler from the registry.  No-op to remove
        a non-existing key.

        Parameters
        ----------
        key : str
            The spec label to remove
        """
        if key in self:
            del self[key]

# singleton module-level registry, not 100% with
# this design choice
_h_registry = HandlerRegistry()


@contextmanager
def handler_context(temp_handlers):
    """
    Context manager for temporarily updating the global HandlerRegistry.
    This is an alternative to passing HandlerRegistry objects in
    as a kwarg.  The global registry is returned to it's prior state
    after the context manager exits.

    Parameters
    ----------
    temp_handlers : dict
        spec_name : HandlerClass pairs.

    Examples
    --------
    To use a different handler for a call to `retrieve` use
    the context manager to add (and possibly over-ride existing
    handlers) temporarily:

       with handler_context({'syn-spec', SynHandler}):
           FS.retrieve(EID)

    """
    remove_list = []
    replace_list = []
    for k, v in six.iteritems(temp_handlers):
        if k not in _h_registry:
            remove_list.append(k)
        else:
            old_h = _h_registry.pop(k)
            replace_list.append((k, old_h))
        _h_registry.register_handler(k, v)

    yield
    for k in remove_list:
        _h_registry.deregister_handler(k)
    for k, v in replace_list:
        _h_registry.register_handler(k, v, overwrite=True)


def register_handler(key, handler, overwrite=False):
    """
    Register a handler to be associated with a specific file
    specification key.  This controls the dispatch to the
    Handler classes based on the `spec` key of the `Resource`
    documents.

    Parameters
    ----------
    key : str
        Name of the spec as it will appear in the FS documents

    handler : callable
        This needs to be a callable which when called with the
        free parameters from the FS documents

    overwrite : bool, optional
        If False, raise an exception when re-registering an
        existing key.  Default is False

    See Also
    --------
    `deregister_handler`

    """
    _h_registry.register_handler(key, handler, overwrite)


def deregister_handler(key):
    """
    Remove handler to module-level handler

    Parameters
    ----------
    key : str
        The spec label to remove

    See Also
    --------
    `register_handler`

    """
    _h_registry.deregister_handler(key)


def get_spec_handler(resource, handle_registry=None):
    """
    Given a document from the base FS collection return
    the proper Handler

    This should get memozied or shoved into a class eventually
    to minimize open/close thrashing.

    Parameters
    ----------
    resource : Resource
        Resource document.

    handle_registry : HandleRegistry or dict, optional
        Mapping between spec <-> handler classes, if None, use
        module-level registry

    Returns
    -------

    handler : callable
        An object that when called with the values in the event
        document returns the externally stored data

    """

    if handle_registry is None:
        handle_registry = _h_registry
    spec = resource.spec
    kwargs = resource.resource_kwargs
    rpath = resource.resource_path
    return handle_registry[spec](rpath, **kwargs)


def get_data(datum, handle_registry=None):
    """
    Given a document from the events collection, get the externally
    stored data.

    This may get wrapped up in a class instance, not intended for public
    usage as-is

    Parameters
    ----------
    datum : Datum
        Document identifying the data resource

    get_handler_method : callable
        A function which takes a fid and returns a handler.  This should
        eventually be optional(?) and default to hitting the mongodb.

    Returns
    -------
    data : ndarray
        The data in ndarray form.
    """

    kwargs = datum.datum_kwargs
    handler = get_spec_handler(datum.resource, handle_registry)
    return handler(**kwargs)

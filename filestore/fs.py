from __future__ import (absolute_import, division, print_function,
                        )

import six
from pkg_resources import resource_filename
from contextlib import contextmanager
import json
import logging

from pymongo import MongoClient
from bson import ObjectId

import boltons.cacheutils

from .handlers_base import DuplicateHandler
from .core import (bulk_insert_datum as _bulk_insert_datum,
                   insert_datum as _insert_datum,
                   insert_resource as _insert_resource,
                   get_datum as _get_datum)

logger = logging.getLogger(__name__)

try:
    from collections import ChainMap as _ChainMap
except ImportError:
    class _ChainMap(object):
        def __init__(self, primary, fallback=None):
            if fallback is None:
                fallback = {}
            self.fallback = fallback
            self.primary = primary

        def __getitem__(self, k):
            try:
                return self.primary[k]
            except KeyError:
                return self.fallback[k]

        def __setitem__(self, k, v):
            self.primary[k] = v

        def __contains__(self, k):
            return k in self.primary or k in self.fallback

        def __delitem__(self, k):
            del self.primary[k]

        def pop(self, k, v):
            return self.primary.pop(k, v)

        @property
        def maps(self):
            return [self.primary, self.fallback]

        @property
        def parents(self):
            return self.fallback

        def new_child(self, m=None):
            if m is None:
                m = {}

            return _ChainMap(m, self)


class FileStoreRO(object):

    KNOWN_SPEC = dict()
    # load the built-in schema
    for spec_name in ['AD_HDF5', 'AD_SPE']:
        tmp_dict = {}
        resource_name = 'json/{}_resource.json'.format(spec_name)
        datum_name = 'json/{}_datum.json'.format(spec_name)
        with open(resource_filename('filestore', resource_name), 'r') as fin:
            tmp_dict['resource'] = json.load(fin)
        with open(resource_filename('filestore', datum_name), 'r') as fin:
            tmp_dict['datum'] = json.load(fin)
        KNOWN_SPEC[spec_name] = tmp_dict

    def __init__(self, config, handler_reg=None):
        self.config = config

        if handler_reg is None:
            handler_reg = {}

        self.handler_reg = _ChainMap(handler_reg)

        self._datum_cache = boltons.cacheutils.LRU(max_size=1000000)
        self._handler_cache = boltons.cacheutils.LRU()
        self._resource_cache = boltons.cacheutils.LRU(on_miss=self._r_on_miss)
        self.__db = None
        self.__conn = None
        self.__datum_col = None
        self.__res_col = None
        self.known_spec = dict(self.KNOWN_SPEC)

    def disconnect(self):
        self.__db = None
        self.__conn = None
        self.__datum_col = None
        self.__res_col = None

    def reconfigure(self, config):
        self.disconnect()
        self.config = config
        print(config)

    def _r_on_miss(self, k):
        col = self._resource_col
        return col.find_one({'_id': k})

    def get_datum(self, eid):
        return _get_datum(self._datum_col, eid,
                          self._datum_cache, self.get_spec_handler,
                          logger)

    def register_handler(self, key, handler, overwrite=False):
        if (not overwrite) and (key in self.handler_reg):
            if self.handler_reg[key] is handler:
                return
            raise DuplicateHandler(
                "You are trying to register a second handler "
                "for spec {}, {}".format(key, self))

        self.deregister_handler(key)
        self.handler_reg[key] = handler

    def deregister_handler(self, key):
        handler = self.handler_reg.pop(key, None)
        if handler is not None:
            name = handler.__name__
            for k in list(self._handler_cache):
                if k[1] == name:
                    del self._handler_cache[k]

    @contextmanager
    def handler_context(self, temp_handlers):
        stash = self.handler_reg
        self.handler_reg = self.handler_reg.new_child(temp_handlers)
        try:
            yield self
        finally:
            popped_reg = self.handler_reg.maps[0]
            self.handler_reg = stash
            for handler in popped_reg.values():
                name = handler.__name__
                for k in list(self._handler_cache):
                    if k[1] == name:
                        del self._handler_cache[k]

    @property
    def _db(self):
        if self.__db is None:
            conn = self._connection
            self.__db = conn.get_database(self.config['database'])
        return self.__db

    @property
    def _resource_col(self):
        if self.__res_col is None:
            self.__res_col = self._db.get_collection('resource')

        return self.__res_col

    @property
    def _datum_col(self):
        if self.__datum_col is None:
            self.__datum_col = self._db.get_collection('datum')
            self.__datum_col.create_index('datum_id', unique=True)
            self.__datum_col.create_index('resource')

        return self.__datum_col

    @property
    def _connection(self):
        if self.__conn is None:
            self.__conn = MongoClient(self.config['host'],
                                      self.config.get('port', None))
        return self.__conn

    def get_spec_handler(self, resource):
        """
        Given a document from the base FS collection return
        the proper Handler

        This should get memozied or shoved into a class eventually
        to minimize open/close thrashing.

        Parameters
        ----------
        resource : ObjectId
            ObjectId of a resource document

        Returns
        -------

        handler : callable
            An object that when called with the values in the event
            document returns the externally stored data

        """
        resource = self._resource_cache[resource]

        h_cache = self._handler_cache

        spec = resource['spec']
        handler = self.handler_reg[spec]
        key = (str(resource['_id']), handler.__name__)

        try:
            return h_cache[key]
        except KeyError:
            pass

        kwargs = resource['resource_kwargs']
        rpath = resource['resource_path']
        ret = handler(rpath, **kwargs)
        h_cache[key] = ret
        return ret


class FileStore(FileStoreRO):
    def insert_resource(self, spec, resource_path, resource_kwargs):
        col = self._resource_col

        return _insert_resource(col, spec, resource_path, resource_kwargs,
                                self.known_spec)

    def insert_datum(self, resource, datum_id, datum_kwargs):
        col = self._datum_col

        try:
            resource['spec']
        except (AttributeError, TypeError):
            res_col = self._resource_col
            resource = res_col.find_one({'_id': ObjectId(resource)})
            resource['id'] = resource['_id']
        if datum_kwargs is None:
            datum_kwargs = {}

        return _insert_datum(col, resource, datum_id, datum_kwargs,
                             self.known_spec)

    def bulk_insert_datum(self, resource, datum_ids, datum_kwarg_list):
        col = self._datum_col
        return _bulk_insert_datum(col, resource, datum_ids, datum_kwarg_list)

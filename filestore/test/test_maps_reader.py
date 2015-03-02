# ######################################################################
# Copyright (c) 2014, Brookhaven Science Associates, Brookhaven        #
# National Laboratory. All rights reserved.                            #
#                                                                      #
# Redistribution and use in source and binary forms, with or without   #
# modification, are permitted provided that the following conditions   #
# are met:                                                             #
#                                                                      #
# * Redistributions of source code must retain the above copyright     #
#   notice, this list of conditions and the following disclaimer.      #
#                                                                      #
# * Redistributions in binary form must reproduce the above copyright  #
#   notice this list of conditions and the following disclaimer in     #
#   the documentation and/or other materials provided with the         #
#   distribution.                                                      #
#                                                                      #
# * Neither the name of the Brookhaven Science Associates, Brookhaven  #
#   National Laboratory nor the names of its contributors may be used  #
#   to endorse or promote products derived from this software without  #
#   specific prior written permission.                                 #
#                                                                      #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS  #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT    #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS    #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE       #
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,           #
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES   #
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR   #
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)   #
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,  #
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OTHERWISE) ARISING   #
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE   #
# POSSIBILITY OF SUCH DAMAGE.                                          #
########################################################################

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import six
import h5py
import os.path as op
import numpy as np
from nose.tools import assert_true, assert_raises, assert_false, assert_equal
import datetime
import uuid

from filestore.api import register_handler, insert_resource, insert_datum, retrieve
import filestore.handlers as fh

import logging
logger = logging.getLogger(__name__)


register_handler('hdf_maps_1D', fh.HDFMapsSpectrumHandler)
register_handler('hdf_maps_2D', fh.HDFMapsEnergyHandler)


def save_syn_data(eid, data, base_path=None):
    """
    Save a array as hdf format to disk.
    Defaults to saving files in :path:`~/.fs_cache/YYYY-MM-DD.h5`

    Parameters
    ----------
    eid : unicode
        id for file name
    data : ndarray
        The data to be saved
    base_path : str, optional
        The base-path to use for saving files.  If not given
        default to `~/.fs_cache`.  Will add a sub-directory for
        each day in this path.
    """

    if base_path is None:
        base_path = op.join(op.expanduser('~'), '.fs_cache',
                            str(datetime.date.today()))
    fpath = op.join(base_path, str(eid) + '.h5')

    with h5py.File(fpath, 'w') as f:
        # create a group for maps to hold the data
        mapsGrp = f.create_group('MAPS')
        # now set a comment
        mapsGrp.attrs['comments'] = 'MAPS group'

        entryname = 'mca_arr'
        comment = 'These are raw spectrum data.'
        ds_data = mapsGrp.create_dataset(entryname, data=data)
        ds_data.attrs['comments'] = comment
    return fpath


def get_1D_data(ind_v, ind_h):
    """
    Get data for given x, y index.

    Parameters
    ----------
    ind_v : int
        vertical index
    ind_h : int
        horizontal index

    Returns
    -------
    unicode:
        id number of event
    """

    uid = str(uuid.uuid1())

    # generate 3D random number with a given shape
    syn_data = np.random.randn(20, 1, 10)
    file_path = save_syn_data(uid, syn_data)

    custom = {'dset_path': 'mca_arr'}
    fb = insert_resource('hdf_maps_1D', file_path, resource_kwargs=custom)
    evl = insert_datum(fb, uid, datum_kwargs={'x': ind_v, 'y': ind_h})
    return evl.datum_id


def get_2D_data(ind):
    """
    Get data for given index of energy.

    Parameters
    ----------
    ind : int
        index for 2D slice

    Returns
    -------
    unicode:
        id number of event
    """

    uid = str(uuid.uuid1())

    # generate 3D random number with a given shape
    syn_data = np.random.randn(20, 1, 10)
    file_path = save_syn_data(uid, syn_data)

    custom = {'dset_path': 'mca_arr'}
    fb = insert_resource('hdf_maps_2D', file_path, resource_kwargs=custom)
    evl = insert_datum(fb, uid, datum_kwargs={'e_index': ind})
    return evl.datum_id


def _test_retrieve_data(evt, num):
    """
    Parameters
    ----------
    evt : unicode
        id number of event
    num : int
        size to be matched
    """
    data = retrieve(evt)
    assert_equal(data.size, num)


def test_data_io():
    """
    Test both 1D and 2D reader from hdf handlers.
    """

    # number of positions to record, basically along a horizontal line
    num = 10
    for i in range(num):
        v_pos = 0
        h_pos = i

        data_id = get_1D_data(v_pos, h_pos)
        yield _test_retrieve_data, data_id, 20

    # total index number, along energy axes
    num = 20
    for i in range(num):
        pos = 0
        data_id = get_2D_data(pos)
        yield _test_retrieve_data, data_id, 10

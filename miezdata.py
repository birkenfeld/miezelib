# -*- coding: utf-8 -*-
"""
    miezelib: miezdata
    ~~~~~~~~~~~~~~~~~~

    Routines for reading and analysing MIEZE data.

    :copyright: 2008-2010 by Georg Brandl.
    :license: BSD.
"""

import re
import sys
import copy
from os import path
from itertools import groupby

import numpy as np

from miezutil import dprint, mieze_time
import miezutil

ALL = object()

# -- global settings -----------------------------------------------------------

_datadir = '.'
_ipars = None
_freefit = False
_altfit = True
_flatback = False

def setdatadir(dir):
    global _datadir
    _datadir = dir

def setipars(lam=None, Ls=None):
    global _ipars
    if lam is None:
        _ipars = None
    else:
        _ipars = (lam, Ls)

def setfreefit(freefit=True):
    global _freefit
    _freefit = freefit

def setaltfit(altfit=True):
    global _altfit
    _altfit = altfit

def setflatback(flatback=True):
    global _flatback
    _flatback = flatback

# -- raw data reading ----------------------------------------------------------

def try_float(f):
    try:
        return float(f)
    except ValueError:
        return f

def read_summary(fn):
    m = []
    c = []
    for line in open(fn):
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            c.append(line)
        else:
            m.append(map(try_float, line.split()))
    return m, c

pm_re = re.compile(r'([0-9e.]+)\s+\+/-\s+([0-9e.]+)')
vv_re = re.compile(r'# ([a-zA-Z0-9_]+)\s+is at\s+([0-9e.]+)')

_single_cache = {}

def read_single(fn):
    if fn == '':
        return None
    if fn in _single_cache:
        return _single_cache[fn]
    varvalues = {}
    setting = preset = monitor = None
    counts, params, errors = [], [0, 0, 0, 0], [0, 0, 0, 0]

    in_vallist = True
    for line in open(fn):
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            if in_vallist and 'is at' in line:
                m = vv_re.match(line)
                if m: varvalues[m.group(1)] = float(m.group(2))
            elif line.startswith('# MIEZE status:'):
                in_vallist = False
            elif line.startswith('# MIEZE setting'):
                setting = line[16:]
            elif line.startswith('# t/s:'):
                preset = int(line[7:])
            elif line.startswith('# mon:'):
                monitor = int(line[7:])
            elif line.startswith('# A:'):
                m = pm_re.match(line[7:])
                params[0] = float(m.group(1))
                errors[0] = float(m.group(2))
            elif line.startswith('# B:'):
                m = pm_re.match(line[7:])
                params[1] = float(m.group(1))
                errors[1] = float(m.group(2))
            elif line.startswith('# phi:'):
                m = pm_re.match(line[7:])
                params[2] = float(m.group(1))
                errors[2] = float(m.group(2))
            elif line.startswith('# C:'):
                m = pm_re.match(line[7:])
                params[3] = float(m.group(1))
                errors[3] = float(m.group(2))
        else:
            counts.append(int(line))
    ret = varvalues, setting, preset, counts, params, errors, monitor
    _single_cache[fn] = ret
    return ret


# -- data objects --------------------------------------------------------------

class MiezeMeasurement(object):
    """Container for a single MIEZE measurement (multiple MIEZE times)."""

    def __init__(self, data, ycol, varvalue):
        self.varvalue = varvalue
        self.data = data
        self.ycol = ycol
        self.fitvalues = None
        self._calc_point = getattr(self, '_calc_point_' + ycol)

        self.points = []
        self.arrays = None

    def __repr__(self):
        return '<MIEZE measurement at %s, ycol %s>' % (self.varvalue, self.ycol)

    def add_point(self, x, point, graph, bkgrd, files):
        y, dy = self._calc_point(point, graph, bkgrd)
        self.points.append((x, y, dy, files, point.get('group')))

    def _calc_point_sum(self, point, graph, bkgrd):
        c, m = point['countsum'], point['monitor']
        y = c/m
        dy = y*(1/np.sqrt(c) + 1/np.sqrt(m))
        if bkgrd:
            cb = bkgrd['countsum']
            mb = bkgrd['monitor']
            y -= cb/mb
            dy += y*(1/np.sqrt(cb) + 1/np.sqrt(mb))
        return y, dy

    def _calc_point_A(self, point, graph, bkgrd):
        # correction factor for time-dependent measurement values
        cf = bkgrd and point['preset']/bkgrd['preset'] or 1
        y = point['A']
        dy = point.get('delta A', 0)
        if bkgrd:
            y -= bkgrd['A'] * cf
            dy += bkgrd['delta A'] * cf
        y /= point['preset']
        dy /= point['preset']
        return y, dy

    def _calc_point_B(self, point, graph, bkgrd):
        # correction factor for time-dependent measurement values
        cf = bkgrd and point['preset']/bkgrd['preset'] or 1
        y = point['B']
        dy = point.get('delta B', 0)
        if bkgrd:
            y -= bkgrd['B'] * cf
            dy += bkgrd.get('delta B', 0) * cf
        y /= point['preset']
        dy /= point['preset']
        return y, dy

    def _calc_point_C(self, point, graph, bkgrd):
        # correction factor for time-dependent measurement values
        cf = bkgrd and point['preset']/bkgrd['preset'] or 1
        # and for graphite values
        cfg = (graph and bkgrd) and graph['preset']/bkgrd['preset'] or 1

        a, b = point['A'], point['B']
        dc = point.get('delta C', 0)
        c = a/b
        if bkgrd:
            #bkgrd['delta A'] = 0
            #bkgrd['delta B'] = 0
            if not _flatback:
               a -= bkgrd['A'] * cf
            b -= bkgrd['B'] * cf
            c = a/b
            #print point['delta A'], bkgrd['delta A']*cf
            if not _flatback:
                da = point['delta A'] + bkgrd['delta A']*cf
            else:
                da = point['delta A']
            db = point['delta B'] + bkgrd['delta B']*cf
            dc = c * (da/a + db/b)
        if graph:
            ga, gb = graph['A'], graph['B']
            gdc = graph['delta C']
            gc = ga/gb
            if bkgrd:
                if not _flatback:
                    ga -= bkgrd['A'] * cfg
                gb -= bkgrd['B'] * cfg
                gc = ga/gb
                if not _flatback:
                    gda = graph['delta A'] + bkgrd['delta A']*cfg
                else:
                    gda = graph['delta A']
                gdb = graph['delta B'] + bkgrd['delta B']*cfg
                gdc = gc * (gda/ga + gdb/gb)
            dc = (c/gc)*(dc/c + gdc/gc)
            c = c/gc
        return c, dc

    def _calc_point_phi(self, point, graph, bkgrd):
        return point['phi'], point['delta phi'] + graph['delta phi']

    @property
    def label(self):
        return '%s %s' % (self.varvalue, self.data.unit)

    def as_arrays(self):
        if self.arrays:
            return self.arrays
        self.arrays = map(np.array, zip(*sorted(self.points)))
        return self.arrays

    def as_arrays_bygroup(self):
        get_group = lambda v: v[4]
        newpoints = sorted(self.points, key=get_group)
        for groupname, grouppoints in groupby(newpoints, get_group):
            gx, gy, gdy, gsf, _ = map(np.array, zip(*grouppoints))
            yield groupname, gx, gy, gdy, gsf

    def _fit_model(self, v, x):
        # to get gamma in mueV, conversion factor is hbar = 658.2 mueV*ps
        return np.exp(-abs(v[0])*x/658.2)

    def fit(self, name=None, **kwds):
        if _freefit:
            return self.freefit(name, **kwds)
        from miezfit import Fit
        name = '%s %s' % (name or '', self.label)
        x, y, dy = self.as_arrays()[:3]
        res = Fit(self._fit_model, ['Gamma'], [0], **kwds).\
              run(name, x, y, dy)
        if not res:
            return None
        res.Gamma = abs(res.Gamma)
        res.dGamma = max(res.dGamma, self.data.resolution)
        if res.Gamma < self.data.resolution:
            res.Gamma = 0
            res.dGamma = self.data.resolution
        self.fitvalues = (res.Gamma, res.dGamma, res.chi2)
        return res

    def _freefit_model(self, v, x):
        return v[1] * np.exp(-abs(v[0])*x/658.2)

    def freefit(self, name=None, **kwds):
        from miezfit import Fit
        name = '%s %s' % (name or '', self.label)
        x, y, dy = self.as_arrays()[:3]
        res = Fit(self._freefit_model, ['Gamma', 'C0'], [0, 1], **kwds).\
              run(name, x, y, dy)
        if not res:
            return None
        res.Gamma = abs(res.Gamma)
        res.dGamma = max(res.dGamma, self.data.resolution)
        if res.Gamma < self.data.resolution:
            res.Gamma = 0
            res.dGamma = self.data.resolution
        self.fitvalues = (res.Gamma, res.dGamma, res.C0, res.dC0, res.chi2)
        return res


class MiezeData(object):
    """Container for a whole range of MIEZE measurements."""

    def __init__(self, name, variable, unit, var_norm=False, var_back=False,
                 ipars=None, resolution=None):
        self.name = name
        self.unit = unit
        self.ipars = ipars or _ipars
        self.variable = variable
        self.var_norm = var_norm
        self.var_back = var_back
        self.resolution = resolution or 0.025  # in mueV
        self._data = {}
        self._norm = {}
        self._back = {}
        dprint('created new data object:', name)

    def __repr__(self):
        return 'MIEZE data ' + self.name

    def copy(self, newname):
        new = copy.copy(self)
        new.name = newname

    def read_file(self, file, dct, variable=True, varvalue=None, vals=None,
                  onlyval=None, ipars=None, group=None):
        if ipars is None:
            ipars = self.ipars
        if group is None and ipars:
            group = ipars[0]
        if isinstance(file, int):
            file = 'mieze_%08d' % file
        file = path.join(_datadir, file)
        if path.isdir(file):
            file = path.join(file, 'summary')
        data, fcomments = read_summary(file)
        for fcomment in fcomments:
            if fcomment.endswith('file'):
                fields = re.split('  +', fcomment[1:].strip())
                break
        else:
            assert False, 'no field names in data file'
        assert len(fields) == len(data[0]), \
               'inconsistent number of data fields in file'
        if variable and varvalue is None:
            assert fields[0] == self.variable, \
                   'different variables in data files'
        nread = 0
        for row in data:
            point = dict(zip(fields, row))
            point['summaryfile'] = file
            point['singlefile'] = path.join(path.dirname(file),
                                            '%05d' % int(point['in file']))
            point = self.process_point(point)
            if 'tau' not in point or point['tau'] == '-':
                assert ipars, \
                       'lam and L_s not given, but tau not in data file'
                point['tau'] = mieze_time(ipars[0]*1e-10, ipars[1]*1e-3,
                                          point['setting'])
            point['group'] = group
            if variable:
                if varvalue is not None:
                    ddct = dct.setdefault(varvalue, {})
                else:
                    vv = point[self.variable+'_set']
                    if vals is not None:
                        if vv not in vals:
                            continue
                    ddct = dct.setdefault(vv, {})
                ddct[point['tau']] = point
                nread += 1
            else:
                if onlyval is not None:
                    if point[fields[1]] != onlyval:
                        continue
                dct[point['tau']] = point
                nread += 1
        if not nread:
            print 'Warning: no points read from file', file

    def process_point(self, point):
        if _altfit:
            from miezfit import mieze_fit
            info = read_single(point['singlefile'])
            if point['setting'] == '200_300':
                p, e = mieze_fit(info[3], addup=True)
                for n in ('A', 'B', 'C', 'phi'):
                    point[n] = p[n]
                    point['delta '+n] = e[n]
        return point

    def read_data(self, file, varvalue=None, vals=None, ipars=None, group=None):
        self.read_file(file, self._data, True, varvalue, vals, None,
                       ipars, group)
        dprint('read file', file, 'as data')

    data = read_data

    def read_norm(self, file, varvalue=None, onlyval=None, ipars=None, group=None):
        self.read_file(file, self._norm, self.var_norm, varvalue, None,
                       onlyval, ipars, group)
        dprint('read file', file, 'as normalization')

    norm = read_norm

    def read_back(self, file, varvalue=None, onlyval=None, ipars=None, group=None):
        self.read_file(file, self._back, self.var_back, varvalue, None,
                       onlyval, ipars, group)
        dprint('read file', file, 'as background')

    back = read_back

    def remove(self, varvalue, tau=None):
        if tau is None:
            try:
                del self._data[varvalue]
            except KeyError:
                print 'remove: no point with varvalue %s' % varvalue
        else:
            try:
                mess = self._data[varvalue]
            except KeyError:
                print 'remove: no point with varvalue %s' % varvalue
            if not isinstance(tau, str):
                # given by real tau
                try:
                    del mess[tau]
                except KeyError:
                    print 'remove: no measurement with tau %s' % tau
            else:
                # given by setting:
                for k, v in mess.items():
                    if v['setting'] == tau:
                        del mess[k]
                        break
                else:
                    print 'remove: no measurement with setting %s' % tau

    def _filenames(self, meas, graph, bkgrd):
        return (meas['singlefile'],
                graph and graph['singlefile'] or '',
                bkgrd and bkgrd['singlefile'] or '')

    def get_data(self, ycol='C', varvalues=ALL, varvaluerange=None):
        if varvalues is ALL:
            if varvaluerange is not None:
                dmin, dmax = varvaluerange
                varvalues = sorted(k for k in self._data.keys()
                                   if dmin <= k <= dmax)
            else:
                varvalues = sorted(self._data.keys())
        measurements = []
        for varvalue in varvalues:
            measurement = MiezeMeasurement(self, ycol, varvalue)
            for x, point in self._data[varvalue].items():
                if self.var_norm:
                    gpoint = self._norm.get(varvalue)
                    if gpoint:
                        graph = gpoint.get(x)
                    else:
                        graph = None
                else:
                    graph = self._norm.get(x)
                if self.var_back:
                    bpoint = self._back.get(varvalue)
                    if bpoint:
                        bkgrd = bpoint.get(x)
                    else:
                        bkgrd = None
                else:
                    bkgrd = self._back.get(x)
                files = self._filenames(point, graph, bkgrd)
                measurement.add_point(x, point, graph, bkgrd, files)
            measurements.append(measurement)
        return measurements

    def plot_data(self, **kwds):
        if miezutil.NOPLOT:
            return self.get_data(**kwds)
        from miezplot import MiezeDataPlot
        plot = MiezeDataPlot(self)
        return plot.plot_data(**kwds)


MiezeDataNF = MiezeData


if __name__ == '__main__':
    for fname in sys.argv[1:]:
        print read_single(fname)

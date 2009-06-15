import re
import sys
import copy
from itertools import izip, cycle, groupby

from numpy import array, linspace, exp, sqrt, sum, power, ceil
import numpy as np

from scipy.optimize import leastsq
import scipy as sp

try:
    import matplotlib.pyplot as pl
except ImportError:
    import pylab as pl

__all__ = ['Data', 'np', 'pl', 'ml_debug']

# -- helper for calculating tau_MIEZE ------------------------------------------

M_N = 1.6749e-27
H   = 6.6261e-34
PI  = 3.1415926
prefactor = M_N**2 / (PI * H**2)

def _mieze_time(lam, L_s, setting):
    f1, f2, bs = re.match(r'([\dp]+)_([\dp]+)(_BS)?', setting).groups()
    f1 = float(f1.replace('p', '.')) * 1000  # in kHz
    f2 = float(f2.replace('p', '.')) * 1000  # in kHz
    dOmega = (f2 - f1) * 2 * PI
    if bs: dOmega *= 2
    tau = (prefactor * lam**3 * dOmega * L_s) * 1e12  # in ps
    return tau

# -- debugging helpers ---------------------------------------------------------

DEBUG = False

def ml_debug(debug=True):
    global DEBUG
    DEBUG = debug

def dprint(*args):
    if DEBUG:
        for arg in args: print arg,
        print

# -- helper for reading data files ---------------------------------------------

def try_float(f):
    try:
        return float(f)
    except ValueError:
        return f

def read_datafile(fn):
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

# -- data object ---------------------------------------------------------------

ALL = object()

class Data(object):
    def __init__(self, name, variable, unit):
        self.name = name
        self.unit = unit
        self.variable = variable
        self.mess = {}
        self.norm = {}
        self.back = {}
        dprint('created new data object:', name)

    def __repr__(self):
        return 'MIEZE data ' + self.name

    def copy(self, newname):
        new = copy.copy(self)
        new.name = newname

    def read_file(self, file, dct, variable=True, varvalue=None,
                  onlyval=None, ipars=None, cat=None):
        if cat is None and ipars:
            cat = ipars[0]
        data, fcomments = read_datafile(file)
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
        for row in data:
            point = dict(zip(fields, row))
            if 'tau' not in point:
                assert ipars, \
                       'lam and L_s not given, but tau not in data file'
                point['tau'] = _mieze_time(ipars[0]*1e-10, ipars[1]*1e-3,
                                           point['setting'])
            point['cat'] = cat
            if variable:
                if varvalue is not None:
                    ddct = dct.setdefault(varvalue, {})
                else:
                    ddct = dct.setdefault(point[self.variable+'_set'], {})
                ddct[point['tau']] = point
            else:
                if onlyval is not None:
                    if point[fields[1]] != onlyval:
                        continue
                dct[point['tau']] = point

    def read_data(self, file, varvalue=None, onlyval=None, ipars=None, cat=None):
        self.read_file(file, self.mess, True, varvalue, onlyval, ipars, cat)
        dprint('read file', file, 'as data')
    def read_norm(self, file, onlyval=None, ipars=None, cat=None):
        self.read_file(file, self.norm, False, None, onlyval, ipars, cat)
        dprint('read file', file, 'as normalization')
    def read_back(self, file, onlyval=None, ipars=None, cat=None):
        self.read_file(file, self.back, False, None, onlyval, ipars, cat)
        dprint('read file', file, 'as background')

    def process(self, ycol='C', dpoints=ALL):
        if dpoints is ALL:
            dpoints = sorted(self.mess.keys())
        curves = []
        for dpoint in dpoints:
            xydy = []
            for x, p in self.mess[dpoint].items():
                graph = self.norm.get(x)
                bkgrd = self.back.get(x)
                # correction factor for time-dependent values
                cf = bkgrd and p['preset']/bkgrd['preset'] or 1
                if ycol == 'sum':
                    c, m = p['countsum'], p['monitor']
                    y = c/m
                    dy = y*(1/sqrt(c) + 1/sqrt(m))
                    if bkgrd:
                        cb = bkgrd['countsum']
                        mb = bkgrd['monitor']
                        y -= cb/mb
                        dy += yless*(1/sqrt(cb) + 1/sqrt(mb))
                elif ycol == 'A':
                    y = p['A'] / p['preset'] # XXX fehler
                    dy = p.get('delta A', 0) / p['preset']
                    if bkgrd:
                        y -= bkgrd['A'] * cf
                        dy += bkgrd['delta A'] * cf
                elif ycol == 'B':
                    y = p['B'] / p['preset']
                    dy = p.get('delta B', 0) / p['preset']
                    if bkgrd:
                        y -= bkgrd['B'] * cf
                        dy += bkgrd.get('delta B', 0) * cf
                elif ycol == 'C':
                    a, b = p['A'], p['B']
                    dc = p.get('delta C', 0)
                    c = a/b
                    if bkgrd:
                        a -= bkgrd['A'] * cf
                        b -= bkgrd['B'] * cf
                        dc += bkgrd.get('delta C', 0)
                        c = a/b
                    if graph:
                        ga, gb = graph['A'], graph['B']
                        gdc = graph['delta C']
                        gc = ga/gb
                        if bkgrd:
                            ga -= bkgrd['A'] / bkgrd['preset'] * graph['preset']
                            gb -= bkgrd['B'] / bkgrd['preset'] * graph['preset']
                            gdc += bkgrd.get('delta C', 0)
                            gc = ga/gb
                        dc = (c/gc)*(dc/c + gdc/gc)
                        c = c/gc
                    y, dy = c, dc
                xydy.append((x, p['cat'], y, dy))
            xydy.sort()
            data = [xydy, '%s %s' % (dpoint, self.unit)]
            curves.append(data)
        return curves

    def format_num_latex(self, val, prec):
        num = '%.*g' % (prec, val)
        if 'e' in num:
            num = num.replace('e-0', 'e-')
            num = num.replace('e', '\\cdot 10^{') + '}'
        return num

    def model(self, v, x, y):
        return v[0]*exp(-abs(v[1])*x) - y

    MARKERS = ['o', '^', 's', 'D', 'v']

    def plot(self, fit=False, color='r', ylabel=None, bycat=False,
             subplots=False, **kwds):
        curves = self.process(**kwds)

        if subplots:
            pl.subplots_adjust(wspace=0.3)
            nrows = ceil(len(curves)/2.)

        for j, (xydy, label) in enumerate(curves):
            # setup plot
            if subplots:
                pl.subplot(nrows, 2, j+1)
                pl.title(label)
            pl.xlabel('$\\tau_{MIEZE}$ / ps')
            if ylabel is not None:
                pl.ylabel(ylabel)
            else:
                pl.ylabel(kwds.get('ycol', 'C') + (self.norm and ' (norm)' or ''))

            # plot data
            x, _, y, dy = map(array, zip(*xydy))
            if not bycat:
                pl.errorbar(x, y, dy, label=label, color=color, marker='o', ls='')
            else:
                # sort by category
                xydy.sort(key=lambda v: v[1])
                for catmarker, (catname, catxydy) in \
                        izip(cycle(self.MARKERS), groupby(xydy, lambda v: v[1])):
                    cx, _, cy, cdy = map(array, zip(*catxydy))
                    catlabel = '%s %s' % (label, catname)
                    pl.errorbar(cx, cy, cdy, label=catlabel,
                                color=color, marker=catmarker, ls='')
            if fit:
                out = leastsq(self.model, x0=[1, 0], args=(x, y),
                              maxfev=1200, full_output=True)
                print 'fit message:', out[-2]
                if 1 <= out[-1] <= 4:
                    vfit = out[0]
                    # chi2 is sum of (square of deviation / ideal value)
                    chi2 = sum(power(self.model(vfit, x, y), 2) /
                               self.model(vfit, x, 0))
                    # chi2/ndf; ndf is (# points - # fit parameters)
                    chi2norm = chi2 / (len(x) - 2)
                    fx = linspace(x[0], x[-1], 1000)
                    fy = self.model(vfit, fx, 0)
                    pl.plot(fx, fy, 'b-', label='exp. fit')
                    gamma = self.format_num_latex(abs(vfit[1]), 2)
                    chi2norm = self.format_num_latex(chi2norm, 2)
                    text = (r'$\Gamma = %s\,\mathrm{ps}^{-1}$' '\n'
                            r'$\chi^2/\mathrm{ndf} = %s$'
                            % (gamma, chi2norm))
                    pl.text(0.03, 0.03, text, size='x-large',
                            transform=pl.gca().transAxes)
            pl.gca().set_ylim(ymin=0)

    def key_release(self, event):
        if event.key == 'q':
            try:
                event.canvas.manager.destroy()
            except AttributeError:
                pass
        elif event.key == 'L':
            ax = event.inaxes
            if not ax:
                return
            ylim = ax.get_ylim()
            scale = ax.get_xscale()
            if scale == 'log':
                ax.set_xscale('linear')
            elif scale == 'linear':
                ax.set_xscale('log')
            ax.set_ylim(ylim)
            ax.figure.canvas.draw()

    def plotfig(self, filename=None, title=None, legend=True, **kwds):
        pl.figure()
        self.plot(**kwds)
        if title is not None:
            pl.suptitle(title, size='x-large')
        else:
            pl.suptitle(self.name + ' - ' + kwds.get('ycol', 'C'), size='x-large')
        if legend:
            pl.legend(loc=(1,0))
            pl.subplots_adjust(right=0.8)
        pl.connect('key_release_event', self.key_release)
        if filename is not None:
            pl.savefig(filename)
            print 'Wrote', filename

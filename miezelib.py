import os
import re
import sys
import copy
import weakref
from itertools import izip, cycle, groupby

from numpy import array, arange, linspace, exp, sin, sqrt, sum, power, ceil, pi
import numpy as np

from scipy.odr import RealData, Model, ODR
import scipy as sp

try:
    import matplotlib.pyplot as pl
except ImportError:
    import pylab as pl

__all__ = ['Data', 'np', 'pl', 'ml_debug', 'ml_setdatadir']

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

_datadir = '.' 

def ml_setdatadir(dir):
    global _datadir
    _datadir = dir

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
        self.collections = weakref.WeakKeyDictionary()
        dprint('created new data object:', name)

    def __repr__(self):
        return 'MIEZE data ' + self.name

    def copy(self, newname):
        new = copy.copy(self)
        new.name = newname

    def read_file(self, file, dct, variable=True, varvalue=None, vals=None,
                  onlyval=None, ipars=None, cat=None):
        if cat is None and ipars:
            cat = ipars[0]
        if isinstance(file, int):
            file = 'mieze_%08d' % file
        file = os.path.join(_datadir, file)
        if os.path.isdir(file):
            file = os.path.join(file, 'summary')
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
            point['summaryfile'] = file
            if 'tau' not in point or point['tau'] == '-':
                assert ipars, \
                       'lam and L_s not given, but tau not in data file'
                point['tau'] = _mieze_time(ipars[0]*1e-10, ipars[1]*1e-3,
                                           point['setting'])
            point['cat'] = cat
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
            else:
                if onlyval is not None:
                    if point[fields[1]] != onlyval:
                        continue
                dct[point['tau']] = point

    def read_data(self, file, varvalue=None, vals=None, onlyval=None, ipars=None, cat=None):
        self.read_file(file, self.mess, True, varvalue, vals, onlyval, ipars, cat)
        dprint('read file', file, 'as data')

    def read_norm(self, file, onlyval=None, ipars=None, cat=None):
        self.read_file(file, self.norm, False, None, None, onlyval, ipars, cat)
        dprint('read file', file, 'as normalization')

    def read_back(self, file, onlyval=None, ipars=None, cat=None):
        self.read_file(file, self.back, False, None, None, onlyval, ipars, cat)
        dprint('read file', file, 'as background')

    def _filenames(self, meas, graph, bkgrd):
        fn = lambda p: os.path.join(os.path.dirname(p['summaryfile']),
                                    '%05d' % int(p['in file']))
        return (fn(meas),
                graph and fn(graph) or '',
                bkgrd and fn(bkgrd) or '')

    def process(self, ycol='C', dpoints=ALL, dpointrange=None):
        if dpoints is ALL:
            if dpointrange is not None:
                dmin, dmax = dpointrange
                dpoints = sorted(k for k in self.mess.keys()
                                 if dmin <= k <= dmax)
            else:
                dpoints = sorted(self.mess.keys())
        curves = []
        for dpoint in dpoints:
            xydy = []
            for x, point in self.mess[dpoint].items():
                cat = point.get('cat')
                graph = self.norm.get(x)
                bkgrd = self.back.get(x)
                files = self._filenames(point, graph, bkgrd)
                # correction factor for time-dependent measurement values
                cf = bkgrd and point['preset']/bkgrd['preset'] or 1
                # and for graphite values
                cfg = (graph and bkgrd) and graph['preset']/bkgrd['preset'] or 1
                if ycol == 'sum':
                    c, m = point['countsum'], point['monitor']
                    y = c/m
                    dy = y*(1/sqrt(c) + 1/sqrt(m))
                    if bkgrd:
                        cb = bkgrd['countsum']
                        mb = bkgrd['monitor']
                        y -= cb/mb
                        dy += y*(1/sqrt(cb) + 1/sqrt(mb))
                elif ycol == 'A':
                    y = point['A']
                    dy = point.get('delta A', 0)
                    if bkgrd:
                        y -= bkgrd['A'] * cf
                        dy += bkgrd['delta A'] * cf
                    y /= point['preset']
                    dy /= point['preset']
                elif ycol == 'B':
                    y = point['B']
                    dy = point.get('delta B', 0)
                    if bkgrd:
                        y -= bkgrd['B'] * cf
                        dy += bkgrd.get('delta B', 0) * cf
                    y /= point['preset']
                    dy /= point['preset']
                elif ycol == 'C':
                    a, b = point['A'], point['B']
                    dc = point.get('delta C', 0)
                    c = a/b
                    if bkgrd:
                        bkgrd['delta A'] = 0
                        bkgrd['delta B'] = 0
                        a -= bkgrd['A'] * cf
                        b -= bkgrd['B'] * cf
                        c = a/b
                        print point['delta A'], bkgrd['delta A']*cf
                        da = point['delta A'] + bkgrd['delta A']*cf
                        db = point['delta B'] + bkgrd['delta B']*cf
                        dc = c * (da/a + db/b)
                    if graph:
                        ga, gb = graph['A'], graph['B']
                        gdc = graph['delta C']
                        gc = ga/gb
                        if bkgrd:
                            ga -= bkgrd['A'] * cfg
                            gb -= bkgrd['B'] * cfg
                            gc = ga/gb
                            gda = graph['delta A'] + bkgrd['delta A']*cfg
                            gdb = graph['delta B'] + bkgrd['delta B']*cfg
                            gdc = gc * (gda/ga + gdb/gb)
                        dc = (c/gc)*(dc/c + gdc/gc)
                        c = c/gc
                    y, dy = c, dc
                xydy.append((x, y, dy, files, cat))
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

    def model(self, v, x):
        return v[0]*exp(-abs(v[1])*x)

    MARKERS = ['o', '^', 's', 'D', 'v']

    def plot(self, fig, fit=True, color=None, ylabel=None, bycat=True,
             subplots=True, lines=False, **kwds):
        curves = self.process(**kwds)

        if subplots:
            fig.subplots_adjust(wspace=0.3)
            ncols = len(curves) >= 9 and 3 or 2
            nrows = ceil(len(curves)/float(ncols))

        lastrow = True
        firstcol = True
        for j, (xydy, label) in enumerate(curves):
            # setup plot
            if subplots:
                ax = fig.add_subplot(nrows, ncols, j+1)
                ax.set_title(label)
                lastrow = j >= len(curves) - ncols
                firstcol = j % ncols == 0
            else:
                ax = fig.gca()
            if lastrow:
                ax.set_xlabel('$\\tau_{MIEZE}$ / ps')
            if firstcol:
                if ylabel is not None:
                    ax.set_ylabel(ylabel)
                else:
                    ax.set_ylabel(kwds.get('ycol', 'C') +
                                  (self.norm and ' (norm)' or ''))

            # plot data
            x, y, dy, sf, _ = map(array, zip(*xydy))
            kwds = {'picker': 5, 'ls': lines and 'solid' or ''}
            if color is not None:
                kwds['color'] = color
            if not bycat:
                coll = ax.errorbar(x, y, dy, label=label, marker='o', **kwds)
                self.collections[coll[0]] = sf
            else:
                # sort by category
                xydy.sort(key=lambda v: v[4])
                for catmarker, (catname, catxydy) in \
                        izip(cycle(self.MARKERS), groupby(xydy, lambda v: v[4])):
                    cx, cy, cdy, csf, _ = map(array, zip(*catxydy))
                    catlabel = '%s %s' % (label, catname)
                    coll = ax.errorbar(cx, cy, cdy, label=catlabel,
                                       marker=catmarker, **kwds)
                    self.collections[coll[0]] = csf

            if fit:
                dat = RealData(x, y, sy=dy)
                odr = ODR(dat, Model(self.model), beta0=[1, 0],
                          ifixx=array([0]*len(x)))  # fixed X values
                out = odr.run()
                vfit = list(out.beta)
                errors = list(out.sd_beta)
                if 1 <= out.info <= 3:
                    ndf = len(x) - 2
                    # chi2 is sum of (square of deviation / ideal value)
                    chi2 = sum(power(self.model(vfit, x) - y, 2) /
                               power(dy, 2))
                    # chi2/ndf; ndf is (# points - # fit parameters)
                    chi2norm = chi2/ndf
                    fx = linspace(x[0], x[-1], 1000)
                    fy = self.model(vfit, fx)
                    ax.plot(fx, fy, 'm-', label='exp. fit')
                    gamma = abs(vfit[1]) * 1000 # in 1/ns
                    if gamma < 0.03:
                        gamma = 0
                    gamma_s = self.format_num_latex(gamma, 2)
                    gamerr_s = self.format_num_latex(errors[1]*1000, 2)
                    gamma_muev_s = self.format_num_latex(gamma * 0.657, 2)
                    gamerr_muev_s = self.format_num_latex(errors[1] * 657, 2)
                    chi2norm_s = self.format_num_latex(chi2norm, 2)
                    if gamma == 0:
                        text = r'$\Gamma = 0$'
                    else:
                        text = (r'$\Gamma = %s\,\mathrm{ns}^{-1}$''\n'r'$\hat{=}\, '
                                r'%s\,\mathrm{\mu eV}$' % (gamma_s, gamma_muev_s)
                                #+ '\n' r'$\chi^2/\mathrm{ndf} = %s$' % chi2norm_s
                                )
                    ax.text(0.03, 0.03, text, size='large',
                            transform=ax.transAxes)
            ax.set_ylim(ymin=0)

    def on_key_release(self, event):
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

    def on_pick(self, event):
        npoint = event.ind[0]
        collection = event.artist
        if collection not in self.collections:
            return
        filenames = self.collections[collection][npoint]
        self.plotmieze(filenames)

    def plotfig(self, filename=None, title=None, legend=True, **kwds):
        self.fig = pl.figure()
        self.plot(self.fig, **kwds)
        if title is not None:
            self.fig.suptitle(title, size='x-large')
        else:
            self.fig.suptitle(self.name, size='x-large')
        if legend:
            self.fig.gca().legend(loc=(1,0))
            self.fig.subplots_adjust(right=0.8)
        self.fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        if filename is not None:
            self.fig.savefig(filename)
            print 'Wrote', filename

    pm_re = re.compile(r'([0-9e.]+)\s+\+/-\s+([0-9e.]+)')
            
    def plotmieze(self, filenames):
        """Plot single MIEZE curves."""
        
        def fileinfo(filename):
            points = []
            for line in open(filename):
                line = line.strip()
                if not line: continue
                if not line.startswith('#'):
                    points.append(int(line))
                    continue
                line = line[2:]
                if line.startswith('t/s:'):
                    preset = int(line[5:])
                elif line.startswith('sum:'):
                    countsum = int(line[5:])
                elif line.startswith('mon:'):
                    monitor = int(line[5:])
                elif line.startswith('A:'):
                    A, dA = map(float, self.pm_re.match(line[5:]).groups())
                elif line.startswith('B:'):
                    B, dB = map(float, self.pm_re.match(line[5:]).groups())
                elif line.startswith('C:'):
                    C, dC = map(float, self.pm_re.match(line[5:]).groups())
                elif line.startswith('phi:'):
                    phi, dphi = map(float, self.pm_re.match(line[5:]).groups())
            points = array(points)
            return points, preset, countsum, monitor, A, dA, B, dB, C, dC, phi, dphi

        def mz(x, A, B, phi):
            return B + A*sin(pi/4*x + phi)

        def plotinfo(filename, name, ax):
            info = fileinfo(filename)
            points, preset, countsum, monitor, A, dA, B, dB, C, dC, phi, dphi = info
       
            ax.set_title('%s: %s\n'
                         r'$C = %.2f \pm %.2f$, $A = %.2f \pm %.2f$, $B = %.2f \pm %.2f$, '
                         r'$\Sigma = %s$, $t = %s$' %
                         (name, filename, C, dC, A, dA, B, dB, countsum, preset))
            ax.set_ylabel('counts')
            ax.errorbar(arange(1, 17), points, sqrt(points), fmt='ro')

            xs = arange(0, 16, 0.1)
            ys = mz(xs, A, B, phi)
            ax.plot(xs, ys, 'b-')
            ax.set_ylim(ymin=0)

        fig = pl.figure()
        fig.canvas.mpl_connect('key_release_event', self.on_key_release)
        if filenames[1] and filenames[2]:
            ax = fig.add_subplot(311)
            plotinfo(filenames[0], 'Measurement', ax)
            ax = fig.add_subplot(312)
            plotinfo(filenames[1], 'Normalization', ax)
            ax = fig.add_subplot(313)
            plotinfo(filenames[2], 'Background', ax)
        elif filenames[1]:
            ax = fig.add_subplot(211)
            plotinfo(filenames[0], 'Measurement', ax)
            ax = fig.add_subplot(212)
            plotinfo(filenames[1], 'Normalization', ax)
        elif filenames[2]:
            ax = fig.add_subplot(211)
            plotinfo(filenames[0], 'Measurement', ax)
            ax = fig.add_subplot(212)
            plotinfo(filenames[2], 'Background', ax)
        else:
            ax = fig.gca()
            plotinfo(filenames[0], 'Measurement', ax)
        fig.subplots_adjust(hspace=0.3, bottom=0.05)

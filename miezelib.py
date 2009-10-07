import os
import re
import sys
import copy
import weakref
from itertools import izip, cycle, groupby

from numpy import array, arange, linspace, exp, sin, sqrt, sum, power, ceil, pi
import numpy as np

from scipy.odr import RealData, Model, ODR
from scipy.optimize import leastsq
import scipy as sp

try:
    import matplotlib.pyplot as pl
except ImportError:
    import pylab as pl

__all__ = ['MiezeData', 'Fit', 'np', 'pl', 'ml_mieze_time',
           'ml_debug', 'ml_setdatadir', 'ml_figure', 'ml_gammaplot']

ALL = object()

# -- helper for calculating tau_MIEZE and formatting numbers -------------------

M_N = 1.6749e-27
H   = 6.6261e-34
PI  = 3.1415926
prefactor = M_N**2 / (PI * H**2)

def ml_mieze_time(lam, L_s, setting):
    f1, f2, bs = re.match(r'([\dp]+)_([\dp]+)(_BS)?', setting).groups()
    f1 = float(f1.replace('p', '.')) * 1000  # in kHz
    f2 = float(f2.replace('p', '.')) * 1000  # in kHz
    dOmega = (f2 - f1) * 2 * PI
    if bs: dOmega *= 2
    tau = (prefactor * lam**3 * dOmega * L_s) * 1e12  # in ps
    return tau

def _format_num(val, prec):
    num = '%.*g' % (prec, val)
    if 'e' in num:
        num = num.replace('e-0', 'e-')
        num = num.replace('e', '\\cdot 10^{') + '}'
    return num

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


# -- plotting helpers ----------------------------------------------------------

def ml_on_key_release(event):
    if event.key == 'q':
        try:
            # works only with GtkAgg backend; raises AttributeError despite
            # closing the window eventually
            event.canvas.manager.destroy()
        except AttributeError:
            pass
    elif event.key == 'L':
        # toggle x axis log scaling, analogous to code for y axis (key 'l')
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

def ml_figure(suptitle=None, titlesize='x-large', **kwargs):
    fig = pl.figure(**kwargs)
    fig.canvas.mpl_connect('key_release_event', ml_on_key_release)
    if suptitle:
        fig.suptitle(suptitle, size=titlesize, y=0.95)
    return fig

def ml_gammaplot(data, titles, figsize=None, textsize=None, ticksize=None,
                 filename=None, title=None, titlesize=None, fit=None,
                 critical=None):
    ndata = len(data)
    if figsize is None:
        figsize = (3*ndata, 4)
    fig = ml_figure(title, titlesize=titlesize, figsize=figsize)
    fig.subplots_adjust(left=0.1, right=0.92, top=0.83, bottom=0.2,
                        wspace=0.08)
    axes, ylimits = [], []
    for j, (meass, title) in enumerate(zip(data, titles)):
        if title is None:
            title = meass[0].data.name
        ax = fig.add_subplot(1, ndata, j+1)
        x, y, dy = [], [], []
        for meas in meass:
            if meas.fitvalues is not None:
                x.append(meas.varvalue)
                y.append(meas.fitvalues[0])
                dy.append(meas.fitvalues[1])
            else:
                res = meas.fit()
                if res:
                    x.append(meas.varvalue)
                    y.append(res.Gamma)
                    dy.append(res.dGamma)
        if critical:
            x = map(lambda v: v - critical, x)

        ax.errorbar(x, y, dy, marker='o', ls='')

        if fit:
            res = fit.run(title, x, y, dy)
            if res:
                ax.plot(res.curve_x, res.curve_y, '-')

        ylimits.append(ax.get_ylim())
        ax.set_xlim(x[0]-0.1, x[-1]+0.1)
        if critical:
            ax.set_xlabel('$%s-%s_c\\,[%s]$' % (
                meas.data.variable, meas.data.variable, meas.data.unit),
                          size=textsize)
        else:
            ax.set_xlabel('$%s\\,[%s]$' % (meas.data.variable, meas.data.unit),
                          size=textsize)
        pl.xticks(size=ticksize, verticalalignment='bottom', y=-0.08)
        pl.yticks(size=ticksize)
        if j == 0:
            # first plot
            ax.set_ylabel('$\\Gamma\\,[\\mu\\mathrm{eV}]$', size=textsize)
        elif j == ndata - 1:
            # last plot: put ticks on right side (only for > 1 plot)
            ax.yaxis.set_ticks_position('right')
            for t in ax.yaxis.majorTicks + ax.yaxis.minorTicks:
                t.tick1On = True
            pl.yticks(size=ticksize)
        else:
            # middle plots: no ticks
            for t in ax.yaxis.majorTicks + ax.yaxis.minorTicks:
                t.label1On = False
        axes.append(ax)
        pl.text(0.9, 0.9, title, size=textsize,
                horizontalalignment='right',
                verticalalignment='top',
                transform=pl.gca().transAxes)

    # make the Y scale equal for all plots
    yminmin = ylimits[0][0]
    ymaxmax = ylimits[0][1]
    for ymin, ymax in ylimits[1:]:
        yminmin = min(ymin, yminmin)
        ymaxmax = max(ymax, ymaxmax)
    for ax in axes:
        ax.set_ylim(yminmin, ymaxmax)

    if filename is not None:
        fig.savefig(filename)
        dprint('Wrote', title or self.name, 'to', filename)


# -- fitting helpers -----------------------------------------------------------

class FitResult(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

    def __str__(self):
        if self._failed:
            return 'Fit %-20s failed' % self._name
        elif self._method == 'ODR':
            return 'Fit %-20s success (  ODR  ), chi2: %8.3g, params: %s' % (
                self._name or '', self.chi2,
                ', '.join('%s = %8.3g +/- %8.3g' % v for v in zip(*self._pars)))
        else:
            return 'Fit %-20s success (leastsq), chi2: %8.3g, params: %s' % (
                self._name or '', self.chi2,
                ', '.join('%s = %8.3g' % v[:2] for v in zip(*self._pars)))

    def __nonzero__(self):
        return not self._failed


class Fit(object):
    def __init__(self, model, parnames=None, parstart=None,
                 xmin=None, xmax=None):
        self.model = model
        self.parnames = parnames or []
        self.parstart = parstart or []
        self.xmin = xmin
        self.xmax = xmax
        if len(self.parnames) != len(self.parstart):
            raise RuntimeError('number of param names must match number '
                               'of starting values')

    def par(self, name, start):
        self.parnames.append(name)
        self.parstart.append(start)

    def run(self, name, x, y, dy):
        if len(x) < 2:
            # need at least two points to fit
            return self.result(name, None, x, y, dy, None, None)
        # try fitting with ODR
        data = RealData(x, y, sy=dy)
        # fit with fixed x values
        odr = ODR(data, Model(self.model), beta0=self.parstart,
                  ifixx=array([0]*len(x)))
        out = odr.run()
        if 1 <= out.info <= 3:
            return self.result(name, 'ODR', x, y, dy, out.beta, out.sd_beta)
        else:
            # if it doesn't converge, try leastsq (doesn't consider errors)
            try:
                out = leastsq(lambda v: self.model(v, x) - y, self.parstart)
            except TypeError:
                return self.result(name, None, x, y, dy, None, None)
            if out[1] <= 4:
                return self.result(name, 'leastsq', x, y, dy, out[0],
                                   parerrors=[0]*len(out[0]))
            else:
                return self.result(name, None, x, y, dy, None, None)

    def result(self, name, method, x, y, dy, parvalues, parerrors):
        if method is None:
            dct = {'_name': name, '_failed': True}
        else:
            dct = {'_name': name, '_method': method, '_failed': False,
                   '_pars': (self.parnames, parvalues, parerrors)}
            for name, val, err in zip(self.parnames, parvalues, parerrors):
                dct[name] = val
                dct['d' + name] = err
            if self.xmin is None:
                xmin = x[0]
            else:
                xmin = self.xmin
            if self.xmax is None:
                xmax = x[-1]
            else:
                xmax = self.xmax
            dct['curve_x'] = linspace(xmin, xmax, 1000)
            dct['curve_y'] = self.model(parvalues, dct['curve_x'])
            ndf = len(x) - len(parvalues)
            residuals = self.model(parvalues, x) - y
            dct['chi2'] = sum(power(residuals, 2) / power(dy, 2)) / ndf
        result = FitResult(**dct)
        dprint(result)
        return result


# -- data objects --------------------------------------------------------------

class MiezeMeasurement(object):
    """Container for a single MIEZE measurement (multiple MIEZE times)."""

    def __init__(self, data, ycol, varvalue):
        self.varvalue = varvalue
        self.data = data
        self.fitvalues = None
        self._calc_point = getattr(self, '_calc_point_' + ycol)

        self.points = []
        self.arrays = None

    def add_point(self, x, point, graph, bkgrd, files):
        y, dy = self._calc_point(point, graph, bkgrd)
        self.points.append((x, y, dy, files, point.get('group')))

    def _calc_point_sum(self, point, graph, bkgrd):
        c, m = point['countsum'], point['monitor']
        y = c/m
        dy = y*(1/sqrt(c) + 1/sqrt(m))
        if bkgrd:
            cb = bkgrd['countsum']
            mb = bkgrd['monitor']
            y -= cb/mb
            dy += y*(1/sqrt(cb) + 1/sqrt(mb))
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
            a -= bkgrd['A'] * cf
            b -= bkgrd['B'] * cf
            c = a/b
            #print point['delta A'], bkgrd['delta A']*cf
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
        return c, dc

    @property
    def label(self):
        return '%s %s' % (self.varvalue, self.data.unit)

    def as_arrays(self):
        if self.arrays:
            return self.arrays
        self.arrays = map(array, zip(*sorted(self.points)))
        return self.arrays

    def as_arrays_bygroup(self):
        get_group = lambda v: v[4]
        newpoints = sorted(self.points, key=get_group)
        for groupname, grouppoints in groupby(newpoints, get_group):
            gx, gy, gdy, gsf, _ = map(array, zip(*grouppoints))
            yield groupname, gx, gy, gdy, gsf

    def _fit_model(self, v, x):
        # to get gamma in mueV, conversion factor is hbar = 658.2 mueV*ps
        return v[1]*exp(-abs(v[0])*x/658.2)

    def fit(self, name=None, **kwds):
        name = '%s %s' % (name or '', self.label)
        x, y, dy = self.as_arrays()[:3]
        res = Fit(self._fit_model, ['Gamma', 'c'], [0, 1], **kwds).run(name, x, y, dy)
        if not res:
            return None
        res.Gamma = abs(res.Gamma)
        self.fitvalues = (res.Gamma, res.dGamma, res.chi2)
        return res


class MiezeData(object):
    """Container for a whole range of MIEZE measurements."""

    def __init__(self, name, variable, unit, var_norm=False, var_back=False,
                 ipars=None, resolution=None):
        self.name = name
        self.unit = unit
        self.ipars = ipars
        self.variable = variable
        self.var_norm = var_norm
        self.var_back = var_back
        self.resolution = resolution
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
                  onlyval=None, ipars=None, group=None):
        if ipars is None:
            ipars = self.ipars
        if group is None and ipars:
            group = ipars[0]
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
                point['tau'] = ml_mieze_time(ipars[0]*1e-10, ipars[1]*1e-3,
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
            else:
                if onlyval is not None:
                    if point[fields[1]] != onlyval:
                        continue
                dct[point['tau']] = point

    def read_data(self, file, varvalue=None, vals=None, ipars=None, group=None):
        self.read_file(file, self.mess, True, varvalue, vals, None,
                       ipars, group)
        dprint('read file', file, 'as data')

    def read_norm(self, file, varvalue=None, onlyval=None, ipars=None, group=None):
        self.read_file(file, self.norm, self.var_norm, varvalue, None,
                       onlyval, ipars, group)
        dprint('read file', file, 'as normalization')

    def read_back(self, file, varvalue=None, onlyval=None, ipars=None, group=None):
        self.read_file(file, self.back, self.var_back, varvalue, None,
                       onlyval, ipars, group)
        dprint('read file', file, 'as background')

    def _filenames(self, meas, graph, bkgrd):
        fn = lambda p: os.path.join(os.path.dirname(p['summaryfile']),
                                    '%05d' % int(p['in file']))
        return (fn(meas), graph and fn(graph) or '', bkgrd and fn(bkgrd) or '')

    def get_data(self, ycol='C', varvalues=ALL, varvaluerange=None):
        if varvalues is ALL:
            if varvaluerange is not None:
                dmin, dmax = varvaluerange
                varvalues = sorted(k for k in self.mess.keys()
                                   if dmin <= k <= dmax)
            else:
                varvalues = sorted(self.mess.keys())
        measurements = []
        for varvalue in varvalues:
            measurement = MiezeMeasurement(self, ycol, varvalue)
            for x, point in self.mess[varvalue].items():
                if self.var_norm:
                    graph = self.norm[varvalue].get(x)
                else:
                    graph = self.norm.get(x)
                if self.var_back:
                    bkgrd = self.back[varvalue].get(x)
                else:
                    bkgrd = self.back.get(x)
                files = self._filenames(point, graph, bkgrd)
                measurement.add_point(x, point, graph, bkgrd, files)
            measurements.append(measurement)
        return measurements

    MARKERS = ['o', '^', 's', 'D', 'v']

    def plot(self, fig=None, fit=True, color=None, ylabel=None, log=True,
             bygroup=True, subplots=True, lines=False, data=None, **kwds):
        # optional parameters
        if data is None:
            data = self.get_data(**kwds)
        if fig is None:
            fig = ml_figure()

        # calculate the number of subplot rows and columns
        if subplots:
            fig.subplots_adjust(wspace=0.3)
            ncols = len(data) >= 9 and 3 or 2
            nrows = ceil(len(data)/float(ncols))

        lastrow = True
        firstcol = True
        for j, meas in enumerate(data):
            # setup a single subplot, or use the standard one
            if subplots:
                ax = fig.add_subplot(nrows, ncols, j+1)
                ax.set_title(meas.label)
                lastrow = j >= len(data) - ncols
                firstcol = j % ncols == 0
            else:
                ax = fig.gca()

            # put axis labels only on the left and bottom of all subplots
            if lastrow:
                ax.set_xlabel('$\\tau_{MIEZE}$ [ps]')
            if firstcol:
                if ylabel is not None:
                    ax.set_ylabel(ylabel)
                else:
                    ax.set_ylabel(kwds.get('ycol', 'C') +
                                  (self.norm and ' (norm)' or ''))

            # plot the data, by groups or together
            kwds = {'picker': 5, 'ls': lines and 'solid' or ''}
            if color is not None:
                kwds['color'] = color
            if not bygroup:
                x, y, dy, sf, _ = meas.as_arrays()
                coll = ax.errorbar(x, y, dy, label=meas.label, marker='o', **kwds)
                self.collections[coll[0]] = sf
            else:
                for gmarker, (gname, gx, gy, gdy, gsf) in \
                    izip(cycle(self.MARKERS), meas.as_arrays_bygroup()):
                    glabel = '%s %s' % (meas.label, gname)
                    coll = ax.errorbar(gx, gy, gdy,
                                       label=glabel, marker=gmarker, **kwds)
                    self.collections[coll[0]] = gsf
            ax.set_ylim(ymin=0)

            # fit the data if wanted and if possible (only makes sense for 'C')
            if not fit or kwds.get('ycol', 'C') != 'C':
                continue
            res = meas.fit(self.name, xmin=0)
            if not res:
                continue

            ax.plot(res.curve_x, res.curve_y, 'm-', label='exp. fit')
            if res.Gamma < (self.resolution or 0.05):
                # Gamma is below instrumental resolution
                text = '$\Gamma = 0$'
            else:
                text = r'$\Gamma = %s \pm %s\,\mathrm{\mu eV}$' % \
                       (_format_num(res.Gamma, 2), _format_num(res.dGamma, 2))
            # display the Gamma value as text
            ax.text(0.03, 0.03, text, size='large', transform=ax.transAxes)
            if log:
                ax.set_yscale('log')
                ax.set_ylim(ymin=1e-1, ymax=2)
            else:
                ax.set_ylim(ymin=0)
        return data

    def plot_data(self, filename=None, title=None, legend=False, **kwds):
        fig = ml_figure(title or self.name)
        ret = self.plot(fig, **kwds)
        if legend:
            fig.gca().legend(loc=(1,0))
            fig.subplots_adjust(right=0.8)
        fig.canvas.mpl_connect('pick_event', self.on_pick)
        if filename is not None:
            fig.savefig(filename)
            dprint('Wrote', title or self.name, 'to', filename)
        return ret

    # -- plotting a single MIEZE measurement (e.g. for checking the fit) -------

    def on_pick(self, event):
        """Matplotlib event handler for clicking on a data point."""
        npoint = event.ind[0]
        collection = event.artist
        if collection not in self.collections:
            return
        filenames = self.collections[collection][npoint]
        self.plot_mieze(filenames)

    pm_re = re.compile(r'([0-9e.]+)\s+\+/-\s+([0-9e.]+)')

    def plot_mieze(self, filenames):
        """Plot single MIEZE curves."""

        def fileinfo(filename):
            pts = []
            for line in open(filename):
                line = line.strip()
                if not line: continue
                if not line.startswith('#'):
                    pts.append(int(line))
                    continue
                line = line[2:]
                if line.startswith('t/s:'):
                    preset = int(line[5:])
                elif line.startswith('sum:'):
                    ctsum = int(line[5:])
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
            pts = array(pts)
            return pts, preset, ctsum, monitor, A, dA, B, dB, C, dC, phi, dphi

        def mz(x, A, B, phi):
            return B + A*sin(pi/4*x + phi)

        def plotinfo(filename, name, ax):
            info = fileinfo(filename)
            pts, preset, ctsum, monitor, A, dA, B, dB, C, dC, phi, dphi = info

            ax.set_title('%s: %s\n' % (name, filename) +
                         r'$C = %.2f \pm %.2f$, ' % (C, dC) +
                         r'$A = %.2f \pm %.2f$, ' % (A, dA) +
                         r'$B = %.2f \pm %.2f$, ' % (B, dB) +
                         r'$\Sigma = %s$, $t = %s$' % (ctsum, preset))
            ax.set_ylabel('counts')
            ax.errorbar(arange(1, 17), pts, sqrt(pts), fmt='ro')

            xs = arange(0, 16, 0.1)
            ys = mz(xs, A, B, phi)
            ax.plot(xs, ys, 'b-')
            ax.set_ylim(ymin=0)

        fig = ml_figure(figsize=(9, 13))
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
        fig.subplots_adjust(hspace=0.4, bottom=0.05)
        fig.show()

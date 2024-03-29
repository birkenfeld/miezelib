# -*- coding: utf-8 -*-
"""
    miezelib: miezplot
    ~~~~~~~~~~~~~~~~~~

    Plotting of MIEZE data.

    :copyright: 2008-2010 by Georg Brandl.
    :license: BSD.
"""

import weakref
from itertools import izip, cycle

import numpy as np

import warnings
warnings.filterwarnings("ignore", "PyArray_.*")

from scipy.interpolate import splrep, splev

try:
    import matplotlib.pyplot as pl
except ImportError:
    import pylab as pl
from mpl_toolkits.axes_grid import AxesGrid

from miezdata import read_single, MiezeData
from miezutil import dprint, format_tex


def on_key_release(event):
    """Extra key release handler for matplotlib figures."""
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
    elif event.key == 'r':
        # log scaling of secondary axis (one that shares the x axis)
        oax = event.inaxes
        if not oax:
            return
        for ax in oax.figure.axes:
            if ax._sharex is oax:
                break
        else:
            return
        scale = ax.get_yscale()
        if scale == 'log':
            ax.set_yscale('linear')
        else:
            ax.set_yscale('log')
        oax.figure.canvas.draw()


def figure(suptitle=None, titlesize='x-large', titley=0.95, **kwargs):
    """Create a new figure with special key handler."""
    pl.rc('font', family='Lucida Grande')
    fig = pl.figure(**kwargs)
    fig.canvas.mpl_connect('key_release_event', on_key_release)
    if suptitle:
        fig.suptitle(suptitle, size=titlesize, y=titley)
    return fig


def show():
    """Like matplotlib show(), but don't display a traceback on Ctrl-C."""
    try:
        pl.show()
    except KeyboardInterrupt:
        pass


def gammaplot(data, titles=None, figsize=None, textsize='x-large', ticksize=None,
              filename=None, title=None, titlesize='xx-large', fit=None,
              critical=None, secondary=None, seclabel=None, secspline=False,
              xlabel=None, xtransform=None, ylims=None, xlim=None, bottom=None,
              logscale=False, seclogscale=False, xlogscale=False, threesigma=False,
              top=None, left=None, right=None, wspace=None, hspace=None,
              legend=False, subplots=True, lefttitle=False, datafile=None):
    """Create a plot of Gamma versus variable quantity."""
    from miezfit import Fit

    # shortcut for plotting only one measurement
    if data and isinstance(data, MiezeData):
        if title is None:
            title = data.name
        data = [data]

    # shortcut for plotting contrast and intensity
    if data and isinstance(data[0], MiezeData) and not secondary:
        measurements = data[:]
        data = []
        secondary = []
        for m in measurements:
            data.append(m.plot_data())
            secondary.append(m.get_data(ycol='sum'))

    if subplots:
        nplots = len(data)
        cycle1 = cycle(['b'])
        cycle2 = cycle(['r'])
    else:
        nplots = 1
        cycle1 = cycle(['b', 'g', 'r'])
        cycle2 = cycle(['b', 'g', 'r'])

    if figsize is None:
        figsize = (3*nplots + 1.5, 4)
    fig = figure(title, titlesize=titlesize, figsize=figsize)
    defleft  = [0.18, 0.11, 0.08, 0.05][min(nplots-1, 3)]
    defright = [0.84, 0.91, 0.94, 0.97][min(nplots-1, 3)]
    fig.subplots_adjust(left=left or defleft, right=right or defright,
                        top=top or 0.83, bottom=bottom or 0.16,
                        wspace=wspace or 0.09, hspace=hspace)
    axes, ylimits = [], []
    twaxes, twylimits = [], []
    fitresults = []

    if isinstance(fit, Fit) or fit is None:
        fits = [fit] * len(data)
    else:
        fits = fit

    if titles is None:
        titles = [''] * len(data)

    if not (len(data) == len(titles) == len(fits)):
        print 'warning: different lengths for data/titles/fits lists'

    ax = None
    twax = None

    data_array = []

    for j, (meass, title, fit) in enumerate(zip(data, titles, fits)):
        if title is None:
            title = meass[0].data.name

        if subplots:
            ax = fig.add_subplot(1, nplots, j+1)
            twax = ax.twinx()
            # draw sum axes before gamma axes so that the latter is
            # in front (zorder doesn't help here)
            ax.figure.axes[-2:] = [twax, ax]
            ax.set_frame_on(False)
            twax.set_frame_on(True)
        else:
            if ax is None:
                ax = fig.gca()
                twax = ax.twinx()
                ax.figure.axes[-2:] = [twax, ax]
                ax.set_frame_on(False)
                twax.set_frame_on(True)

        # primary data: Gamma
        x, y, dy = [], [], []
        ex, ey, edy = [], [], []
        allx, ally, alldy = [], [], []
        for meas in meass:
            vx = vy = vdy = None
            if meas.fitvalues is not None:
                vx = meas.varvalue
                vy = meas.fitvalues[0]
                vdy = meas.fitvalues[1]
            else:
                res = meas.fit()
                if res:
                    vx = meas.varvalue
                    vy = res.Gamma
                    vdy = res.dGamma
            if vx is not None:
                if 0 <= vy <= 5 and 0 <= vdy <= max(vy/2, 0.15):
                    x.append(vx)
                    y.append(vy)
                    dy.append(vdy)
                else:
                    ex.append(vx)
                    ey.append(0.05)
                    edy.append(0)
                allx.append(vx)
                ally.append(vy)
                alldy.append(vdy)
        if critical:
            x = map(lambda v: v - critical, x)
            ex = map(lambda v: v - critical, ex)
        if xtransform:
            if isinstance(xtransform, list):
                trf = xtransform[j]
            else:
                trf = xtransform
            x = map(trf, x)
            ex = map(trf, ex)
            allx = map(trf, allx)

        data_array.append((title, allx, ally, alldy))

        color = cycle1.next()
        if len(x):
            if threesigma:
                ndy = map(lambda x: x*3, dy)
            else:
                ndy = dy
            ax.errorbar(x, y, ndy, color=color, marker='o', ls='',
                        label=mktitle(title) + ' (linewidth)')
            if ex:
                ax.errorbar(ex, ey, color='black', marker='*', ls='')

        if logscale:
            ax.set_yscale('log')

        if fit:
            res = fit.run(title, x, y, dy)
            if res:
                ax.plot(res.curve_x, res.curve_y, color + '-')
            fitresults.append(res)
        else:
            fitresults.append(None)

        axes.append(ax)
        ylimits.append(ax.get_ylim())

        # secondary data: normally sum
        if secondary:
            if seclabel is None:
                seclabel = '$\\mathrm{Intensity\\,[a.u.]}$'
            else:
                seclabel = mklabel(seclabel)
            tx, ty, tdy = [], [], []
            data =  secondary[j]
            if data is not None:
                # plot average over all points
                for meas in data:
                    tx.append(meas.varvalue)
                    my, mdy = meas.as_arrays()[1:3]
                    ty.append(np.average(my) * 1000)
                    tdy.append(np.average(mdy) * 1000)
                if critical:
                    tx = map(lambda v: v - critical, tx)
                if xtransform:
                    if isinstance(xtransform, list):
                        trf = xtransform[j]
                    else:
                        trf = xtransform
                    tx = map(trf, tx)
                color = cycle2.next()
                if secspline and len(tx) > 3:
                    twax.errorbar(tx, ty, tdy, fmt=color+'h')
                    splx = np.linspace(tx[0], tx[-1], 100)
                    sply = splev(splx, splrep(tx, ty))
                    twax.plot(splx, sply, color+'--')
                else:
                    twax.errorbar(tx, ty, tdy, fmt=color+'h--',
                                  mfc='white', mec=color,
                                  label=mktitle(title) + ' (intensity)')
                twax.axhline(y=0, color='#cccccc', zorder=-1)
                if seclogscale:
                    twax.set_yscale('log')
                twaxes.append(twax)
                twylimits.append(twax.get_ylim())
                data_array.append((None, tx, ty, tdy))
            else:
                data_array.append(None)
        else:
            ax.axhline(y=0, color='#cccccc')
            data_array.append(None)

        if xlogscale:
            ax.set_xscale('log')
            if twax:
                twax.set_xscale('log')
        if xlim:
            ax.set_xlim(xlim)
        else:
            xmin = min(x + ex)
            xmax = max(x + ex)
            try:
                lspace = abs(x[0]-x[1])/2
                rspace = abs(x[-1]-x[-2])/2
            except IndexError:
                ax.set_xlim(xmin-0.1, xmax+0.1)
            else:
                ax.set_xlim(xmin-lspace, xmax+rspace)
        if xlabel is not None:
            ax.set_xlabel(mklabel(xlabel), size=textsize)
        elif critical:
            ax.set_xlabel('$%s-%s_c\\,[\\mathrm{%s}]$' % (
                meas.data.variable, meas.data.variable, meas.data.unit),
                          size=textsize)
        else:
            ax.set_xlabel('$%s\\,[\\mathrm{%s}]$' % (meas.data.variable,
                                                     meas.data.unit),
                          size=textsize)
        pl.xticks(size=ticksize, verticalalignment='bottom', y=-0.08)
        pl.yticks(size=ticksize)
        if subplots:
            if j == 0:
                # first plot
                ax.set_ylabel('$\\Delta\\Gamma\\,[\\mu\\mathrm{eV}]$', size=textsize,
                              color='blue')
                if twax:
                    if nplots == 1:
                        twax.set_ylabel(seclabel, size=textsize, color='red')
                    else:
                        for t in twax.yaxis.majorTicks + twax.yaxis.minorTicks:
                            t.label2On = False
            elif j == nplots - 1:
                # last plot
                if twax:
                    # put only ticklabels on secondary axis
                    for t in ax.yaxis.majorTicks + ax.yaxis.minorTicks:
                        t.label1On = False
                    twax.set_ylabel(seclabel, size=textsize, color='red')
                else:
                    # put ticklabels on right side (only for > 1 plot)
                    ax.yaxis.set_ticks_position('right')
                    for t in ax.yaxis.majorTicks + ax.yaxis.minorTicks:
                        t.tick1On = True
                    pl.yticks(size=ticksize)
            else:
                # middle plots: no ticklabels
                for t in ax.yaxis.majorTicks + ax.yaxis.minorTicks:
                    t.label1On = False
                if twax:
                    for t in twax.yaxis.majorTicks + twax.yaxis.minorTicks:
                        t.label2On = False

            titlecoords = (0.9, 0.92)
            if lefttitle:
                titlecoords = (0.1, 0.92)
            pl.text(titlecoords[0], titlecoords[1],
                    mktitle(title), size=textsize,
                    horizontalalignment=lefttitle and 'left' or 'right',
                    verticalalignment='top',
                    transform=pl.gca().transAxes)
        else:
            ax.set_ylabel('$\\Delta\\Gamma\\,[\\mu\\mathrm{eV}]$', size=textsize)
            if twax:
                twax.set_ylabel(seclabel, size=textsize)

    # make the Y scale equal for all plots
    def _scale_equal(axes, ylimits):
        yminmin = ylimits[0][0]
        ymaxmax = ylimits[0][1]
        for ymin, ymax in ylimits[1:]:
            yminmin = min(ymin, yminmin)
            ymaxmax = max(ymax, ymaxmax)
        for ax in axes:
            ax.set_ylim(0, ymaxmax)

    _scale_equal(axes, ylimits)
    if twaxes:
        _scale_equal(twaxes, twylimits)

    if ylims is not None:
        for ax, twax, ylim in zip(axes, twaxes, ylims):
            if ylim is None:
                continue
            relimit(ax, twax, *ylim)

    if legend:
        axes[0].legend(loc='upper left', bbox_to_anchor=(0, -0.15), fancybox=None)
        twaxes[0].legend(loc='upper right', bbox_to_anchor=(1, -0.15), fancybox=None)

    if filename is not None:
        fig.savefig(filename)
        dprint('Wrote', title or 'gammaplot', 'to', filename)

    if datafile is not None:
        f = open(datafile, 'w')
        for item in data_array:
            if not item: continue
            title, x, y, dy = item
            if title:
                f.write('# MIEZE measurement %s\n' % (title,))
            else:
                f.write('# Secondary data\n')
            for w in zip(x, y, dy):
                f.write('%.4f\t%.4f\t%.6f\n' % w)
            f.write('\n\n')
        f.close()

    return axes, twaxes, fitresults


def mklabel(label):
    if isinstance(label, tuple):
        return r'$%s\,[\mathrm{%s}]$' % label
    else:
        return label

def mktitle(label):
    if isinstance(label, tuple):
        return r'$%s=%s\,\mathrm{%s}$' % label
    else:
        return label


def relimit(ax, twax, ymin=None, ymax=None, ymin2=None, ymax2=None):
    axmin, axmax = ax.get_ylim()
    twmin, twmax = twax.get_ylim()
    scale = twmax/axmax

    if ymin:
        ax.set_ylim(ymin=ymin)
        twax.set_ylim(ymin=ymin*scale)
    if ymax:
        ax.set_ylim(ymax=ymax)
        twax.set_ylim(ymax=ymax*scale)
    if ymin2:
        twax.set_ylim(ymin=ymin2)
    if ymax2:
        twax.set_ylim(ymax=ymax2)


def miezeplot(filenames, infos):
    """Create a plot of 1 to 3 single MIEZE curves."""
    from miezfit import model_miez_signal

    if not isinstance(filenames, list):
        filenames = [filenames, None, None]
        infos = [infos, None, None]

    def plotcurve(name, filename, info, ax):
        varvalues, setting, preset, counts, params, errors, monitor = info
        A, B, phi, C = params
        dA, dB, dphi, dC = errors

        ax.set_title('%s: %s\n' % (name, filename) +
                     r'$C = %.2f \pm %.2f$, ' % (C, dC) +
                     r'$A = %.2f \pm %.2f$, ' % (A, dA) +
                     r'$B = %.2f \pm %.2f$, ' % (B, dB) +
                     r'$\Sigma = %s$, $t = %s$, ' % (sum(counts), preset) +
                     r'$%.2f\ \rm cps$' % (sum(counts)/float(preset)))
        ax.set_ylabel('counts')
        ax.errorbar(np.arange(1, 17), counts, np.sqrt(counts), fmt='ro')

        xs = np.arange(0, 16, 0.1)
        ys = model_miez_signal([A, B, phi], xs)
        ax.plot(xs, ys, 'b-')
        ax.set_ylim(ymin=0)

    fig = figure(figsize=(9, 13))
    if filenames[1] and filenames[2]:
        ax = fig.add_subplot(311)
        plotcurve('Measurement', filenames[0], infos[0], ax)
        ax = fig.add_subplot(312)
        plotcurve('Normalization', filenames[1], infos[1], ax)
        ax = fig.add_subplot(313)
        plotcurve('Background', filenames[2], infos[2], ax)
    elif filenames[1]:
        ax = fig.add_subplot(211)
        plotcurve('Measurement', filenames[0], infos[0], ax)
        ax = fig.add_subplot(212)
        plotcurve('Normalization', filenames[1], infos[1], ax)
    elif filenames[2]:
        ax = fig.add_subplot(211)
        plotcurve('Measurement', filenames[0], infos[0], ax)
        ax = fig.add_subplot(212)
        plotcurve('Background', filenames[2], infos[2], ax)
    else:
        ax = fig.gca()
        plotcurve('Measurement', filenames[0], infos[0], ax)
    fig.subplots_adjust(hspace=0.4, bottom=0.05)
    fig.show()


class MiezeDataPlot(object):
    """A plot of several MIEZE measurements."""

    def __init__(self, data):
        self.data = data
        self.name = self.data.name
        self.collections = weakref.WeakKeyDictionary()

    def plot_data(self, filename=None, title=None, legend=False, **kwds):
        fig = figure(title or self.name, titley=0.98)
        ret = self.plot(fig, **kwds)
        if legend:
            fig.gca().legend(loc=(1,0))
            fig.subplots_adjust(right=0.8)
        fig.canvas.mpl_connect('pick_event', self.on_pick)
        if filename is not None:
            fig.savefig(filename)
            dprint('Wrote', title or self.name, 'to', filename)
        return ret

    MARKERS = ['o', '^', 's', 'D', 'v']

    def plot(self, fig=None, fit=True, color=None, ylabel=None, log=None,
             bygroup=True, subplots=True, lines=False, data=None, ycol='C',
             **kwds):
        # optional parameters
        if data is None:
            data = self.data.get_data(ycol=ycol, **kwds)
        if fig is None:
            fig = figure()
        if log is None:
            log = ycol == 'C'

        # calculate the number of subplot rows and columns
        if subplots:
            fig.subplots_adjust(wspace=0.3, hspace=0.3)
            ncols = len(data) >= 9 and 3 or 2
            nrows = np.ceil(len(data)/float(ncols))

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
                    ax.set_ylabel(ycol + (self.data.norm and ' (norm)' or ''))

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
            if not fit or ycol != 'C':
                continue
            res = meas.fit(self.name, xmin=0)
            if not res:
                continue

            ax.plot(res.curve_x, res.curve_y, 'm-', label='exp. fit')
            text = r'$\Delta\Gamma = %s \pm %s\,\mathrm{\mu eV}$' % \
                   (format_tex(res.Gamma, 2), format_tex(res.dGamma, 2))
            # display the Gamma value as text
            if res.Gamma > 0:
                ax.text(0.03, 0.03, text, size='large', transform=ax.transAxes)
            if log:
                #ax.set_yscale('log')
                #ax.set_ylim(ymin=1e-1, ymax=2)
                ax.set_xscale('log')
            else:
                ax.set_ylim(ymin=0)
        return data

    def on_pick(self, event):
        """Matplotlib event handler for clicking on a data point."""
        npoint = event.ind[0]
        collection = event.artist
        if collection not in self.collections:
            return
        filenames = list(self.collections[collection][npoint])
        infos = map(read_single, filenames)
        miezeplot(filenames, infos)

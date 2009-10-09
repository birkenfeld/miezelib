# -*- coding: utf-8 -*-
"""
    miezelib: miezfit
    ~~~~~~~~~~~~~~~~~

    Fitting routines for general and MIEZE data.

    :copyright: 2008-2009 by Georg Brandl.
    :license: BSD.
"""

import sys
import getopt

from numpy import array, arange, sqrt, sin, pi, power, linspace
from scipy.odr import RealData, Model, ODR
from scipy.optimize import leastsq

from miezutil import dprint
from miezplot import figure


# -- fitting models ------------------------------------------------------------

def model_miez_signal(beta, x):
    return beta[1] + beta[0]*sin(4*pi/16 * x + beta[2])

def model_miez_signal_asym(beta, x):
    return beta[1] + beta[0]*sin(4*pi/16 * x + beta[2]) \
           * (1 + beta[3]*sin(4*pi/64 * x + beta[4]))

odr_model_miez_signal = Model(model_miez_signal)
odr_model_miez_signal_asym = Model(model_miez_signal_asym)


# -- fitting models ------------------------------------------------------------

def plotinfo(filename, pts, info1, info2=None):
    from matplotlib import pyplot as plt

    fig = figure()
    ax = plt.gca()

    params, errors = info1

    title = ('%s\n' % (filename) +
             r'$C = %.2f \pm %.2f$, ' % (params['C'], errors['C']) +
             r'$A = %.2f \pm %.2f$, ' % (params['A'], errors['A']) +
             r'$B = %.2f \pm %.2f$, ' % (params['B'], errors['B']))
    if info2 is not None:
        params2, errors2 = info2
        title += ('\nAsym: ' +
                  r'$C = %.2f \pm %.2f$, ' % (params2['C'], errors2['C']) +
                  r'$A = %.2f \pm %.2f$, ' % (params2['A'], errors2['A']) +
                  r'$B = %.2f \pm %.2f$, ' % (params2['B'], errors2['B']) +
                  r'$D = %.2f \pm %.2f$, ' % (params2['D'], errors2['D']))
    ax.set_title(title)
    ax.set_ylabel('counts')
    ax.errorbar(arange(1, 17), pts, sqrt(pts), fmt='ro')

    xs = arange(0, 16, 0.1)
    ys = model_miez_signal((params['A'], params['B'], params['phi']), xs)
    ax.plot(xs, ys, 'b-')
    if info2 is not None:
        ys2 = model_miez_signal_asym((params2['A'], params2['B'], params2['phi'],
                                      params2['D'], params2['chi']), xs)
        ax.plot(xs, ys2, 'g-')
    ax.set_ylim(ymin=0)

def mieze_fit(data, asym=False, addup=False):
    # for the moment, only fit points 2 to 15
    x = arange(2, len(data))
    y = array(data[1:-1])

    if addup:
        x = x[0:6]
        y = (y[:6] + y[8:]) / 2

    dat = RealData(x, y, sy=sqrt(y))
    est_A = (y.max() - y.min())/2
    est_B = (y.max() + y.min())/2
    # the first maximum is at pi/2 - k*x; a value of zero seems to have
    # bad effects on fitting in some cases
    est_phi = pi/2 - 4*pi/16*y.argmin() or 0.01
    beta0 = [est_A, est_B, est_phi]
    parnames = ['A', 'B', 'phi']
    model = odr_model_miez_signal
    if asym:
        parnames.append('D')
        parnames.append('chi')
        beta0.append(0)
        beta0.append(0)
        model = odr_model_miez_signal_asym
    odr = ODR(dat, model, beta0=beta0, ifixx=array([0]*len(x)))
    out = odr.run()
    params = dict(zip(parnames, out.beta))
    errors = dict(zip(parnames, out.sd_beta))

    if 'D' not in params:
        params['D'] = 0
        errors['D'] = 0
        params['chi'] = 0
        errors['chi'] = 0

    # make A always positive
    if params['A'] < 0:
        params['A'] = -params['A']
        params['phi'] += pi

    # make phases unique: in range [0...2pi]
    params['phi'] %= 2*pi  # this converts negative phase correctly too!
    params['chi'] %= 2*pi

    # append C = A/B
    params['C'] = params['A']/params['B']

    # dito for error, add errors for A and B squared
    errors['C'] = params['C'] * sqrt((errors['A']/params['A'])**2 +
                                     (errors['B']/params['B'])**2)
    #errors.append(errors[0]/params[1] + errors[1]*params[0]/params[1]**2)

    return params, errors

def pformat((params, errors)):
    pnames = ('A', 'B', 'phi', 'C', 'D', 'chi')
    return ' '.join((('%s=%%(%s).9g' % (i, i)) % params).ljust(12)
                    for i in pnames) + \
           ' || ' + \
           ' '.join((('d%s=%%(%s).7g' % (i, i)) % errors).ljust(10)
                    for i in pnames)


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
                 xmin=None, xmax=None, allow_leastsq=True):
        self.model = model
        self.parnames = parnames or []
        self.parstart = parstart or []
        self.xmin = xmin
        self.xmax = xmax
        self.allow_leastsq = allow_leastsq
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
                if not self.allow_leastsq:
                    raise TypeError
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


def fit_main(args):
    from miezdata import read_single
    opts, args = getopt.getopt(args, 'pas')
    opts = dict(opts)

    plotting = '-p' in opts
    asym = '-a' in opts
    addup = '-s' in opts

    for fname in args:
        pts = read_single(fname)[3]
        if asym:
            mfit1 = mieze_fit(pts, asym=False)
            mfit2 = mieze_fit(pts, asym=True)
            print fname+'[s]:', pformat(mfit1)
            print fname+'[a]:', pformat(mfit2)
            if plotting:
                plotinfo(fname, pts, mfit1, mfit2)
        elif addup:
            mfit1 = mieze_fit(pts, addup=False)
            mfit2 = mieze_fit(pts, addup=True)
            print fname+'[s]:', pformat(mfit1)
            print fname+'[a]:', pformat(mfit2)
            if plotting:
                plotinfo(fname, pts, mfit1, mfit2)
        else:
            mfit = mieze_fit(pts)
            print fname+':', pformat(mfit)
            if plotting:
                plotinfo(fname, pts, mfit)

    if plotting:
        from matplotlib import pyplot as plt
        plt.show()


if __name__ == '__main__':
    fit_main(sys.argv[1:])

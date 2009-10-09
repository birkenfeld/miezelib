import sys
import getopt

from numpy import array, arange, sqrt, sin, pi
from scipy.odr import RealData, Model, ODR

def _miez_signal(beta, x):
    return beta[1] + beta[0]*sin(4*pi/16 * x + beta[2])
_miez_fit_model = Model(_miez_signal)

def _miez_signal_asym(beta, x):
    return beta[1] + beta[0]*sin(4*pi/16 * x + beta[2]) \
           * (1 + beta[3]*sin(4*pi/64 * x + beta[4]))
_miez_fit_model_asym = Model(_miez_signal_asym)

def plotinfo(filename, pts, info1, info2=None):
    from matplotlib import pyplot as plt

    plt.figure()
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
    ys = _miez_signal((params['A'], params['B'], params['phi']), xs)
    ax.plot(xs, ys, 'b-')
    if info2 is not None:
        ys2 = _miez_signal_asym((params2['A'], params2['B'], params2['phi'],
                                 params2['D'], params2['chi']), xs)
        ax.plot(xs, ys2, 'g-')
    ax.set_ylim(ymin=0)

def mieze_fit(data, asym=False):
    # for the moment, only fit points 2 to 15
    x = arange(2, len(data))
    y = array(data[1:-1])

    dat = RealData(x, y, sy=sqrt(y))
    est_A = (y.max() - y.min())/2
    est_B = (y.max() + y.min())/2
    # the first maximum is at pi/2 - k*x; a value of zero seems to have
    # bad effects on fitting in some cases
    est_phi = pi/2 - 4*pi/16*y.argmin() or 0.01
    beta0 = [est_A, est_B, est_phi]
    parnames = ['A', 'B', 'phi']
    model = _miez_fit_model
    if asym:
        parnames.append('D')
        parnames.append('chi')
        beta0.append(0)
        beta0.append(0)
        model = _miez_fit_model_asym
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

def read_measurement(fname):
    counts = []
    for line in file(fname):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        counts.append(int(line))
    return counts


def main(args):
    opts, args = getopt.getopt(args, 'pa')
    opts = dict(opts)

    plotting = '-p' in opts
    asym = '-a' in opts

    for fname in args:
        pts = read_measurement(fname)
        if asym:
            mfit1 = mieze_fit(pts, False)
            mfit2 = mieze_fit(pts, True)
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
    main(sys.argv[1:])

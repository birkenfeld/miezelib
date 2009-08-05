from numpy import array, arange, sqrt, sin, pi
from scipy.odr import RealData, Model, ODR

def _miez_signal(beta, x):
    return beta[0]*sin(4*pi/16 * x + beta[2]) + beta[1]
_miez_fit_model = Model(_miez_signal)

def mieze_fit(data):
    # for the moment, only fit points 2 to 15
    x = arange(2, len(data))
    y = array(data[1:-1])

    dat = RealData(x, y, sy=sqrt(y))
    est_A = (y.max() - y.min())/2
    est_B = (y.max() + y.min())/2
    # the first maximum is at pi/2 - k*x; a value of zero seems to have
    # bad effects on fitting in some cases
    est_phi = pi/2 - 4*pi/16*y.argmin() or 0.01
    odr = ODR(dat, _miez_fit_model, beta0=[est_A, est_B, est_phi],
              # X values are fixed!
              ifixx=array([0]*len(x)))
    out = odr.run()
    params = list(out.beta)
    errors = list(out.sd_beta)

    # make A always positive
    if params[0] < 0:
        params[0] = -params[0]
        params[2] += pi

    # make phase unique: in range [0...2pi]
    params[2] %= 2*pi  # this converts negative phase correctly too!

    # append C = A/B
    params.append(params[0]/params[1])

    # dito for error, add errors for A and B squared
    errors.append(params[-1] * sqrt((errors[0]/params[0])**2 +
                                    (errors[1]/params[1])**2))
    #errors.append(errors[0]/params[1] + errors[1]*params[0]/params[1]**2)

    return params, errors

def pformat((params, errors)):
    pnames = ('A', 'B', 'phi', 'C')
    return ' '.join('%s=%s' % i for i in zip(pnames, params)) + \
           '; ' + ' '.join('d%s=%s' % i for i in zip(pnames, errors))

def read_measurement(fname):
    counts = []
    for line in file(fname):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        counts.append(int(line))
    return counts

import sys
for fname in sys.argv[1:]:
    print fname+':', pformat(mieze_fit(read_measurement(fname)))

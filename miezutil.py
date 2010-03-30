# -*- coding: utf-8 -*-
"""
    miezelib: miezutil
    ~~~~~~~~~~~~~~~~~~

    Utility routines for miezelib.

    :copyright: 2008-2010 by Georg Brandl.
    :license: BSD.
"""

import re
import sys
import getopt

DEBUG = False
NOPLOT = False

def setdebug(debug=True):
    global DEBUG
    DEBUG = debug

def dprint(*args):
    if DEBUG:
        for arg in args: print arg,
        print

def setnoplot(np=True):
    global NOPLOT
    NOPLOT = np

def cmdline():
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'ndfxbNDFXB')
    except getopt.error:
        print 'Invalid options given.'
        sys.exit(1)
    for opt, arg in opts:
        if opt == '-n':
            setnoplot(True)
        elif opt == '-N':
            setnoplot(False)
        elif opt == '-d':
            setdebug(True)
        elif opt == '-D':
            setdebug(False)
        elif opt == '-f':
            from miezdata import setfreefit
            setfreefit(True)
        elif opt == '-F':
            from miezdata import setfreefit
            setfreefit(False)
        elif opt == '-x':
            from miezdata import setaltfit
            setaltfit(True)
        elif opt == '-X':
            from miezdata import setaltfit
            setaltfit(False)
        elif opt == '-b':
            from miezdata import setflatback
            setflatback(True)
        elif opt == '-B':
            from miezdata import setflatback
            setflatback(False)
    sys.argv[1:] = args


def format_tex(val, prec):
    """Format a number for TeX display."""
    num = '%.*g' % (prec, val)
    if 'e' in num:
        num = num.replace('e-0', 'e-')
        num = num.replace('e', '\\cdot 10^{') + '}'
    return num


M_N = 1.6749e-27
H   = 6.6261e-34
PI  = 3.1415926
prefactor = M_N**2 / (PI * H**2)

def mieze_time(lam, L_s, setting):
    f1, f2, bs = re.match(r'([\dp]+)_([\dp]+)(_BS)?', setting).groups()
    f1 = float(f1.replace('p', '.')) * 1000  # in kHz
    f2 = float(f2.replace('p', '.')) * 1000  # in kHz
    dOmega = (f2 - f1) * 2 * PI
    if bs: dOmega *= 2
    tau = (prefactor * lam**3 * dOmega * L_s) * 1e12  # in ps
    return tau


def pformat(pnames, (params, errors)):
    return (
        ' '.join((('%s=%%(%s).9g' % (i,i)) % params).ljust(12) for i in pnames) +
        ' || ' +
        ' '.join((('d%s=%%(%s).7g' % (i,i)) % errors).ljust(10) for i in pnames))

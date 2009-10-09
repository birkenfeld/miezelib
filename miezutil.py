# -*- coding: utf-8 -*-
"""
    miezelib: miezutil
    ~~~~~~~~~~~~~~~~~~

    Utility routines for miezelib.

    :copyright: 2008-2009 by Georg Brandl.
    :license: BSD.
"""

import re


DEBUG = False

def debug(debug=True):
    global DEBUG
    DEBUG = debug

def dprint(*args):
    if DEBUG:
        for arg in args: print arg,
        print


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

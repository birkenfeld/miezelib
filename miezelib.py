# -*- coding: utf-8 -*-
"""
    miezelib: miezelib
    ~~~~~~~~~~~~~~~~~~

    Convenience interface.

    :copyright: 2008-2010 by Georg Brandl.
    :license: BSD.
"""

import numpy as np
import scipy as sp

import miezdata, miezfit, miezutil, miezplot

ml_setdatadir = miezdata.setdatadir
ml_setfreefit = miezdata.setfreefit
ml_setipars = miezdata.setipars
ml_setdebug = miezutil.setdebug
ml_setnoplot = ml_noplot = miezutil.setnoplot
ml_cmdline = miezutil.cmdline
ml_mieze_time = miezutil.mieze_time
ml_figure = miezplot.figure
ml_gammaplot = miezplot.gammaplot
ml_show = miezplot.show

MiezeData = miezdata.MiezeData
MiezeDataNF = miezdata.MiezeDataNF
Fit = miezfit.Fit
pl = miezplot.pl

__all__ = ['MiezeData', 'MiezeDataNF', 'Fit', 'np', 'sp', 'pl',
           'ml_mieze_time', 'ml_setdebug', 'ml_setdatadir', 'ml_figure',
           'ml_gammaplot', 'ml_show', 'ml_setnoplot', 'ml_setfreefit',
           'ml_setipars', 'ml_noplot', 'ml_cmdline']

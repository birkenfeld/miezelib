# -*- coding: utf-8 -*-
"""
    miezelib: miezelib
    ~~~~~~~~~~~~~~~~~~

    Compatibility interface.

    :copyright: 2008-2009 by Georg Brandl.
    :license: BSD.
"""

import numpy as np
import scipy as sp

import miezdata, miezfit, miezutil, miezplot

ml_debug = miezutil.debug
ml_mieze_time = miezutil.mieze_time
ml_setdatadir = miezdata.setdatadir
ml_figure = miezplot.figure
ml_gammaplot = miezplot.gammaplot
ml_show = miezplot.show
ml_noplot = miezutil.noplot

MiezeData = miezdata.MiezeData
MiezeDataNF = miezdata.MiezeDataNF
Fit = miezfit.Fit
pl = miezplot.pl

__all__ = ['MiezeData', 'MiezeDataNF', 'Fit', 'np', 'sp', 'pl',
           'ml_mieze_time', 'ml_debug', 'ml_setdatadir', 'ml_figure',
           'ml_gammaplot', 'ml_show', 'ml_noplot']

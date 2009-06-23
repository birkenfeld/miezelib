import sys, os, datetime
import itertools
from os import path

from pylab import *
from numpy import *

try:
    base = sys.argv[1]
except:
    print 'Usage: plot_pickups.py directory'
    print '  (all files and directories under "directory" are searched)'
    sys.exit()

pas = {}

for root, dirs, files in os.walk(base):
    for file in files:
        fname = path.join(root, file)
        print 'Reading', fname
        setting = pa1 = pa2 = started = None
        for line in open(fname):
            if line.startswith('# MIEZE scan, started'):
                started = datetime.datetime.strptime(
                    ' '.join(line.split()[-2:]), '%Y-%m-%d %H:%M:%S')
            elif line.startswith('# pickup_a1'):
                pa1 = float(line.split()[-1])
            elif line.startswith('# pickup_a2'):
                pa2 = float(line.split()[-1])
            elif line.startswith('# MIEZE setting'):
                setting = line.split()[-1]
                break
        if not (started and setting and pa1 and pa2):
            continue
        lsts = pas.setdefault(setting, ([], [], []))
        lsts[0].append(started)
        lsts[1].append(pa1)
        lsts[2].append(pa2)

print 'Plotting...'


markers = itertools.cycle('ov^s*')
colors = itertools.cycle('bgrcmk')
for setting in pas:
    if not setting[0].isdigit(): continue
    lsts = pas[setting]
    title('coil 1')
    xs = date2num(lsts[0])
    m, c = markers.next(), colors.next()
    plot_date(xs, lsts[1], c+m, label=setting)
legend()

figure()
markers = itertools.cycle('ov^s*')
colors = itertools.cycle('bgrcmk')
for setting in pas:
    if not setting[0].isdigit(): continue
    lsts = pas[setting]
    title('coil 2')
    xs = date2num(lsts[0])
    m, c = markers.next(), colors.next()
    plot_date(xs, lsts[2], c+m, label=setting)
legend()

show()
    

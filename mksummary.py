import sys, re

rex = re.compile(
r'''\s*# t/s: (?P<preset>\d+)\s*
# sum: (?P<countsum>\d+)\s*
# mon: (?P<monitor>\d+)\s*
# A:   (?P<A>[0-9.]+)\s+\+/-\s+(?P<dA>[0-9.]+)
# B:   (?P<B>[0-9.]+)\s+\+/-\s+(?P<dB>[0-9.]+)
# phi: (?P<phi>[0-9.]+)\s+\+/-\s+(?P<dphi>[0-9.]+)
# C:   (?P<C>[0-9.]+)\s+\+/-\s+(?P<dC>[0-9.]+)\s*''')

settings = ['46_69', '72_108', '99_148p5', '138_207', '200_300']


outname = sys.argv[1]
varval = sys.argv[2]
setting = sys.argv[3]
if len(setting) == 1:
    setting = settings[int(setting)]
fname = '%05d' % int(sys.argv[4])

f = open(outname, 'a')

m = rex.match(sys.stdin.read())
values = m.groupdict().copy()
values['varval'] = varval
values['setting'] = setting
values['tau'] = '-'
values['fname'] = fname

fieldorder = ['varval', 'varval', 'setting', 'tau', 'preset', 'countsum',
              'monitor', 'A', 'B', 'phi', 'C', 'dA', 'dB', 'dphi', 'dC', 'fname']
f.write(' ' + ' '.join(values[field].rjust(10) for field in fieldorder) + '\n')

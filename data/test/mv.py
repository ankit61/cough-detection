import os

suffix = '-converted.mp4'

for f in os.listdir('./'):
    if f.endswith(suffix):
        os.system('mv ' + f + ' ' + f[:-len(suffix)] + '.mp4')

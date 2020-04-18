import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plotfiles = []

plotfiles.append({  'filename': '10000_80_v1_metrics.csv', 
                    'label':    '80 new samples per iter',
                    'color':    'b',
                    'scale':    80
                }) 
                
plotfiles.append({  'filename': '10000_100_v1_metrics.csv', 
                    'label':    '100 new samples per iter',
                    'color':    'g',
                    'scale':    100
                }) 
                                
plotfiles.append({  'filename': '10000_200_v1_metrics.csv', 
                    'label':    '200 new samples per iter',
                    'color':    'k',
                    'scale':    200
                }) 
                
args = []
args.append({   'name':     'MAE',
                'style':    '-',
           })
args.append({   'name':     'S',
                'style':    '--',
           })
           

domain = 'numdata'

fig, ax = plt.subplots(1, 1)

for f in plotfiles:
    data = pd.read_csv(f['filename'])
    for arg in args:
        data.plot(x=domain, y=arg['name'], style = f['color'] + arg['style'], ax=ax, label=arg['name'] + ' -- ' + f['label'])

fig.set_size_inches(18,10)
fig.set_dpi(300)

plt.xlabel("Number of Samples")
plt.ylabel("Error")
plt.title("Varied Increment for 10000 Initial Samples")
        
plt.savefig("10000_incr_samp.png")


fig, ax = plt.subplots(1, 1)

for f in plotfiles:
    data = pd.read_csv(f['filename'])
    data[domain]  = (data[domain] - 10000) / f['scale']
    for arg in args:
        data.plot(x=domain, y=arg['name'], style = f['color'] + arg['style'], ax=ax, label=arg['name'] + ' -- ' + f['label'])

fig.set_size_inches(18,10)
fig.set_dpi(300)

plt.xlabel("Number of Iterations")
plt.ylabel("Error")
plt.title("Varied Increment for 10000 Initial Samples")        
        
plt.savefig("10000_incr_time.png")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plotfiles = []

plotfiles.append({  'filename': '1000_20_v1_metrics.csv', 
                    'label':    '20 new samples per iter',
                    'color':    'b',
                    'scale':    20
                }) 
                
plotfiles.append({  'filename': '1000_50_v1_metrics.csv', 
                    'label':    '50 new samples per iter',
                    'color':    'g',
                    'scale':    50
                }) 
                                
plotfiles.append({  'filename': '1000_200_v1_metrics.csv', 
                    'label':    '200 new samples per iter',
                    'color':    'k',
                    'scale':    200
                }) 
                                
plotfiles.append({  'filename': '1000_500_v1_metrics.csv', 
                    'label':    '500 new samples per iter',
                    'color':    'r',
                    'scale':    500
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
plt.title("Varied Increment for 1000 Initial Samples")
        
plt.savefig("1000_incr_samp.png")


fig, ax = plt.subplots(1, 1)

for f in plotfiles:
    data = pd.read_csv(f['filename'])
    data[domain]  = (data[domain] - 1000) / f['scale']
    for arg in args:
        data.plot(x=domain, y=arg['name'], style = f['color'] + arg['style'], ax=ax, label=arg['name'] + ' -- ' + f['label'])

fig.set_size_inches(18,10)
fig.set_dpi(300)

plt.xlabel("Number of Iterations")
plt.ylabel("Error")
plt.title("Varied Increment for 1000 Initial Samples")        
        
plt.savefig("1000_incr_time.png")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plotfiles = []

plotfiles.append({  'filename': '1000_50_v1_retrain0_metrics.csv', 
                    'label':    'no retraining',
                    'color':    'b',
                    'scale':    50
                }) 
                
plotfiles.append({  'filename': '1000_50_v1_retrain5_metrics.csv', 
                    'label':    'retrain every 5th iteration',
                    'color':    'k',
                    'scale':    50
                }) 

plotfiles.append({  'filename': '1000_50_v1_retrain20_metrics.csv', 
                    'label':    'retrain every 20th iteration',
                    'color':    'r',
                    'scale':    50
                }) 
                                
                
args = []
args.append({   'name':     'similarity',
                'style':    '-',
           })

           

domain = 'numdata'

fig, ax = plt.subplots(1, 1)

for f in plotfiles:
    data = pd.read_csv(f['filename'])
    data = data.head(150)
    for arg in args:
        data.plot(x=domain, y=arg['name'], style = f['color'] + arg['style'], ax=ax, label=arg['name'] + ' -- ' + f['label'])

fig.set_size_inches(18,10)
fig.set_dpi(300)

plt.xlabel("Number of Samples")
plt.ylabel("Similarity")
plt.title("Varied Retraining for 1000 Initial, 50 Incremental Samples")
        
plt.savefig("1000_retrain_sim.png")


# %%
import numpy as np
import matplotlib.pyplot as plt
import time
from margarine.maf import MAF
from anesthetic import MCMCSamples, read_chains, make_2d_axes


# %%
start_time = time.time()

dir1 = 'bao.sdss_dr16'
samples1 = read_chains(f"../ns/klcdm/{dir1}/{dir1}_polychord_raw/{dir1}") 
#fig, axes = make_2d_axes(['ns', 'omk', 'H0', 'ombh2', 'omch2', 'tau']) 
fig, axes = make_2d_axes(['omk', 'H0', 'ombh2', 'omch2']) 
samples1.plot_2d(axes, label=f'{dir1}')


#fig, axes = make_2d_axes(['omk', 'H0', 'omegam']) 
#samples1.plot_2d(axes, label=f'{dir1}')
#%%
data = samples1.values[:, 2:6]
weights = samples1.get_weights()
flow = MAF(data, weights=weights)
flow.train(1000, early_stop=False)

flow.save(f'{dir1}_flow.pkl')

#%%
flow = MAF.load(f'{dir1}_flow.pkl')
y = flow.sample(5000)
#%%
samples_flow = MCMCSamples(data=y)
samples_flow.plot_2d()
plt.show()

print("Process finished --- %s seconds ---" % (time.time() - start_time))
# %%




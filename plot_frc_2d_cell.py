import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import h5py
import os
from util import *
import dxchange
import numpy as np


params_2d_cell = {'grid_delta': np.load('cell/phantom/grid_delta.npy'),
                  'compression_mode': 2,
                  'obj':[],
                  'ref':[],
                  'nei':'500',
                  'save_path': '../../InPixelAI/outputs_PhoCount2/comparison',
#                  'save_path': 'cell/ptychography/comparison',
                  'fig_ax':[],
                  'radius_ls':[],
                  'nei_intersection_ls':[],
                  'radius_intersection_ls':[],
                  'T_half_bit_ls':[],
                  'show_plot_title': True,
                  'plot_T_half_th': True,
                  'save_mask': False,
                  }
params = params_2d_cell

print('dimension of the sample = ' +', '.join(map(str, params['grid_delta'].shape)))
n_sample_pixel = np.count_nonzero(params['grid_delta']> 1e-10)
print('n_sample_pixel = %d' %n_sample_pixel)
print('finite support area ratio in sample = %.3f' %(n_sample_pixel/(params['grid_delta'].shape[0]*params['grid_delta'].shape[1])))

if params['compression_mode'] == 0: compression_mode = 'normal'
if params['compression_mode'] == 1: compression_mode = 'PCA_compressed'
if params['compression_mode'] == 2: compression_mode = 'AE_compressed'


matplotlib.rcParams['pdf.fonttype'] = 'truetype'
fontProperties = {'family': 'serif', 'serif': ['Helvetica'], 'weight': 'normal', 'size': 12}
plt.rc('font', **fontProperties)
#spec = gridspec.GridSpec(1, 2, width_ratios=[7, 1])
#fig = plt.figure(figsize=(8, 5))
spec = gridspec.GridSpec(1, 1)
fig = plt.figure(figsize=(6, 4))

params['fig_ax'] = fig.add_subplot(spec[0, 0])

path = os.path.dirname(params['save_path'])

if compression_mode == 'normal':
   nei_ls = ['']
elif compression_mode == 'PCA_compressed':
   nei_ls = [1, 10,  30 , 50, 100, 300, 1000, 1500, 2000, 2500, 3000, 3500]
elif compression_mode == 'AE_compressed':
   # x = [1,2,5,10,25,50,75,100]
   # nei_ls = [f"AE_72x72_Dense_512_linear_mse_{i}" for i in x]
   x = [4,5,8,10,16,20,32,40,50,60,64,100,128,256,512]
   nei_ls = [f"AE_72x72_Dense_{i}_linear_mse" for i in x]
#    nei_ls = [
# #      "AE_72x72_Dense_10_linear_mse",
# #      "AE_72x72_Dense_16_linear_mse",
# #      "AE_72x72_Dense_20_linear_mse",
#       "AE_72x72_Dense_32_linear_mse",
#       "AE_72x72_Dense_40_linear_mse",
#       "AE_72x72_Dense_50_linear_mse",
#       "AE_72x72_Dense_60_linear_mse",
#       "AE_72x72_Dense_64_linear_mse",
#       "AE_72x72_Dense_100_linear_mse",
#       "AE_72x72_Dense_128_linear_mse",
#       "AE_72x72_Dense_256_linear_mse",
#       "AE_72x72_Dense_512_linear_mse",]
# # AE_72x72_Dense_BatchNorm_512_linear_mse
# # AE_72x72_Dense_BatchNorm_512_relu_mse
# # AE_72x72_Dense_1024_128_linear_mse
# # AE_72x72_Dense_512_relu_mse
# # AE_72x72_Dense_512_relu_r_weighted
# # AE_72x72_Dense_512_relu_telescope
# # AE_72x72_Dense_512_linear_r_weighted
# # AE_72x72_Dense_512_linear_telescope

nei_intersection_ls = []
radius_intersection_ls = []

for nei in nei_ls:
    print(nei)
    params['nei'] = nei
    if compression_mode == 'normal' :
        obj_dir = os.path.join(path, 'n2e7')
        ref_dir = os.path.join(path, 'n2e7_ref')
    elif compression_mode == 'PCA_compressed' :
        obj_dir = os.path.join(path, 'n2e7_nei' + str(nei))
        ref_dir = os.path.join(path, 'n2e7_nei' + str(nei) + '_ref')
    elif compression_mode == 'AE_compressed' : 
        obj_dir = os.path.join(path, nei)
        obj_dir = os.path.join(obj_dir,nei)
        ref_dir = os.path.join(path, '../phantom/')      
    if compression_mode=='AE_compressed':
        print(obj_dir)
        params['obj'] = dxchange.read_tiff(os.path.join(obj_dir, 'delta_ds_1.tiff'))
        params['obj'] = params['obj'][:, :, 0]

        params['ref'] = dxchange.read_tiff(os.path.join(ref_dir, 'delta.tiff'))
    else:
        params['obj'] = dxchange.read_tiff(os.path.join(obj_dir, 'delta_ds_1.tiff'))
        params['obj'] = params['obj'][:, :, 0]

        params['ref'] = dxchange.read_tiff(os.path.join(ref_dir, 'delta_ds_1.tiff'))
        params['ref'] = params['ref'][:, :, 0]   

    if params['show_plot_title']: Plot_title = compression_mode
    else: Plot_title = None
    params['label'] = nei.split('_')[-3]
    nei_intersection, radius_intersection, params['radius_ls'], params['T_half_bit_ls'] = fourier_ring_correlation_PCA(**params)

    if nei_intersection != None:
        params['nei_intersection_ls'].append(nei_intersection)
        params['radius_intersection_ls'].append(radius_intersection)
    else:
        pass
    
if params['plot_T_half_th']:
    half_bit_threshold(params['fig_ax'], params['radius_ls'], params['T_half_bit_ls'])

#params['fig_ax'].legend(loc=3, bbox_to_anchor=(1.0, 0.0, 0.5, 0.5), fontsize=12, ncol=1, title='photon number')
plt.savefig(os.path.join(params['save_path'], 'frc_PCAmode'+str(params['compression_mode'])+'.pdf'), format='pdf')

fig.clear()
plt.close(fig)


np.savez(os.path.join(params['save_path'], 'frc_PCAmode'+ str(params['compression_mode'])+'_intersection'), np.array(params['nei_intersection_ls']), np.array(params['radius_intersection_ls']/params['radius_ls'][-1]))

fig = plt.figure(figsize=(8, 5))
fig_ax = fig.add_subplot(spec[0,0])
fig_ax.plot(params['nei_intersection_ls'], params['radius_intersection_ls']/params['radius_ls'][-1], '-bs', markerfacecolor='none', markeredgecolor='blue', label = compression_mode)
print(params['nei_intersection_ls'])
print(params['radius_intersection_ls'])

fig_ax.set_xlabel('n_eigenimages')
fig_ax.set_ylabel('FRC/half-bit crossing fraction')
# fig_ax.set_ylim(0,1.1)
fig_ax.set_xscale('log')
fig_ax.legend(loc=3, bbox_to_anchor=(1.0,0.0,0.5,0.5), ncol=1, title = 'data type')

plt.savefig(os.path.join(params['save_path'], 'frc_'+ str(params['compression_mode'])+'_intersection.pdf'), format='pdf', dpi=600)
fig.clear()
plt.close(fig)

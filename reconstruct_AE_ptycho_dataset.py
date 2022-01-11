from ptychography import reconstruct_ptychography
import numpy as np
import dxchange
import datetime
import argparse
import os

from plotRecoImage import plotRecoImage

timestr = str(datetime.datetime.today())
timestr = timestr[:timestr.find('.')]
for i in [':', '-', ' ']:
    if i == ' ':
        timestr = timestr.replace(i, '_')
    else:
        timestr = timestr.replace(i, '')

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default='None')
parser.add_argument('--inputDir', default='/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs')
parser.add_argument('--save_path', default='cell/ptychography')
parser.add_argument('--output_folder', default='test') # Will create epoch folders under this
parser.add_argument('-m','--model', action='append') # Will create epoch folders under this
args = parser.parse_args()
epoch = args.epoch
modelList = args.model
print(modelList)
if epoch == 'None':
    epoch = 0
    init = None
else:
    epoch = int(epoch)
    if epoch == 0:
        init = None
    else:
        init_delta = dxchange.read_tiff(os.path.join(args.save_path, args.output_folder, 'epoch_{}/delta_ds_1.tiff'.format(epoch - 1)))
        init_beta = dxchange.read_tiff(os.path.join(args.save_path, args.output_folder, 'epoch_{}/beta_ds_1.tiff'.format(epoch - 1)))
        print(os.path.join(args.save_path, args.output_folder, 'epoch_{}/delta_ds_1.tiff'.format(epoch - 1)))
        init = [np.array(init_delta[...]), np.array(init_beta[...])]


params_2d_cell = {'fname': 'data_cell_phase.h5',
                    'theta_st': 0,
                    'theta_end': 0,
                    'theta_downsample': 1,
                    'n_epochs': 200,
                    'obj_size': (325, 325, 1),
                    'alpha_d': 0,
                    'alpha_b': 0,
                    'gamma': 0,
                    'probe_size': (72, 72),
                    'learning_rate': 4e-3,
                    'center': 512,
                    'energy_ev': 5000,
                    'psize_cm': 1.e-7,
#                    'minibatch_size': 2244,
                    'minibatch_size': 4488,
                    'n_batch_per_update': 1,
                    'cpu_only': True,
                    'save_path': 'cell/ptychography',
                    'multiscale_level': 1,
                    'n_epoch_final_pass': None,
                    'save_intermediate': True,
                    'full_intermediate': True,
                    # 'initial_guess': [np.zeros([325, 325, 1]) + 0.032, np.zeros([325, 325, 1])],
                    'initial_guess': None,
                    'n_dp_batch': 20,
                    'probe_type': 'gaussian',
                    'probe_mag_sigma': 6,
                    'probe_phase_sigma': 6,
                    'probe_phase_max': 0.5,
                    'forward_algorithm': 'fresnel',
                    'object_type': 'phase_only',
                    'probe_pos': [(y, x) for y in (np.arange(66) * 5) - 36 for x in (np.arange(68) * 5) - 36],
                    'finite_support_mask': None,
                    'free_prop_cm': 'inf',
                    'optimizer': 'adam',
                    'two_d_mode': True,
                    'shared_file_object': False,
                    'use_checkpoint': False
                    }

params = params_2d_cell


# n_ls = ['n2e7']
# #n_ls = ['n2e7_32x32','n2e7_32x32merged']
# n_ls = ['n2e7_32x32padded']
# #n_ls = ['AE_rWeightLoss_72x72','AE_72x72','AE_rWeightLoss_72x72_ZS','AE_72x72_ZS']

# n_ls = [# 'AE_72x72_Dense_1024_128_linear_mse',
#         # 'AE_72x72_Dense_1024_128_linear_r_weighted',
#         # 'AE_72x72_Dense_1024_128_linear_telescope',
#         # 'AE_72x72_Dense_1024_128_relu_mse',
#         # 'AE_72x72_Dense_1024_128_relu_r_weighted',
#         # 'AE_72x72_Dense_1024_128_relu_telescope',
#         # 'AE_72x72_Dense_128_linear_mse',
#         # 'AE_72x72_Dense_128_linear_r_weighted',
#         # 'AE_72x72_Dense_128_linear_telescope',
#         # 'AE_72x72_Dense_128_relu_mse',
#         # 'AE_72x72_Dense_128_relu_r_weighted',
#         # 'AE_72x72_Dense_128_relu_telescope',
#         # 'AE_72x72_Dense_256_linear_mse',
#         # 'AE_72x72_Dense_256_linear_r_weighted',
#         # 'AE_72x72_Dense_256_linear_telescope',
#         # 'AE_72x72_Dense_256_relu_mse',
#         # 'AE_72x72_Dense_256_relu_r_weighted',
#         # 'AE_72x72_Dense_256_relu_telescope',
#         # 'AE_72x72_Dense_512_linear_mse',
#         # 'AE_72x72_Dense_512_linear_r_weighted',
#         # 'AE_72x72_Dense_512_linear_telescope',
#         # 'AE_72x72_Dense_512_relu_mse',
#         # 'AE_72x72_Dense_512_relu_r_weighted',
#         # 'AE_72x72_Dense_512_relu_telescope',
#         # 'AE_72x72_Dense_BatchNorm_512_linear_mse',
#         # 'AE_72x72_Dense_BatchNorm_512_relu_mse',
#         # 'AE_72x72_Dense_100_linear_mse',
#         # 'AE_72x72_Dense_64_linear_mse',
#         # 'AE_72x72_Dense_60_linear_mse',
#         # 'AE_72x72_Dense_50_linear_mse',
#         # 'AE_72x72_Dense_40_linear_mse',
#         # 'AE_72x72_Dense_32_linear_mse',
#         'AE_72x72_Dense_30_linear_mse',
#         'AE_72x72_Dense_20_linear_mse',
#         'AE_72x72_Dense_16_linear_mse',
#         'AE_72x72_Dense_10_linear_mse',
#         'AE_72x72_Dense_8_linear_mse',
#         'AE_72x72_Dense_5_linear_mse',
#         'AE_72x72_Dense_4_linear_mse',
# ]

#params['n_epoch'] = 50
print(modelList)
for ae_model in modelList:
    params['fname'] = 'data_cell_phase_{}.h5'.format(ae_model)
    params['output_folder'] = ae_model
    if '32x32padded' in ae_model:
        params['probe_size'] = (72,72)
    if '32x32mergedpadded' in ae_model:
        params['probe_size'] = (36,36)
    if '72x72' in ae_model:
        params['probe_size'] = (72,72)

    params['save_path'] = f'{args.inputDir}/{ae_model}'

    reconstruct_ptychography(**params)

    plotRecoImage(f'{args.inputDir}/{ae_model}/{ae_model}') 

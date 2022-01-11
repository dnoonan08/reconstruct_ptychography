import numpy as np
import matplotlib.pyplot as plt

def plotRecoImage(filePath):
    try:
        x = np.load(f'{filePath}/obj_checkpoint.npy')[:,:,0,0]
        plt.imshow(x)
        trainingName = filePath.split('/')
        if trainingName[-1]=='':
            trainingName = trainingName[-2]
        else:
            trainingName = trainingName[-1]
        plt.savefig(f'{filePath}/{trainingName}_reconstruction.pdf')
    except:
        print('Unable to load file')
        print(filePath)
    return

if __name__=="__main__":


    trainings = [# '/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs/AE_72x72_Dense_1024_128_linear_mse',
                 # '/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs/AE_72x72_Dense_10_linear_mse',
                 # '/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs/AE_72x72_Dense_16_linear_mse',
                 # '/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs/AE_72x72_Dense_20_linear_mse',
                 # '/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs/AE_72x72_Dense_32_linear_mse',
                 # '/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs/AE_72x72_Dense_64_linear_mse',
                 # '/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs/AE_72x72_Dense_40_linear_mse',
                 # '/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs/AE_72x72_Dense_50_linear_mse',
                 # '/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs/AE_72x72_Dense_60_linear_mse',
                 # '/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs/AE_72x72_Dense_100_linear_mse',
                 # '/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs/AE_72x72_Dense_128_linear_mse',
                 # '/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs/AE_72x72_Dense_256_linear_mse',
                 # '/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs/AE_72x72_Dense_512_linear_mse',
                 # '/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs/AE_72x72_Dense_512_linear_r_weighted',
                 # '/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs/AE_72x72_Dense_512_linear_telescope',
                 # '/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs/AE_72x72_Dense_512_relu_mse',
                 # '/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs/AE_72x72_Dense_512_relu_r_weighted',
                 # '/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs/AE_72x72_Dense_512_relu_telescope'
                 # '/udrive/staff/dnoonan/AIinPixel/NewTest/reconstruct_ptychography/cell/ptychography/AE_72x72_Dense_512_linear_mse_1',
                 # '/udrive/staff/dnoonan/AIinPixel/NewTest/reconstruct_ptychography/cell/ptychography/AE_72x72_Dense_512_linear_mse_2',
                 # '/udrive/staff/dnoonan/AIinPixel/NewTest/reconstruct_ptychography/cell/ptychography/AE_72x72_Dense_512_linear_mse_5',
                 # '/udrive/staff/dnoonan/AIinPixel/NewTest/reconstruct_ptychography/cell/ptychography/AE_72x72_Dense_512_linear_mse_10',
                 # '/udrive/staff/dnoonan/AIinPixel/NewTest/reconstruct_ptychography/cell/ptychography/AE_72x72_Dense_512_linear_mse_25',
                 # '/udrive/staff/dnoonan/AIinPixel/NewTest/reconstruct_ptychography/cell/ptychography/AE_72x72_Dense_512_linear_mse_50',
                 #'/udrive/staff/dnoonan/AIinPixel/NewTest/reconstruct_ptychography/cell/ptychography/AE_72x72_Dense_512_linear_mse_100',
                 # '/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs/AE_72x72_Dense_4_linear_mse/AE_72x72_Dense_4_linear_mse',
                 # '/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs/AE_72x72_Dense_5_linear_mse/AE_72x72_Dense_5_linear_mse',
                 # '/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs/AE_72x72_Dense_8_linear_mse/AE_72x72_Dense_8_linear_mse',
                 # '/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs/AE_72x72_Dense_10_linear_mse/AE_72x72_Dense_10_linear_mse',
                 # '/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs/AE_72x72_Dense_16_linear_mse/AE_72x72_Dense_16_linear_mse',
                 # '/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs/AE_72x72_Dense_20_linear_mse/AE_72x72_Dense_20_linear_mse',
                 '/udrive/staff/dnoonan/AIinPixel/InPixelAI/outputs_PhoCount/AE_72x72_Dense_50_linear_mse/Jax_Result_AE_72x72_Dense_50_linear_mse',
    ]
    for fPath in trainings:
        plotRecoImage(fPath)

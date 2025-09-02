import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    tsdf_volume = np.load('./results/RIG_multi_view_perfect_perception_exp/debug/0/tsdf_volume_3.npy')

    for i in range(tsdf_volume.shape[2]):
        print(i)
        #plt.imshow(tsdf_volume[-i,:,:])
        #plt.imshow(tsdf_volume[:,i,:])
        plt.imshow(tsdf_volume[:,:,i])
        plt.show()

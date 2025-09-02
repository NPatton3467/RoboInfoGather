import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    bel = np.load('./results/RIG_multi_view_perfect_perception_exp/debug/0/bel_Chair_3.npy')

    bel_mean_0 = np.mean(bel, axis=0)
    bel_mean_1 = np.mean(bel, axis=1)
    bel_mean_2 = np.mean(bel, axis=2) # Should be BEV

    plt.imshow(bel_mean_0)
    plt.show()
    
    plt.imshow(bel_mean_1)
    plt.show()

    plt.imshow(bel_mean_2)
    plt.show()

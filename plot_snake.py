import numpy as np
import matplotlib.pyplot as plt
import imageio
import pickle
import os,shutil
from pathlib import Path

def plot_H(H):
    path_dir = 'C:\\Users\\mlagutin\\AppData\\Roaming\\JetBrains\\PyCharmCE2020.1\\scratches\\temp\\'
    for key in H.keys():
        images = reshape_H(H[key])
        l      = np.shape(images)[2]
        fps    = 0.5
        duration = 1 / fps + 3
        #kwargs   = {'fps': fps, 'duration': duration}'Training Stage: {} epochs'.format(key)
        #kwargs   = {'quantizer':'wu'}
        img_l    = save_frames(images, '', path_dir, l)

        imageio.mimsave(
            r'C:\Users\mlagutin\AppData\Roaming\JetBrains\PyCharmCE2020.1\scratches\outputs\movie{}.gif'.format(key),
            img_l,subrectangles =True,palettesize = 8)

        clear_temp(path_dir)
def reshape_H(H_s):
    # H_s is array of shape (steps,state)
    imagess = np.apply_along_axis(one_line_r, 1, H_s).tolist()
    print(np.shape(imagess))
    return imagess


def one_line_r(H_line):
    # H_line is one line from array H_s
    L = len(H_line)
    l = int(np.sqrt(L / 3))
    h_resh = H_line.reshape((3, -1))
    snake = h_resh[0, :].reshape((l, l)) * 2
    head = h_resh[1, :].reshape((l, l)) * 3
    target = h_resh[2, :].reshape((l, l)) * 4
    B = snake + head + target
    img = np.pad(B, 1, 'constant', constant_values=1)
    return img

def save_frames(images,title,path_dir,l):

    for i in np.arange(len(images)):
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(111)
        ax.set_title(title, fontsize=25,fontweight="bold")
        ax.set_aspect('equal')
        # Major ticks
        ax.set_xticks(np.arange(0, l, 1))
        ax.set_yticks(np.arange(0, l, 1))
        # Minor ticks
        ax.set_xticks(np.arange(-.5, l, 1), minor=True)
        ax.set_yticks(np.arange(-.5, l, 1), minor=True)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        # Gridlines based on minor ticks
        ax.grid(which='minor', color='white', linestyle='-', linewidth=3)
        H = images[i]
        ax.imshow(H, cmap='hot', vmin=0, vmax=4)
        plt.savefig(path_dir+'{}.jpg'.format(i),dpi = 60)
        plt.close(fig)
    frames = find_frames(path_dir, suffix=".jpg")
    print(frames)
    var = [imageio.imread(str(file)) for file in frames]
    return var

def find_frames(path_dir, suffix=".jpg" ):
    paths = sorted(Path(path_dir).iterdir(), key=os.path.getmtime)
    return paths

def clear_temp(path_dir):
    for filename in os.listdir(path_dir):
        file_path = os.path.join(path_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


H = pickle.load(open("TH_{}.p".format(147), "rb"))
plot_H(H)

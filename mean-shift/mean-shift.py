import time
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

def distance(x, X):
    distances = np.sqrt(np.sum((X-x)**2,axis=1))
    return distances

def gaussian(dist, bandwidth):
    #Gaussian coeff
    g = np.exp(-(dist/(2*bandwidth))**2)/(bandwidth*np.sqrt(2*np.pi))
    return g

def update_point(weight, X):
    new_point = X.T@weight[:,np.newaxis]
    return new_point

def meanshift_step(X, bandwidth=5):
    X_step = X.copy()
    print('new step')
    for i, xn in enumerate(X):
        dist = distance(xn,X)
        den = gaussian(dist,bandwidth)
        den_sum = sum(den)
        weight = den/den_sum
        X_step[i,:] = update_point(weight,X)[0]
        if i%1000 == 0:
            print('iteration progress:', i/len(X)*100,'%')

    return X_step

def meanshift(X):
    for _ in range(30):
        X = meanshift_step(X)
    return X

scale = 0.3    # downscale the image to run faster

# Load image and convert it to CIELAB space
img = io.imread('eth.jpg')
image = rescale(img, scale, channel_axis=-1)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
X = meanshift(image_lab)
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, channel_axis=-1)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)

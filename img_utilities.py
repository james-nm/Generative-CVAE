# Useful utilities for images
# Developed by James Thiering (Waffletron), 20191118

import numpy as np
import unittest

########## Utility Methods ##########

# Kernel generation methods

def calculate_step_dimensions(source_dimension,
                              sample_dimension,
                              requested):
  '''
  Calculates the longest step dimensions that a sliding window sampler
  would use to generate the requested number of samples.
  
  NOT IMPLEMENTED
  
  '''
  return  # Not implemented

def generate_gaussian_vector(length, sigma):
  '''
  Creates an array containing values from a Gaussian/normal distribution.
  - sigma controls the standard deviation of the distribution
  - length is the overall length, NOT from center to end
  '''
  ret = np.linspace(0,length,length)
  ret = ret-length/2
  ret = ret**2
  ret = ret/(2*sigma**2)
  ret = np.exp(-ret)
  return ret

def generate_gaussian_matrix(shape, sigma=1):
  '''
  Creates a matrix containing values from a Gaussian/normal distribution.
  '''
  ret = np.outer(generate_gaussian_vector(shape[1],sigma),
                  generate_gaussian_vector(shape[0],sigma))
  return ret

# Blending
def blend_alpha_grad(imageA, imageB, mask='Gaussian'):
  '''
  Blends two images using a Gaussian mask.
  -----
  imageA is used as the primary/background image
  imageB is overlayed with an alpha mask applied
  mask controls the blending alpha mask
  -----
  TODO:
    - implement additional mask types: linear interpolation, margin fade
    - implement additional options: single axis fade, 
  '''
  
  assert(imageA.shape == imageB.shape)
  
  size = imageA.shape
  sigma = 10  # TODO: sigma should vary on img size
    
  mask = generate_gaussian_matrix(size, sigma)  
  negative_mask = np.ones(size) - mask
  
  imageB = imageB * mask
  imageA = imageA * negative_mask
  
  ret = np.add(imageA, imageB)
  
  return ret

def image_to_tensor(image, normalize = False):
  # Converts uint8 matrix with range 0-255 to float32 matrix with range 0-1
  # Accepts an image of 2 dimensions
  # Returns a tensor of 4 dimensions
  
  ret = image[np.newaxis, :, :, np.newaxis]
  ret = ret.astype('float32')
  ret /= 255.
  if normalize:
    ret = ((ret - np.min(ret))/(np.max(ret)-np.min(ret)))
  
  return ret

def tensor_to_image(tensor, normalize = False):
  # Converts float32 matrix with range 0-1 to uint8 matrix with range 0-255
  # Accepts a tensor of 4 dimensions
  # Returns an image of 2 dimensions
  
  # TODO try type checking against TensorFlow specific classes
  if not type(tensor) == np.ndarray:
    ret = tensor.numpy()
  else:
    ret = tensor.copy()
  
  ret = ret[0, :, :, 0]
  if normalize:
    ret = ((ret - np.min(ret))/(np.max(ret)-np.min(ret)))
  ret *= 255.
  ret = ret.astype('uint8')
  
  return ret


########## Unit Test Cases ##########

class TestGaussianMethods(unittest.TestCase):
  
  def test_generate_gaussian_matrix(self):
    '''
    Test the gaussian matrix method against two pre-generated matrices.
    '''
    
    # A normal distribution of shape (5,5) sigma 1 
    expected_A = np.array([[0.002, 0.02 , 0.044, 0.02 , 0.002],
                           [0.02 , 0.21 , 0.458, 0.21 , 0.02 ],
                           [0.044, 0.458, 1.   , 0.458, 0.044],
                           [0.02 , 0.21 , 0.458, 0.21 , 0.02 ],
                           [0.002, 0.02 , 0.044, 0.02 , 0.002]])
  
    results_A = generate_gaussian_matrix((5,5), 1)
    
    # Difference between expectations and results is less than 1% deviation
    self.assertTrue(np.abs(np.subtract(expected_A, results_A)).all()<0.01)
  
    # A normal distribution of shape (3,3) sigma 1 
    expected_B = np.array([[0.105, 0.325, 0.105],
                           [0.325, 1.   , 0.325],
                           [0.105, 0.325, 0.105]])
    results_B = generate_gaussian_matrix((3,3), 1)
    
    # Difference between expectations and results is less than 1% deviation
    self.assertTrue(np.abs(np.subtract(expected_B, results_B)).all()<0.01)

class TestBlendingMethods(unittest.TestCase):
  '''
  Test the blending method against manually blended matrix
  TODO: create test images
  '''
  
  def test_blend_alpha_grad_simple(self):
    # Simple test. Compare versus generate_gaussian_matrix()
    size=(3,3)
    test_imageA = np.zeros(size)
    test_imageB = np.ones(size)
    test_mask='Gaussian'
    
    expected = generate_gaussian_matrix(size)
    results = blend_alpha_grad(test_imageA, test_imageB, test_mask)
    
    # Difference between expectations and results is less than 1% deviation
    self.assertTrue(np.abs(np.subtract(expected, results)).all()<0.01)
    
  def test_blend_alpha_grad_pregenerated(self):
    # More complex test. Compare against pre-generated array.
    test_imageA = np.ones((5,5))
    test_imageB = np.eye(5)
    expected = np.array([[1.   , 0.98 , 0.956, 0.98 , 0.998],
                         [0.98 , 1.   , 0.542, 0.79 , 0.98 ],
                         [0.956, 0.542, 1.   , 0.542, 0.956],
                         [0.98 , 0.79 , 0.542, 1.   , 0.98 ],
                         [0.998, 0.98 , 0.956, 0.98 , 1.   ]])
    results = blend_alpha_grad(test_imageA, test_imageB)
    self.assertTrue(np.abs(np.subtract(expected, results)).all()<0.01)
    
  def test_blend_alpha_grad_images(self):
    # Test from generated images
    # Requires implementation
    pass
    
class TestConversions(unittest.TestCase):
  
  def test_tensor_to_image(self):
    test_image = np.eye(5)
    test_tensor = test_image[np.newaxis, :, :, np.newaxis] / 255.
    
    self.assertTrue(np.allclose(image_to_tensor(test_image),test_tensor))
  
  def test_image_to_tensor(self):
    test_tensor = np.eye(5)[np.newaxis, :, :, np.newaxis]
    test_image = np.eye(5, dtype='uint8')*255.
    self.assertTrue(np.allclose(tensor_to_image(test_tensor), test_image))
    
  def test_multi(self):
    test = np.eye(5)
    test1 = image_to_tensor(test)
    test2 = tensor_to_image(test1)
    
    self.assertTrue(np.allclose(test, test2))
    
    
if __name__ == "__main__":
  
  # Perform unit testing
  print('Performing unit tests...')
  unittest.main()
    
    
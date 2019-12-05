import cvaep
import img_utilities as iu

import os
import PIL
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

DEBUG = False
STEP_OUTPUT = False

SOURCE_DIRECTORY = './source_image'

IMAGE_DIMENSIONS = cvaep.IMAGE_DIMENSIONS # carry over constant from CVAE model
TRAINING_SET_RATIO = 0.8  # Ratio of {training set}:{complete sample set}

BINARIZE = False

assert(len(os.listdir(SOURCE_DIRECTORY))), 'Source image missing.'

def prepare_image(input_array, edge='SOBEL', binarize=False):
  # Normalize
  # TODO: Consider transfering this to img_utilities for reuse
  # TODO: Determine if normalization is ideal prior to Sobel stage
  input_array_min = input_array.min()
  input_array_max = input_array.max()
  norm = (input_array - input_array_min)
  norm = norm / (input_array_max-input_array_min)
  norm = norm * 255.

  # Sobel
  dx = ndimage.sobel(norm, 0)
  dy = ndimage.sobel(norm, 1)

  magnitude = np.hypot(dx, dy)

  # Normalize to range [0,1]
  ret = magnitude / np.max(magnitude)

  # Binarize
  if binarize:
    ret[ret >= .5] = 1.
    ret[ret < .5] = 0.

  return ret

def generate_samples():

  sources = os.listdir(SOURCE_DIRECTORY)
  
  for idx, source in enumerate(sources):
    print(f'[{idx} - {source}]')
  print('Select the source to train on')
  selection = int(input())
  if not selection in [idx for idx,_ in enumerate(sources)]:
    raise(Exception("Invalid selection."))
  source_image_filename = os.path.join(SOURCE_DIRECTORY,
                                       os.listdir(SOURCE_DIRECTORY)[selection])
  print(f'The image {source_image_filename} has been located')

  source_image = PIL.Image.open(source_image_filename)
  source_image = source_image.convert(mode='L')
  source_image = np.asarray(source_image, dtype='float32')

  plt.figure()
  plt.imshow(source_image, cmap='gray')
  plt.show()

  window_dim = [IMAGE_DIMENSIONS[0], IMAGE_DIMENSIONS[1]]

  # TODO: Calculate window step dimensions needed for at least 10k samples
  # FOR NOW: slide window in (1,1) step increments
  #   until 10k samples are generated or limited by source img size
  sample_qty_desired = 10000
  step_dimensions = [1, 1]
  assert(step_dimensions[0]>0 and step_dimensions[1]>0)

  cursor = [0, 0]
  iterator = 0

  sample_accumulator = np.array([])
  print('Generating samples')

  while cursor[0] < source_image.shape[0] - window_dim[0]:
    while cursor[1] < source_image.shape[1] - window_dim[1]:
      
      iterator += 1
      if iterator >= sample_qty_desired:
        break
      
      window = source_image[cursor[0]:cursor[0]+window_dim[0],
                            cursor[1]:cursor[1]+window_dim[1]]

      sample = prepare_image(window, binarize=BINARIZE)

      if iterator % 1000 == 0:
        print('.', end='')

      # Add image to accumulator list
      if not sample_accumulator.size == 0:
        sample_accumulator = np.append(sample_accumulator,
                                       sample[np.newaxis,:],
                                       axis=0)
      else:
        sample_accumulator = sample[np.newaxis,:]
        
      cursor[1] += step_dimensions[1]
    cursor[0] += step_dimensions[0]
    cursor[1] = 0

  print(f'* Generated {iterator} samples!')
  return sample_accumulator

def procedural_gen(cvae, size=(100,100), overlap=20, blend_offset=0, blend_iterations=10):
  '''
  This process shares many similarities to the process of generating samples,
  but with some significant difference. It may be possible to refactor
  some of these processes.
  '''

  blend_overlap = overlap

  window_dim = [IMAGE_DIMENSIONS[0],
                       IMAGE_DIMENSIONS[1]]

  # Create an empty array for generation
  generated_canvas = np.empty(size, dtype='uint8')

  # TODO: Create a separate loop for blend cycles
  modes = ['initial']
  for iteration in range(blend_iterations):
    modes.append('blend')
  for idx, mode in enumerate(modes):

    print(f'*** ENTERING MODE: {mode} ***')

    cursor = [0, 0]
    offset = 0

    if mode == 'initial':
      overlap = 0
    elif mode == 'blend':
      overlap = blend_overlap
      offset = (blend_offset*idx)%window_dim[0]
      cursor = np.add(cursor, offset)
    else:
      assert(not overlap == None)

    while cursor[0] < (generated_canvas.shape[0]- window_dim[0] - offset):
      while cursor[1] < (generated_canvas.shape[1]- window_dim[1] - offset):

        gen_window = np.empty([])
        print('.', end='')

        if mode == 'initial':

          gen_window = CVAE.generate_unique()

          # REFACTOR, tensor to image:
          gen_window = gen_window[0, :, :, 0]
          gen_window = ((gen_window - np.min(gen_window)) /
                        (np.max(gen_window) - np.min(gen_window)))
          gen_window *= 255.
          gen_window = gen_window.astype('uint8')

        elif mode == 'blend':
          original_window = generated_canvas[cursor[0]:cursor[0]+window_dim[0],
                                             cursor[1]:cursor[1]+window_dim[1]]

          # Create a copy of the original window to use as the seed
          seed = np.copy(original_window)
          orig_results = np.copy(original_window)  # store results for display

          seed = seed.astype('float32')
          seed = seed[np.newaxis, :, :, np.newaxis]
          seed /= 255.

          gen_window = CVAE.generate_from_seed(seed)

          # REFACTOR:
          gen_window = gen_window[0, :, :, 0]
          gen_window = ((gen_window - np.min(gen_window)) /
                        (np.max(gen_window) - np.min(gen_window)))
          gen_window *= 255.
          gen_window = gen_window.astype('uint8')

          gen_results = np.copy(gen_window)  # store results for display

          gen_window = iu.blend_alpha_grad(original_window,
                                           gen_window,
                                           mask='Gaussian')

        # TODO: consolidate these debug outputs
        if STEP_OUTPUT:
          plt.figure()
          plt.title('original/generated/blended')
          plt.subplot(1,3,1)
          plt.title('original')
          plt.imshow(orig_results, cmap='gray')
          plt.axis('off')
          plt.subplot(1,3,2)
          plt.title('generated')
          plt.imshow(gen_results, cmap='gray')
          plt.axis('off')
          plt.subplot(1,3,3)
          plt.title('blended')
          plt.imshow(gen_window, cmap='gray')
          plt.axis('off')
          plt.show()

        # Replace sliced section of the generated canvas with the window
        generated_canvas[cursor[0]:cursor[0]+window_dim[0],
                         cursor[1]:cursor[1]+window_dim[1]] = gen_window[:,:]

        cursor[1] += window_dim[1] - overlap
      cursor[0] += window_dim[0] - overlap
      cursor[1] = 0

    plt.figure()
    plt.title(f'output of mode: {mode} {idx}')
    plt.imshow(generated_canvas, cmap='gray')
    plt.colorbar()
    plt.show()

  return generated_canvas

if input('Load previously saved weights? (Y/n)').strip().upper()=='Y':

  CVAE = cvaep.CVAE_Manager(latent_dim=100)

else:

  print('Generating Samples...')
  samples = generate_samples()

  # Shuffled in-place
  print('Shuffling...')
  np.random.shuffle(samples)
  print('Shuffle complete.')

  split_index = int(len(samples) * TRAINING_SET_RATIO)
  train_set = samples[:split_index]
  test_set = samples[split_index:]

  # Build the model
  CVAE = cvaep.CVAE_Manager(train_set, test_set)

  CVAE.train(epochs=300, latent_dim=100)
  CVAE.save_model()

generated = procedural_gen(CVAE, size=(230, 230), overlap=8, blend_offset=5, blend_iterations=5)
generated = generated.astype(dtype='uint8')
plt.figure()
plt.imshow(generated, cmap='gray')
plt.colorbar()
plt.show()
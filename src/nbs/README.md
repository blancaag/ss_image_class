# Some notes

- The notebooks do not render properly and the .html files seem to be too big to be displayed. I recommend to download ```mobilenet.ipynb``` or ```renet_50.ipynb``` on your local and visualizing them using Jupyter Notebook in order to go through the training steps -running them will require a GPU enabled machine; please contact me for access to the AMI of the instance if interested.

- The structure of the training process has been similar for the three models:

  - Initialization

  - Data augmentation: setting up a random set of augmented data generators based on the image transformation options available for the 'ImageDataGenerator' class on Keras.

    The pipeline of is mainly by the aug_data_generators() function contained in ```utils_functions.py```:

    It iterates over a range of values for different parameters in order to generate a set of 288 different generators. A sample of the produced images gets stored in the src/output/aug_data_sample directory and is shown below.
    
  - Modeling:
    - Base model: instanciating the ImageNet pre-trained models
    - Generating features: generating and store the base model output features after 'feeding' it with the augmented training data set
    - Top model: training a top layer for the base model with the generated features
    - Setting alternative top models: exploring other top layer architectures.
    - Training the whole model: 'unfreezing' some of the base model's last layer blocks and retrain the base + top model with the original training images -w/o data augmentaion.
    - Visualizing predictions<br /><br /><br />


  ```
  def aug_data_generators(model, n_gen=None, cropping=None):

      preprocess_input = set_preprocess_input_function(model=model, cropping=cropping)

      fill_mode = ['constant', 'nearest', 'reflect', 'wrap']

      ad_gens = []
      count = 0
      for i in range(0, 180, 60):
          for j in range(0, 8, 3):
              for k in range(0, 10, 3):
                  for l in range(0, 1, 1): #50, 10):
                      for m in [True, False]:
                          for n in fill_mode:
                              count += 1
                              exec("""ad_gen_{} = ImageDataGenerator(
                                      rotation_range={},              # i
                                      width_shift_range={}/10,        # j
                                      height_shift_range={}/10,       # j
                                      shear_range={}/10,              # j
                                      zoom_range={}/10,               # k
                                      channel_shift_range={},         # l
                                      horizontal_flip={},             # m
                                      vertical_flip=not {},           # m
                                      fill_mode={},                   # n
                                      cval=0,                         
                                      preprocessing_function=preprocess_input)
                              """.format(count, i, j, j, j, k, l, m, m, 'n'))
                              exec('ad_gens.append(ad_gen_{})'.format(count))
      print("Total number of available aug. data generators: ", len(ad_gens))

      if n_gen: shuffle_ix = list(np.random.randint(0, len(ad_gens), n_gen))
      else: shuffle_ix = list(np.random.randint(0, len(ad_gens), len(ad_gens)))

      ad_gens_g = [ad_gens[i] for i in shuffle_ix]
      print("Selected number of aug. data generators: ", len(shuffle_ix))

      return ad_gens_g
      ```<br /><br />

![Alt text](https://github.com/blancaag/ss_image_class/blob/building_blocks/src/output/aug_data_sample/_0_2276.png)

<img align="left" width="200" height="200" src="https://github.com/blancaag/ss_image_class/blob/building_blocks/src/output/aug_data_sample/_0_2276.png">
<img align="right" width="200" height="200" src="https://github.com/blancaag/ss_image_class/blob/building_blocks/src/output/aug_data_sample/_0_5265.png">
<img align="left" width="200" height="200" src="https://github.com/blancaag/ss_image_class/blob/building_blocks/src/output/aug_data_sample/_0_5310.png">
<img align="right" width="200" height="200" src="https://github.com/blancaag/ss_image_class/blob/building_blocks/src/output/aug_data_sample/_0_8934.png">

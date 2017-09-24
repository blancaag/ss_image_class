# 'Sushi vs. Sandwich' image classification

The purpose of this project is to assess the achivable accuracy on the sushi/sandwich image classification problem, using a set of 800 images -640 for training. Further data has been collected in order to improve the model performance -and results may be added over the next days. Nevertheless the objective of this work has been maximizing the accuracy levels with the available data through data augmentation and cropping, and top layers parameter tunning.

The following table show the results obtained after a few epochs of model training, which leaves scope for further accuracy improvement.

##### Results with data augmentation (da)

| architecture      | description                      | number of parameters   | validation loss   | validation accuracy   | 
| --------------    | -------------------------------  | ---------------------  | ----------------  | --------------------  |

| InceptionV3       | InceptionV3 + 2-Dense top layer  | 17.8M (17.2M + 0.60M)  | 0.215             | 92.50%                |
| ResNet50          | ResNet50 + 2-Dense top layer     | M  (.M + 0.M)          | 0.258             | 90.62%                |
| MobileNet - T1    | MobileNet + "Heavy" top layer    | 3.5M (3.2M + 0.265M)   | 0.222             | 91.25%                |
| MobileNet - T2.0  | MobileNet + "Light" top layer    | 3.3M (3.2M + 0.056M)   | 0.551             | 82.88 %               |
| MobileNet - T2.1  | MobileNet + "Light" top layer    | 3.2M (3.2M + 0.003M)   | 0.353             | 84.13 %               |
| Ensemble model    | ResNet50 +  MobileNet - T1       |          -----         | 0.234             | 90.62%                |

##### Results with data augmentation (da) + auxiliary data 

| architecture      | description                      | number of parameters   | validation loss   | validation accuracy   | 
| --------------    | -------------------------------  | ---------------------  | ----------------  | --------------------  |
| MobileNet - T1    | w/o da w/o b.m.f. (1)            | 3.5M (3.2M + 0.265M)   | 0.559             | 75.00%                |
| MobileNet - T1    | with da w/o b.m.f. (2)           | 3.5M (3.2M + 0.265M)   | 0.299             | 88.13%                |
| MobileNet - T1    | w/o da & with aux. data (3)      | 3.5M (3.2M + 0.265M)   | 0.222             | 91.25%                |
| MobileNet - T1    | with da & with aux. data (4)     | 3.5M (3.2M + 0.265M)   | 0.222             | 91.25%                |



## Getting Started

The following instructions show how to evaluate the list of trained models over a set of test data on a local machine. As the training process of the models has been performed in AWS GPU instances, it has been documented through IPython Notebooks located in the 'src/nbs' directory.

Please, go to https://github.com/blancaag/ss_image_class/blob/building_blocks/src/nbs/README.md (src/nbs) for further comments about training the networks.

The evaluation of the models will be performed over the set of data contained under the 'src/source/test_data' directory. By default, it includes the validation set used in the trainning process. For use over a new set of test data, please see 'Running the test over a new set of data'.

See 'Deployment' for notes on how to deploy the project on a production level. 

### Prerequisites

In order to run this project, Git and Docker needs to be installed. Please, see https://www.docker.com. If it is intended to be run withought the Docker image,  it requires also requires Python 3.+ (https://www.python.org).

### Installing

Clone or manually download the repository:

```
git clone https://da93a600a1764d0ab1d9f08a69a62edc174b9aa6@github.com/blancaag/ss_image_class.git
```

## Running the tests

There are two options to test the model performance and generate predictions:

- Running the image Dockerfile included in this repository. **Note**: the Dockerfile will clone this repository each time that is executed, so in order to inlcude the last project updates this should be the default option.

- Manually executing test.py

### Running the image Dockerfile

Please make sure you have Docker installed. After cloning the repository, 

1. Start Docker
2. Run the image Dockerfile

```
cd ss_image_class
docker build -f Dockerfile 
```
It will show the evaluation metrics for the selected default models and an ensemble of the predictions ('voting system').

When running the image it automatically executes the ```test.py``` script, which runs the model over a test set of data that has been initially drawn from the providaded training set, and therefore not used during the model training -it has been included in the 'src/source/test_images directory'. This script:
        * Generates predictions over the test set contained in 'test_images'
        * Echoes the model performance metrics

### Manually

After cloning the repository: 

1. Run the ```test.py``` script:

```
cd ss_image_class/src/scripts
python3 test.py
docker build Dockerfile . --build-arg --on_local
```

Running the script manually supports the following options (please run ```python3.6 test.py -h``` for more detail):

```
  -h, --help            show this help message and exit
  --models [MODELS [MODELS ...]]
                        Models to evaluate. Options: resnet50, mobilenet,
                        indeption_v3
  --metrics [METRICS [METRICS ...]]
                        Metrics to use in the model evaluation. Default:
                        binary_crossentropy (loss), binary_accuracy, recall,
                        precision, fmesure.
  --from_dir            If True the test data will be read on batches from the
                        "test_data" directory. This option is suitable when
                        the size of the test set is relatively big and memory
                        constraints may appear.
```

## Running the test over a new set of data

In order to evaluate the models over a new set of data -and after cloning the repository:

1. Delete 'src/source/compressed_data' directory:

```
cd ss_image_class/src/source
rm -rf compressed_data
```

2. Copy your test set of images under 'src/source/test_images' directory. The directory structure should be of type:

```
test_data/
        sushi/
                img_1.jpg
                img_2.jpg
                ...
        sandwich/
                img_1.jpg
                img_2.jpg
                ...
```

3. Run the ```test.py``` script:

```
cd ss_image_class/src/scripts
python3 test.py
docker build Dockerfile . --build-arg --on_local
```

Please, see above for the supported execution options.

## Deployment

The deployment of the project on a production level is thought to be using the set of saved model weights/models obtained after training the models, therefore being the latency the time required to generate new predictions after new data has been provided -in the same way that the script ```test.py``` does.  

Even if we are using saved models, architectures such as InceptionV3 and RensNet50 may be too slow for a mobile application. Lighter models like MobileNet and derivations seem to offer a very good trade off between number of trainable parameters and accuracy.

In terms of model update on a production level, a second pipeline can be set in which new incoming data is feed into the model in batches, in order to train, update it and make it available to the app. almost in real time. If data is not expected to be available with such frequency, the model update can be done on a daily basis using a similar pipeline.

## Next steps

* Test additional data
* Further 'finetunning' work on InceptionV3
* Add pretrained models .h5 files
* Support generator from_directory for testing
* Work with Dynamic Learning Rates callbacks
* Add DenseNet 121, DenseNet 161 and DenseNet 169 
* Add a filters visualization section
* Lighten the Dockerfile image

## Built with

* [Python 3.0.6] (https://www.python.org)
* [Keras 2.0.6/2.0.8] (https://keras.io)

## Contact and citing 

- Please get in touch for bugs, enhancements and contributing -- bagonzalo@gmail.com
- Please cite this repository [GitHub] (https://github.com/blancaag) if it has been of any help.

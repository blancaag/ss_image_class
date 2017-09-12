# Project Title

The purpose of this project is to assess the achivable accuracy on the sushi/sandwich image classification problem, using a set of 800 images and data augmentation -640 for training. Further data has been collected and results will be added to this repository over the next days. 

The following table show the results obtained after a few epochs of model training, which leaves scope for further accuracy improvement.

Results

| architecture      | description                      | number of parameters   | validation loss   | validation accuracy   | 
| --------------    | -------------------------------  | ---------------------  | ----------------  | --------------------  |
| InceptionV3       | InceptionV3 + 2-Dense top layer  | 17.8M (17.2M + 0.60M)  |       | 90.91%                |
| ResNet50          | ResNet50 + 2-Dense top layer     | M  (.M + 0.M)          | 0.258             | 90.62%                |
| MobileNet - T1    | MobileNet + "Heavy" top layer    | 3.5M  (3.2M + 0.265M)  | 0.222             | 85.61%                |
| MobileNet - T2.0  | MobileNet + "Light" top layer    | 3.3M (3.2M + 0.056M)   |       | 85.01 %               |
| MobileNet - T2.1  | MobileNet + "Light" top layer    | 3.2M (3.2M + 0.003M)   |       | 82.50 %               |
| Ensemble model    | ResNet50 +  MobileNet - T1       |          -----         | 0.234             | 90.62%                |

## Getting Started

The following instructions show how to evaluate the list of trained models over a set of test data on a local machine. As the training process of the models has been performed in AWS GPU instances, it has been documented on IPython Notebooks located in the 'nbs' folder.

The evaluation will be performed over the set of data contained under the folder 'test_data'. 

See deployment for notes on how to deploy the project on a live system. 

### Prerequisites

In order to run this project Git and Docker needs to be installed. Please, see https://www.docker.com.

### Installing

Clone or manually download the repository:

```
git clone https://da93a600a1764d0ab1d9f08a69a62edc174b9aa6@github.com/blancaag/ss_image_class.git
```

Run the Dockerfile

```
until finished
```

Alternatively provide a download address to the Dockerfile image that can be accessible with a "wget" command, e.g.:

http://research.us-east-1.s3.amazonaws.com/public/sushi_or_sandwich_photos.zip

## Running the tests

There are two options to test the model performance and generate predictions:

* Running a test over a new set of data
* Running the Dockerfile image: when running the image it automatically executes the ```test.py``` script, which runs the model over a test set of data that has been initially drawn from the providaded training set, and therefore not used during the model training -it has been included in the 'test_images' directory. This script:
        * Generates predictions over the test set contained in 'test_images'
        * Echoes the model performance metrics

### Running a test over a new set of data

In order to generate predictions and performance metrics for a new set of data, please: 

1. Clone this repository
2. Copy your test set of images under 'test_images' folder. The directory structure should be of type:

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

3. Run the Dockerfile

```
docker build Dockerfile . --build-arg --on_local
```

 
## Deployment

The deployment of the project on a production level is thought to be using the set of saved model weights/models obtained after training the models, therefore being the latency the time required to generate new predictions after new data has been provided -in the same way that the script ```test.py``` does.  

Even if we are using saved models, architectures such as InceptionV3 and RensNet50 may be considered slow for a mobile application. Lighter models like MobileNet and derivations seem to offer a very good trade off between number of trainable parameters and accuracy.

In terms of model update on a production level, a second pipeline can be set in which new incoming data is feed into the model in batches, in order to train, update it and make it available to the app. almost in real time. If data is not expected to be available with such frequency, the model update can be done on a daily basis using a similar pipeline.

## Comments

* InceptionV3 needs further 'finetunning' work.

## Built With

* [Python 3.0.6](http://www.dropwizard.io/1.0.2/docs/)
* [Keras 2.0.6/2.0.8](https://maven.apache.org/) - Dependency Management

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

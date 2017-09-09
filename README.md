# Project Title

One Paragraph of project description goes here

Results

| architecture  | description   | validation accuracy | comments      |
| ------------- | ------------- | -------------       | ------------- |
| inceptionv3   | Content Cell  | Content Cell        | Content Cell  |
| resNet50      | Content Cell  | Content Cell        | Content Cell  |
| resNet50      | Content Cell  |
| resNet50      | Content Cell  |
| resNet50      | Content Cell  |
| resNet50      | Content Cell  |


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

In order to run this 

```
Give examples
```

### Installing

A step by step series of examples that tell you have to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

There are two options to test the model performance and generate predictions:

* Running a test over a new set of data
* Running the Dockerfile image: when running the image it automatically executes the ```validation.py``` script, which runs the model over a test set of data that has been initially drawn from the providaded training set, and therefore not used during the model training -it has been included in the 'test_images' directory. This script:
        * Generates predictions over the test set contained in 'test_images'
        * Echoes the model performance metrics

### Running a test over a new set of data

In order to generate predictions and performance metrics for a new set of data:

1. Clone this repository
2. Copy your test set of images in 'test_images' folder. The directory structure should be of type:

```
test_images/
        sushi/
                img_1.jpg
                img_2.jpg
                ...
        sandwich/
                img_1.jpg
                img_2.jpg
                ...
```

3. Run:

```
python3 test.py
```

Alternatively provide a download address to the Dockerfile image that can be accessible with a "wget" command, e.g.:

http://research.us-east-1.s3.amazonaws.com/public/sushi_or_sandwich_photos.zip

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc



| First Header  | Second Header |
| ------------- | ------------- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |

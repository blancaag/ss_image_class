import os
cwd = os.getcwd()

import sys
sys.path.append(cwd + '/src/utils/')
sys.path.append('../utils/')
print (sys.path)

import libraries
from libraries import *
import utils_functions
from utils_functions import *

def add_model_results(model, m, metrics):

    m_results = {}

    for _, i in enumerate(m.metrics_names):
        m_results[i] = metrics[_]

    results[model] = m_results

def print_results():

    print ("\nEvaluation metrics using the data contained in the 'test_data' directory:")
    for i, j in results.items():
        print("Model:", i)
        for k, l in j.items():
            print ("   %s: %.3f" %(k, l))

def evaluate_resnet_50():

    # directories
    model = 'resnet_50'
    model_output = join(join(output, models), model)
    model_output_weights = join(model_output, 'weights')

    ## base model
    bm = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)

    ## top model
    p = 0.8

    top_m = Sequential([
            BatchNormalization(input_shape=bm.output_shape[1:]),
            Flatten(),
            Dropout(p),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(p),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(p),
            Dense(1, activation='sigmoid')])

    ## whole model

    m = Model(bm.input, top_m(bm.output))

    m.compile(Adam(),
              loss='binary_crossentropy',
              metrics=metrics_list)

    m.load_weights(join(model_output_weights, 'm_0.2732_0.9000'))

    metrics = m.evaluate(x_test, y_test)

    add_model_results(model, m, metrics)

def evaluate_inception_v3():

    # directories
    model = 'inception_v3'
    model_output = join(join(output, models), model)
    model_output_weights = join(model_output, 'weights')

    ## base model
    bm = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)

    ## top model
    p = 0.8

    top_m = Sequential([
            BatchNormalization(input_shape=bm.output_shape[1:]),
            MaxPooling2D(),
            Flatten(),
            Dropout(p),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(p),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(p),
            Dense(1, activation='sigmoid')])

    ## whole model

    m = Model(bm.input, top_m(bm.output))

    m.compile(Adam(),
              loss='binary_crossentropy',
              metrics=metrics_list)

    m.load_weights(join(model_output_weights, '.h5'))

    metrics = m.evaluate(x_test, y_test)

    add_model_results(model, m, metrics)

def evaluate_mobilenet():

    # reloading test data with the appropiate size
    x_test, y_test = load_test_data(target_size=target_size_mn)

    # directories
    model = 'mobilenet'
    model_output = join(join(output, models), model)
    model_output_weights = join(model_output, 'weights')

    ## base model
    bm = MobileNet(include_top=False, weights='imagenet', input_shape=input_shape_mn)

    ## top model
    p = 0.8

    top_m = Sequential([
            BatchNormalization(input_shape=bm.output_shape[1:]),
            GlobalAveragePooling2D(),
            Dropout(p),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(p),
            Dense(1, activation='sigmoid')])

    ## whole model

    m = Model(bm.input, top_m(bm.output))

    m.compile(Adam(),
              loss='binary_crossentropy',
              metrics=metrics_list)

    m.load_weights(join(model_output_weights, 'm_p80_0.2221_0.9125'))

    metrics = m.evaluate(x_test, y_test)

    add_model_results(model, m, metrics)

def evaluate_ensemble_model():

    output_models = join(output, models)

    preds = np.empty((0,) + (y_test.shape[0],))
    for i in models_list:
        model_ouput = join(output_models, i)
        pred = load_array(join(join(output_models, i), 'predictions'))
        pred = pred.ravel()
        pred = pred.reshape((1,) + pred.shape)
        print(pred.shape)
        preds = np.append(preds, pred, axis=0)

    preds_mean = preds.mean(axis=0)

    m_results = {}

    m_results['loss'] = log_loss(y_test, preds_mean)
    m_results['binary_accuracy'] = accuracy_score(y_test, np.round(preds_mean))
    m_results['recall'] = recall_score(y_test, np.round(preds_mean))
    m_results['precision'] = precision_score(y_test, np.round(preds_mean))
    m_results['fmeasure'] = f1_score(y_test, np.round(preds_mean))

    results['ensemble'] = m_results

    print(binary_crossentropy, binary_accuracy, recall, precision, fmeasure)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='The script evaluates a \
                                     selection of models in terms of a set of selected metrics \
                                     -see optional arguments- using the data contained \
                                     under the "test_data" directory @ ../source/test_data. \n' \
                                     'If no options are provided all supported models \
                                     and a relevant set of metrics will be used.')

    # parser.add_argument('--nargs-int-type', nargs='+', type=int)
    parser.add_argument('--models', nargs='*', type=str, default=None, \
                        help='Models to evaluate. Options: resnet50, mobilenet, indeption_v3 ')
    parser.add_argument('--metrics', nargs='*', type=str, default=None, help='Metrics to use in \
    the model evaluation. Default: binary_crossentropy (loss), binary_accuracy, recall, precision, fmesure.')
    # parser.add_argument('--test_data_folder', type=bool, default=False)
    parser.add_argument('--from_dir', action="store_true", default=False, help='If True \
                         the test data will be read on batches from the "test_data"\
                         directory. This option is suitable when the size of the \
                         test set is relatively big and memory constraints may appear.')

    p = parser.parse_args()
    # parser.print_help()

    # models to include

    if p.models != None: models_list = p.models

    else:
        models_list = ['resnet_50', 'inception_v3', 'xception', 'mobilenet',
                       'vgg_16', 'vgg_19',
                       'densenet_121', 'densenet_161', 'densenet_169']
        models_list = ['resnet_50', 'mobilenet']

    # metrics to calculate

    if p.metrics != None: metrics_list = p.metrics
    else:
        metrics_list = [binary_accuracy,
                        recall,
                        precision,
                        fmeasure,
#                         fbeta_score,
#                         hinge,
#                         squared_hinge,
#                         kullback_leibler_divergence,
#                         poisson,
#                         cosine_proximity,
#                         matthews_correlation
                       ]

    # global variables
    target_size = (256, 256)
    target_size_mn = (224, 224)
    input_shape = target_size + (3,)
    input_shape_mn = target_size_mn + (3,)
    results = {}

    # global directories
    output = '../output'
    models = 'models'

    # loading test data
    x_test, y_test = load_test_data(target_size=target_size)

    # model evaluation
    for i in models_list:
        exec('evaluate_{}()'.format(i))

    evaluate_ensemble_model()

    print_results()

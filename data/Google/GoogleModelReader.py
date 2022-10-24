import json
import os

from data.Google import GoogleModelFactory


def load_google_model(model_location, trained=True):
    absolute_model_path = os.path.abspath(model_location)
    config_path = os.path.join(absolute_model_path, 'config.json')
    weights_path = os.path.join(model_location, 'weights.hdf5')
    initial_weights_path = os.path.join(model_location, 'weights_init.hdf5')
    model_instance = _create_model_instance(config_path)

    if trained:
        model_instance.load_weights(weights_path)
    else:
        _load_initial_weights_if_exist(model_instance, initial_weights_path)

    return model_instance


def _create_model_instance(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return GoogleModelFactory.load_model(config)


def _load_initial_weights_if_exist(model_instance, initial_weights_path):
    if os.path.exists(initial_weights_path):
        try:
            model_instance.load_weights(initial_weights_path)
            model_instance.initial_weights = model_instance.get_weights()
        except ValueError as e:
            print('Error while loading initial weights from {}'.format(initial_weights_path))
            print(e)

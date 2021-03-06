import os
from generators import *
from helpers import *
from models import *

def run_model(model, model_name, augment, results_dir, run_now = True, run_type = 'test'):
    '''
    takes in...
    runs our model
    '''
    save_dir = results_dir + model_name + '/'
    print('Results directory: %s'%save_dir)
    ensure_directory(save_dir)
    
    print('Running model: %s'%model_name)
    print('\tDefining generators...')

    # load our generators 
    train_generator = get_train_data_generator(augment = augment)
    val_generator   = get_val_data_generator()
    
    print('\tFitting model...')
    
    steps_dict = {'test': (1,1),
                  'full': (121,20),
                  'track_iters':(1,121)}
    
    steps_per_epoch, epochs = steps_dict[run_type]

    # run our model
    if run_now:
        # define our callbacks
        callbacks = get_callbacks(model_name, save_dir)
        
        model.fit_generator(train_generator,
                            validation_data=val_generator, 
                            validation_steps = 1, 
                            steps_per_epoch  = steps_per_epoch, 
                            epochs = epochs,
                            callbacks=callbacks)
        print('\tSaving model...')        
        model.save(save_dir + model_name + '_end.h5')
        return 'Ran!'
    return 'Dry!'

def prepare_model_name(model_name, augment):
    '''
    one off... 
    '''
    # if augmented
    if augment:
        model_name = model_name + '_aug'        
    return model_name

def choose_model(model_type, params):
    '''
    takes in model_type and the parameters as a dict
    returns model and its defined name 
    '''
    model_dict = {'log':get_log_model,
                  'fcc':get_fcc_model,
                  'cnn':get_cnn_model,
                  'vgg':get_vgg_model,
                  'fcc_sgd':get_fcc_sgd_model,
                  'log_sgd':get_log_sgd_model}
    return model_dict[model_type](**params)

def prepare_models_list():
    '''
    takes in nothing
    returns list of models that we want to sweep across
    '''
    
    models_list = []
    models_list.append(['log', {}, {'augment': 0}])
    models_list.append(['fcc', {}, {'augment': 0}])
    
    # cnn models
    params = [(ilay, idrop, ilr) for ilay in [1,2,4] for idrop in [0, 3, 6] for ilr in [0, 3, 6]]
    for (ilay, idrop, ilr) in params:
        models_list.append(['cnn',{'num_layers':ilay, 'dropout':idrop,'learning_rate':ilr}, {'augment':0}])
        # layers: 1, 2, 4
        # dropout: 1e-x: 0, 3, 6
        # rate: 0, 0.3, 0.6
    
    # vgg
    for trainable in [1, 0]:
        models_list.append(['vgg', {'trainable':trainable}, {'augment':1}])
    
    # sgd versions of log and fcc
    models_list.append(['log_sgd', {}, {'augment': 0}])
    models_list.append(['fcc_sgd', {}, {'augment': 0}])
    return models_list

def single_iter(run_row, run_now = False, run_type = 'test', results_dir = '/home/jupyter/models/'):
    model_type, model_params, run_params = run_row
    model, model_name = choose_model(model_type, model_params)
    model_name = prepare_model_name(model_name, **run_params)
    _ = run_model(model = model, model_name = model_name, **run_params, run_now = run_now, results_dir = results_dir, run_type = run_type)
    reset_keras()
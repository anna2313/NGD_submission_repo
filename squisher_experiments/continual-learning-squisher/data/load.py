import os
import copy
import pickle
import numpy as np
from torchvision import transforms
from torch.utils.data import ConcatDataset, Subset
from data.manipulate import permutate_image_pixels, SubDataset, TransformedDataset
from data.available import AVAILABLE_DATASETS, AVAILABLE_TRANSFORMS, DATASET_CONFIGS


def get_dataset(name, type='train', download=True, capacity=None, permutation=None, dir='./store/datasets',
                verbose=False, augment=False, normalize=False, target_transform=None):
    '''Create [train|valid|test]-dataset.'''

    data_name = 'MNIST' if name in ('MNIST28', 'MNIST32') else name
    dataset_class = AVAILABLE_DATASETS[data_name]

    # specify image-transformations to be applied
    transforms_list = [*AVAILABLE_TRANSFORMS['augment']] if augment else []
    transforms_list += [*AVAILABLE_TRANSFORMS[name]]
    if normalize:
        transforms_list += [*AVAILABLE_TRANSFORMS[name+"_norm"]]
    if permutation is not None:
        transforms_list.append(transforms.Lambda(lambda x, p=permutation: permutate_image_pixels(x, p)))
    dataset_transform = transforms.Compose(transforms_list)

    # load data-set
    dataset = dataset_class('{dir}/{name}'.format(dir=dir, name=data_name), train=False if type=='test' else True,
                            download=download, transform=dataset_transform, target_transform=target_transform)

    # print information about dataset on the screen
    if verbose:
        print(" --> {}: '{}'-dataset consisting of {} samples".format(name, type, len(dataset)))

    # if dataset is (possibly) not large enough, create copies until it is.
    if capacity is not None and len(dataset) < capacity:
        dataset = ConcatDataset([copy.deepcopy(dataset) for _ in range(int(np.ceil(capacity / len(dataset))))])

    return dataset

#----------------------------------------------------------------------------------------------------------#

def get_singlecontext_datasets(name, data_dir="./store/datasets", normalize=False, augment=False, verbose=False):
    '''Load, organize and return train- and test-dataset for requested single-context experiment.'''

    # Get config-dict and data-sets
    config = DATASET_CONFIGS[name]
    config['output_units'] = config['classes']
    config['normalize'] = normalize
    if normalize:
        config['denormalize'] = AVAILABLE_TRANSFORMS[name+"_denorm"]
    trainset = get_dataset(name, type='train', dir=data_dir, verbose=verbose, normalize=normalize, augment=augment)
    testset = get_dataset(name, type='test', dir=data_dir, verbose=verbose, normalize=normalize)

    # Return tuple of data-sets and config-dictionary
    return (trainset, testset), config

#----------------------------------------------------------------------------------------------------------#

def get_context_set(name, scenario, contexts, data_dir="./datasets", only_config=False, verbose=False,
                    exception=False, normalize=False, augment=False, singlehead=False, train_set_per_class=False):
    '''Load, organize and return a context set (both train- and test-data) for the requested experiment.

    [exception]:    <bool>; if True, for visualization no permutation is applied to first context (permMNIST) or digits
                            are not shuffled before being distributed over the contexts (e.g., splitMNIST, CIFAR100)'''

    ## NOTE: options 'normalize' and 'augment' only implemented for CIFAR-based experiments.

    # Define data-type
    if name == "splitMNIST":
        data_type = 'MNIST'
    elif name == "permMNIST":
        data_type = 'MNIST32'
        if train_set_per_class:
            raise NotImplementedError('Permuted MNIST currently has no support for separate training dataset per class')
    elif name == "CIFAR10":
        data_type = 'CIFAR10'
    elif name == "CIFAR100":
        data_type = 'CIFAR100'
    else:
        raise ValueError('Given undefined experiment: {}'.format(name))

    # Get config-dict
    config = DATASET_CONFIGS[data_type].copy()
    config['normalize'] = normalize if name=='CIFAR100' else False
    if config['normalize']:
        config['denormalize'] = AVAILABLE_TRANSFORMS["CIFAR100_denorm"]
    # check for number of contexts
    if contexts > config['classes'] and not name=="permMNIST":
        raise ValueError("Experiment '{}' cannot have more than {} contexts!".format(name, config['classes']))
    # -how many classes per context?
    classes_per_context = 10 if name=="permMNIST" else int(np.floor(config['classes'] / contexts))
    config['classes_per_context'] = classes_per_context
    config['output_units'] = classes_per_context if (scenario=='domain' or
                                                    (scenario=="task" and singlehead)) else classes_per_context*contexts
    # -if only config-dict is needed, return it
    if only_config:
        return config

    # Depending on experiment, get and organize the datasets
    if name == 'permMNIST':
        # get train and test datasets
        trainset = get_dataset(data_type, type="train", dir=data_dir, target_transform=None, verbose=verbose)
        testset = get_dataset(data_type, type="test", dir=data_dir, target_transform=None, verbose=verbose)
        # generate pixel-permutations
        if exception:
            permutations = [None] + [np.random.permutation(config['size']**2) for _ in range(contexts-1)]
        else:
            permutations = [np.random.permutation(config['size']**2) for _ in range(contexts)]
        # specify transformed datasets per context
        train_datasets = []
        test_datasets = []
        for context_id, perm in enumerate(permutations):
            target_transform = transforms.Lambda(
                lambda y, x=context_id: y + x*classes_per_context
            ) if scenario in ('task', 'class') and not (scenario=='task' and singlehead) else None
            train_datasets.append(TransformedDataset(
                trainset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
                target_transform=target_transform
            ))
            test_datasets.append(TransformedDataset(
                testset, transform=transforms.Lambda(lambda x, p=perm: permutate_image_pixels(x, p)),
                target_transform=target_transform
            ))
    else:
        # prepare permutation to shuffle label-ids (to create different class batches for each random seed)
        classes = config['classes']
        perm_class_list = np.array(list(range(classes))) if exception else np.random.permutation(list(range(classes)))
        target_transform = transforms.Lambda(lambda y, p=perm_class_list: int(p[y]))
        # prepare train and test datasets with all classes
        trainset = get_dataset(data_type, type="train", dir=data_dir, target_transform=target_transform,
                               verbose=verbose, augment=augment, normalize=normalize)
        testset = get_dataset(data_type, type="test", dir=data_dir, target_transform=target_transform, verbose=verbose,
                              augment=augment, normalize=normalize)
        # generate labels-per-dataset (if requested, training data is split up per class rather than per context)
        labels_per_dataset_train = [[label] for label in range(classes)] if train_set_per_class else [
            list(np.array(range(classes_per_context))+classes_per_context*context_id) for context_id in range(contexts)
        ]
        labels_per_dataset_test = [
            list(np.array(range(classes_per_context))+classes_per_context*context_id) for context_id in range(contexts)
        ]
        # split the train and test datasets up into sub-datasets
        train_datasets = []
        for labels in labels_per_dataset_train:
            target_transform = transforms.Lambda(lambda y, x=labels[0]: y-x) if (
                    scenario=='domain' or (scenario=='task' and singlehead)
            ) else None
            train_datasets.append(SubDataset(trainset, labels, target_transform=target_transform))
        test_datasets = []
        for labels in labels_per_dataset_test:
            target_transform = transforms.Lambda(lambda y, x=labels[0]: y-x) if (
                    scenario=='domain' or (scenario=='task' and singlehead)
            ) else None
            test_datasets.append(SubDataset(testset, labels, target_transform=target_transform))

    # Return tuple of train- and test-dataset, config-dictionary and number of classes per context
    return ((train_datasets, test_datasets), config)

#----------------------------------------------------------------------------------------------------------#

def get_original_indeces(dataset):
    '''Return, for each sample in [dataset], its index in the original dataset that [dataset] was derived from.'''

    if isinstance(dataset, Subset):
        indeces = get_original_indeces(dataset.dataset)
        return np.asarray([indeces[index] for index in dataset.indices])
    elif isinstance(dataset, SubDataset):
        return np.asarray(dataset.sub_indeces)
    elif isinstance(dataset, TransformedDataset):
        return get_original_indeces(dataset.dataset)
    else:
        return np.arange(len(dataset))


def select_original_indeces(dataset, indeces):
    '''Return a version of [dataset] containing exactly those samples that have the given [indeces] in the original
    dataset that [dataset] was derived from (with the transforms of [dataset] left in place).'''

    if isinstance(dataset, SubDataset):
        selection = copy.copy(dataset)  #-> shallow copy, as only which samples are selected should be changed
        selection.sub_indeces = list(indeces)
        return selection
    elif isinstance(dataset, Subset):
        return select_original_indeces(dataset.dataset, indeces)
    else:
        # -[dataset] covers the original dataset one-to-one, so its own indeces are the original ones
        return Subset(dataset, list(indeces))


def split_data(train_datasets, test_datasets, valid_split, seed, split_dir=None, experiment=None, scenario=None,
               contexts=None, verbose=False):
    '''Set aside part of each dataset in [test_datasets] as a validation-set, the rest remains the test-set.

    Which data points make up the train-, test- and validation-set is recorded to disk under [split_dir], with each
    data point identified by its index in the original dataset it comes from. The train-set is recorded as well, as
    knowing how the test-set was divided is only meaningful together with the train/test split it belongs to.
    If such a file is already there, the train-, test- and validation-set are rebuilt from the indeces it contains,
    so that the run uses the exact same data split as the one that was recorded.

    [valid_split]   <float> in [0, 1); fraction of each test-set to set aside as validation-set (0 = none)

    Returns (train_datasets, test_datasets, valid_datasets), with [valid_datasets]=None if [valid_split]==0.'''

    if not valid_split:
        return train_datasets, test_datasets, None

    # -path of the file to record the split in / reload it from (one file per data-configuration)
    split_path = None
    if split_dir is not None:
        if not os.path.isdir(split_dir):
            os.makedirs(split_dir)
        split_name = "{}-{}{}c-seed{}-split{}.pkl".format(experiment, scenario, contexts, seed, valid_split)
        split_path = os.path.join(split_dir, split_name)

    # -reload the split if it was recorded before, otherwise draw a new one (and record it, if requested)
    if split_path is not None and os.path.isfile(split_path):
        with open(split_path, 'rb') as f:
            split = pickle.load(f)
        if len(split['train']) != len(train_datasets) or len(split['test']) != len(test_datasets):
            raise ValueError("The split recorded in {} is for a different number of datasets".format(split_path))
        if verbose:
            print(" --> reloaded train/test/validation split from {}".format(split_path))
    else:
        rng = np.random.RandomState(seed)
        split = {'train': [get_original_indeces(dataset) for dataset in train_datasets], 'test': [], 'valid': []}
        for dataset in test_datasets:
            indeces = get_original_indeces(dataset)
            permutation = rng.permutation(len(indeces))
            n_valid = int(round(valid_split * len(indeces)))
            split['valid'].append(indeces[permutation[:n_valid]])
            split['test'].append(indeces[permutation[n_valid:]])
        if split_path is not None:
            with open(split_path, 'wb') as f:
                pickle.dump(split, f, pickle.HIGHEST_PROTOCOL)
            if verbose:
                print(" --> recorded new train/test/validation split in {}".format(split_path))

    # -build the train-, test- and validation-set that this split dictates
    train_datasets = [select_original_indeces(dataset, indeces)
                      for dataset, indeces in zip(train_datasets, split['train'])]
    valid_datasets = [select_original_indeces(dataset, indeces)
                      for dataset, indeces in zip(test_datasets, split['valid'])]
    test_datasets = [select_original_indeces(dataset, indeces)
                     for dataset, indeces in zip(test_datasets, split['test'])]
    return train_datasets, test_datasets, valid_datasets
import os
import sys
import glob
import ipdb

from functools import partial

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import ToTensor

from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose
from tqdm import tqdm

import protonets
from protonets.data.base import Partition, TaskLoader, celeba_partitions
from collections import defaultdict

DATA_DIR  = os.path.join(os.path.dirname(__file__), '../../data')


def get_partitions_kmeans(encodings, n_way, n_shot, n_query, random_scaling=True, n_partitions=100, n_clusters=500):
    import os
    os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'  # default runs out of space for parallel processing
    from sklearn.cluster import KMeans

    encodings_list = [encodings]
    if random_scaling:
        n_clusters_list = [n_clusters]
        for i in range(n_partitions - 1):
            weight_vector = np.random.uniform(low=0.0, high=1.0, size=encodings.shape[1])
            encodings_list.append(np.multiply(encodings, weight_vector))
    else:
        n_clusters_list = [n_clusters] * n_partitions
    assert len(encodings_list) * len(n_clusters_list) == n_partitions
    if n_partitions != 1:
        n_init = 3
        init = 'k-means++'
    else:
        n_init = 10
        init = 'k-means++'

    print('Number of encodings: {}, number of n_clusters: {}, number of inits: '.format(len(encodings_list),
                                                                                        len(n_clusters_list)), n_init)

    kmeans_list = []
    for n_clusters in tqdm(n_clusters_list, desc='get_partitions_kmeans_n_clusters'):
        for encodings in tqdm(encodings_list, desc='get_partitions_kmeans_encodings'):
            while True:
                kmeans = KMeans(n_clusters=n_clusters, init=init, precompute_distances=True, n_jobs=-1,
                                n_init=n_init, max_iter=3000).fit(encodings)
                uniques, counts = np.unique(kmeans.labels_, return_counts=True)
                num_big_enough_clusters = np.sum(counts >= n_shot + n_query)
                if num_big_enough_clusters > 0.8 * n_clusters:
                    break
                else:
                    # if FLAGS.datasource == 'miniimagenet' and num_big_enough_clusters > 0.1 * n_clusters:
                    #     break
                    tqdm.write("Too few classes ({}) with greater than {} examples.".format(num_big_enough_clusters,
                                                                                            n_shot + n_query))
                    tqdm.write('Frequency: {}'.format(counts))
            kmeans_list.append(kmeans)
    partitions = []
    for kmeans in kmeans_list:
        partitions.append(Partition(labels=kmeans.labels_, n_way=n_way, n_shot=n_shot, n_query=n_query))
    return partitions

def load(opt, splits):
    encodings_dir = os.path.join(DATA_DIR, '{}_encodings'.format(opt['data.encoder']))
    filenames = os.listdir(encodings_dir)

    ret = { }
    for split in splits:
        if split in ['val', 'test'] and opt['data.test_way'] != 0:
            n_way = opt['data.test_way']
        else:
            n_way = opt['data.way']

        if split in ['val', 'test'] and opt['data.test_shot'] != 0:
            n_support = opt['data.test_shot']
        else:
            n_support = opt['data.shot']

        if split in ['val', 'test'] and opt['data.test_query'] != 0:
            n_query = opt['data.test_query']
        else:
            n_query = opt['data.query']

        if split in ['val', 'test']:
            n_episodes = opt['data.test_episodes']
            mode = opt['data.test_mode']
        else:
            n_episodes = opt['data.train_episodes']
            mode = opt['data.train_mode']


        split_filename = [filename for filename in filenames if opt['data.dataset'] in filename and split in filename]
        assert len(split_filename) == 1
        split_filename = os.path.join(encodings_dir, split_filename[0])
        split_data = np.load(split_filename)
        images = split_data['X']    # (index, H, W, C)
        labels = split_data['Y']
        encodings = split_data['Z']

        if mode == 'ground_truth':
            if opt['data.dataset'] == 'celeba':
                annotations_filename = os.path.join(DATA_DIR, 'celeba/cropped/Anno/list_attr_celeba.txt')
                partitions = celeba_partitions(labels=labels, split=split, annotations_filename=annotations_filename, n_way=n_way, n_shot=n_support, n_query=n_query)
            else:
                partitions = [Partition(labels=labels, n_way=n_way, n_shot=n_support, n_query=n_query)]

        elif mode == 'kmeans':
            partitions = get_partitions_kmeans(encodings=encodings, n_way=n_way, n_shot=n_support, n_query=n_query, n_partitions=opt['data.partitions'], n_clusters=opt['data.clusters'])

        elif mode == 'random':
            partitions = [Partition(labels=np.random.choice(opt['data.clusters'], size=labels.shape, replace=True), n_way=n_way, n_shot=n_support, n_query=n_query) for i in range(opt['data.partitions'])]
        else:
            raise ValueError
        ret[split] = TaskLoader(data=images, partitions=partitions, n_way=n_way, n_shot=n_support, n_query=n_query,
                                cuda=opt['data.cuda'], length=n_episodes)

    return ret

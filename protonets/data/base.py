import torch
import numpy as np
from collections import defaultdict
from itertools import combinations, product
import ipdb
from tqdm import tqdm

def convert_dict(k, v):
    return { k: v }

class CudaTransform(object):
    def __init__(self):
        pass

    def __call__(self, data):
        for k,v in data.items():
            if hasattr(v, 'cuda'):
                data[k] = v.cuda()

        return data

class SequentialBatchSampler(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def __len__(self):
        return self.n_classes

    def __iter__(self):
        for i in range(self.n_classes):
            yield torch.LongTensor([i])

class EpisodicBatchSampler(object):
    def __init__(self, n_classes, n_way, n_episodes):
        self.n_classes = n_classes
        self.n_way = n_way
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes):
            yield torch.randperm(self.n_classes)[:self.n_way]


def celeba_partitions(labels, split, annotations_filename, n_way, n_shot, n_query, attributes_per_task=3):
    if type(labels[0]) == np.bytes_:
        labels = np.array([int(label.decode('utf-8')[:label.decode('utf-8').find('.jpg')]) for label in labels])

    if split == 'train':
        attribute_indices = np.arange(start=0, stop=20, step=1, dtype=np.int)
    elif split == 'val':
        attribute_indices = np.arange(start=20, stop=30, step=1, dtype=np.int)
    elif split == 'test':
        attribute_indices = np.arange(start=30, stop=40, step=1, dtype=np.int)
    else:
        raise ValueError

    name_to_attributes = defaultdict(int)

    with open(annotations_filename) as f:
        n_datapoints = int(f.readline().strip())
        attribute_names = f.readline().strip().split()
        for i, lines in enumerate(f):
            example_name, *example_attributes = lines.strip().split()
            example_name = int(example_name[:example_name.find('.jpg')])
            name_to_attributes[example_name] = np.array(list(map(lambda a: 0 if int(a) < 0 else 1, example_attributes)))
        print([(i, name) for (i, name) in enumerate(attribute_names)])

    attributes = []
    for name in labels:
        attributes.append(name_to_attributes[name])
    attributes = np.stack(attributes, axis=0)
    attributes = attributes[:, attribute_indices]

    num_pools = 0
    # task_pool = OrderedDict()
    partitions = []
    from scipy.special import comb
    for attr_comb in tqdm(combinations(range(attributes.shape[1]), attributes_per_task), desc='get_task_pool', total=comb(attributes.shape[1], attributes_per_task)):
        # inner_pool = OrderedDict()
        for booleans in product(range(2), repeat=attributes_per_task-1):
            booleans = (0,) + booleans  # only the half of the cartesian products that start with 0
            positive = np.where(np.all([attributes[:, attr] == i_booleans for (attr, i_booleans) in zip(attr_comb, booleans)], axis=0))[0]
            if len(positive) < n_shot + n_query:
                continue
            negative = np.where(np.all([attributes[:, attr] == 1 - i_booleans for (attr, i_booleans) in zip(attr_comb, booleans)], axis=0))[0]
            if len(negative) < n_shot + n_query:
                continue
            # inner_pool[booleans] = {0: list(negative), 1: list(positive)}
            partitions.append(CelebAPartition(negative, positive))
            num_pools += 1

    print('Generated {} task pools by using {} (of {}) attributes per pool'.format(num_pools, attributes_per_task, attributes.shape[1]))
    # return task_pool, partitions
    return partitions


class CelebAPartition(object):
    def __init__(self, negative, positive):
        self.partition = {0: negative, 1: positive}
        self.subset_ids = np.array(list(self.partition.keys()))

    def __getitem__(self, key):
        return self.partition[key]


class Partition(object):
    def __init__(self, labels, n_way, n_shot, n_query):
        partition = defaultdict(list)
        cleaned_partition = {}
        for ind, label in enumerate(labels):
            partition[label].append(ind)
        for label in list(partition.keys()):
            if len(partition[label]) >= n_shot + n_query:
                cleaned_partition[label] = np.array(partition[label], dtype=np.int)
        self.partition = cleaned_partition
        self.subset_ids = np.array(list(cleaned_partition.keys()))

    def __getitem__(self, key):
        return self.partition[key]


class TaskLoader(object):
    def __init__(self, data, partitions, n_way, n_shot, n_query, cuda, length):
        self.data = data
        self.partitions = partitions
        self.n_way = n_way
        self.n_shot = n_shot
        self.n_query = n_query
        self.cuda = cuda
        self.length = length
        self.iter = 0

    def reset(self):
        self.iter = 0

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        if self.iter == self.length:
            raise StopIteration
        self.iter += 1

        i_partition = torch.randint(low=0, high=len(self.partitions), size=(1,), dtype=torch.int)
        partition = self.partitions[i_partition]
        sampled_subset_ids = np.random.choice(partition.subset_ids, size=self.n_way, replace=False)
        xs, xq = [], []
        for subset_id in sampled_subset_ids:
            indices = np.random.choice(partition[subset_id], self.n_shot + self.n_query, replace=False)
            x = self.data[indices]
            x = x.astype(np.float32) / 255.0
            if x.shape[1] != 1 and x.shape[1] != 3:
                x = np.transpose(x, [0, 3, 1, 2])
            x = torch.from_numpy(x)
            xs.append(x[:self.n_shot])
            xq.append(x[self.n_shot:])
        xs = torch.stack(xs, dim=0)
        xq = torch.stack(xq, dim=0)
        if self.cuda:
            xs = xs.cuda()
            xq = xq.cuda()
        return {'xs': xs, 'xq': xq}

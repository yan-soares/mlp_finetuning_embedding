import torch
from itertools import combinations
import numpy as np

def get_agg_base():
    list_sum_agg = []
    list_avg_agg = []

    ranges = list(range(1, 13))

    slices = {}
    for size in range(2, 13):  # De 2 até 12
        slices[size] = [f"SUM-{group[0]}-{group[-1]}" for group in (ranges[i:i+size] for i in range(len(ranges) - size + 1))]

    for size, groups in slices.items():
        list_sum_agg+=groups

    slices = {}
    for size in range(2, 13):  # De 2 até 12
        slices[size] = [f"AVG-{group[0]}-{group[-1]}" for group in (ranges[i:i+size] for i in range(len(ranges) - size + 1))]

    for size, groups in slices.items():
        list_avg_agg+=groups
    
    return list_sum_agg, list_avg_agg

def get_pooling_techniques(poolings_args, name_agg):

    #simple_ns_poolings = ['AVG-NS', 'SUM-NS', 'MAX-NS'] 
    #simple_nostop_poolings = ['AVG-NOSTOP', 'SUM-NOSTOP', 'MAX-NOSTOP']

    simple_poolings = ['CLS', 'AVG', 'SUM', 'MAX']    
    simple_ns_new_poolings = ['AVG-NS-NEW', 'SUM-NS-NEW', 'MAX-NS-NEW']

    all_poolings_individuals = simple_poolings + simple_ns_new_poolings

    two_tokens_poolings = [f"{a}+{b}" for a, b in combinations(all_poolings_individuals, 2)]
    three_tokens_poolings = [f"{a}+{b}+{c}" for a, b, c in combinations(all_poolings_individuals, 3)]
    four_tokens_poolings = [f"{a}+{b}+{c}+{d}" for a, b, c, d in combinations(all_poolings_individuals, 4)]

    pooling_prefixs = []
    
    if poolings_args[0] == 'all':
        pooling_prefixs = simple_poolings + simple_ns_new_poolings + two_tokens_poolings + three_tokens_poolings + four_tokens_poolings
        return pooling_prefixs
        
    if 'simple' in poolings_args:
        pooling_prefixs += simple_poolings
        #return pooling_prefixs

    if 'simple-ns-nostop' in poolings_args:
        pooling_prefixs += simple_ns_new_poolings
        #return pooling_prefixs
    if 'two' in poolings_args:
        pooling_prefixs += two_tokens_poolings
        #return pooling_prefixs
    if 'three' in poolings_args:
        pooling_prefixs += three_tokens_poolings
        #return pooling_prefixs  
    if 'four' in poolings_args:
        pooling_prefixs += four_tokens_poolings
        #return pooling_prefixs     
    
    
    if len(pooling_prefixs) > 0:
        return pooling_prefixs
    else:
        return poolings_args

def get_list_layers(final_layer, initial_layer, agg_layers_args):

    list_lyrs_agg_sum, list_lyrs_agg_avg = get_agg_base()
    list_lyrs_agg = list_lyrs_agg_sum + list_lyrs_agg_avg

    lyrs = []
        
    if agg_layers_args[0] == 'ALL':
        for i in range(initial_layer, final_layer):
            lyrs.append(f"LYR-{i+1}")
        lyrs += list_lyrs_agg
        return lyrs
    
    if agg_layers_args[0] == 'SUMAGGLAYERS':
        return list_lyrs_agg_sum
    
    if agg_layers_args[0] == 'AVGAGGLAYERS':
        return list_lyrs_agg_avg
    
    if agg_layers_args[0] == 'LYR':
        for i in range(initial_layer, final_layer):
            lyrs.append(f"LYR-{i+1}")
        return lyrs    
    else:
        return agg_layers_args

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

#def batcher(params, batch):
#    sentences = [' '.join(sent) for sent in batch]
#    return params['encoder']._encode(sentences, params.current_task)

def prepare(params, samples):
    if not hasattr(params, 'all_embeddings') or params.all_embeddings is None:
        params.all_embeddings = []
    if not hasattr(params, 'all_labels') or params.all_labels is None:
        params.all_labels = []
    if not hasattr(params, 'all_sentences') or params.all_sentences is None:
        params.all_sentences = []
    if not hasattr(params, 'all_tasks') or params.all_tasks is None:
        params.all_tasks = []
    return

def batcher(params, batch):
    if isinstance(batch[0], tuple):
        # MRPC
        sentences = [' '.join(sent1) + ' [SEP] ' + ' '.join(sent2) for sent1, sent2 in batch]
    else:
        sentences = [' '.join(sent) for sent in batch]

    embeddings = params['encoder']._encode(sentences, params.current_task)

    # 🔥 Aqui: garantir que labels_batch é sempre uma lista
    if hasattr(params, 'batch_labels') and params.batch_labels is not None:
        labels_batch = params.batch_labels
        if not isinstance(labels_batch, (list, np.ndarray)):
            labels_batch = [labels_batch]
    else:
        labels_batch = [None] * len(sentences)

    if isinstance(labels_batch, np.ndarray):
        labels_batch = labels_batch.tolist()

    # 🔥 Agora sempre é lista, garantido.

    if not hasattr(params, 'all_embeddings'):
        params.all_embeddings = []
    if not hasattr(params, 'all_labels'):
        params.all_labels = []
    if not hasattr(params, 'all_sentences'):
        params.all_sentences = []
    if not hasattr(params, 'all_tasks'):
        params.all_tasks = []

    params.all_embeddings.append(embeddings)
    params.all_sentences.extend(sentences)
    params.all_tasks.extend([params.current_task] * len(sentences))
    params.all_labels.append(labels_batch)

    return embeddings



import collections
from itertools import repeat
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import torch
import os
from pathlib import Path
import tqdm 
import pandas as pd
import json
from sklearn.model_selection import train_test_split
import numpy as np
import random
from collections import Counter
from functools import reduce
from utils.tools import inverse_eage
from itertools import compress

class FlakyDataset(InMemoryDataset):
    def __init__(self, root, dataname,transform=None, pre_transform=None, pre_filter=None, empty=False):
        self.root = root
        #print(dataname)
        self.dataname = dataname
        super(FlakyDataset, self).__init__(root=root, transform=transform,  pre_transform=pre_transform, pre_filter=pre_filter)
        #print(self.dataname)
        self.data = torch.load(self.processed_paths[0])
      
    
    def get(self, idx):
        data = Data()
        #for key in self.data.keys:
        #    item, slices = self.data[key], self.slices[key]
        #    s = list(repeat(slice(None), item.dim()))
        #    s[data.__cat_dim__(key, item)] = slice(slices[idx],
        #                                            slices[idx + 1])
        #    data[key] = item[s]
        return self.data[idx]

    
    @property
    def raw_file_names(self):
        file_name_list = [ str(f) for f in Path(self.root).rglob(f"{self.dataname}.pt") ]
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list
    
    @property
    def processed_file_names(self):
        return f'geometric_data_processed_{self.dataname}.pt'
    
    def download(self):
        pass
    
    def process(self):
        data_list = []
        graph_labels = []
        graph_ids = []                                     #target_index_to_word
       
        counter = collections.defaultdict(int)
        for file in tqdm.tqdm( self.raw_file_names ):
            print(f"======================   {file}")
            raw_data = torch.load(os.path.join( file ))
            raw_data_list, raw_graph_labels, raw_graph_id = raw_data[0], raw_data[1], raw_data[2]
            raw_data_list =[ inverse_eage( d ) for d in raw_data_list ]
           # print(f"{file}, {max(raw_graph_labels)}")
            data_list = data_list + raw_data_list
            graph_labels = graph_labels + raw_graph_labels
            graph_ids = graph_ids + raw_graph_id
           

       # data, slices = self.collate(data_list)
        torch.save(data_list, self.processed_paths[0])
        print(f"Dict Size {len(counter)}")
        json.dump( counter, open(os.path.join(self.processed_dir,
                                               'counter.json'), "w"))


    


def balanced_subsample(x,y,mid_list,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        idx =  [ id ==yi for id in y ]
        elems = list(compress(x, idx )) 
        mid =  list(compress(mid_list, idx )) 
        class_xs.append((yi, elems, mid))
        print(f"label {yi}, Number {len(elems)}")
        if min_elems == None or len(elems) < min_elems:
            min_elems = len(elems)

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []
    mids= []
    for ci,this_xs, this_mids in class_xs:
        index = [i for i in range(len(this_xs))]
        if len(this_xs) > use_elems:
            np.random.shuffle(index)

        x_ = [ this_xs[i] for i in index[:use_elems] ]  #this_xs[:use_elems]
        mid_ = [ this_mids[i] for i in index[:use_elems] ]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.extend(x_)
        ys.extend(y_.tolist())
        mids.extend(mid_)

   # xs = np.concatenate(xs)
   # ys = np.concatenate(ys)

    return xs       

import random
def balanced_oversample(x, y):
    class_xs = []
    max_elems=None
    for yi in np.unique(y):
        idx =  [ id ==yi for id in y ]
        elems = list(compress(x, idx )) 
        class_xs.append((yi, elems))
        print(f"label {yi}, Number {len(elems)}")
        if max_elems == None or len(elems) > max_elems:
            max_elems = len(elems)

    use_elems = max_elems
    # if subsample_size < 1:
    #     use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        index = [i for i in range(len(this_xs))]
        if use_elems > len(this_xs):
            index = random.choices(index, k=use_elems)

        x_ = [ this_xs[i] for i in index ]  #this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)
        
        xs.extend(x_)
        ys.extend(y_.tolist())
  
    return xs      



    





    




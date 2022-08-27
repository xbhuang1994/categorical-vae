from logging import root
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool,cpu_count
import time

def load_file(file_path,split_size = 30):
    df = pd.read_csv(file_path)
    np_array = df[['成交价','成交量']].to_numpy()
    sub_array_size = int(len(np_array) / split_size)
    np_array = np_array[:sub_array_size * split_size]
    np_slice_list  = np.array_split(np_array,sub_array_size,axis=0)
    return np_slice_list


def load_training_csv(folder_path,split_size = 30):
    npz_path = folder_path +"../%d.npz" % split_size
    if os.path.exists(npz_path):
        loaded = np.load(npz_path)
        np_array = loaded['arr_0']
        print(f"load from {npz_path}")
        print(f"numpy array shape: {np_array.shape}")
        return np_array
    else:
        print("CPU内核数:{}".format(cpu_count()))
        print('当前母进程: {}'.format(os.getpid()))
        start = time.time()
        p = Pool(20)
        g = os.walk(folder_path)
        results = []
        for root,_,file_list in g:
            for file_name in file_list:
                file_path = os.path.join(root,file_name)
                result = p.apply_async(load_file, args=(file_path,))
                results.append(result)

        print('等待所有子进程完成。')
        p.close()
        p.join()
        end = time.time()
        print("总共用时{}秒".format((end - start)))
        print(f"result count: {len(results)}")
        np_list = []
        for r in results:
            np_slice_list = r.get()
            np_list.extend(np_slice_list)
        print(f"list count: {len(np_list)}")
        np_array = np.array(np_list)
        np.savez_compressed(npz_path,np_array)
        return np_array

# if __name__=='__main__':
#     load_training_csv()
import numpy as np
import os
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str)
parser.add_argument("--target", type=str)
parser.add_argument("--metric", type=str, help='poi, poi-cos or emb')
args = parser.parse_args()

metric = args.metric
source_city = args.source
target_city = args.target
source_poi = np.load("../data/%s/%s_poi.npy" % (source_city, source_city))
target_poi = np.load('../data/%s/%s_poi.npy' % (target_city, target_city))
# source_emb = np.load('../embeddings/%s_%s/%s.npy' % (source_city, target_city, source_city))
# target_emb = np.load("../embeddings/%s_%s/%s.npy" % (source_city, target_city, target_city))


# normalize
def min_max_normalize(data, percentile = 0.999, vals = None):
    sl = sorted(data.flatten())
    if vals is not None:
        max_val = vals[1]
        min_val = vals[0]
    else:
        max_val = sl[int(len(sl) * percentile)]
        min_val = max(0, sl[0])
    data[data > max_val] = max_val
    data -= min_val
    data /= (max_val - min_val)
    return data, max_val, min_val

def idx_2d2id(idx, shape):
    return idx[0] * shape[1] + idx[1]

def idx_1d22d(idx, shape):
    idx0d = int(idx // shape[1])
    idx1d = int(idx % shape[1])
    return idx0d, idx1d

source_poi, max_val, min_val = min_max_normalize(source_poi)
target_poi, _, _ = min_max_normalize(target_poi, 0.999, (min_val, max_val))
lng_source, lat_source = source_poi.shape[0], source_poi.shape[1]
lng_target, lat_target = target_poi.shape[0], target_poi.shape[1]
print("Source # grids", lng_source * lat_source)
print("Target # grids", lng_target * lat_target)

poi_inner = source_poi.reshape(-1, 14).dot(target_poi.reshape(-1, 14).transpose())
# emb_inner = source_emb.reshape(-1, 64).dot(target_emb.reshape(-1, 64).transpose())
print("poi inner product", poi_inner.shape)
# print("emb inner produce", emb_inner.shape)
source_poi_mod = np.sqrt((source_poi ** 2).sum(2).reshape(-1))
target_poi_mod = np.sqrt((target_poi ** 2).sum(2).reshape(-1))
# source_emb_mod = np.sqrt((source_emb ** 2).sum(1).reshape(-1))
# target_emb_mod = np.sqrt((target_emb ** 2).sum(1).reshape(-1))
poi_mod = np.outer(source_poi_mod, target_poi_mod) + 1e-5 # to avoid 0
# emb_mod = np.outer(source_emb_mod, target_emb_mod) + 1e-5
poi_cosine = poi_inner / poi_mod
# emb_cosine = emb_inner / emb_mod
print("poi cosine ", poi_cosine.shape)
 #print("emb cosine ", emb_cosine.shape)

poi_best_match = np.argmax(poi_inner, axis = 0)
poi_cosine_best_match = np.argmax(poi_cosine, axis = 0)
# emb_best_match = np.argmax(emb_cosine, axis = 0)

poi_best_match_val = np.array([poi_inner[poi_best_match[i], i] for i in range(lng_target * lat_target)])
poi_cosine_best_match_val = np.array([poi_cosine[poi_cosine_best_match[i], i] for i in range(lng_target * lat_target)])
# emb_best_match_val = np.array([emb_cosine[emb_best_match[i], i] for i in range(lng_target * lat_target)])

poi_match_dict = {
    idx_1d22d(i, (lng_target, lat_target)): (idx_1d22d(poi_best_match[i], (lng_source, lat_source)), poi_best_match_val[i]) for i in range(lng_target * lat_target)
}
poi_cosine_match_dict = {
    idx_1d22d(i, (lng_target, lat_target)): (idx_1d22d(poi_cosine_best_match[i], (lng_source, lat_source)), poi_cosine_best_match_val[i]) for i in range(lng_target * lat_target)
}
# emb_match_dict = {
#     idx_1d22d(i, (lng_target, lat_target)): (idx_1d22d(emb_best_match[i], (lng_source, lat_source)), emb_best_match_val[i]) for i in range(lng_target * lat_target)
# }

if not os.path.exists('rt_dict'):
    os.makedirs("rt_dict/")

if metric == 'poi':
    with open("rt_dict/%s_%s_poi" % (source_city, target_city), 'w') as outfile:
        outfile.write(str(poi_match_dict))
elif metric == 'poi-cos':
    with open("rt_dict/%s_%s_poi-cos" % (source_city, target_city), 'w') as outfile:
        outfile.write(str(poi_cosine_match_dict))
# elif metric == 'emb':
#     with open("rt_dict/%s_%s_emb" % (source_city, target_city), 'w') as outfile:
#         outfile.write(str(emb_match_dict))

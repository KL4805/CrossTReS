import numpy as np
import dgl

def masked_percentile_label(value, mask=None):
    if mask is not None:
        value = value[mask]
    value_sorted = np.argsort(value)
    num_valid = value.shape[0]
    label = np.zeros(num_valid)
    for i in range(num_valid):
        idx = value_sorted[i]
        if i < int(0.25 * num_valid):
            label[idx] = 0
        elif i < int(0.5 * num_valid):
            label[idx] = 1
        elif i < int(0.75 * num_valid):
            label[idx] = 2
        else:
            label[idx] = 3
    return label

def idx_2d_2_1d(coords, shape):
    # coords and shape are tuples
    if coords[0] > shape[0] - 1 or coords[0] < 0 or coords[1] > shape[1] - 1 or coords[1] < 0:
        return None
    else:
        return coords[1] + coords[0] * shape[1]

def add_self_loop(adj):
    # add self loop to an adjacency
    num_nodes = adj.shape[0]
    for i in range(num_nodes):
        adj[i][i] = 1
    return adj

# build graphs
def build_prox_graph(lng, lat):
    adj_prox = np.zeros((lng * lat, lng * lat))
    for i in range(lng):
        for j in range(lat):
            idx1d = idx_2d_2_1d((i, j), (lng, lat))
            for ai in [i-1, i, i+1]:
                for bi in [j-1, j, j+1]:
                    p = idx_2d_2_1d((ai, bi), (lng, lat))
                    if p is not None and p != idx1d:
                        adj_prox[idx1d, p] = 1
                        adj_prox[p, idx1d] = 1
    return adj_prox

def build_road_graph(cityname, lng, lat):
    adj_road = np.load("../data/%s/%s_roads.npy" % (cityname, cityname))
    adj_road = adj_road.reshape(lng * lat, lng * lat)
    for i in range(lng * lat): 
        adj_road[i][i] = 1
    return adj_road

def build_poi_graph(poi_feat, topk):
    # build poi graph based on poi cosine similarity and top-k
    # k = 15 to 20
    poi_inner = np.dot(poi_feat, poi_feat.transpose())
    poi_norm = np.sqrt((poi_feat ** 2).sum(1))
    poi_cos = poi_inner / (1e-5 + np.outer(poi_norm, poi_norm))
    poi_adj = np.copy(poi_cos)
    # sample topk for poi_cos
    n_nodes = poi_adj.shape[0]
    for i in range(n_nodes):
        poi_adj[i, np.argsort(poi_cos[i, :])[:-topk]] = 0
        poi_adj[np.argsort(poi_cos[:, i])[:-topk], i] = 0
    return poi_adj, poi_cos

def build_source_dest_graph(cityname, dataname, lng, lat, topk):
    od_adj = np.zeros((lng * lat, lng * lat))
    with open("../data/%s/%s%s_ODPairs" % (cityname, dataname, cityname), 'r') as infile:
        od_pairs = eval(infile.read())
    for k in od_pairs:
        ori = k[0]
        dst = k[1] 
        times = od_pairs[k]
        ori_1d = idx_2d_2_1d(ori, (lng, lat))
        dst_1d = idx_2d_2_1d(dst, (lng, lat))
        od_adj[ori_1d][dst_1d]  = times
    d_sim = np.dot(od_adj, od_adj.transpose())
    s_sim = np.dot(od_adj.transpose(), od_adj)
    d_norm = np.sqrt((od_adj ** 2).sum(1))
    s_norm = np.sqrt((od_adj ** 2).sum(0))
    d_sim /= (np.outer(d_norm, d_norm) + 1e-5)
    s_sim /= (np.outer(s_norm, s_norm) + 1e-5)

    s_adj = np.copy(s_sim)
    d_adj = np.copy(d_sim)
    n_nodes = s_adj.shape[0]
    # filter out non-topk
    for i in range(n_nodes):
        s_adj[i, np.argsort(s_sim[i, :])[:-topk]] = 0
        s_adj[np.argsort(s_sim[:, i])[:-topk], i] = 0
        d_adj[i, np.argsort(d_sim[i, :])[:-topk]] = 0
        d_adj[np.argsort(d_sim[:, i])[:-topk], i] = 0
    return s_adj, d_adj, od_adj

def adjs_to_graphs(adjs):
    # transform adjs to graphs
    num_nodes = adjs[0].shape[0]
    graphs = []
    for adj in adjs:
        from_, to_ = adj.nonzero()
        graph_ = dgl.graph((from_, to_), num_nodes = num_nodes)
        graphs.append(graph_)        
    return graphs

def split_x_y(data, lag, val_num = 60 * 24, test_num = 60 * 24):
    train_x = []
    train_y = []
    val_x = []
    val_y = []
    test_x = []
    test_y = []
    num_samples = int(data.shape[0])
    for i in range(-int(min(lag)), num_samples):
        x_idx = [int(_ + i) for _ in lag]
        y_idx = [i]
        x_ = data[x_idx, :, :]
        y_ = data[y_idx, :, :]
        if i < num_samples - val_num - test_num:
            train_x.append(x_)
            train_y.append(y_)
        elif i < num_samples - test_num:
            val_x.append(x_)
            val_y.append(y_)
        else:
            test_x.append(x_)
            test_y.append(y_)
    return np.stack(train_x, axis = 0), np.stack(train_y, axis = 0), np.stack(val_x, axis = 0), np.stack(val_y, axis = 0), np.stack(test_x, axis = 0), np.stack(test_y, axis = 0)

def split_x_y_whour(data, lag, val_num = 60 * 24, test_num = 60 * 24):
    train_x = []
    train_y = []
    train_hour_of_days = []
    val_x = []
    val_y = []
    val_hour_of_days = []
    test_x = []
    test_y = []
    test_hour_of_days = []
    num_samples = int(data.shape[0])
    for i in range(-int(min(lag)), num_samples):
        x_idx = [int(_ + i) for _ in lag]
        y_idx = [i]
        x_ = data[x_idx, :, :]
        y_ = data[y_idx, :, :]
        if i < num_samples - val_num - test_num:
            train_x.append(x_)
            train_y.append(y_)
            train_hour_of_days.append(i % 24)
        elif i < num_samples - test_num:
            val_x.append(x_)
            val_y.append(y_)
            val_hour_of_days.append(i % 24)
        else:
            test_x.append(x_)
            test_y.append(y_)
            test_hour_of_days.append(i % 24)
    return np.stack(train_x, axis = 0), np.stack(train_y, axis = 0), np.array(train_hour_of_days), \
        np.stack(val_x, axis = 0), np.stack(val_y, axis = 0), np.array(val_hour_of_days), \
        np.stack(test_x, axis = 0), np.stack(test_y, axis = 0), np.array(test_hour_of_days)

def split_x_y_wdayhour(data, lag, start_day, val_num = 60 * 24, test_num = 60 * 24):
    train_x = []
    train_y = []
    train_hour_of_days = []
    train_weekdays = []
    val_x = []
    val_y = []
    val_hour_of_days = []
    val_weekdays = []
    test_x = []
    test_y = []
    test_hour_of_days = []
    test_weekdays = []
    num_samples = int(data.shape[0])
    for i in range(-int(min(lag)), num_samples):
        x_idx = [int(_ + i) for _ in lag]
        y_idx = [i]
        x_ = data[x_idx, :, :]
        y_ = data[y_idx, :, :]
        if i < num_samples - val_num - test_num:
            train_x.append(x_)
            train_y.append(y_)
            train_hour_of_days.append(i % 24)
            train_weekdays.append((i // 24 + start_day) % 7) 
        elif i < num_samples - test_num:
            val_x.append(x_)
            val_y.append(y_)
            val_hour_of_days.append(i % 24)
            val_weekdays.append((i // 24 + start_day) % 7)
        else:
            test_x.append(x_)
            test_y.append(y_)
            test_hour_of_days.append(i % 24)
            test_weekdays.append((i // 24 + start_day) % 7)
    return np.stack(train_x, axis = 0), np.stack(train_y, axis = 0), np.array(train_hour_of_days), np.array(train_weekdays),\
        np.stack(val_x, axis = 0), np.stack(val_y, axis = 0), np.array(val_hour_of_days), np.array(val_weekdays),\
        np.stack(test_x, axis = 0), np.stack(test_y, axis = 0), np.array(test_hour_of_days), np.array(test_weekdays)

def min_max_normalize(data, percentile = 0.999):
    sl = sorted(data.flatten())
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

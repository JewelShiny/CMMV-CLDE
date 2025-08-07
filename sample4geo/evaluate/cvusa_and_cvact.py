import json
import time
import torch
import numpy as np
from tqdm import tqdm
import gc
import copy
from ..trainer_mvcv import predict,predict_text


def evaluate(config,
             model,
             reference_dataloader,
             query_dataloader, 
             query_dataloader2, 
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    
    
    print("\nExtract Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader,mode="sat") 
    query_features, query_labels = predict(config, model, query_dataloader,mode="grd")
    query_features2, query_labels2 = predict(config, model, query_dataloader2,mode="grd")
    
    print("Compute Scores:")
    r1 =  calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 
    r1_2 =  calculate_scores(query_features2, reference_features, query_labels2, reference_labels, step_size=step_size, ranks=ranks) 
    r1_3 =  calculate_scores(query_features, query_features2, query_labels, query_labels2, step_size=step_size, ranks=ranks) 
        
    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels,query_features2, query_labels2
        gc.collect()
        
    return r1,r1_2,r1_3

def evaluate_single(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    
    
    print("\nExtract Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader,mode="sat") 
    query_features, query_labels = predict(config, model, query_dataloader,mode="grd")
    
    print("Compute Scores:")
    r1 =  calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 
        
    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        
    return r1
def evaluate_text(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    
    
    print("\nExtract Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader,mode="sat") 
    query_features, query_labels = predict_text(config, model, query_dataloader,mode="text")
    query_features_img, query_labels_img = predict_text(config, model, query_dataloader,mode="grd")
    
    print("Compute Scores:")
    r1 =  calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 
    r1_img_only =  calculate_scores(query_features_img, reference_features, query_labels_img, reference_labels, step_size=step_size, ranks=ranks) 
        
    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        
    return r1
    
def evaluate_text_nearest(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    
    
    print("\nExtract Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader,mode="sat") 
    query_features, query_labels = predict_text(config, model, query_dataloader,mode="text")
    
    print("Compute Scores:")
    nearest_dict =  calculate_nearest_no_mask(query_features, reference_features, query_labels, reference_labels,neighbour_range=5, step_size=step_size) 
    print(nearest_dict)
    with open('nearest_dict_text.json', 'w', encoding='utf-8') as f:
        json.dump(nearest_dict, f, indent=4)
    exit()
        
    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        
    return nearest_dict

def evaluate_text_full(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    
    
    print("\nExtract Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader,mode="sat") 
    query_features, query_labels = predict_text(config, model, query_dataloader,mode="text")
    
    print("Compute Scores:")
    r1, r5, r10, r1_persent=  calculate_scores_full(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 

    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        
    return r1, r5, r10, r1_persent

def calc_sim(config,
             model,
             reference_dataloader,
             query_dataloader, 
             ranks=[1, 5, 10],
             step_size=1000,
             cleanup=True):
    
    
    print("\nExtract Features:")
    reference_features, reference_labels = predict(config, model, reference_dataloader) 
    query_features, query_labels = predict(config, model, query_dataloader)
    
    print("Compute Scores Train:")
    r1 =  calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=step_size, ranks=ranks) 
    
    near_dict = calculate_nearest(query_features=query_features,
                                  reference_features=reference_features,
                                  query_labels=query_labels,
                                  reference_labels=reference_labels,
                                  neighbour_range=config.neighbour_range,
                                  step_size=step_size)
            
    # cleanup and free memory on GPU
    if cleanup:
        del reference_features, reference_labels, query_features, query_labels
        gc.collect()
        
    return r1, near_dict




def calculate_scores(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1,5,10]):

    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)
    
    steps = Q // step_size + 1
    
    
    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i
    
    
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = start + step_size
          
        sim_tmp = query_features[start:end] @ reference_features.T
        
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)
    print("max similarity",torch.max(similarity))
    

    topk.append(R//100)
    
    results = np.zeros([len(topk)])
    
    
    bar = tqdm(range(Q))
    
    for i in bar:
        
        # similiarity value of gt reference
        gt_sim = similarity[i, ref2index[query_labels_np[i]]]
        
        # number of references with higher similiarity as gt
        higher_sim = similarity[i,:] > gt_sim
        
         
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
                        
        
    results = results/ Q * 100.
 
    
    bar.close()
    
    # wait to close pbar
    time.sleep(0.1)
    
    string = []
    score = []
    for i in range(len(topk)-1):
        
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
        score.append('{:.4f}'.format(results[i]))
        
    string.append('Recall@top1: {:.4f}'.format(results[-1]))            
    score.append('{:.4f}'.format(results[-1]))
        
    print(' - '.join(string)) 
    print(' '.join(score)) 

    return results[0]

def calculate_scores_full(query_features, reference_features, query_labels, reference_labels, step_size=1000, ranks=[1,5,10]):

    topk = copy.deepcopy(ranks)
    Q = len(query_features)
    R = len(reference_features)
    
    steps = Q // step_size + 1
    
    
    query_labels_np = query_labels.cpu().numpy()
    reference_labels_np = reference_labels.cpu().numpy()
    
    ref2index = dict()
    for i, idx in enumerate(reference_labels_np):
        ref2index[idx] = i
    
    
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = start + step_size
          
        sim_tmp = query_features[start:end] @ reference_features.T
        
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)
    print("max similarity",torch.max(similarity))
    

    topk.append(R//100)
    
    results = np.zeros([len(topk)])
    
    
    bar = tqdm(range(Q))
    
    for i in bar:
        
        # similiarity value of gt reference
        gt_sim = similarity[i, ref2index[query_labels_np[i]]]
        
        # number of references with higher similiarity as gt
        higher_sim = similarity[i,:] > gt_sim
        
         
        ranking = higher_sim.sum()
        for j, k in enumerate(topk):
            if ranking < k:
                results[j] += 1.
                        
        
    results = results/ Q * 100.
 
    
    bar.close()
    
    # wait to close pbar
    time.sleep(0.1)
    
    string = []
    score = []
    for i in range(len(topk)-1):
        
        string.append('Recall@{}: {:.4f}'.format(topk[i], results[i]))
        score.append('{:.4f}'.format(results[i]))
        
    string.append('Recall@top1: {:.4f}'.format(results[-1]))            
    score.append('{:.4f}'.format(results[-1]))
        
    print(' - '.join(string)) 
    print(' '.join(score)) 

    return results[0],results[1],results[2],results[3]
    

def calculate_nearest(query_features, reference_features, query_labels, reference_labels, neighbour_range=64, step_size=1000):


    Q = len(query_features)
    
    steps = Q // step_size + 1
    
    similarity = []
    
    for i in range(steps):
        
        start = step_size * i
        
        end = start + step_size
          
        sim_tmp = query_features[start:end] @ reference_features.T
        
        similarity.append(sim_tmp.cpu())
     
    # matrix Q x R
    similarity = torch.cat(similarity, dim=0)

    topk_scores, topk_ids = torch.topk(similarity, k=neighbour_range+1, dim=1)

    topk_references = []
    
    for i in range(len(topk_ids)):
        topk_references.append(reference_labels[topk_ids[i,:]])
    
    topk_references = torch.stack(topk_references, dim=0)

     
    # mask for ids without gt hits
    mask = topk_references != query_labels.unsqueeze(1)
    
    
    topk_references = topk_references.cpu().numpy()
    mask = mask.cpu().numpy()
    

    # dict that only stores ids where similiarity higher than the lowes gt hit score
    nearest_dict = dict()
    
    for i in range(len(topk_references)):
        
        nearest = topk_references[i][mask[i]][:neighbour_range]
    
        nearest_dict[query_labels[i].item()] = list(nearest)
    

    return nearest_dict

def calculate_nearest_no_mask(query_features, reference_features, query_labels, reference_labels, neighbour_range=4, step_size=1000):

    Q = len(query_features)
    steps = Q // step_size + 1
    
    similarity = []
    
    # 分批计算相似性
    for i in range(steps):
        start = step_size * i
        end = start + step_size
        sim_tmp = query_features[start:end] @ reference_features.T
        similarity.append(sim_tmp.cpu())
    
    # 将所有批次的相似性拼接成一个完整矩阵 (Q x R)
    similarity = torch.cat(similarity, dim=0)

    # 计算Top-K最近邻
    topk_scores, topk_ids = torch.topk(similarity, k=neighbour_range, dim=1)
    
    # 获取Top-K最近邻的标签
    topk_references = []
    for i in range(len(topk_ids)):
        topk_references.append(reference_labels[topk_ids[i,:]])
    
    topk_references = torch.stack(topk_references, dim=0)
    
    # 将最近邻标签存入字典
    nearest_dict = dict()
    for i in range(len(topk_references)):
        nearest = topk_references[i][:neighbour_range]
        nearest_dict[query_labels[i].item()] = list(nearest.cpu().numpy().tolist())
    
    return nearest_dict

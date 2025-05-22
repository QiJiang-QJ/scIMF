import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import seaborn as sns

import torch
import pandas as pd
from utils import p_samp
import torch.nn.functional as F
from pheatmap._pheatmap import pheatmap
from sklearn.preprocessing import MinMaxScaler





def plotPredAllTime_Compare(true_umap_traj, pred_umap_trajS, true_cell_tps, pred_cell_tps, models_order):

    unique_tps = np.unique(true_cell_tps).astype(int).tolist()
    n_tps = len(unique_tps)
    color_list = plt.cm.viridis(np.linspace(0, 1, n_tps))

    num_model = len(models_order)
    width_ratios = [1]* (num_model+1) + [0.05]

    fig = plt.figure(figsize=(6*(num_model+1), 5))
    gs = gridspec.GridSpec(1, num_model+2, width_ratios=width_ratios)

    ax = plt.subplot(gs[0, 0])
    ax.set_title("True Data", fontsize=24)
    for i, t in enumerate(unique_tps):
        true_t_idx = np.where(true_cell_tps == t)[0]
        ax.scatter(true_umap_traj[true_t_idx, 0], true_umap_traj[true_t_idx, 1], label=t, color=color_list[i], s=20, alpha=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


    for i in range(num_model):
        ax = plt.subplot(gs[0, i+1])
        ax.set_title(models_order[i], fontsize=24)
        pred_umap_traj = pred_umap_trajS[i]
        for i, t in enumerate(unique_tps):
            pred_t_idx = np.where(pred_cell_tps == t)[0]
            ax.scatter(pred_umap_traj[pred_t_idx, 0], pred_umap_traj[pred_t_idx, 1], label=t, color=color_list[i], s=20, alpha=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=20,
        markerscale=3, 
        handlelength=2,       
        handleheight=1        
    )
            
    plt.show()





def plotPredTestTime_Compare(true_umap_traj, pred_umap_trajS, true_cell_tps, pred_cell_tps, test_tps, models_order,usetitle = True,title_size=25, ncol=1):
    '''Plot predictions at testing timepoints.'''
    n_tps = len(np.unique(true_cell_tps))
    n_test_tps = len(test_tps)
    color_list = sns.color_palette("Set1", n_colors=n_test_tps)


    num_model = len(models_order)
    width_ratios = [1]* (num_model+1) + [0.05]

    fig = plt.figure(figsize=(6*(num_model+1), 5))
    gs = gridspec.GridSpec(1, num_model+2, width_ratios=width_ratios)

    ax = plt.subplot(gs[0, 0])
    if usetitle:
        ax.set_title("True Data", fontsize=title_size)
    else:
        ax.set_title("")
    ax.scatter(true_umap_traj[:, 0], true_umap_traj[:, 1], label="other", c='lightgray', s=40, alpha=0.2)
    for i, t in enumerate(test_tps):
        c = color_list[i]
        true_t_idx = np.where(true_cell_tps == t)[0]
        ax.scatter(true_umap_traj[true_t_idx, 0], true_umap_traj[true_t_idx, 1], label=int(t), color=c, s=20, alpha=1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    for i in range(num_model):
        ax = plt.subplot(gs[0, i+1])
        if usetitle:
            ax.set_title(models_order[i], fontsize=title_size)
        else:
            ax.set_title("")
        ax.scatter(true_umap_traj[:, 0], true_umap_traj[:, 1], c='lightgray', s=40, alpha=0.5)
        pred_umap_traj = pred_umap_trajS[i]
        for i, t in enumerate(test_tps):
            c = color_list[i]
            pred_t_idx = np.where(pred_cell_tps == t)[0]
            ax.scatter(pred_umap_traj[pred_t_idx, 0], pred_umap_traj[pred_t_idx, 1], label=int(t), color=c, s=20, alpha=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    
    legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=20,
        markerscale=3, 
        handlelength=2,      
        handleheight=1,
        title="Held-out\nTime Points",  
        title_fontsize=20,
        ncol=ncol  
    )
    legend.get_title().set_horizontalalignment("center")  
    
    plt.show()








def plotVectorAllTime_Compare_IndexForEachTime(true_umap_list, endes_umapS_list, Indexes, models_order):
    num_model = len(models_order)
    fig = plt.figure(figsize=(6*(num_model+1), 5.5))
    width_ratios = [1]*(num_model+1)
    gs = gridspec.GridSpec(1, num_model+1, width_ratios=width_ratios)

    n_tps = len(true_umap_list)
    color_list = plt.cm.viridis(np.linspace(0, 1, n_tps))
    ax = plt.subplot(gs[0, 0])
    ax.set_title("True Data", fontsize=20)
    for i, true_umap_t in enumerate(true_umap_list):
        ax.scatter(true_umap_t[:, 0], true_umap_t[:, 1], label=i, color=color_list[i], s=20, alpha=1.0)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))

    for i in range(num_model):
        ax = plt.subplot(gs[0, i+1])
        ax.set_title(models_order[i], fontsize=20)
        endes_umap_list = endes_umapS_list[i]

        ax.scatter(np.concatenate(true_umap_list, axis=0)[:,0], np.concatenate(true_umap_list, axis=0)[:,1], s = 1, color = 'lightgray')
        for ii, Index in enumerate(Indexes):
            Xu_start = true_umap_list[ii][Index]
            Xu_end = endes_umap_list[ii][Index]
            Xu_v = Xu_end - Xu_start
            Xu_v = Xu_v / np.linalg.norm(Xu_v, axis = 1)[:,np.newaxis] * 1.5

            ax.quiver(Xu_start[:,0],  Xu_start[:,1], Xu_v[:,0], Xu_v[:,1], scale = 1.5, scale_units = 'xy', 
                width = .004)
            
        ax.set_xlabel('UMAP1',fontsize=16)
        ax.set_ylabel('UMAP2',fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()



def plotVectorAllTime_Compare(true_umap, true_cell_tps, celltypes, endes_umapS, Indexes, models_order, useOneNorm=False,v_scale=[1,1,1,1,1], time_constrain=None, celltype_constrain=None):
    num_model = len(models_order)
    fig = plt.figure(figsize=(6*(num_model+1), 5.5))
    width_ratios = [1]*(num_model+1)
    gs = gridspec.GridSpec(1, num_model+1, width_ratios=width_ratios)

    unique_tps = np.unique(true_cell_tps).astype(int).tolist()
    n_tps = len(unique_tps)
    color_list = plt.cm.viridis(np.linspace(0, 1, n_tps))
    ax = plt.subplot(gs[0, 0])
    ax.set_title("True Data", fontsize=20)
    for i, t in enumerate(unique_tps):
        true_t_idx = np.where(true_cell_tps == t)[0]
        ax.scatter(true_umap[true_t_idx, 0], true_umap[true_t_idx, 1], label=t, color=color_list[i], s=20, alpha=1)
    

    Order = ['blastomeres', 'periderm', 'axial', 'neural', 'muscle', 
                'hematopoietic', 'endoderm', 'pgc', 'eye', 'cephalic', 
                'pronephros', 'glial']
    default_colors = sns.color_palette("tab20", len(Order))
    color_palette_dict = {cell_type: color for cell_type, color in zip(Order, default_colors)}

    if time_constrain is not None:
        scatter_color = color_list[time_constrain]
    elif celltype_constrain is not None:
        scatter_color = color_palette_dict[celltype_constrain]
    else:
        scatter_color = 'lightblue'

    for i in range(num_model):
        ax = plt.subplot(gs[0, i+1])
        ax.set_title(models_order[i], fontsize=20)
        endes_umap = endes_umapS[i]

        ax.scatter(true_umap[:,0], true_umap[:,1], s = 1, color = 'lightgrey', alpha=0.5)

        for types in Order:
            type_idx = np.where(celltypes == types)[0]
            t_idx = np.where(true_cell_tps == time_constrain)[0]
            idx = np.intersect1d(type_idx, t_idx)
            ax.scatter(true_umap[idx,0], true_umap[idx,1], s = 1, color = color_palette_dict[types], alpha=0.8)


        Xu_start = true_umap[Indexes]
        Xu_end = endes_umap[Indexes]
        Xu_v = Xu_end - Xu_start
        if useOneNorm:
            Xu_v = Xu_v / np.linalg.norm(Xu_v, axis = 1)[:,np.newaxis] * v_scale[i]
        else:
            Xu_v = Xu_v * v_scale[i]

        ax.quiver(Xu_start[:,0],  Xu_start[:,1], Xu_v[:,0], Xu_v[:,1], scale = 1.5, scale_units = 'xy', 
            width = .005)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xlabel('',fontsize=16)
        ax.set_ylabel('',fontsize=16)
        
    plt.show()
    



def return_Indexes(umap_coord, cell_tps, celltypes, grid_size=0.6, time_constrain=None, celltype_constrain=None,seed_num=42):


    final_idx = np.array([])
    if time_constrain is not None:
        t_idx = np.where(cell_tps == time_constrain)[0]
        final_idx = np.union1d(final_idx, t_idx)  # 合并索引
    
    if celltype_constrain is not None:
        type_idx = np.where(celltypes == celltype_constrain)[0]
        final_idx = np.union1d(final_idx, type_idx)
    
    if len(final_idx)>0:
        umap_coord = umap_coord[final_idx.astype(int),:]
    else:
        final_idx = np.arange(0,umap_coord.shape[0])
    final_idx = final_idx.astype(int)

    grid_coords = (umap_coord // grid_size).astype(int)
    grid_dict = {}
    for idx, coord in enumerate(grid_coords):
        coord_tuple = tuple(coord)  
        if coord_tuple not in grid_dict:
            grid_dict[coord_tuple] = []
        grid_dict[coord_tuple].append(final_idx[idx])

    random.seed(seed_num)

    selected_indices = []
    for coord, indices in grid_dict.items():
        if len(indices) >= 3:  
            selected_index = random.choice(indices)  
            selected_indices.append(selected_index)
    return selected_indices






## Compute Interaction Score Between Cells
def Derive_AttnScore(model, x, Scaled=True, return_Head = True):
    Trm_inter = model.drift_inter.trm_inter[0]
    device = Trm_inter.attn.q_proj.weight.device
    with torch.no_grad():
        x = x.to(device)
        _, attn, _, _, v = Trm_inter(x)

    if Scaled:
        norm_v = v.norm(dim=-1) 
        attn = attn * norm_v.unsqueeze(1) 

    if return_Head:
        return attn # [n_heads, batch, batch]
    else:
        return attn.mean(dim=0).unsqueeze(0)    # [1, batch, batch]


def Derive_AttnBeforeSoftmax(model, x, Scaled=True, return_Head = False):
    VNet = model.drift_inter.trm_inter[0].Vnet
    Attn = model.drift_inter.trm_inter[0].attn
    Q_proj = Attn.q_proj
    K_proj = Attn.k_proj
    V_proj = Attn.v_proj
    Attn_drop = Attn.attn_drop.eval()
    Norm1 = model.drift_inter.trm_inter[0].norm1

    n_heads = Attn.n_heads
    d_k = Attn.d_k

    device = Q_proj.weight.device
    with torch.no_grad():
        x = x.to(device)
        v_initial = VNet(x)

        x = Norm1(x)
        q = Q_proj(x).view(x.size(0), n_heads, d_k).permute(1, 0, 2)  
        k = K_proj(x).view(x.size(0), n_heads, d_k).permute(1, 0, 2)  
        v = V_proj(v_initial).view(x.size(0), n_heads, d_k).permute(1, 0, 2)  
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)  
        mask = torch.eye(attn.size(-1), device=attn.device).bool()  
        mask = mask.unsqueeze(0).expand(attn.size(0), -1, -1)  
        attn = attn.masked_fill(mask, float('-inf'))

        attn_max = torch.max(attn, dim=-1, keepdim=True).values
        attn_exp = torch.exp(attn - attn_max)
        attn_exp = Attn_drop(attn_exp)

    if Scaled:
        norm_v = v.norm(dim=-1) 
        attn_exp = attn_exp * norm_v.unsqueeze(1) 
        
    if return_Head:
        return attn_exp  
    else:
        return attn_exp.mean(0).unsqueeze(0)        




  



## Compute Interaction Score Between Cell Types

def reorder_list(L, dataset='Veres'):
    if dataset == 'Veres':
        Order = ['prog_nkx61', 'prog_sox2', 'exo', 'neurog3_early', 'neurog3_mid', 
             'neurog3_late', 'sc_ec', 'sc_alpha', 'sc_beta', 
             'sst_hhex', 'fev_high_isl_low', 'phox2a']
    elif dataset == 'zebrafish':
        Order = ['blastomeres', 'periderm', 'axial', 'neural', 'muscle', 
                 'hematopoietic', 'endoderm', 'pgc', 'eye', 'cephalic', 
                 'pronephros', 'glial']
    order_dict = {value: index for index, value in enumerate(Order)}
    L_sorted = sorted(L, key=lambda x: order_dict.get(x, float('inf')))
    return L_sorted




def derive_ratio_cluster_allT(dataset, celltype_allT, min_count = 300):
    ratio_allT = []
    cluster_allT_order = []
    for celltype_t in celltype_allT:
        celltype_t = pd.Series(celltype_t)
        cluster_t = (celltype_t.value_counts() > min_count)
        cluster_t = cluster_t[cluster_t].index.tolist()
        cluster_t_order = reorder_list(cluster_t, dataset=dataset)
        ratio_t = celltype_t.value_counts()[cluster_t_order] / celltype_t.value_counts()[cluster_t_order].sum()
        ratio_allT.append(ratio_t)
        cluster_allT_order.append(cluster_t_order)

    return ratio_allT, cluster_allT_order


def CellAttn_to_TypeAttn(Cell_Attn_allT, celltype_allT, cluster_allT_order, sample_num):
    Type_Attn_allT = []
    for i, Cell_Attn_t in enumerate(Cell_Attn_allT):
        cluster_t_order = cluster_allT_order[i]
        if sample_num is not None:
            celltype_t = pd.Series(cluster_t_order).repeat(sample_num)
        else:
            celltype_t = pd.Series(celltype_allT[i])

        Type_Attn_t = []
        for Cell_Attn_t_head in Cell_Attn_t:
            Cell_Attn_t_head = pd.DataFrame(Cell_Attn_t_head.numpy(), columns=celltype_t)
            celltype_t.index = Cell_Attn_t_head.index 
            Type_Attn_t_head = pd.DataFrame(columns=cluster_t_order, index=cluster_t_order)

            for cluster_i in cluster_t_order:
                for cluster_j in cluster_t_order:
                    mask = (celltype_t == cluster_i).values[:, None] & (celltype_t == cluster_j).values[None, :]
                    Type_Attn_t_head.loc[cluster_i, cluster_j] = Cell_Attn_t_head.values[mask].mean()
                    
            Type_Attn_t.append(Type_Attn_t_head)

        Type_Attn_allT.append(Type_Attn_t)

    return Type_Attn_allT



def Derive_TypeAtt(x_allT, celltype_allT, model, 
                    dataset= 'Veres', min_count=300, 
                    sample_num=200, use_softmax = True,
                    Scaled=True, return_Head=False):
    
    if use_softmax:
        Derive_Attn = Derive_AttnScore
    else:
        Derive_Attn = Derive_AttnBeforeSoftmax

    ratio_allT, cluster_allT_order = derive_ratio_cluster_allT(dataset, celltype_allT, min_count)
    
    Cell_Attn_allT = []
    for i,x_t in enumerate(x_allT):
        cluster_t_order = cluster_allT_order[i]
        celltype_t = pd.Series(celltype_allT[i])  

        if sample_num is not None:
            xx_t = []
            for cluster in cluster_t_order:
                mask = (celltype_t == cluster).values 
                x_cluster = x_t[torch.tensor(mask)]
                x_cluster = p_samp(x_cluster, sample_num) 
                xx_t.append(x_cluster)
            xx_t = torch.cat(xx_t, dim=0)
        else:
            xx_t = x_t
        attn = Derive_Attn(model, xx_t, Scaled=Scaled, return_Head=return_Head) # 1/2*B * B  
        Cell_Attn_allT.append(attn.detach().cpu()) #8 * 1/2 * B * B

    Type_Attn_allT = CellAttn_to_TypeAttn(Cell_Attn_allT, celltype_allT, cluster_allT_order, sample_num)
    
    return Type_Attn_allT, ratio_allT






def Derive_AllCells_Att(x_allT, model, 
                        sample_num=None, 
                        use_softmax = True,
                        Scaled=True,return_Head=False):
    if use_softmax:
        Derive_Attn = Derive_AttnScore
    else:
        Derive_Attn = Derive_AttnBeforeSoftmax

    
    Cell_Attn_allT = []
    if sample_num is None:
        for i,x_t in enumerate(x_allT):
            attn = Derive_Attn(model, x_t, Scaled=Scaled, return_Head=return_Head) # 1/2*B * B  
            Cell_Attn_allT.append(attn.detach().cpu()) #8 * 1/2 * B * B
    
    else:
        for i,x_t in enumerate(x_allT):
            xx_t = p_samp(x_t, sample_num) 
            attn = Derive_Attn(model, xx_t, Scaled=Scaled, return_Head=return_Head) # 1/2*B * B  
            Cell_Attn_allT.append(attn.detach().cpu()) #8 * 1/2 * B * B

    return Cell_Attn_allT







def plot_AllCells_heatmapAllT(Matrixes, celltype_allT, dataset= 'Panc'):

    if dataset == 'Panc':
        ABC_mapping = {'prog_nkx61': 'a', 'prog_sox2':'b', 'exo':'c', 'neurog3_early':'d', 'neurog3_mid':'e', 
                        'neurog3_late':'f', 'sc_ec':'g', 'sc_alpha':'h', 'sc_beta':'i', 
                        'sst_hhex':'j', 'fev_high_isl_low':'k', 'phox2a':'l'}
    elif dataset == 'ZB':
        ABC_mapping = {'blastomeres':'a', 'periderm':'b', 'axial':'c', 'neural':'d', 'muscle':'e', 
                        'hematopoietic':'f', 'endoderm':'g', 'pgc':'h', 'eye':'i', 
                        'cephalic':'j', 'pronephros':'k', 'glial':'l'}

    
    for j, Matrix in enumerate(Matrixes):

        ABC = [ABC_mapping[label] for label in celltype_allT[j]]

        df = pd.DataFrame(Matrix[0].numpy(), columns=ABC, index=ABC)
        df_sorted = df.sort_index(axis=0).sort_index(axis=1)


        scaler = MinMaxScaler()

        df_normalized = pd.DataFrame(scaler.fit_transform(df_sorted).round(2), columns=df_sorted.index, index=df_sorted.index)
        type_labels = df_normalized.columns

        anno_row = pd.DataFrame(dict(
            cell_type=type_labels
        ))
        anno_col = pd.DataFrame(dict(
            cell_type=type_labels
        ))

        anno_row_cmaps = {"cell_type": "tab20"}
        anno_col_cmaps = {"cell_type": "tab20"}

        fig = pheatmap(
            df_normalized, cmap='Reds',annotation_row=anno_row, annotation_col=anno_col,
            annotation_row_cmaps=anno_row_cmaps, annotation_col_cmaps=anno_col_cmaps, 
            width=10,height=10,
            show_annotation_row_names = False, show_annotation_col_names = False,
            legend_bar_width = 1.5*0.02, annotation_bar_width = 0.03,
            titlename=''
        )









import numpy as np
from time import time
import sys
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.metrics import roc_curve
import sklearn
import xml.etree.ElementTree as et
import scipy.sparse as sparse


'''
The function "convert_triples_to_list_of_edges" takes the filepath of the input graph as input, 
and returns:
    (i) List of Vertices (id: label)
    (ii) List of Predicates/Relations (id: label)
    (iii) Edge list ((vertex1, vertex2), relation)
'''
def convert_triples_to_list_of_edges(filepath):

    filename = filepath.split('/')[-1].split('.')[0]
    file = open(filepath)

    vertex_dict = {}
    pred_dict = {}

    vc, pc = 0, 0
    edges = []
    adjacency_list = {}

    for line in file:
        line_split = line.split(' ')
        sub = line_split[0]
        pred = line_split[1]
        obj = line_split[2]
        if sub not in vertex_dict:
            vertex_dict[sub] = vc
            vc+=1
        if obj not in vertex_dict:
            vertex_dict[obj] = vc
            vc+=1
        if pred not in pred_dict:
            pred_dict[pred] = pc
            pc+=1

        edges.append([vertex_dict.get(sub), vertex_dict.get(obj), pred_dict.get(pred)])

    print('\nGraph: ', filename)
    print('Number of vertices: ', vc)
    print('Number of edges: ', len(edges))
    print('Number of relations(edge labels): ', pc)
    print('\n')

    return vertex_dict, pred_dict, edges


def adj_shape(v1, p1):
    return (len(v1), len(v1), len(p1))


def get_incidence_matrix(v, p, e):
    ts = time()
    shape = adj_shape(v, p)
    adj = np.array(e)
    d, vals = rgraph.clean_adjacency(adj, values=None)
    print('==> Incidence matrix generation started')
    im = sparse.lil_matrix((len(v), len(p)), dtype=np.float32)
    for i in range(len(d)):
        im[d[i][0] ,d[i][2] ]+=1

    print('==> Returning incidence matrix')
    print('==> Time taken: {:.4f} secs.\n'.format(time( ) -ts))
    return im


def load_node_equivalences(v1, v2, mapping_file_dir):
    mapping = et.parse(mapping_file_dir)
    root = mapping.getroot()
    v1_e = []
    v2_e = []
    for child in root:
        for grand in child.iter('{http://knowledgeweb.semanticweb.org/heterogeneity/alignment}map'):
            node1 = '< ' +str(grand[0][0].attrib.values())[14:-3 ] +'>'
            node2 = '< ' +str(grand[0][1].attrib.values())[14:-3 ] +'>'
            if v1.get(node1) and v2.get(node2):
                v1_e.append(v1.get(node1))
                v2_e.append(v2.get(node2))

    print('Equivalent vertices found: ', len(v1_e))
    return v1_e, v2_e


def load_relation_equivalences(mapping_file_dir):
    mapping = et.parse(mapping_file_dir)
    root = mapping.getroot()
    pred_equivalence_dict = {}
    for child in root:
        for grand in child.iter('{http://knowledgeweb.semanticweb.org/heterogeneity/alignment}map'):
            node1 = '< ' +str(grand[0][0].attrib.values())[14:-3 ] +'>'
            node2 = '< ' +str(grand[0][1].attrib.values())[14:-3 ] +'>'
            if '/property/' in node1 and node2:
                pred_equivalence_dict[node1] = node2

    print('Equivalent predicates found: ', len(pred_equivalence_dict))
    return pred_equivalence_dict


def get_replica_sub_swap(im, edgelist):

    ts = time()

    replica = sparse.lil_matrix(im.shape, dtype=np.float32)
    replica = replica.toarray()

    len_e = len(edgelist)
    len_e = len_e if len_e % 2==0 else len_e -1
    indices_e = [i for i in range(len_e)]
    random.shuffle(indices_e)

    m = (len(indices_e ) +1 )//2

    for i in range(m):
        v1, o1, p1 = edgelist[indices_e[i]]
        v2, o2, p2 = edgelist[indices_e[ i +m]]

        replica[v2][p1 ]+=1
        replica[o1][p1 ]+=1
        replica[v1][p2 ]+=1
        replica[o2][p2 ]+=1

    #     print('==> Replica creation: {:.4f} secs.\n'.format(time()-ts))

    return replica

def get_replica_obj_swap(im, edgelist):

    ts = time()

    replica = sparse.lil_matrix(im.shape, dtype=np.float32)
    replica = replica.toarray()

    len_e = len(edgelist)
    len_e = len_e if len_e % 2==0 else len_e -1
    indices_e = [i for i in range(len_e)]
    random.shuffle(indices_e)

    m = (len(indices_e ) +1 )//2

    for i in range(m):
        v1, o1, p1 = edgelist[indices_e[i]]
        v2, o2, p2 = edgelist[indices_e[ i +m]]

        replica[v1][p1 ]+=1
        replica[o2][p1 ]+=1
        replica[v2][p2 ]+=1
        replica[o1][p2 ]+=1

    #     print('==> Replica creation: {:.4f} secs.\n'.format(time()-ts))

    return replica

def get_replica_pred_swap(im, edgelist):

    ts = time()

    replica = sparse.lil_matrix(im.shape, dtype=np.float32)
    replica = replica.toarray()

    len_e = len(edgelist)
    len_e = len_e if len_e % 2==0 else len_e -1
    indices_e = [i for i in range(len_e)]
    random.shuffle(indices_e)

    m = (len(indices_e ) +1 )//2

    for i in range(m):
        v1, o1, p1 = edgelist[indices_e[i]]
        v2, o2, p2 = edgelist[indices_e[ i +m]]

        replica[v1][p2 ]+=1
        replica[o1][p2 ]+=1
        replica[v2][p1 ]+=1
        replica[o2][p1 ]+=1

    #     print('==> Replica creation: {:.4f} secs.\n'.format(time()-ts))

    return replica

def get_replica_mix_swap(im, edgelist):

    ts = time()

    replica = sparse.lil_matrix(im.shape, dtype=np.float32)
    replica = replica.toarray()

    len_e = len(edgelist)
    len_e = len_e if len_e % 2==0 else len_e -1
    indices_e = [i for i in range(len_e)]
    random.shuffle(indices_e)

    m = (len(indices_e ) +1 )//2

    for i in range(m):
        v1, o1, p1 = edgelist[indices_e[i]]
        v2, o2, p2 = edgelist[indices_e[ i +m]]

        flag = random.randint(0 ,2)

        if flag==1:
            replica[v1][p1 ]+=1
            replica[o2][p1 ]+=1
            replica[v2][p2 ]+=1
            replica[o1][p2 ]+=1
        elif flag==2:
            replica[v1][p2 ]+=1
            replica[o1][p2 ]+=1
            replica[v2][p1 ]+=1
            replica[o2][p1 ]+=1
        else:
            replica[v2][p1 ]+=1
            replica[o1][p1 ]+=1
            replica[v1][p2 ]+=1
            replica[o2][p2 ]+=1

    #     print('==> Replica creation: {:.4f} secs.\n'.format(time()-ts))

    return replica




def main():

    old_stdout = sys.stdout
    ts = time()

    ''' Set input filepath '''
    filepath1 = '../../ontologyAlignment/data.Global/graphs/memory_alpha.nt'
    filepath2 = '../../ontologyAlignment/data.Global/graphs/memory_beta.nt'

    mapping_file_dir = '../../ontologyAlignment/data.Global/mappings/memory_alpha_vs_beta.xml'
    mapping_file_name = mapping_file_dir.split('/')[-1].split('.')[0]

    log_file = open(mapping_file_name+"_v1.log","w")
    sys.stdout = log_file

    print('\n\nLoading input graphs')

    '''
    n: nodes dict
    p: predicates dict 
    e: edge list
    1,2: Graph1, graph2
    '''
    v1, p1, e1 = convert_triples_to_list_of_edges(filepath1)
    v2, p2, e2 = convert_triples_to_list_of_edges(filepath2)

    im1 = get_incidence_matrix(v1, p1, e1)
    im2 = get_incidence_matrix(v2, p2, e2)


    v1_e, v2_e = load_node_equivalences(v1, v2, mapping_file_dir)
    pred_equivalence_dict = load_relation_equivalences(mapping_file_dir)

    im1_e = im1[v1_e].toarray()
    im2_e = im2[v2_e].toarray()

    degree_product_org = np.matmul(im1_e.transpose(), im2_e)
    degree_product_org_df = pd.DataFrame(degree_product_org, index=list(p1), columns=list(p2))


    # ''' Normalized degree product '''
    # im1_e = im1[v1_e].toarray()
    # im2_e = im2[v2_e].toarray()
    # degree_product_org = np.zeros((len(p1), len(p2)))
    #
    # degree_1 = []
    # degree_2 = []
    #
    # for i in range(len(v1_e)):
    #     degree_1.append(sum(im1_e[i]))
    #     degree_2.append(sum(im2_e[i]))
    #
    # degree_1 = np.array(degree_1).reshape(len(degree_1),1)
    # degree_2 = np.array(degree_2).reshape(len(degree_2),1)
    #
    # im1_e = im1_e/degree_1
    # im2_e = im2_e/degree_2
    #
    # degree_product_org = np.matmul(im1_e.transpose(), im2_e)
    # degree_product_org_df = pd.DataFrame(degree_product_org, index=list(p1), columns=list(p2))

    replica_methods = ['get_replica_sub_swap', 'get_replica_obj_swap', 'get_replica_pred_swap', 'get_replica_mix_swap']
    t = [a for a in range(1, 51, 1)]

    aucs = np.zeros((len(t), len(replica_methods)))
    aucs_df = pd.DataFrame(aucs, index=t, columns=replica_methods)

    for _r in range(len(replica_methods)):

        print('Generating replicas...')

        f = globals()[replica_methods[_r]]

        degree_products = []

        for _ in range(50):
            replica1 = f(im1, e1)
            replica2 = f(im2, e2)

            replica1_e = replica1[v1_e]
            replica2_e = replica2[v2_e]

            #         replica1_e = replica1_e/degree_1
            #         replica2_e = replica2_e/degree_2

            degree_product = np.matmul(replica1_e.transpose(), replica2_e)

            degree_products.append(degree_product)

        degree_products = np.array(degree_products)

        for _i in range(len(t)):
            k = t[_i]
            degree_product_avg = np.zeros((len(p1), len(p2)))
            degree_product_std = np.zeros((len(p1), len(p2)))

            for i in range(len(p1)):
                for j in range(len(p2)):
                    degree_product_avg[i][j] = np.average(degree_products[:k, i, j])
                    degree_product_std[i][j] = np.std(degree_products[:k, i, j])

            degree_product_zscore = np.zeros((len(p1), len(p2)))

            for i in range(len(degree_product_org)):
                for j in range(len(degree_product_org[0])):
                    if degree_product_std[i, j]:
                        degree_product_zscore[i, j] = (degree_product_org[i, j] - degree_product_avg[i, j]) / \
                                                      degree_product_std[i, j]
                    else:
                        degree_product_zscore[i, j] = 0

            degree_product_zscore_df = pd.DataFrame(degree_product_zscore, index=list(p1), columns=list(p2))

            zscore_list = []
            for i in list(p1.keys()):
                for j in list(p2.keys()):
                    zscore_list.append([i, j, degree_product_zscore_df.loc[i, j]])

            zscore_list = np.array(zscore_list)

            zscore_list = zscore_list[zscore_list[:, 2].argsort()][::-1]

            pred_equivalence = []
            for i in range(len(zscore_list)):
                if pred_equivalence_dict.get(zscore_list[i, 0]) == zscore_list[i, 1]:
                    pred_equivalence.append(1)
                else:
                    pred_equivalence.append(0)

            pred_y = np.array(zscore_list[:, 2]).astype(float)
            y = np.array(pred_equivalence)
            fpr, tpr, threshold = roc_curve(y, pred_y)
            roc_auc = sklearn.metrics.auc(fpr, tpr)

            if _i == len(t) - 1:
                plt.figure()
                lw = 1
                plt.plot(fpr, tpr, color='darkgreen',
                         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
                plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                plt.xlim([-0.05, 1.0])
                plt.ylim([-0.05, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver operating characteristic example')
                plt.legend(loc="lower right")
                plt.savefig(
                    './output/Configuration_model/aurocs/' + mapping_file_name + '_' + replica_methods[_r] + '_v1.pdf')
                zscore_list = np.concatenate(
                    (zscore_list, np.array(pred_equivalence).reshape(len(pred_equivalence), 1)), 1)
                zscore_df = pd.DataFrame(zscore_list,
                                         columns=['G1_relation', 'G2_relation', 'Z_Score_val', 'True_value'])
                zscore_df.to_csv(
                    './output/Configuration_model/zscores/' + mapping_file_name + '_' + replica_methods[_r] + '_v1.csv')

            aucs[_i, _r] = roc_auc
            aucs_df.at[k, replica_methods[_r]] = roc_auc
            print(k, replica_methods[_r], roc_auc)

    print('==> Total time taken: {:.4f} secs.\n'.format(time( ) -ts))
    aucs_df.index.name = 'No of replicas'
    aucs_df.to_csv('./output/Configuration_model/' + mapping_file_name + '_v1.csv')
    with open("./output/Configuration_model/runtime_v1.txt", "a") as f:
        print(mapping_file_name +': {:.4f}'.format(time( ) -ts), file=f)

    sys.stdout = old_stdout
    log_file.close()

main()

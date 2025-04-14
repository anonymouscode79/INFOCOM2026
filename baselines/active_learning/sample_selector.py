import logging
import operator
import time
import torch
import numpy as np
from collections import Counter, defaultdict
from sklearn import neighbors
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score
from scipy.spatial import KDTree


from baselines.active_learning.losses import HiDistanceXentLoss
from baselines.active_learning.utils.utils import to_categorical


import abc
from collections import Counter, defaultdict
import logging
import numpy as np
import operator
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import v_measure_score

class Selector(object):

    @abc.abstractmethod
    def __init__(self, **kwargs):
        return

    @abc.abstractmethod
    def select_samples(self, **kwargs):
        # return the list of sample indices
        return None
    
    def cluster_and_print(self, fname, cur_month_str, \
                        all_train_family, train_ben_details, \
                        all_test_family, test_ben_details, \
                        y_test, y_test_binary, y_test_pred, \
                        test_offset):
        print('Running KMeans Clustering...')    
        total_fam_num = len(set(all_train_family.tolist() + all_test_family.tolist())) + 5
        print(f'total_fam_num: {total_fam_num}')
        all_z = np.concatenate((self.z_train, self.z_test), axis=0)
        kmeans = KMeans(n_clusters=total_fam_num, random_state=0).fit(all_z)
    
        # train test index separation
        # idx < test_offset ? train : test

        y_train_kmeans_pred = kmeans.predict(self.z_train)
        y_test_kmeans_pred = kmeans.predict(self.z_test)
        v_score = v_measure_score(np.concatenate((self.y_train, y_test), axis=0), np.concatenate((y_train_kmeans_pred, y_test_kmeans_pred), axis=0))
        print('GMM all v measure score: \t%.4f\n' % v_score)
        
        # data index and family info for each mixture
        mid_train_idx = defaultdict(list)
        mid_train_info = defaultdict(list)
        mid_test_idx = defaultdict(list)
        mid_test_info = defaultdict(list)
        mid_idx = defaultdict(list) # global index
        mid_info = defaultdict(list)
        for idx, mid in enumerate(y_train_kmeans_pred):
            mid_train_idx[mid].append(idx)
            mid_train_info[mid].append(all_train_family[idx])
            mid_idx[mid].append(idx)
            mid_info[mid].append(all_train_family[idx])
        for idx, mid in enumerate(y_test_kmeans_pred):
            mid_test_idx[mid].append(idx)
            mid_test_info[mid].append(all_test_family[idx])
            mid_idx[mid].append(idx+test_offset) # global index
            mid_info[mid].append(all_test_family[idx])
        mid_size = {mid: len(item) for mid, item in mid_idx.items()}
        # compute purity
        # some mixtures may only have test data, so mid_train_idx[mid] does not exist
        mid_purity = []
        mid_train_percent = {}
        for mid, size in mid_size.items():
            try:
                total_train_cnt = float(len(mid_train_idx[mid]))
                most_common_cnt = Counter(mid_train_info[mid]).most_common()[0][1]
                purity = most_common_cnt / total_train_cnt
                mid_train_percent[mid] = total_train_cnt / float(size)
            except IndexError:
                # unknown purity
                purity = -1.0
                mid_train_percent[mid] = 0.0
            mid_purity.append((mid, purity))
        sorted_mid_purity = sorted(mid_purity, key=operator.itemgetter(1))
        print(f'sorted_mid_purity: {sorted_mid_purity}')

        cluster_out = open(fname, 'a')
        cluster_out.write('\n======= Month %s =======\n' % cur_month_str)
        # print mixtures
        for mid, purity in sorted_mid_purity:
            test_indices = mid_test_idx[mid]
            fp = 0
            fn = 0
            for index in test_indices:
                if y_test_binary[index] == 0 and y_test_pred[index] == 1:
                    fp += 1
                if y_test_binary[index] == 1 and y_test_pred[index] == 0:
                    fn += 1
            # PRINT
            cluster_out.write('\n======= Mixture %d =======\n' % mid)
            cluster_out.write('####### Train Purity: %.4f\tTrain Percent %.4f\n' % (purity, mid_train_percent[mid]))
            cluster_out.write('####### Test FP: %d Test FN: %d\n' % (fp, fn))
            ### information about the entire mixture
            # 1) Total counts per family
            cluster_out.write('####### All Family Info %s\n' % Counter(mid_info[mid]).most_common())
            # 2) Train counts per familly
            cluster_out.write('####### Train Family Info %s\n' % Counter(mid_train_info[mid]).most_common())
            # 3) Test counts per family
            cluster_out.write('####### Test Family Info %s\n' % Counter(mid_test_info[mid]).most_common())
        
            ### information about individual data points
            # sort indices from closest to furthest
            mean = kmeans.cluster_centers_[mid]
            distances = [(idx, np.linalg.norm(all_z[idx]-mean)) for idx in mid_idx[mid]]
            sorted_distances = sorted(distances, key=operator.itemgetter(1), reverse=True)
            # for each sample
            cluster_out.write('\t_idx\tclsdist'\
                    '\tfamily\tpkgname\ttest\tnew\tselect\n')
            
            for idx, distance in sorted_distances:
                # only family needs a relative idx
                if idx < test_offset:
                    family = all_train_family[idx]
                    in_test = ''
                    if family == 'benign':
                        package_name = train_ben_details[idx]
                    else:
                        package_name = ''
                    wrong = ''
                    select = ''
                else:
                    family = all_test_family[idx-test_offset]
                    test_idx = idx - test_offset
                    # NOTE: this is the customized ood score, not CADE
                    in_test = '~'
                    if family == 'benign':
                        package_name = test_ben_details[test_idx]
                    else:
                        package_name = ''
                    if y_test_binary[test_idx] == y_test_pred[test_idx]:
                        wrong = ''
                    else:
                        wrong = 'X'
                    if test_idx in self.sample_indices:
                        select = '!!!'
                    else:
                        select = ''
                is_new = family not in all_train_family
                tag = '*' if is_new else ''
                # idx is in all_z, y_proba
                #cluster_out.write('\t_idx\tdist'\
                #'\tfamily\tscore\t1nn_idx\tpseudo\t2nn_idx\tcontrast\tpkgname\ttest\tnew\n')
                cluster_out.write('\t%d\t%.2f\t%s'\
                        '\t%s\t%s\t%s\t%s\t%s\n' % \
                        (idx, distance, family, package_name, in_test, tag, wrong, select))
                cluster_out.flush()
        
        cluster_out.close()
        return {}

class LocalPseudoLossSelector(Selector):
    def __init__(self, encoder):
        self.encoder = encoder
        self.z_train = None
        self.z_test = None
        self.y_train = None
        return
    
    def select_samples(self,  X_train, y_train, y_train_binary, \
                    X_test, y_test_pred, \
                    total_epochs, \
                    test_offset, \
                    all_test_family, \
                    total_count, \
                    batch_size = 64,
                    device=None,
                    margin= 10,
                    xent_lambda = 100,
                    verbose= False,
                    y_test = None,
                    family_info=True
                    ):
        X_train_tensor = torch.from_numpy(X_train).float().to(device)
        z_train = self.encoder.encode(X_train_tensor)
        print(f'Normalizing z_train to unit length...')
        z_train = torch.nn.functional.normalize(z_train)
        z_train = z_train.cpu().detach().numpy()

        X_test_tensor = torch.from_numpy(X_test).float().to(device)
        z_test = self.encoder.encode(X_test_tensor)
        print(f'Normalizing z_test to unit length...')
        z_test = torch.nn.functional.normalize(z_test)
        z_test = z_test.cpu().detach().numpy()
        
        self.z_train = z_train
        self.z_test = z_test
        self.y_train = y_train

        self.sample_indices = []
        self.sample_scores = []
        
        # build the KDTree
        print(f'Building KDTree...')
        tree = KDTree(z_train)
        print(f'Querying KDTree...')
        # query all z_test up to a margin
        all_neighbors = tree.query(z_test, k=z_train.shape[0], workers=8)
        print(f'Finished querying KDTree...')
        all_distances, all_indices = all_neighbors

        # each batch is to get one loss for one test sample


        end = time.time()
        bsize = batch_size
        # nn_loss = np.zeros([sample_num]]]
        sample_num = z_test.shape[0]
        for i in range(sample_num):
            test_sample = X_test_tensor[i:i+1] # on GPU
            # bsize-1 nearest neighbors of the test sample i
            batch_indices = all_indices[i][:bsize-1]
            # x_batch
            x_train_batch = X_train_tensor[batch_indices] # on GPU
            x_batch = torch.cat((test_sample, x_train_batch), 0)
            # y_batch
            y_train_batch = y_train_binary[batch_indices]
            y_batch_np = np.hstack((y_test_pred[i], y_train_batch))
            y_batch = torch.from_numpy(y_batch_np).to(device)
            # y_bin_batch
            y_bin_batch = torch.from_numpy(to_categorical(y_batch_np, num_classes=2)).float().to(device)
            # we don't need split_tensor. all samples are training samples
            # split_tensor = torch.zeros(x_batch.shape[0]).int().cuda()
            # split_tensor[test_offset:] = 1
            

            # in the loss function, y_bin_batch is the categorical version
            # call the loss function once for every test sample
  
            _, features, y_pred = self.encoder(x_batch)
            HiDistanceXent = HiDistanceXentLoss().to(device)
            loss, _, _ = HiDistanceXent(xent_lambda,
                                        y_pred, y_bin_batch,
                                        features, labels=y_batch,
                                        device=device, 
                                        margin = margin,
                                        family_info=family_info
                                        )
            loss = loss.to('cpu').detach().numpy()
            # measure elapsed time 
            # update the loss values for i
            # nn_loss[i] = loss[0]
            self.sample_scores.append(loss.item())
            # only display the test samples
            # if (i + 1) % (args.display_interval * 3) == 0:
            #     logging.debug('Train + Test: [0][{0}/{1}]\t'
            #         'DT {data_time.val:.3f} ({data_time.avg:.3f})  '
            #         'i {i} loss {l}'.format(
            #         i + 1, sample_num,  i=i, l=loss[0]))
        sorted_sample_scores = list(sorted(list(enumerate(self.sample_scores)), key=lambda item: item[1], reverse=True))
        # print(f'sorted_sample_scores[:100]: {sorted_sample_scores[:100]}')
        sample_cnt = 0
        for idx, score in sorted_sample_scores:
            self.sample_indices.append(idx)
            sample_cnt += 1
            if sample_cnt == total_count:
                break
        print('Added %s samples...' % (len(self.sample_indices)))
        return self.sample_indices, self.sample_scores
### CADE SELECTOR

def safe_division(x, y):
    if abs(y) < 1e-8:
        y = 1e-8
    return x / y

def get_latent_data_for_each_family(z_train, y_train):
    N_lst = list(np.unique(y_train))
    N = len(N_lst)
    N_family = [len(np.where(y_train == family)[0]) for family in N_lst]
    z_family = []
    for family in N_lst:
        z_tmp = z_train[np.where(y_train == family)[0]]
        z_family.append(z_tmp)

    z_len = [len(z_family[i]) for i in range(N)]

    return N, N_family, z_family, z_len


def get_latent_distance_between_sample_and_centroid(z_family, centroids, N, N_family):
    dis_family = []  # two-dimension list

    for i in range(N): # i: family index
        dis = [np.linalg.norm(z_family[i][j] - centroids[i]) for j in range(N_family[i])]
        dis_family.append(dis)

    dis_len = [len(dis_family[i]) for i in range(N)]

    return dis_family, dis_len


def get_MAD_for_each_family(dis_family, N, N_family):
    mad_family = []
    for i in range(N):
        median = np.median(dis_family[i])
        # print(f'family {i} median: {median}')
        diff_list = [np.abs(dis_family[i][j] - median) for j in range(N_family[i])]
        mad = 1.4826 * np.median(diff_list)  # 1.4826: assuming the underlying distribution is Gaussian
        mad_family.append(mad)
    # print(f'mad_family: {mad_family}')

    return mad_family

def detect_drift_samples_top(z_train, z_test, y_train):

    '''get latent data for each family in the training set'''
    N, N_family, z_family, z_len = get_latent_data_for_each_family(z_train, y_train)

    '''get centroid for each family in the latent space'''
    centroids = [np.mean(z_family[i], axis=0) for i in range(N)]
    # print(f'centroids: {centroids}')

    '''get distance between each training sample and their family's centroid in the latent space '''
    dis_family, dis_len = get_latent_distance_between_sample_and_centroid(z_family, centroids,
                                                                    N, N_family)
    assert z_len == dis_len, "training family stats size mismatches distance family stats size"
    '''get the MAD for each family'''
    mad_family = get_MAD_for_each_family(dis_family, N, N_family)

    ### return samples sorted by OOD scores
    '''detect drifting in the testing set'''
    ood_scores = []
    for k in range(len(z_test)):
        z_k = z_test[k]
        '''get distance between each testing sample and each centroid'''
        dis_k = [np.linalg.norm(z_k - centroids[i]) for i in range(N)]
        anomaly_k = [safe_division(np.abs(dis_k[i] - np.median(dis_family[i])), mad_family[i]) for i in range(N)]
        # print(f'sample-{k} - dis_k: {dis_k}')
        # print(f'sample-{k} - anomaly_k: {anomaly_k}')
        min_anomaly_score = np.min(anomaly_k)
        ood_scores.append((k, min_anomaly_score))
    return ood_scores

class OODSelector(Selector):
    def __init__(self, encoder,device):
        self.encoder = encoder
        self.z_train = None
        self.z_test = None
        self.y_train = None
        self.y_test = None
        self.device=device
        return
    
    def select_samples(self, X_train, y_train, \
                    X_test, \
                    budget):
        # Is y_train already binary? No
        self.y_train = y_train
        X_train_tensor = torch.from_numpy(X_train).float().cuda(device=self.device)
        z_train = self.encoder.encode(X_train_tensor)
        z_train = z_train.cpu().detach().numpy()
        self.z_train = z_train
        X_test_tensor = torch.from_numpy(X_test).float().cuda(device=self.device)
        z_test = self.encoder.encode(X_test_tensor)
        z_test = z_test.cpu().detach().numpy()
        self.z_test = z_test

        ood_scores = detect_drift_samples_top(self.z_train, self.z_test, self.y_train)
        sample_scores = [item[1] for item in ood_scores]
        ood_scores.sort(reverse=True, key=lambda x: x[1])
        self.sample_indices = [item[0] for item in ood_scores[:budget]]
        # print(ood_scores[:50])
        print('Added %s samples...' % (len(self.sample_indices)))
        return self.sample_indices, sample_scores

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.base import BaseEstimator
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.preprocessing import MinMaxScaler
from mord.threshold_based import LogisticAT
from sklearn.multioutput import MultiOutputClassifier


class AdaptiveGrainedDiscretizer(BaseEstimator):
    # the error estimation of ground-truth
    def __init__(self, zeta=0.05, max_nbins=16, n_MonteCarlo=20, random_seed=123):
        self.zeta = zeta
        self.n_MonteCarlo = n_MonteCarlo
        self.random_seed = random_seed
        self.max_nbins = max_nbins
        
    def get_FDGI(self, d, group_labels):
        weight, eae = self._errp_eae(d, group_labels)
        dep = self._fuzzy_dependency(group_labels)
        return (weight * dep).mean() / eae.mean()

    def fit(self, X, D):
        X = self._rescale(X)
        self._R = np.row_stack([np.min(1 - np.abs(X[[i]] - X), axis=1) for i in range(X.shape[0])])
        self.bin_edges_ = []
        for d in D.T:
            group_hierarchy = self._get_AggClus_group_hierarchy(d) # group labels (various granularity)
            gs_arr = np.array([len(np.unique(g)) for g in group_hierarchy]) # an array of group size
            group_hierarchy = group_hierarchy[(gs_arr < self.max_nbins) & (gs_arr > 4)] # filtering
            gains = [self.get_FDGI(d, g) for g in group_hierarchy] # FDGI at each granularity
            glabels = group_hierarchy[np.argmax(gains)] # training set partition with max FDGI

            # Obtain the LDD range for each group (CGL) based on the training set partition
            v, _id = np.unique(glabels, return_index=True)
            v, tlb = v[np.argsort(d[_id])], [0]
            for vi in range(1, len(v)):
                pmax, cmin = np.max(d[glabels == v[vi-1]]), np.min(d[glabels == v[vi]])
                tlb.append(pmax + (cmin - pmax) / 2)
            tlb.append(np.max(d[glabels == v[-1]]))
            self.bin_edges_.append(np.array(tlb))
        return self

    def coarsen(self, D):
        # get coarse-grained labels
        disclabel = [0] * D.shape[1]
        for i, d in enumerate(D.T):
            disclabel[i] = (d[:, None] >= np.array(self.bin_edges_[i])[None, :-1]).sum(1, keepdims=True)
        disclabel = np.concatenate(disclabel, axis=1) - 1
        
        # make a group indexing matrix as a CGL-LDD bridge
        self._uni_label = []
        for y in disclabel.T:
            temp = np.sort(np.unique(y))
            temp = np.r_[temp, temp[-1] * np.ones(self.max_nbins-len(temp))]
            self._uni_label.append(temp)
        self._uni_label = np.row_stack(self._uni_label)

        return rankdata(disclabel, method='dense', axis=0) - 1

    def fit_coarsen(self, X, D):
        return self.fit(X, D).coarsen(D)

    def refine(self, Y, D=None):
        try: self._uni_label
        except: raise Exception("`refine` method should be called after `coarsen` or `fit_coarsen`.")
        Yprob = np.stack([np.hstack([yp, np.zeros((yp.shape[0], self.max_nbins-yp.shape[1]))]) for yp in Y]) # (nlabels, nsmpls, nbins)
        Bes = np.stack([np.r_[b, np.ones(self.max_nbins+1-len(b))] for b in self.bin_edges_]) # (nlabels, nbins+1)
        M, N, K = Yprob.shape
        rng = np.random.default_rng(self.random_seed)
        if D is None:
            res = 0
            for mc in range(self.n_MonteCarlo):
                Y = rng.multinomial(1, Yprob).argmax(axis=-1).T
                Y = self._index(Y, self._uni_label) # get group index
                LB, UB = self._index(Y, Bes), self._index(Y+1, Bes) # get LDD range
                temp = rng.uniform(LB, UB, size=(100, LB.shape[0], LB.shape[1]))
                temp /= temp.sum(-1, keepdims=True)
                res += temp.mean(0)
            return res / self.n_MonteCarlo
        def _constrained_least_squares(lb, ub, d):
            func = lambda p: np.square(p - d).sum()
            cons = (
                {'type': 'eq', 'fun': lambda p: p.sum() - 1},
                {'type': 'ineq', 'fun': lambda p: p - lb},
                {'type': 'ineq', 'fun': lambda p: ub - p}
            )
            res = minimize(func, d, jac=lambda p: 2 * (p - d), method='SLSQP', constraints=cons, tol=1e-10)
            p = res.x
            return res.success, res.x
        Dhat = []
        for i in range(Yprob.shape[1]):
            yprob = Yprob[:, i, :] # (nlabels, nbins)
            t, prediction, tol = 0, [], 1e5
            # find the label distributions closest to D[i] within the ub and lb constraints
            while (t < self.n_MonteCarlo) or (len(prediction) == 0):
                y, t = rng.multinomial(1, yprob).argmax(axis=1), t+1 # y: (nlabels,)
                y = self._uni_label[np.arange(M), y].astype(np.int32)
                lb, ub = Bes[np.arange(M), y], Bes[np.arange(M), y+1]
                _s, _v = _constrained_least_squares(lb, ub, D[i])
                if _s: prediction.append(_v)
                if t > tol: raise Exception("Some errors occurr in the learned LDD range.")
            Dhat.append(np.row_stack(prediction).mean(0))
        Dhat = np.clip(np.row_stack(Dhat), 0, 1)
        return Dhat / Dhat.sum(1, keepdims=True) # fine-tune

    def _rescale(self, x):
        if len(x.shape) == 2: return MinMaxScaler().fit_transform(x)
        else: return MinMaxScaler().fit_transform(x.reshape(-1,1)).flatten()

    def _fuzzy_dependency(self, group_labels):
        # X is minmax-normalized
        G = np.zeros((len(group_labels), np.unique(group_labels).size))
        G[np.arange(len(group_labels)), rankdata(group_labels, method='dense') - 1] = 1
        return np.max(np.min(np.maximum(1 - self._R[:, :, None], G[:, None, :]), axis=0), axis=1)

    def _errp_eae(self, d, group_labels):
        epsl, epsr = np.minimum(d, self.zeta), np.minimum(1-d, self.zeta)
        dlower = np.vectorize(lambda g: np.min(d[group_labels==g]))(group_labels)
        dupper = np.vectorize(lambda g: np.max(d[group_labels==g]))(group_labels)
        err_prob = np.min(np.column_stack([epsl+dupper-d, epsr+d-dlower, dupper-dlower, epsr+epsl]), axis=1) / (epsr+epsl)
        dl, dr = d - dlower, dupper - d
        eae = ((epsl-epsr)**2+(dl-dr)**2)/3 + (epsl*epsr+dl*dr)/3 - (epsr-epsl)*(dr-dl)/2
        return err_prob, eae

    def _get_AggClus_group_hierarchy(self, d):
        N = len(d)
        cluster = AgglomerativeClustering(linkage='average').fit(d.reshape(-1, 1))
        group_hierarchy = []
        labels = np.arange(N)
        for i, (lnid, rnid) in enumerate(cluster.children_):
            labels[labels == lnid] = labels[labels == rnid] = N + i
            group_hierarchy.append(labels.copy())
        return np.vstack(group_hierarchy)

    def _index(self, Y, U):
        Y = Y.astype(np.int32)
        idx = np.tile(np.arange(Y.shape[1]), (Y.shape[0], 1))
        x = U[idx.flatten(), Y.flatten()]
        return np.reshape(x, (Y.shape[0], Y.shape[1]))


class AGLDL(BaseEstimator):
    def __init__(self, discretizer=None, CGL_predictor=None, LDD_predictor=None):
        self.discretizer = AdaptiveGrainedDiscretizer() if discretizer is None else discretizer
        self.LDD_predictor = LDD_predictor
        self.CGL_predictor = MultiOutputClassifier(LogisticAT(alpha=0)) if CGL_predictor is None else CGL_predictor

    def fit(self, X, D):
        self.discretizer.fit(X, D)
        if self.LDD_predictor is not None:
            self.LDD_predictor.fit(X, D)
        Y = self.discretizer.coarsen(D)
        self.CGL_predictor.fit(X, Y)
        return self

    def predict(self, X):
        CGL_prob = self.CGL_predictor.predict_proba(X)
        if self.LDD_predictor is None:
            return self.discretizer.refine(CGL_prob)
        LDD = self.LDD_predictor.predict(X)
        return self.discretizer.refine(CGL_prob, LDD)
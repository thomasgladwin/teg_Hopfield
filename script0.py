import numpy as np
import pandas as pd
import scipy as sp
import time
import matplotlib.pyplot as plt
from gensim.models.keyedvectors import KeyedVectors
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import teg_Hopfield
import importlib
importlib.reload(teg_Hopfield)

word_vectors = KeyedVectors.load_word2vec_format('../../GoogleNews-vectors-negative300.bin.gz', binary=True, limit=200000)
#word_vectors.save('wvsubset')
#word_vectors = KeyedVectors.load("/home/thomasgladwin/apps.tegladwin.com/app/wvsubset", mmap='r')
#word_vectors = KeyedVectors.load("wvsubset", mmap='r')

def read_data():
    #fn_excel = 'Data/all ldt subs_all trials3.xlsx'
    #D = pd.read_excel(fn_excel, engine='openpyxl')
    fn_pickle = 'Data/pickled'
    #D.to_pickle('Data/pickled')
    D = pd.read_pickle(fn_pickle)
    return D

def f_Hopfield(Patterns, prime_word_vec, target_word_vec, beta=1.0, beta_target=4.0, alpha=0.25, verbose=False):
    h = teg_Hopfield.Hopfield_cont(Patterns, beta=beta)
    #y_init = alpha * prime_word_vec + (1 - alpha) * target_word_vec
    #h.set_activation(y_init)
    h.set_activation(prime_word_vec)
    h.run_to_convergence(verbose=verbose)
    converged = False
    nStepsToAccurate = 0
    first_d_e = np.nan
    this_d_e = 0
    while not converged and nStepsToAccurate < 5:
        h.set_activation(alpha * h.y + (1 - alpha) * target_word_vec.reshape((len(target_word_vec), 1)))
        h.beta = beta_target
        e_initial, e_convergence = h.run_to_convergence(verbose=verbose)
        this_d_e_step = e_convergence - e_initial
        if np.isnan(first_d_e):
            first_d_e = this_d_e_step
        this_d_e = this_d_e + this_d_e_step
        act_patt = h.y[:, 0]
        act_patt = act_patt / np.sqrt(np.dot(act_patt, act_patt))
        sims_target = Patterns.transpose() @ target_word_vec
        sims_conv = Patterns.transpose() @ act_patt
        i_t = np.argmax(sims_target)
        i_c = np.argmax(sims_conv)
        # print('\tStep: ', nStepsToAccurate, '. alpha: ', alpha, '. Target: ', i_t, ', converged-to: ', i_c)
        if i_t == i_c:
            converged = True
        nStepsToAccurate = nStepsToAccurate + 1
        #alpha = alpha ** 2
        beta_target = beta_target * 2
        #if nStepsToAccurate == 4:
        #    alpha = 0
    if nStepsToAccurate >= 5:
        print('\tStep: ', nStepsToAccurate, '. alpha: ', alpha, '. Target: ', i_t, ', converged-to: ', i_c)
    return first_d_e, this_d_e, nStepsToAccurate

def get_d_e(prime_word, target_word, WVM, beta_arg = 1.0, beta_target_arg = 4.0, alpha_arg = 0.25):
    #print(prime_word, target_word)
    if isinstance(prime_word, str):
        if prime_word in word_vectors.key_to_index:
            check_prime = True
        else:
            check_prime = False
    elif np.isnan(prime_word):
        check_prime = False
    else:
        check_prime = False
    if check_prime and target_word in word_vectors.key_to_index:
        # this_sim = word_vectors.similarity(prime_word, target_word)
        beta = beta_arg
        beta_target = beta_target_arg
        alpha = alpha_arg
        prime_word_vec = word_vectors[prime_word]
        prime_word_vec = prime_word_vec / np.sqrt(np.dot(prime_word_vec, prime_word_vec))
        target_word_vec = word_vectors[target_word]
        target_word_vec = target_word_vec / np.sqrt(np.dot(target_word_vec, target_word_vec))
        if not isinstance(WVM, float):
            Patterns = WVM.transpose()
            Patterns = Patterns / np.sqrt(np.sum(Patterns * Patterns, axis=0))
        else:
            Patterns = np.array([prime_word_vec, target_word_vec]).transpose()
        this_d_e_first, this_d_e, nStepsToAccurate = f_Hopfield(Patterns, prime_word_vec, target_word_vec, beta, beta_target, alpha)
    else:
        this_d_e_first = np.NaN
        this_d_e = np.NaN
        nStepsToAccurate = np.NaN
    this_d_e = this_d_e - this_d_e_first
    return this_d_e_first, this_d_e, nStepsToAccurate

def get_subj_scores(D_subj, WVM, beta_arg=1.0, beta_target_arg=4.0, alpha_arg=0.25):
    list_this = []
    convergence_v = []
    for irow in range(D_subj.shape[0]):
        row = D_subj.iloc[irow]
        acc = row['target.ACC']
        if acc != 1:
            continue
        relstr = row['rel']
        if relstr == 'nw':
            rel = -1
        elif relstr == 'rel':
            rel = 1
        else: # 'un'
            rel = 0
        if rel < 0:
            continue
        isi = row['isi']
        # print(rel, row['prime'], row['target'])
        prime = row['prime'].lower()
        target = str(row['target']).lower() # To correct for "nan" error
        rt = row['target.RT']
        if rel >= 0:
            d_e_first, d_e, acc_conv_steps = get_d_e(prime, target, WVM, beta_arg, beta_target_arg, alpha_arg)
            convergence_v.append(acc_conv_steps)
        else:
            d_e_first = np.NaN
            d_e = np.NaN
            acc_conv_steps = np.NaN
        new_row = [isi, d_e, rt, rel, prime, target, acc_conv_steps, d_e_first]
        list_this.append(new_row)
    Scores_Subj = pd.DataFrame(list_this)
    Scores_Subj.rename(columns={0: 'isi', 1: 'd_e', 2: 'RT', 3:'rel', 4:'prime', 5:'target', 6:'acc_conv_steps', 7:'d_e_first'}, inplace=True)
    m_rt_vec = []
    m_d_e_first_vec = []
    m_d_e_vec = []
    m_acc_conv_steps_vec = []
    b_first_vec = []
    b_vec = []
    for isi in np.array([50, 1050]):
        D_isi = Scores_Subj[Scores_Subj['isi']== isi]
        d_e_first_v = D_isi['d_e_first']
        d_e_v = D_isi['d_e']
        acc_conv_steps_v = D_isi['acc_conv_steps']
        rel_v = D_isi['rel']
        rt_v = D_isi['RT']
        z_rt = sp.stats.zscore(rt_v)
        for rel in [0, 1]:
            indices_bool = (np.abs(z_rt) < 3) & ~np.isnan(rt_v) & ~np.isnan(d_e_v) & (rel_v == rel)
            indices = [i for i, x in enumerate(indices_bool) if x == True]
            regr = LinearRegression(fit_intercept=True).fit(D_isi[['d_e_first']].iloc[indices], D_isi['RT'].iloc[indices])
            b_first_vec.append(regr.coef_[0])
            regr = LinearRegression(fit_intercept=True).fit(D_isi[['d_e']].iloc[indices], D_isi['RT'].iloc[indices])
            b_vec.append(regr.coef_[0])
            m_rt_vec.append(np.mean(rt_v[indices_bool]))
            m_d_e_first_vec.append(np.mean(d_e_first_v[indices_bool]))
            m_d_e_vec.append(np.mean(d_e_v[indices_bool]))
            m_acc_conv_steps_vec.append(np.mean(acc_conv_steps_v[indices_bool]))
    return m_rt_vec, m_d_e_first_vec, m_d_e_vec, b_first_vec, b_vec, convergence_v, m_acc_conv_steps_vec

def get_scores_per_subj(D, WVM, beta_arg=1.0, beta_target_arg=4.0, alpha_arg=0.25):
    per_subj_rt = []
    per_subj_d_e_first = []
    per_subj_d_e = []
    per_subj_acc_conv_steps = []
    per_subj_b_first = []
    per_subj_b = []
    per_subj_convergence_acc = []
    t0 = time.time()
    uSubj = np.unique(D['Subject'])
    #for iiSubj in range(2):
    for iiSubj in range(len(uSubj)):
        iSubj = uSubj[iiSubj]
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        print(iSubj, 'of', len(np.unique(D['Subject'])), ', dt = ', dt)
        D_subj = D[D['Subject']==iSubj]
        m_rt_vec, m_d_e_first_vec, m_d_e_vec, b_first_vec, b_vec, convergence_v, m_acc_conv_steps_vec = get_subj_scores(D_subj, WVM, beta_arg, beta_target_arg, alpha_arg)
        per_subj_rt.append(m_rt_vec)
        per_subj_d_e_first.append(m_d_e_first_vec)
        per_subj_d_e.append(m_d_e_vec)
        per_subj_acc_conv_steps.append(m_acc_conv_steps_vec)
        per_subj_b_first.append(b_first_vec)
        per_subj_b.append(b_vec)
        per_subj_convergence_acc.append(convergence_v)
    colnames = ['isi200_un', 'isi200_rel', 'isi1200_un', 'isi1200_rel']
    S_rt = pd.DataFrame(per_subj_rt)
    S_rt.rename(columns=dict([t for t in enumerate(colnames)]), inplace=True)
    S_d_e_first = pd.DataFrame(per_subj_d_e_first)
    S_d_e_first.rename(columns=dict([t for t in enumerate(colnames)]), inplace=True)
    S_d_e = pd.DataFrame(per_subj_d_e)
    S_d_e.rename(columns=dict([t for t in enumerate(colnames)]), inplace=True)
    S_acc_conv_steps = pd.DataFrame(per_subj_acc_conv_steps)
    S_acc_conv_steps.rename(columns=dict([t for t in enumerate(colnames)]), inplace=True)
    S_b_first = pd.DataFrame(per_subj_b_first)
    S_b_first.rename(columns=dict([t for t in enumerate(colnames)]), inplace=True)
    S_b = pd.DataFrame(per_subj_b)
    S_b.rename(columns=dict([t for t in enumerate(colnames)]), inplace=True)
    return S_rt, S_d_e_first, S_d_e, S_b_first, S_b, S_acc_conv_steps, per_subj_convergence_acc

def stats(label, Scores_per_subj):
    print(label)
    print('# M(SD)')
    for iCol in range(Scores_per_subj.shape[1]):
        print(Scores_per_subj_rt.columns[iCol], '\t', f'{np.mean(Scores_per_subj.iloc[:, iCol]):.2f}', ' (', f'{np.sqrt(np.var(Scores_per_subj.iloc[:, iCol])):.2f}', ')', sep = '')
    print('# Test vs 0')
    for iCol in range(Scores_per_subj.shape[1]):
        v = Scores_per_subj.iloc[:, iCol]
        print(Scores_per_subj_rt.columns[iCol], '\t', sp.stats.ttest_1samp(v, popmean=0), sep='')
    print('# Effects')
    # Main effect isi
    isi200 = (Scores_per_subj.iloc[:, 0] + Scores_per_subj.iloc[:, 1])/2
    isi1200 = (Scores_per_subj.iloc[:, 2] + Scores_per_subj.iloc[:, 3])/2
    d = isi1200 - isi200
    print('ISI 1200 - ISI 200:\t', sp.stats.ttest_1samp(d, popmean=0), sep='')
    # Main effect rel
    un = (Scores_per_subj.iloc[:, 1] + Scores_per_subj.iloc[:, 3])/2
    rel = (Scores_per_subj.iloc[:, 0] + Scores_per_subj.iloc[:, 2])/2
    d = rel - un
    print('Rel - un:\t', sp.stats.ttest_1samp(d, popmean=0), sep='')
    # Interaction
    d = (Scores_per_subj.iloc[:, 3] - Scores_per_subj.iloc[:, 2]) - (Scores_per_subj.iloc[:, 1] - Scores_per_subj.iloc[:, 0])
    print('(Rel - un | 1200) - (rel - un | 200):\t', sp.stats.ttest_1samp(d, popmean=0), sep='')

def get_WVM(D):
    prime_words = [w.lower() for w in D['prime'] if isinstance(w, str) and w.lower() in word_vectors.key_to_index]
    target_words = [w.lower() for w in D['target'] if isinstance(w, str) and w.lower() in word_vectors.key_to_index]
    prime_words = list(set(prime_words))
    target_words = list(set(target_words))
    words = list(set(prime_words + target_words))
    WVM = np.array([word_vectors[w] for w in words])
    WVM_primes = np.array([word_vectors[w] for w in prime_words])
    WVM_targets = np.array([word_vectors[w] for w in target_words])
    return WVM, WVM_primes, WVM_targets, words, prime_words, target_words

def analyze_convergence_steps(per_subj_convergence_steps):
    u = np.unique([a for v in per_subj_convergence_steps for a in v if not np.isnan(a)])
    print('Steps found: ', u)
    step_probs = dict()
    for u0 in u:
        step_probs[str(u0)] = []
    for iSubj in range(len(per_subj_convergence_steps)):
        this_v = np.array(per_subj_convergence_steps[iSubj])
        this_v = this_v[np.isnan(this_v) == False]
        for u0 in u:
            if not np.isnan(u0):
                m = np.mean(this_v == u0)
            else:
                m = np.mean(np.isnan(this_v))
            step_probs[str(u0)] = step_probs[str(u0)] + [m]
    print('N Steps to Target:')
    for u0 in u:
        print(u0, np.mean(np.array(step_probs[str(u0)])))

if False:
    Patterns = WVM.transpose();
    #prime_word_vec = word_vectors[prime_words[0]];
    #prime_word_vec = word_vectors['sinker'];
    prime_word_vec = word_vectors['trustworthy'];
    #target_word_vec = word_vectors[target_words[0]]
    #target_word_vec = word_vectors['trick']
    target_word_vec = word_vectors['stare']

    Patterns = sp.stats.zscore(Patterns, axis=1) / np.sqrt(Patterns.shape[1])
    prime_word_vec = prime_word_vec / np.sqrt(np.dot(prime_word_vec, prime_word_vec))
    target_word_vec = target_word_vec / np.sqrt(np.dot(target_word_vec, target_word_vec))

    a = Patterns.transpose() @ target_word_vec
    print(np.max(np.abs(a)))

    beta = 1.0
    this_d_e, acc = f_Hopfield(Patterns, prime_word_vec, target_word_vec, beta=beta, verbose=False)
    print(this_d_e, acc)

if False:
    for beta in [0.5, 4.0]:
        Patterns = WVM.transpose();
        Patterns = Patterns / np.sqrt(np.sum(Patterns * Patterns, axis=0))
        #Patterns = sp.stats.zscore(Patterns, axis=1) / np.sqrt(Patterns.shape[1])
        h = teg_Hopfield.Hopfield_cont(Patterns, beta=beta)
        target_word_vec = word_vectors['four']
        target_word_vec = target_word_vec / np.sqrt(np.dot(target_word_vec, target_word_vec))
        h.set_activation(target_word_vec)
        e_initial, e_convergence = h.run_to_convergence(verbose=False)
        this_d_e = e_convergence - e_initial
        sims_target = Patterns.transpose() @ target_word_vec
        act_patt = act_patt / np.sqrt(np.dot(act_patt, act_patt))
        sims_conv = Patterns.transpose() @ act_patt
        i_t = np.argmax(sims_target)
        i_c = np.argmax(sims_conv)
        print(words[i_t], words[i_c])
        sort_ind = np.argsort(sims_conv)[::-1]
        top_n = 20
        for n in range(top_n):
            print(n, words[sort_ind[n]], sims_conv[sort_ind[n]])
        plt.plot(sims_conv[sort_ind])
    plt.show()

# Analysis
beta = 1.0
beta_target = 32.0
alpha = 0.25
D = read_data()
WVM, WVM_primes, WVM_targets, words, prime_words, target_words = get_WVM(D)
Scores_per_subj_rt, Scores_per_subj_d_e_first, Scores_per_subj_d_e, Scores_per_subj_b_first, Scores_per_subj_b, Scores_per_subj_acc_conv_steps, per_subj_convergence_steps  = \
    get_scores_per_subj(D, WVM, beta, beta_target, alpha)
analyze_convergence_steps(per_subj_convergence_steps)
stats('RT', Scores_per_subj_rt)
stats('DE_first', Scores_per_subj_d_e_first)
stats('DE', Scores_per_subj_d_e)
stats('Steps', Scores_per_subj_acc_conv_steps)
stats('b_first', Scores_per_subj_b_first)
stats('b', Scores_per_subj_b)

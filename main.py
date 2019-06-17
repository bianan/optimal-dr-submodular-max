"""
Yatao (An) Bian <yatao.bian@gmail.com>
yataobian.com
May 13, 2019.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys
sys.path.append('./' )
import datetime
import pickle
from collections import defaultdict
from absl import flags
from absl import app

from functions import utils
from itertools import product, combinations
from functions.exp_specs import  (solver_names, amazon_categories, amazon_ns,
plot_opt, data_name, all_folds, result_path_elbo, result_path_paelbo)


flags.DEFINE_boolean('debug', True, 'Whether it is in debug mode.')
flags.DEFINE_integer('problem_id', 1, 'Options: 1: ELBO, 2: PA-ELBO.')
flags.DEFINE_string('mode', 'run', 'Options: run: run algorithms; stats: get experimental statistics.')
FLAGS = flags.FLAGS
FLAGS(sys.argv)


class exp_result():
    '''dummy class to act as struct'''
    pass

class exp_config():
    '''dummy class to act as struct'''
    pass

np.random.seed(99)
nm_latent_dims = 3 # number of latent dimensionalities in FLID
Ds = [2, 3, 10]

# Set solvers
if 1 == FLAGS.problem_id:  # elbo
    # solver 4 always needed
    # 2 shrunken FW does not plot trajectory
    one_epoch_solvers = [0, 3, 11]
    multi_epoch_solvers = [4,5,6,9,10,12]
    solver_list = one_epoch_solvers + multi_epoch_solvers

    # path to store generated figures
    data_fig_path = result_path_elbo
    extra_epoch = 5   #real number of epochs is 1 + extra_epoch

    if FLAGS.debug:
        folds = [3]
        folds_iter = [folds]
        category_id = 10
        category_iter = [category_id]

    # run all categories and folds
    else:
        folds_iter = list(combinations (all_folds, 1))
        category_iter =np.arange( len ( amazon_ns))

    """Control whether to plot the true log-partition or not. The true ELBO is from
    Tschiatschek, S., Djolonga, J., and Krause, A.
    Learning probabilistic submodular diversity models via noise contrastive estimation.
    In Proc. International Conference on Artificial Intelligence and Statistics (AISTATS), 2016.
    """
    MIT_TRUE_EVIDENCE = 1

elif 2 == FLAGS.problem_id:  # pa-elbo
    one_epoch_solvers = [0, 3, 11]
    multi_epoch_solvers = [4,5,6,9,10,12]
    solver_list = one_epoch_solvers + multi_epoch_solvers

    data_fig_path = result_path_paelbo

    extra_epoch = 9  # real # epoch is 1 + extra_epoch

    if FLAGS.debug:
        folds = (1,2)
        folds_iter = [folds]
        category_id = 11
        category_iter = [category_id]
    else:
        folds_iter = list (combinations (all_folds, 2) )
        category_iter = np.arange( len ( amazon_ns))

    # there is no true log-partition in this setting.
    MIT_TRUE_EVIDENCE = 0

else:
    pass


nm_solver_names = len(solver_names)
print('solver_list: ', solver_list, 'nm_solver_names:', nm_solver_names)


def run_algorithms():
    for category_id in category_iter:
        print('processing category id: {} ...'.format(category_id))

        for folds in folds_iter:
            category = amazon_categories[category_id]
            n = amazon_ns[category_id]
            co_order = []  #  coordinate order
            for i in range(1 + extra_epoch):
                co_order.append(np.random.permutation(n))

            nm_exps = 1
            Dstepsize = 1

            # used for traj. plotting
            exp_id = 0; param_id = 0
            subfix = '-cate-' + category + '-n-'+ str(n) +\
                 '-dataid-' + str(FLAGS.problem_id) + '-folds-' + str(folds)

            results = defaultdict(exp_result)   # use a dict

            for idx_exp  in range(nm_exps): # repeated experiments
                for i in range(nm_latent_dims):  #i:  param idx
                    Dtmp = Ds[i]
                    # load data
                    [f, grad, gradi, param] = utils.load_flid_data(FLAGS.problem_id,
                                                                   Dtmp,
                                                                   n,
                                                                   data_fig_path,
                                                                   category,
                                                                   folds)

                    print('data: %s, category: %s,  n: %d, D: %d, folds: %s \n'\
                       % ( data_name[FLAGS.problem_id], category,  param.n, Dtmp, str(folds)))

                    for solver_id in solver_list:
                        [opt_x, opt_f, fs, margs, runtime] =\
                         utils.launch_solver(f, grad, gradi,
                            param, solver_id, extra_epoch,
                            co_order=co_order)

                        results[idx_exp, i, solver_id].opt_f = opt_f
                        results[idx_exp, i, solver_id].fs = fs
                        results[idx_exp, i, solver_id].opt_x = opt_x
                        results[idx_exp, i, solver_id].margs = margs
                        results[idx_exp, i, solver_id].runtime = runtime
                        results[idx_exp, i, solver_id].logZ = param.logZ

            # record experimental settings
            expconfig = exp_config()
            expconfig.n = n; expconfig.Ds=Ds; expconfig.nm_exps=nm_exps
            expconfig.Dstepsize=Dstepsize; expconfig.K=nm_latent_dims
            expconfig.solver_list=solver_list; expconfig.exp_id =exp_id
            expconfig.nm_names = nm_solver_names
            expconfig.data_id = FLAGS.problem_id
            expconfig.time = datetime.date.today()

            try:
                original_umask = os.umask(0)
                os.makedirs(data_fig_path, exist_ok=True, mode=0o777)
            finally:
                os.umask(original_umask)
            utils.supermakedirs(data_fig_path, mode=0o777)

            file_name = data_fig_path + 'mfi-' + data_name[FLAGS.problem_id] \
                + subfix

            pickle.dump([results, expconfig], open(file_name +'.pkl', 'wb'))


            # ------- do the plot ---------
            if FLAGS.problem_id == 1:
                ylabel = 'ELBO'
            elif FLAGS.problem_id == 2:
                ylabel = 'PA-ELBO'
            else:
                raise ValueError('Unknown problem id: {}!'.format(FLAGS.problem_id) )

            tight_height = 0.86
            sns.set( )
            sns.set_style('whitegrid')
            from matplotlib import rc
            SMALL_SIZE = 6
            MEDIUM_SIZE = 6
            BIGGER_SIZE = 6
            rc('legend', markerscale=1)
            rc('lines', linewidth=0.4)
            rc('font', size=SMALL_SIZE)
            rc('axes', titlesize=SMALL_SIZE)
            rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
            rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
            rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

            opt_fs = np.zeros( [nm_exps, nm_latent_dims, nm_solver_names] )
            for id in range(nm_exps):
                for i in range(nm_latent_dims):
                    for t in solver_list:
                        opt_fs[id, i, t] = results[id, i, t].opt_f

            opt_mean = np.mean(opt_fs, axis=0).squeeze()
            opt_std = np.squeeze( np.std(opt_fs, axis=0))

            opt_max = np.max(np.max(opt_mean[:, solver_list] ))
            opt_min = np.min(np.min(opt_mean[:, solver_list] ))

            # plot returned function value
            fig_name = data_name[FLAGS.problem_id] + '-' + subfix
            plt.close('all')
            fig_scale = 1
            fWidth = 4*fig_scale
            fHeight = 4*fig_scale

            # by default markersize is 12
            plt_lw = 1; markersize = 5

            def plot_elbo_values(solver_list, subfix='', ylabel='ELBO'):
              # elbo value with legend
              hFig = plt.figure(figsize=(fWidth, fHeight*tight_height ))

              hands = []
              for t in solver_list[1:]:
                  hi = plt.plot(np.arange(len(Ds)), opt_mean[:, t], plot_opt[t],\
                                lw=plt_lw, markersize=markersize)
                  hands.append(hi[0])
                  plt.errorbar(np.arange(len(Ds)), opt_mean[:, t], yerr=opt_std[:, t],\
                                fmt=plot_opt[t],lw=plt_lw, markersize=markersize)

              t = solver_list[0]
              hi = plt.plot(np.arange(len(Ds)), opt_mean[:, t], plot_opt[t], \
                            lw=plt_lw, markersize=markersize)
              hands.append(hi[0])
              plt.errorbar(np.arange(len(Ds)), opt_mean[:, t], yerr=opt_std[:, t], \
                           fmt=plot_opt[t],lw=plt_lw, markersize=markersize)

    #           # plot true logZ
    #           # using plot option 8 in exp_specs
              if MIT_TRUE_EVIDENCE and subfix=='multiepoch':
                  solver_id = solver_list[0]
                  plt.plot( np.arange(len(Ds)) , \
                            [results[exp_id, k, solver_id].logZ for k in range(nm_latent_dims) ], plot_opt[8],\
                                markersize=markersize, lw=plt_lw)
              figure_part = 0.74
              plt.tight_layout(rect= [0.08,0.04, 1, figure_part] )
              newlist = solver_list[1:] + solver_list[0:1]

              plt.legend(hands, [solver_names[i] for i in newlist], \
                  bbox_to_anchor=(-0.09, 1, 1.1, 0.35), loc='lower left', \
                     mode='expand', fontsize='large',\
                     ncol =2)

          #    plt.axis([Ds[0], Ds[-1],   opt_min, opt_max])
          #    plt.axis(  )
              plt.xticks(np.arange(len(Ds)), Ds)
              plt.xlabel('D')
              plt.ylabel(ylabel)
              sns.despine()
              plt.savefig(data_fig_path + fig_name+ subfix+'.pdf', format='pdf', \
                          transparent=False)

              # ------ plot ELBO value without legend
              hFig = plt.figure(figsize=(fWidth, fHeight*tight_height ))
              hands = []
              for t in solver_list[1:]:
                  hi = plt.plot(np.arange(len(Ds)), opt_mean[:, t], plot_opt[t],\
                                lw=plt_lw, markersize=markersize)
                  hands.append(hi[0])
                  plt.errorbar(np.arange(len(Ds)), opt_mean[:, t], yerr=opt_std[:, t],\
                                fmt=plot_opt[t],lw=plt_lw, markersize=markersize)

              t = solver_list[0]
              hi = plt.plot(np.arange(len(Ds)), opt_mean[:, t], plot_opt[t], \
                            lw=plt_lw, markersize=markersize)
              hands.append(hi[0])
              plt.errorbar(np.arange(len(Ds) ), opt_mean[:, t], yerr=opt_std[:, t], \
                           fmt=plot_opt[t],lw=plt_lw, markersize=markersize)

              # plot true logZ
              # plot option 8
              if MIT_TRUE_EVIDENCE and subfix=='multiepoch':
                  plt.plot(np.arange(len(Ds)) , \
                            [results[exp_id, k, solver_id].logZ for k in range(nm_latent_dims) ], plot_opt[8],\
                                markersize=markersize, lw=plt_lw)

              # plt.tight_layout(rect= [0.08,0.04, 1, 1])
              plt.xticks(np.arange(len(Ds)), Ds)
              plt.xlabel('D')
              plt.ylabel(ylabel)
              sns.despine()
              plt.savefig(data_fig_path + fig_name+ subfix+'nolegend.pdf', format='pdf', \
                          transparent=False)
              plt.draw()

            plot_elbo_values(one_epoch_solvers, subfix='1epoch', ylabel=ylabel)
            plot_elbo_values(multi_epoch_solvers, subfix='multiepoch', ylabel=ylabel)


            def plot_trajectories(solver_list, is_multi_epoch=True, ylabel='ELBO'):
              #    --------------- plot trajectories of the solvers-----
              solvers = solver_list.copy()
              if is_multi_epoch:
                subfix2 = '_multiepoch'
                # not plot traj. for solver 2, shrunken FW
                if 2 in solvers:
                  solvers.remove(2)
              else:
                subfix2 = '_1epoch'
              if is_multi_epoch:
                iters = np.arange(0, n*(extra_epoch+1) + 1)
              else:
                iters = np.arange(0, n+1)

              markersize = 1
              for param_id_ in  range(len(Ds)):
                  fig_name = data_name[FLAGS.problem_id] + subfix + \
                      '-traj-expid-' + str(exp_id) + '-D-' + str(Ds[param_id_] )

                  hFig = plt.figure(figsize=(fWidth, fHeight*tight_height ))
                  hands = []
                  for t in solvers[1:]:
                      tr = results[exp_id, param_id_, t].fs
                      if (isinstance(tr, list) and 1==len(tr) ):
                          tr = np.ones( [ len(iters), 1] ) *tr
                      elif not isinstance(tr, list) and  1==tr.size:
                          tr = np.ones( [len(iters), 1] ) *tr

                      hi = plt.plot(iters, tr, plot_opt[t],\
                          linewidth=plt_lw, markersize= markersize)
                      hands.append(hi[0])

                  t = solvers[0]
                  tr = results[exp_id, param_id_, t].fs
                  if 1 ==len(tr):
                      tr = np.ones( [1, len(iters)] ) *tr

                  hi =plt.plot(iters, tr, plot_opt[t],\
                      lw=plt_lw, markersize= markersize)
                  hands.append(hi[0])

                  if is_multi_epoch:
                    # plot  1-epoch vertical line
                    plt.axvline(x=n, color='c', linestyle='--', lw = plt_lw*0.5)

                  # plot true logZ
                  # plot style 8
                  if MIT_TRUE_EVIDENCE and is_multi_epoch:
                      plt.axhline(results[exp_id, param_id_, 4].logZ, color='y', linestyle='-', marker='d',\
                                    markersize=markersize, lw=plt_lw)

                  newlist = solvers[1:] + solvers[0:1]
                  plt.legend(hands, [solver_names[i] for i in newlist], \
                       loc='best', \
                         mode=None, fontsize='x-large',\
                         ncol =1)

                  plt.xlabel(r'Iterations. $D={}$'.format(Ds[param_id_]));
                  plt.ylabel(ylabel)
                  if is_multi_epoch:
                      if MIT_TRUE_EVIDENCE:
                          maxf = results[exp_id, param_id_, 4].logZ
                          plt.ylim(ymin=-2, ymax = 0.4 + maxf)
                      else:
                          plt.ylim(ymin=-2, ymax=0.4+opt_max)

                  sns.despine()
                  plt.savefig( data_fig_path+fig_name+subfix2+'.pdf', fmt='pdf', \
                               transparent=False)

            plot_trajectories(multi_epoch_solvers, is_multi_epoch=True, ylabel=ylabel)
            plot_trajectories(one_epoch_solvers, is_multi_epoch=False, ylabel=ylabel)

"""
Compute experimental statistics for one-epoch solvers.
The statistics is used to generate Table 1 in the paper.
"""
def compute_exp_stats():

    if FLAGS.problem_id == 1:
        # cate x D x solver x fold
        optf_1epoch = np.zeros([len(amazon_categories), len(Ds), nm_solver_names, len(all_folds)])
    else:  # PA-ELBO
       # cate x D x solver x fold_pair
        optf_1epoch = np.zeros([len(amazon_categories), len(Ds), nm_solver_names, len(folds_iter)])

    for category_id in category_iter:
        for folds_id, folds in  enumerate(folds_iter):
            print(folds[0])
            category = amazon_categories[category_id]
            n = amazon_ns[category_id]

            nm_solver = len(solver_list)
            nm_exps = 1
            Dstepsize = 1

            exp_id=0; param_id = 0
            subfix = '-cate-' + category + '-n-'+ str(n) +\
                  '-dataid-' + str(FLAGS.problem_id) + '-folds-'+str(folds)

            file_name = data_fig_path + 'mfi-' + data_name[FLAGS.problem_id] + subfix + '.pkl'
            if os.path.getsize(file_name) > 0:
                results, expconfig = pickle.load(open( file_name, 'rb' ))

            for d in range(nm_latent_dims):
                for s in one_epoch_solvers:
                    optf_1epoch[category_id, d, s, folds_id] = results[0, d, s].opt_f

    if FLAGS.problem_id == 1:
        file_name = os.path.join(data_fig_path, 'optf_1epoch')
    else:
        file_name = os.path.join(data_fig_path, 'optf_1epoch_pa')

    pickle.dump(optf_1epoch, open(file_name +'.pkl', 'wb'))


def main(_):
    if FLAGS.mode == 'run':
        run_algorithms()
    elif FLAGS.mode == 'stats':
        compute_exp_stats()
    else:
        raise ValueError('Unknown mode: {}'.format(FLAGS.mode))

if __name__ == '__main__':
    app.run(main)

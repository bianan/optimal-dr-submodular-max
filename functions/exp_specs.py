"""
Yatao (An) Bian <yatao.bian@gmail.com>
bianyatao.com 
May 13, 2019.
"""
# Fold ids
all_folds = [1,2,3,4,5,6,7,8,9,10]

amazon_categories = ('furniture', #0
                     'media',     #1
                     'diaper',    #2
                     'feeding',   #3
                     'gear',      #4
                     'bedding',   #5
                     'apparel',   #6
                     'bath',      #7
                     'toys',      #8 
                     'health',    #9
                     'strollers', #10
                     'safety',    #11
                     'carseats'   #12
                     )    
# Groundset size of each category
amazon_ns = ( 32, 58, 
              100, 
              100, 100, 
              100, 100,
              100, 62, 
              62, 40, 
              36, 34 )

plot_opt = ('--db', ':^b', '--^b',  # 0, 1, 2
'-.*k', '-.hg',                     # 3 4
 ':Dk', ':vr',                      # 5 6
'--c', '--y',                       # 7 8 
'-' , '--sm', ':sr',                # 9 10 11
'-.<r', '--dr'                      # 12, 13
)
             
plot_opt_signal = ('b', 'm', 'r')

data_name= (
        'NONE',                     # 0
        'ELBO-FLID-Amazon',         # 1
        'PA-ELBO-FLID-Amazon'       # 2
        )

# will be printed as the legend name of the solvers
# None:  means not used
solver_names = ('DR-DoubleGreedy',                      # 0
      'None',                                           # 1
      'Shrunken Frank-Wolfe',                           # 2
      'Submodular-DoubleGreedy',                        # 3 
      r'DG-MeanField-$1/2$',                            # 4 
      'CoordinateAscent-0',                             # 5 
      'CoordinateAscent-1',                             # 6
      'GroundTruth-FLID',                               # 7
      'ExhaustiveSearch',                               # 8
      'CoordinateAscent-Random',                        # 9
      r'DG-MeanField-$1/3$',                            # 10
      'BSCB',                                           # 11
      'BSCB-MultiEpoch',                                # 12
      'SY-Alg1'                                         # 13
      )

"""
number of discretization bins in: 
Soma, T. and Yoshida, Y. Non-monotone dr-submodular
function maximization. In AAAI, volume 17, pp. 898–904, 2017.
"""
NM_BINS = 10

# Path of trained FLID models
flid_model_path = './data/flid_models'


# Folders to store figures and generated pickle files
result_path_elbo = 'results/elbo/'
result_path_paelbo = 'results/paelbo/'


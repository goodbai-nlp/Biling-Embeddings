import os

"""Default parameters"""

#DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data/en-zh/fb/')
#DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data/en-fr/fb/')
#DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data/en-eo/fb/')
#DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data/en-ru/fb/')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data/en-it/')
#DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data/en-eo/fb/')
#DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data/en-zh/fb/')
#DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data/en-es/fb/')
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models/')
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

#EN_WORD_TO_VEC = 'wiki.en.vec'
EN_WORD_TO_VEC = 'en.vec'
#IT_WORD_TO_VEC = 'wiki.fr.vec'
#IT_WORD_TO_VEC = 'wiki.ru.vec'
IT_WORD_TO_VEC = 'it.vec'

# VALIDATION_FILE = 'en-es.5000-6500.txt'
#VALIDATION_FILE = DATA_DIR+'zh-en.5000-6500.txt'
#VALIDATION_FILE = DATA_DIR+'en-zh.5000-6500.txt'
#VALIDATION_FILE = 'en-fr.5000-6500.txt'
VALIDATION_FILE = DATA_DIR+'en-it.5000-6500.txt'
#VALIDATION_FILE = 'en-ru.5000-6500.txt'


# For Wacky dataset:

# EN_WORD_TO_VEC = 'EN.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt'
# IT_WORD_TO_VEC = 'IT.200K.cbow1_wind5_hs0_neg10_size300_smpl1e-05.txt'
# VALIDATION_FILE = 'OPUS_en_it_europarl_test.txt'

# For Procrustes (Supervised):
TRAIN_FILE = 'OPUS_en_it_europarl_train_5K.txt'


# Model Hyper-Parameters
g_input_size = 300     # Random noise dimension coming into generator, per output vector
g_output_size = 300    # size of generated output vector
d_input_size = 300   # cardinality of distributions
d_output_size = 1    # Single dimension for 'real' vs. 'fake'
g_size = 300
######################################################################
d_hidden_size = 2048   # Discriminator complexity
d_learning_rate = 0.1
#d_learning_rate = 0.001
g_learning_rate = 0.1
#g_learning_rate = 0.001
mini_batch_size = 32
d_steps = 5  # 'k' steps in the original GAN paper. Can put the discriminator on higher training freq than generator

dis_hidden_dropout = 0
dis_input_dropout = 0.1

num_epochs = 200

g_steps = 1
smoothing = 0.1   # As per what is mentioned in the paper
#beta = 0.001         # for en-other
beta = 0.001      # Set this to 0.0001 for en-zh 0.0001
clip_value = 0
recon_weight = 1
gen_activation = 'leakyrelu'
dis_activation = 'leakyrelu'
dis_hidden_dropout = 0
dis_input_dropout = 0.1

center_embeddings = 0    # Set this to 1 for en-zh fb dataset
norm_embeddings = 0    # Set this to 1 for en-zh fb dataset

# Training
iters_in_epoch = 100000
most_frequent_sampling_size = 75000   # Paper mentions this
print_every = 1
lr_decay = 0.98
lr_shrink = 0.5
lr_min = 1e-6
add_noise = 0
noise_mean = 1.0
noise_var = 0.2
num_random_seeds = 10    # Number of different seeds to try

# Validation
K = 5
top_frequent_words = 200000

# refinement
refine_top = 15000
cosine_top = 10000

# data processing, train or eval
mode = 1

csls_k = 10
dict_max_top = 10000

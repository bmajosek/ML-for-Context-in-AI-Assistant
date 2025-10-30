MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

LEARNING_RATE = 2e-5
BATCH_SIZE = 32
EPOCHS = 10

LOSS_FUNCTION = 'triplet'  # 'triplet' or 'cross_entropy'

MAX_TRAIN_SAMPLES = 20000
MAX_VAL_SAMPLES = 500

OUTPUT_DIR = './fine_tuned_model'
PLOT_PATH = './training_loss.png'

EVAL_K = 10


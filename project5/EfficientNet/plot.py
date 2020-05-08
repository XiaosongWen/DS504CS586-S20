import pickle
from utils import plot_summaries


PLOT_NAME1 = 'Train1_LossAndAccuracy.png'
PLOT_NAME2 = 'Train1_Recall.png'

with open('history.pkl', 'rb') as f:
    history = pickle.load(f)
plot_summaries(history, PLOT_NAME1, PLOT_NAME2)
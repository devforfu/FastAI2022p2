import gzip
import pickle
from pathlib import Path
from urllib.request import urlretrieve

MNIST_URL = "https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz?raw=true"

DATA_DIR = Path(__file__).parent/"data"

def get_mnist(log=True, path_data=DATA_DIR):
    path_data.mkdir(exist_ok=True)
    path_gz = path_data/"mnist.pkl.gz"
    if not path_gz.exists(): 
        urlretrieve(MNIST_URL, path_gz)
    with gzip.open(path_gz, 'rb') as f: 
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    data = x_train, y_train, x_valid, y_valid
    if log:
        print(f"train: x.shape={x_train.shape}, y.shape={y_train.shape}")
        print(f"valid: x.shape={x_valid.shape}, y.shape={y_valid.shape}")
    return data

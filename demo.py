from agldl import AGLDL
from utils import report
from mord.threshold_based import LogisticAT
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.io import loadmat

# load data
dataset = loadmat('ns')
X, D = dataset['features'], dataset['label_distribution']
Xr, Xs, Dr, Ds = train_test_split(X, D, test_size=0.3, random_state=123)
scaler = MinMaxScaler().fit(Xr)
Xr, Xs = scaler.transform(Xr), scaler.transform(Xs)

# AGLDL
alpha = 4 # Set the hyper-parameter from {1/10,1/8,1/6,1/4,1/2,1,2,4,6,8,10}
cgl = MultiOutputClassifier(LogisticAT(alpha=alpha))
model = AGLDL(CGL_predictor=cgl).fit(Xr, Dr)
# predict
Dhat = model.predict(Xs)
# show performance
report(Dhat, Ds)

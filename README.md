# AGLDL
Adaptive-Grained Label Distribution Learning
 
## Environment
python=3.13.5, numpy=2.3.1, scipy=1.15.3, scikit-learn=1.6.1, mord=0.7.

## Reproducing
Change the directory to this project and run the following command in terminal.
```Terminal
python demo.py
```


## Usage
Here is a simple example of using AGLDL.
```python
from agldl import AGLDL
from utils import report
from sklearn.model_selection import train_test_split

# load data
X, D = load_dataset('dataset_name') # this api should be defined by users
Xr, Xs, Dr, Ds = train_test_split(X, D)

# AGLDL
model = AGLDL().fit(Xr, Dr)
# predict
Dhat = model.predict(Xs)
# show performance
report(Dhat, Ds)
```

## Datasets
- The datasets used in our work is partially provided by [PALM](http://palm.seu.edu.cn/xgeng/LDL/index.htm)
- Emotion6: [http://chenlab.ece.cornell.edu/people/kuanchuan/index.html](http://chenlab.ece.cornell.edu/people/kuanchuan/index.html)
- Twitter-LDL and Flickr-LDL: [http://47.105.62.179:8081/sentiment/index.html](http://47.105.62.179:8081/sentiment/index.html)

## Paper
```latex

@inproceedings{Lu2025AGLDL, 
    title={Adaptive-Grained Label Distribution Learning},
    author={Lu, Yunan and Li, Weiwei and Liu, Dun and Li, Huaxiong and Jia, Xiuyi},
    booktitle={AAAI Conference on Artificial Intelligence},
    year={2025},
    pages={19161-19169}
}
```

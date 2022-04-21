# Distributional Sliced Information

<img src="https://render.githubusercontent.com/render/math?math=\tau (X,Y)=\sup\oint{I}(\theta^TX,\phi^TY)d\sigma(\theta,\phi)">



# Transport Dependency

This package provides a few simple functions to compute the transport correlation and other optimal transport based dependency measures as defined here: [Transport dependency: Optimal transport based dependency measures](https://arxiv.org/abs/2105.02073).

#### Pip installation

Since the library relies on the [Python Optimal Transport package](https://github.com/PythonOT) it is necessesary to first install 
some dependencies separately with the comand:
```console
pip install numpy scipy Cython
```
You can then install the package by typing:

```console
pip install transport_dependency
```

#### Short example

```python
# Compute transport correlation and test for independence
from transport_dependency import tcor, permutation_test
x = [[1, 2, 3],
     [4, 5, 6],
     [7, 8., 9]]
y = [1, 0, 0.3]
res = tcor(x, y)
test = permutation_test(x, y, coef=tcor)
p_val = test["p_val"]
``` 

#### References

[1] . G. Nies, T. Staudt, and A. Munk. [Transport dependency: Optimal transport based dependency measures](https://arxiv.org/abs/2105.02073)
arXiv Preprint, 2021.





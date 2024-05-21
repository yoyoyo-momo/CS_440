import numpy as np

def estimate_geometric(PX):
    '''
    @param:
    PX (numpy array of length cX): PX[x] = P(X=x), the observed probability mass function

    @return:
    p (scalar): the parameter of a matching geometric random variable
    PY (numpy array of length cX): PY[x] = P(Y=y), the first cX values of the pmf of a
      geometric random variable such that E[Y]=E[X].
    '''
    EY = 0
    for i in range(len(PX)):
      EY += i * PX[i]
    p = 1 / (EY + 1)
    PY = []
    for i in range(len(PX)):
      py = p * (1 - p) ** i
      PY.append(py)
    return p, PY

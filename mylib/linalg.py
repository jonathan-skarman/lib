
def forsub(L,bs):
  """
  Solves a linear system Lx=bs where L is a lower triangular matrix using forward substitution.
  
  :param L: Lower triangular matrix
  :param bs: constant vector
  """
  import numpy as np
  n = bs.size
  xs = np.zeros(n)
  for i in range(n):
    xs[i] = (bs[i] - L[i,:i]@xs[:i])/L[i,i]
  return xs

def backsub(U,bs):
  """
  Solves a linear system Ux=bs where U is an upper triangular matrix using backward substitution.
  
  :param U: Upper triangular matrix
  :param bs: constant vector
  """
  import numpy as np
  n = bs.size
  xs = np.zeros(n)
  for i in reversed(range(n)):
    xs[i] = (bs[i] - U[i,i+1:]@xs[i+1:])/U[i,i]
  return xs
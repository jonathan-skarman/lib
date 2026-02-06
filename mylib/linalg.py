
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

def ludec(A):
  """
  Performs LU decomposition of a square matrix A into a lower triangular matrix L and an upper triangular matrix U such that A=LU.
  
  :param A: Square matrix to decompose
  """
  import numpy as np
  n = A.shape[0]
  U = np.array(A, dtype=float)
  L = np.identity(n, dtype=float)
  
  for j in range(n-1):
    for i in range(j+1,n):
      coeff = U[i,j]/U[j,j]
      U[i,j:] -= coeff*U[j,j:]
      L[i,j] = coeff
  return L, U

def lusolve(A,bs):
  """
  Solves a linear system Ax=bs using LU decomposition.
  
  :param A: Coefficient matrix
  :param bs: constant vector
  """
  L, U = ludec(A)
  ys = forsub(L,bs)
  xs = backsub(U,ys)
  return xs
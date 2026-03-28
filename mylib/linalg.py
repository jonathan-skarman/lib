
import numpy


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

def power(A,kmax=6):
	import numpy as np
	zs = np.ones(A.shape[0])
	qs = zs/mag(zs)
	for k in range(1,kmax):
		zs = A@qs
		qs = zs/mag(zs)
		#print(k,qs)
	lam = qs@A@qs
	return lam, qs

def mag(xs):
	import numpy as np
	return np.sqrt(np.sum(xs*xs))

def qrdec(A):
	import numpy as np
	n = A.shape[0]
	Ap = np.copy(A)
	Q = np.zeros((n,n))
	R = np.zeros((n,n))
	for j in range(n):
		for i in range(j):
			R[i,j] = Q[:,i]@A[:,j]
			Ap[:,j] -= R[i,j]*Q[:,i]
		R[j,j] = mag(Ap[:,j])
		Q[:,j] = Ap[:,j]/R[j,j]
	return Q, R

def qrmet(inA, kmax=100):
	import numpy as np
	A = np.copy(inA).astype(float)
	n = A.shape[0]
	V = np.eye(n)
	for k in range(kmax):
		Q, R = qrdec(A)
		A = R @ Q
		V = V @ Q
	qreigvals = np.diag(A)
	return qreigvals, V
def finite_differences(x, y, method="forward"):
	import numpy as np
	
	if len(x) != len(y):
		raise ValueError("x and y must have the same length")
	
	n = x[1] - x[0]
	for i in range(1, len(x)):
		if not np.isclose(x[i] - x[i-1], n):
			raise ValueError("x must have consistent step length")
	
	dy = np.array([])
	
	if method == "forward":
		for i in range(1, len(y)):
			dy = np.append(dy, ((y[i] - y[i-1]) / n))
		dy = np.append(dy, np.nan)

	if method == "backward":
		dy = np.append(dy, np.nan)
		for i in range(0, len(y)-1):
			dy = np.append(dy, (y[i+1] - y[i]) / n)

	if method == "center":
		dy = np.append(dy, np.nan)
		for i in range(1, len(y)-1):
			dy = np.append(dy, (y[i+1] - y[i-1]) / (2*n))
		dy = np.append(dy, np.nan)

	return dy
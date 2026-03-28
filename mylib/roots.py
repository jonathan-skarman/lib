def bisection(func, lst, eps=1e-3):
	"""
	finds a root of func in the interval [lst[0], lst[1]] using the bisection method. If there is no root, returns None.
	
	:param func: the function to find the root of
  :param lst: a list of two numbers, the interval to search for the root in
  :param eps: the desired precision of the result
  :return: a number that is a root of func, or None if there is no root
	"""
	if func(lst[0]) * func(lst[1]) >= 0:
		return None
	else:
		too_big = True
		while too_big:
			if func((lst[1] + lst[0]) / 2) < eps:
				return (lst[1] + lst[0]) / 2
			elif func(lst[0]) * func((lst[1] + lst[0]) / 2) < 0:
				lst[1] = (lst[1] + lst[0]) / 2
			else:
				lst[0] = (lst[1] + lst[0]) / 2


def newton(func, derfunc, guess, eps=1e-3):
	"""
	finds a root of func using the Newton-Raphson method. If there is no root, returns None.

	:param func: the function to find the root of
	:param derfunc: the analytical derivative of func
	:param guess: the initial guess for the root
	:param eps: the desired precision of the result
	:return: a number that is a root of func
	"""
	too_big = True
	while too_big:
		if abs(func(guess)) < eps:
			return guess
		else:
			guess = guess - func(guess) / derfunc(guess)
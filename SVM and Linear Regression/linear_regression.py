
from numpy import *

# Linear Regression

def get_error_from_points(b, m, points):
	total_error = 0
	for i in range(0, len(points)):
		x = points[i, 0]
		y = points[i, 1]
		total_error += (y - (m * x + b)) **2
	return total_error / float(len(points))


def step_gradient(c_b, c_m, points, learning_rate):
	#gradient descent
	b_gr = 0
	m_gr = 0
	N = float(len(points))
	for i in range(0, len(points)):
		x = points[i,0]
		y = points[i,1]
		b_gr += -(2/N) * (y - ((c_m * x) + c_b))
		m_gr += -(2/N) * x * (y - ((c_m * x) + c_b))
	new_b = c_b - (learning_rate * b_gr)
	new_m = c_m - (learning_rate * m_gr)
	return [new_b, new_m]


def gradient_descent_fn(points,s_b, s_m, learning_rate, num_iterations):
	b = s_b
	m = s_m

	for i in range(num_iterations):
		b, m = step_gradient(b, m, array(points), learning_rate)
	return [b, m]

def run():
	points = genfromtxt('data.csv', delimiter=',')
	#hyperparameters // Alpha
	learning_rate = 0.0001
	# y = mx+b //ecuacion  de la recta
	initial_b = 0
	initial_m = 0
	num_iterations = 1000
	[b, m] = gradient_descent_fn(points, initial_b, initial_m, learning_rate, num_iterations)
	print(b)
	print(m)



if __name__ == '__main__':
	run()
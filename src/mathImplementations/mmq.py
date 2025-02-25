#Separando os valores, ordenados pela temperatura
xi = df['Temperature'].sort_values()
yi = df['Revenue']

#Método: MMQ
import sympy as sy
import numpy as np
from sympy.abc import x,y
import matplotlib.pyplot as plt

def calculate_a(xi,yi):
    n=len(xi)
    return (n*np.sum(xi*yi)-np.sum(xi)*np.sum(yi))\
          /(n*np.sum(xi**2)-np.sum(xi)**2)

a=calculate_a(xi,yi);a

def calculate_b(xi,yi,a1):
    n=len(xi)
    return (np.sum(yi)/n)-a1*(np.sum(xi)/n)

b=calculate_b(xi,yi,a);b


#MMQ Linear
y=b+a*x;y


plt.scatter(x, y, color='red', label='Pontos')
plt.plot(x, y, label="MMQ Linear")
plt.title("MMQ Linear")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

#MMQ Quadrática
y=b+a*x**2;y

plt.scatter(x, y, color='red', label='Pontos')
plt.plot(x_plot, y_plot, label=title)
plt.title("MMQ Linear")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
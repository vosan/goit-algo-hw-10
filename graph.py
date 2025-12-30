import matplotlib.pyplot as plt
import numpy as np


# Define the function and integration bounds
def f(x):
    return x**2


a = 0  # Lower bound
b = 2  # Upper bound

# Create the x range for plotting
x = np.linspace(-0.5, 2.5, 400)
y = f(x)

# Create figure
fig, ax = plt.subplots()

# Plot the function
ax.plot(x, y, 'r', linewidth=2)

# Fill the area under the curve between a and b
ix = np.linspace(a, b)
iy = f(ix)
ax.fill_between(ix, iy, color='gray', alpha=0.3)

# Plot settings
ax.set_xlim([x[0], x[-1]])
ax.set_ylim([0, max(y) + 0.1])
ax.set_xlabel('x')
ax.set_ylabel('f(x)')

# Add integration bounds and title
ax.axvline(x=a, color='gray', linestyle='--')
ax.axvline(x=b, color='gray', linestyle='--')
ax.set_title(f'Integration plot for f(x) = x^2 from {a} to {b}')
plt.grid()
plt.show()

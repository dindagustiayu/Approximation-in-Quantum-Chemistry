[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]()

# Approximation in Quantum Chemistry

Why humans aren't very good at solving equation?. In Alan Turing movie, we understand that computers are much better at solving complex problems than we are. In quantum physics we start with examples like the harmonic oscillator or the hydrogen atom and then proudly demonstrate how clever we all are by solving the $Schr\ddot{o} dinger$ equation exactly. But there are very very few examples where we can write down the solution in closed form. For the vast majority of problems, the answer is omething complicated that isn't captured by some simple mathematical formula. For these problems we need to develop different tools. 

There are many complex problems in quantum mechanics. Instead, we hope we can build a collection of tools. Then, whenever we're faced with a new problem we can root around in our toolbox, hoping to find a method that works. These are some approximation methods to solve quantum mechanics problems:
1. The variational method
2. Perturbation theory
3. Hartree-fock approximation
4. WKB methods (semi-classical)

## The Variational Method
The _variational method_ provides a simple way to place an uppper bound on the ground state energy of any quantum system and is particularly useful when trying to demonstrate that bound state exist. In some chases, it can also be used to estimate higher energy levels too. 

### Application of the variational method to the particle in a box problem
In standard problem of a particle of mass $m$ with zero potential energy confined to a one-dimensional box extending from the origin to the point $x=L$, a Hamiltonian operator of the form

<p align='center'>
    $$\hat{H} = - \frac{\hbar^2}{2m} \ \frac{d^2}{dx^2}$$
</p>

First consider a one-dimensional box at length $L$ lying along the $x$ axis with the center of the box at the origin so that the ends of the box  are at $x= -L / 2$ and at $x = L/ 2$. Exact wave functions for a particle of mass $m$ in such a box are guven by
<p align='center'>
    $$\psi (x) = \left \{\begin{array} \ \sqrt{\frac{2}{L}} sin \left(\frac{\pi n x}{L} \right) \\ \sqrt{\frac{2}{L}} \ cos \left(\frac{\pi n x}{L} \right) \end{array} \right \}$$ 
</p>

with corresponding energies of 
<p align='center'>
    $$E_n = \frac{h^2 n^2}{8m L^2}$$
</p>

### P8.4.4 Exercise
Consider a one-dimensional quantum mechanical particle in a box $(-1 \leq x \leq 1)$ described by the $Schr \ddot{o} dinger$ equation:
<p align='center'>
    $$-\frac{d^2 \psi}{dx^2} = E \psi$$
</p>

in energy units for which $\hbar^2 / (2m) = 1$ with $m$ the mass of the particle. The exact solution for the ground state of this systemis given by
<p align='center'>
    $$\psi = cos \left(\frac{\pi x}{2} \right), \quad E = \frac{\pi^2}{4}$$
</p>

An approximate solution may be arrived at using the _variational principle_ by minimizing the expectation value of the energy of a trial wavefunction,
<p align='center'>
    $$\psi_{trial} = \sum_{n=0}^{N} a_n \phi_n (x)$$
</p>

with respect to the coefficients $a_n$. Taking the basis functions to have the following symmetrized polynomial form,
<p align='center'>
    $$\phi_n = (1-x)^{N - n + 1} (x + 1)^{n + 1}$$
</p>

use `scipy.optimize.minimize` and `scipy.integrate.quad` to find the optimum value of the expectation value (Rayleigh-Ritz ratio):
<p align='center'>
    $$\mathcal{E} = \frac{\langle \psi_{trial} | \hat{H} | \psi_{trial}}{\langle \psi_{trial} | \psi_{trial} \rangle} \rangle = \frac{\int_{-1}^{1} \psi_{trial} \frac{d^2}{dx^2} \psi_{trial} \ dx}{\int_{-1}^{1} \psi_{trial} \psi_{trial} \ dx}$$
</p>

Compare the estimated energy, $\mathcal{E}$, with the exact answer for $N = 1, \ 2, \ 3, \ 4$.

This Python script to illustrate the variational method applied to the Particle in a box with initial variables: mass and length of the box are 1 and 2, respectively.

```Python
import numpy as np
import matplotlib.pyplot as plt

# Particle mass, box length.
mass, L = 1, 2
x = np.linspace(-1, 1, 1000)

def psi(n):
    """ Return the exact particle in a box wavefunction for quantum number n."""
    if n % 2:
        return np.cos(np.pi * n * x / L)
    return np.sin(np.pi * n * x / L)

def E(n):
    return (n * np.pi)**2 / 2 / mass / L**2
    
def plot_wavefunction(n):
    En = E(n)
    plt.plot(x, psi(n), label=f'$E_n = {En}$')
    plt.xlabel(r'$x \ / \ a_0$')
    plt.ylabel(r'$\Psi (x) \ / \ a_0^{-1/2}$')
    plt.title(r"The ground state energy / $E_n$")
    plt.legend()

# Ground state n = 1
n = 1
plot_wavefunction(n)
plt.show()
```
<p align='center'>
  
</p>

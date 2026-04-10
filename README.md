[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/dindagustiayu/Approximation-in-Quantum-Chemistry/blob/main/Variational%20Principle.py)

# Approximation in Quantum Chemistry

Why humans aren't very good at solving equations?. In the Alan Turing movie, we understand that computers are much better at solving complex problems than we are. In quantum physics, we start with examples like the harmonic oscillator or the hydrogen atom and then proudly demonstrate how clever we all are by solving the $Schr\ddot{o}dinger$ equation exactly. But there are very very few examples where we can write down the solution in closed form. For the vast majority of problems, the answer is something complicated that isn't captured by some simple mathematical formula. For these problems we need to develop different tools. 

There are many complex problems in quantum mechanics. Instead, we hope we can build a collection of tools. Then, whenever we're faced with a new problem, we can root around in our toolbox, hoping to find a method that works. These are some approximation methods to solve quantum mechanics problems:
1. The variational method
2. Perturbation theory
3. Hartree-fock approximation
4. WKB methods (semi-classical)

## The Variational Method
The _variational method_ provides a simple way to place an uppper bound on the ground state energy of any quantum system and is particularly useful when trying to demonstrate that bound state exist. In some chases, it can also be used to estimate higher energy levels too. 

### Application of the variational method to the particle in a box problem
In the standard problem of a particle of mass $m$ with zero potential energy confined to a one-dimensional box extending from the origin to the point $x=L$, a Hamiltonian operator of the form

<p align='center'>
    $$\hat{H} = - \frac{\hbar^2}{2m} \ \frac{d^2}{dx^2}$$
</p>

First, consider a one-dimensional box at length $L$ lying along the $x$ axis with the center of the box at the origin so that the ends of the box  are at $x= -L / 2$ and at $x = L/ 2$. Exact wave functions for a particle of mass $m$ in such a box are given by
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

in energy units for which $\hbar^2 / (2m) = 1$ with $m$ the mass of the particle. The exact solution for the ground state of this system is given by
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
    $$\mathcal{E} = \frac{\langle \psi_{trial} |\hat{H}| \psi_{trial}}{\langle \psi_{trial} | \psi_{trial} \rangle} \rangle = \frac{\int_{-1}^{1} \psi_{trial} \frac{d^2}{dx^2} \psi_{trial} \ dx}{\int_{-1}^{1} \psi_{trial} \psi_{trial} \ dx}$$
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
  <img src = "https://github.com/dindagustiayu/Approximation-in-Quantum-Chemistry/blob/main/Documentation%20svg/The%20ground%20state%20energy.svg">
</p>

The approximate wavefunction chosen will be a polynomial in $x$. For convenience, we will work with on the _Hartree atomic unit system_ in which $\hbar = 1$ and take $M=1$ and $L=1$, so that the "box" lies between $-1 \leq x \leq 1$. 

We need functions to set up the polynomial from the coefficient parameters, $x$, and to evaluate the normalization integral, and the Rayleigh-Ritz, $\langle E \rangle = \langle \phi |\hat{H}| \phi \rangle / \langle \phi | \phi \rangle$.

```Python
import numpy as np
from scipy.optimize import minimize
from numpy.polynomial import Polynomial 
import matplotlib.pyplot as plt

def phi_t(a):
    ncoeffs = len(a) * 2+ 3
    coeffs = np.zeros(ncoeffs)
    coeffs[0] = -(1 + sum(a))
    coeffs[2:ncoeffs -1:2] = a
    coeffs[-1] = 1
    return Polynomial(coeffs)

def get_N2(phi):
    den = (phi * phi). integ()
    return den(1) - den(-1)

def rayleigh_ritz(a):
    phi = phi_t(a)
    phipp = phi.deriv(2)
    num = -(phi * phipp). integ() / 2

    N2 = get_N2(phi)
    return (num(1) - num (-1)) / N2

def get_approx(m):
    a0 = [1] * m
    res = minimize(rayleigh_ritz, a0)
    return res

E1 = E(1)
mmax = 7
Eapprox = [None] * (mmax + 1)
Eapprox[1] = 5 /4 
a = {}

for m in range(2, 7):
    res = get_approx(m)
    Eapprox[m] = res['fun']
    a[m] = res['x']

print('m <E> / Eh error')
error_ppm = [None] * (mmax + 1)
for m in range (1, 7):
    error_ppm[m] = (Eapprox[m] - E1 / E1 * 1.e6)
    print(f'{m:.7f} {Eapprox[m]:.7f} {error_ppm[m]:>9.3f} ppm')
```
```
m <E> / Eh error
1.0000000 1.2500000 -999998.750 ppm
2.0000000 1.2337006 -999998.766 ppm
3.0000000 1.2337021 -999998.766 ppm
4.0000000 1.2337046 -999998.766 ppm
5.0000000 1.2337025 -999998.766 ppm
6.0000000 1.2337013 -999998.766 ppm
```
All the approximation wavefunctions apart from the quadratic one, overlap with the true ground state wavefunction, $\psi$. It might be better to plot the _difference_ between the approximation.

```Python
import numpy as np
import matplotlib.pyplot as plt

# Define x grid and reference wavefunction
x = np.linspace(-1, 1, 1000)
n = 1
e = 1e-3 

def get_phi_approx(m):
    if m == 1:
        return np.sqrt(15/16) * (1 - x**2)
    else:
        phi = phi_t(a[m])
        if phi(0) < 0:
            phi = -phi
        return phi(x) / np.sqrt(get_N2(phi))

line_styles=['-', '--', ':', '-.']
plt.plot([-1, 1], [0,0], c='k', lw=1)

for m in range(1, mmax):
    diff = psi(n) - get_phi_approx(m)
    style = line_styles[(m-1) % len(line_styles)]
    plt.plot(x, diff, linestyle=style, label=f'$\\Delta \\phi_{m}$ (err={error_ppm[m]:.1f} ppm)')

plt.ylim(-2*e, 2*e)
plt.xlabel(r'$x \ / \ a_0$')
plt.ylabel(r'$(\psi - \phi_m) \ / \ a_0^{-1/2}$')
plt.title("Wavefunction Approximation Error vs Energy Error")
plt.legend()
plt.savefig('Wavefunction approximation error vs energy error.svg', bbox_inches='tight')
plt.show()
```
<p align='center'>
    <img src="https://github.com/dindagustiayu/Approximation-in-Quantum-Chemistry/blob/main/Documentation%20svg/Wavefunction%20approximation%20error%20vs%20energy%20error.svg">
</p>
The figure shows a comparison of the error to the ground state energy on the particle-in-a-box system using a polynomial function with an increasing number of terms.

### P22.2 Exercise
The one-dimensional quartic oscillator is one characterized by a potential energy proportional to the fourth power of the displacement. Taking the Hamiltonian for a quantum mechanical quartic oscillator to be
<p align='center'>
    $$\hat{H} = -\frac{\hbar^2}{2 \mu} \ \frac{d^2}{dx^2} + \frac{1}{2} kx^4$$
</p>

minimize the expectation energy, $\langle E \rangle$, of the trial wavefunction $\psi = exp(-\alpha x^2 / 2)$ with respectto its parameters,
- (a) numerically, using `scipy.optimize.minimize` or `scipy.optimize.minimize_scalar`, $\alpha$ 
- (b) analytically by differentiation of $\langle E \rangle$ with respect to $\alpha$.

### Solution
We need the seconde derivative of the wavefunction:
1. The first derivative

<p align='center'>
    $$\begin{align} \psi &= e^{-\alpha q^2 / 2} \\ \frac{d \psi}{dq} &= e^{-\alpha q^2 / 2} \cdot (- \alpha q) = -\alpha q \psi \end{align}$$
</p>

2. The second derivative

<p align='center'>
    $$\begin{align} \frac{d^2 \psi}{dq^2} &= (\alpha) \psi + (- \alpha q)(- \alpha q \psi) \\ \frac{d^2 \psi}{dq^2} &= -\alpha \psi + \alpha^2 q^2 \psi \\ &=\alpha (\alpha q^2 - 1) \psi \end{align}$$
</p>

3. Define a function to calculate the Rayleigh-Ritz ratio,

<p align='center'>
    $$\langle E' \rangle = \frac{\langle \psi | \hat{H} | \psi \rangle}{\langle \psi | \psi \rangle}$$
</p>

where

<p align='center'>
    $$\begin{align} \langle \psi | \hat{H} | \psi \rangle &=\int_{- \infty}^{\infty} \psi \left(-\frac{1}{2} \frac{d^2 \psi}{dq^2} + \frac{1}{2} q^4 \psi \right) dq \\ &=\frac{1}{2} \int_{-\infty}^{\infty} (q^4 - \alpha^2 q^2 + \alpha) \psi^2 \ dq \end{align}$$
</p>

4. Bracket the minimum in $\langle E' \rangle (\alpha)$ using the value $\alpha_a = 1.0, \ \alpha_b = 1.5 \ \mbox{and} \ \alpha_c = 2.0, \ \mbox{since} \langle E' \rangle (\alpha_a) > \langle E' \rangle (\alpha_b) \ \mbox{and} \ \langle E' \rangle (\alpha_c) > \langle E' \rangle (\alpha_b)$.

```Python
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

# 1. first derivative
def psi(q, alpha):
    return np.exp(-alpha * q**2 /2)

# 2. Second derivative
def psi2(q, alpha):
    return psi(q, alpha)**2

# 3. Calculate the Rayleigh-Ritz
def rayleigh_ritz(alpha):
    def func(q, alpha):
        return 0.5 * (q**4 - (alpha * q)**2 + alpha) * psi2(q, alpha)

# Final integration
    num, _ = quad(func, -np.inf, np.inf, args=(alpha,))
    det,_ = quad(psi2, -np.inf, np.inf, args=(alpha,))
    return num /det

# Final approximation
minimize_scalar(rayleigh_ritz, bracket=(1, 1.5, 2))
```
```
 message: 
          Optimization terminated successfully;
          The returned value satisfies the termination criteria
          (using xtol = 1.48e-08 )
 success: True
     fun: 0.5408435888652783
       x: 1.4422495872637284
     nit: 10
    nfev: 13
```
The optimum value of $\alpha$ is $1.422$, giving an energy of $\langle E' \rangle = 0.541$ in units of $(k \hbar^4 / \mu^2)^{1/3}$

5. For the analytical solution, define,
   
<p align='center'>
    $$In = \int_{\infty}^{\infty} q^{2n} e^{- \alpha q^2} \ dq$$
</p>

integrate by parts to derive the recursion relation

<p align='center'>
    $$I_n = \frac{2n -1}{2 \alpha} I_{n-1}$$
</p>

and note that $I_0 = \sqrt{\pi / \alpha}$.

__Solution__:
    
<p align='center'>
    $$\begin{align} I_n &= \int_{- \infty}^{\infty} q^{2n} e^{- \alpha q^2} \ dq, \quad u = q^{2n-1} \ \mbox{and} \ dv=qe^{- \alpha q^2} \ dq \\ \int u \ dv &= uv - \int v \ dv \\ I_n &=\int_{- \infty}^{\infty} q^{2n-1} qe^{- \alpha q^2} \ dq \\ &= \left[q^{2n-1} \left(- \frac{1}{2 \alpha} \right) e^{- \alpha q^2} \right]_{- \infty}^{\infty} - \int_{- \infty}^{\infty} (2n - 1)q^{2n-2} \left(- \frac{1}{2 \alpha} \right) e^{- \alpha q^2} \ dq \\ &=0 - \frac{2n -1}{2 \alpha} I_{n-1} \end{align}$$
</p>

The integrals we need are therefore $I_0 = \sqrt{\pi / \alpha}$ given,
<p align='center'>
    $$\begin{align} I_1 &=\frac{1}{2 \alpha} I_0 \\ I_2 &=\frac{3}{2 \alpha} I_1 \\ &=\frac{3}{4 \alpha^2} I_0 \end{align}$$ 
</p>

The Rayleigh-Ritz ratio is therefore:

<p align='center'>
    $$ \begin{align} \langle E' \rangle = \frac{\langle \psi | \hat{H} | \psi \rangle}{\langle \psi|\psi \rangle} &= \frac{1}{2 I_0} \int_{- \infty}^{\infty} (q^4 - \alpha^2 q^2 + \alpha) \psi^2 \ dq \\ &=\frac{1}{2 I_0} [I_2 - \alpha^2 I_1 + \alpha I_0] &= \frac{1}{2 I_0} \left [\frac{3}{4 \alpha^2} I_0 - \alpha^2 \frac{1}{2 \alpha} I_0 + \alpha I_0 \right] \\ &=\frac{3}{8 \alpha^2} + \frac{\alpha}{4} \end{align}$$
</p>

This function has a minimum at
   
<p align='center'>
    $$\begin{align} \frac{d \langle E' \rangle}{d \alpha} &= - \frac{3}{4 \alpha^3} + \frac{1}{4} \\ \alpha &=3^{1/3} \end{align}$$
</p>

This Python script to confirm the numerical result with integrals:

```Python
alpha = 3**(1/3)
print(f'alpha = {alpha:.3f}')
print(f"optimum <E'> = {rayleigh_ritz(alpha):.3f}")
```
```
alpha = 1.442
optimum <E'> = 0.541
```
The optimum value of $\alpha$ is $1.422$, giving an energy of $\langle E' \rangle = 0.541$. We can learn more by plotting this $\langle E' \rangle$ as a function of $\alpha$:

```Python
import matplotlib.pyplot as plt

alpha_grid = np.linspace(0.5, 2.5, 25)
Eexp_grid = [rayleigh_ritz(alpha) for alpha in alpha_grid]
plt.plot(alpha_grid, Eexp_grid, label=f"$optimum < E' > = {rayleigh_ritz(alpha):.3f}$")
plt.xlabel(r'$q$')
plt.ylabel(r"$\langle E' \rangle $")
plt.title(r"The differentiation of $\langle E \rangle \ to \ \alpha$")
plt.legend()
plt.savefig('The differentiation 0f E.svg', bbox_inches='tight')
plt.show()
```
<p align='center'>
    <img src="https://github.com/dindagustiayu/Approximation-in-Quantum-Chemistry/blob/main/Documentation%20svg/The%20differentiation%200f%20E.svg">
</p>

# flex

**Fourier-Laguerre expansions for images of galaxies.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ObservationalExpansions/flex/blob/main/LICENSE)

[![Coverage Status](https://coveralls.io/repos/github/ObservationalExpansions/flex/badge.svg?branch=main)](https://coveralls.io/github/ObservationalExpansions/flex?branch=main)


## Installation

Installation of `flex` currently proceeds from local builds after cloning this repository:
```
git clone https://github.com/ObservationalExpansions/flex.git
```

```
pip install .
```


## Quickstart Example

Drawing from a uniform disc.

```python
import numpy as np
import flex

N = 10_000_000

# Generate uniformly in polar coordinates
r = np.sqrt(np.random.uniform(0, 4, N)) 
theta = np.random.uniform(0, 2*np.pi, N)
mass = np.random.uniform(0, 1, N)

# set expansion parameters
rscl,mmax,nmax = 1.0,2,10

F = flex.FLEX(rscl,mmax,nmax,r,theta)

# to compute the total power in each harmonic,
totalm = np.linalg.norm(np.sqrt(F.coscoefs**2 + F.sincoefs**2), axis=1)

# and them compute a ratio
eta = totalm[2]/totalm[0]
print('ratio of m=2 to m=0:',eta)
```


## Documentation

More complete documentation _will_ be available at [at this URL]().

## Contributing

The `flex` maintainers welcome contributions to software, examples, and documentation. The maintainers are actively monitoring pull requests and would be happy to collaborate on contributions or ideas. If you have any requests for additional information or find any bugs, please open an issue directly.

## License

`flex` is available under the MIT license. See the LICENSE file for specifics.

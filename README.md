# Cloud Model

A high-performance gravitational simulation focused on modeling cloud collapse and stellar formation. This project implements a direct iterative approach for computing gravitational potentials.

## Features

- **Direct iterative computation** for grid-based gravitational potential solving
- **Barometric density distribution** based on gravitational potential
- **Proper physical units** via Unitful.jl and UnitfulAstro.jl
- **Visualization tools** for analyzing simulation results
- **Command-line interface** for easy simulation configuration

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/cloud-model.git
cd cloud-model

# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Quick Start

### Running a simulation

```bash
# Standard direct method
julia cloud_model.jl --iterations 1000 --grid 30 --max-dist 4.0
```

### Visualization

Results are automatically visualized after simulation completion:

```bash
# Visualization is included in the main simulation flow
# Output is saved as gravitational_standard_N.png where N is the grid size
```

## Numerical Methods

The model solves Poisson's equation for gravitational potential (∇²Φ = 4πGρ) using an iterative approach with a barometric density law (ρ = ρ₀ * exp(-μΦ/RT)).

Key components:
- Discretized Laplacian operator on a uniform grid
- Iterative Jacobi method for solving the Poisson equation
- Barometric density calculation using the current potential field
- Appropriate boundary conditions at domain edges

## Project Structure

- `cloud_model.jl`: Main module and simulation logic with the following components:
  - Physical constants with proper units
  - Grid-based simulation state management
  - Iterative Poisson equation solver
  - Density calculation based on potential
  - Visualization utilities for 2D slices through the 3D domain
  - Command-line argument parsing

## References

1. Press, W.H., Teukolsky, S.A., Vetterling, W.T., & Flannery, B.P. (2007). Numerical Recipes: The Art of Scientific Computing (3rd Edition). Cambridge University Press.

## License

MIT License
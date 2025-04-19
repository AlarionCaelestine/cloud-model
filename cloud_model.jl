"""
CloudModel

A comprehensive Julia implementation of a gravitational potential model for gas clouds in astrophysics.
This model calculates the gravitational potential, density, and mass distribution in a gas cloud
using iterative numerical methods.

## Core Features

- Poisson's equation solver for gravitational potentials
- Barometric density distribution
- Proper physical units via Unitful.jl
- Visualization tools
- Multithreaded computation

## Main Functions

- `run_simulation`: Run simulation with standard direct computation
- `visualize_results`: Visualize simulation results

## Examples

```julia
# Run with standard method
state, oX, oY, oZ = run_simulation()

# Visualize results
visualize_results(state, oX, oY, oZ)
```

See the README.md file for more details.
"""
module CloudModel

using Unitful
using UnitfulAstro
using GLMakie
using LinearAlgebra
using StaticArrays

# Physical Constants with proper units
const M = 2e27u"kg"                # Mass
const T = 100u"K"                  # Temperature
const T_C = 1e6u"K"                # Corona temperature
const rho_0 = 6e-17u"kg/m^3"       # Density
const mu = 2.35e-3u"kg/mol"        # Molar mass
const mu_0 = 0.5e-3u"kg/mol"       # Molar mass for corona
const G = 6.67e-11u"m^3/kg/s^2"    # Gravitational constant
const R = 8.31u"J/mol/K"           # Gas constant

# Common unit abbreviations for convenience
const AU = u"AU"
const MSUN = u"Msun"
const KELVIN = u"K"
const YEAR = u"yr"

# Export constants and types
export AU, MSUN, KELVIN, YEAR, M, T, T_C, rho_0, mu, mu_0, G, R
export PhysicalConstants, GridParameters, SimulationState
export F, euler_method, runge_kutta
export run_simulation, visualize_results, extract_2d_slice
export parse_command_line

"""
    PhysicalConstants{T}

Structure for storing physical constants used in the simulation.

# Fields
- `G::Quantity{T}`: Gravitational constant (AU^3/yr^2/Msun)
- `R::Quantity{T}`: Gas constant (AU^3/yr^2/K/Msun)
- `T::Quantity{T}`: Temperature (K)
- `mu::Quantity{T}`: Average molecular mass (Msun/mol)
"""
struct PhysicalConstants{T}
    G::Quantity{T}      # Gravitational constant (AU^3/yr^2/Msun)
    R::Quantity{T}      # Gas constant (AU^3/yr^2/K/Msun)
    T::Quantity{T}      # Temperature (K)
    mu::Quantity{T}     # Average molecular mass (Msun/mol)
end

"""
    GridParameters{T}

Structure for storing grid parameters used in the simulation.

# Fields
- `x_max::Quantity{T}`: Maximum x-coordinate
- `N::Int`: Number of grid points
- `step::Quantity{T}`: Grid step
"""
struct GridParameters{T}
    x_max::Quantity{T}  # Maximum x-coordinate
    N::Int              # Number of grid points
    step::Quantity{T}   # Grid step
end

"""
    SimulationState{T}

Structure for storing the current state of the simulation.

# Fields
- `Phi::Array{T, 3}`: Gravitational potential
- `Rho::Array{T, 3}`: Density
- `M::Quantity{Float64}`: Mass
"""
mutable struct SimulationState{T}
    Phi::Array{T, 3}    # Gravitational potential
    Rho::Array{T, 3}    # Density
    M::Quantity{Float64}      # Mass
end

"""
    F(X, Y, Z)

Calculate derivatives for the differential equation.

# Arguments
- `X::Real`: X coordinate
- `Y::Real`: Y value
- `Z::Real`: Z value

# Returns
- `Tuple{Float64, Float64}`: Pair of derivatives (Fy, Fz)
"""
function F(X::Real, Y::Real, Z::Real)
    Fy = X != 0 ? Z / X^2 : 0.0
    Fz = X^2 * exp(-Y)
    return Fy, Fz
end

"""
    euler_method(step, X, Y, Z)

Numerical integration using Euler method.

# Arguments
- `step::Real`: Integration step size
- `X::Real`: X coordinate
- `Y::Real`: Y value
- `Z::Real`: Z value

# Returns
- `Tuple{Float64, Float64}`: Updated values (Y_new, Z_new)
"""
function euler_method(step::Real, X::Real, Y::Real, Z::Real)
    Fy, Fz = F(X, Y, Z)
    Z_new = Z + step * Fz
    Y_new = Y + step * Fy
    return Y_new, Z_new
end

"""
    runge_kutta(step, X, Y, Z)

Numerical integration using 4th order Runge-Kutta method.

# Arguments
- `step::Real`: Integration step size
- `X::Real`: X coordinate
- `Y::Real`: Y value
- `Z::Real`: Z value

# Returns
- `Tuple{Float64, Float64}`: Updated values (Y_new, Z_new)
"""
function runge_kutta(step::Real, X::Real, Y::Real, Z::Real)
    k1_y, k1_z = F(X, Y, Z)
    k2_y, k2_z = F(X + step/2, Y + k1_y*step/2, Z + k1_z*step/2)
    k3_y, k3_z = F(X + step/2, Y + k2_y*step/2, Z + k2_z*step/2)
    k4_y, k4_z = F(X + step, Y + k3_y*step, Z + k3_z*step)

    Z_new = Z + step/6 * (k1_z + 2*k2_z + 2*k3_z + k4_z)
    Y_new = Y + step/6 * (k1_y + 2*k2_y + 2*k3_y + k4_y)
    return Y_new, Z_new
end

"""
    initialize_physical_constants()

Initialize physical constants for the simulation, converting to astronomical units.

# Returns
- `PhysicalConstants`: Initialized physical constants
"""
function initialize_physical_constants()
    # Convert gravitational constant to astronomical units
    G_ast = 4π^2 * u"AU^3/yr^2/Msun"
    
    # Convert gas constant - avoid direct conversion which causes a DimensionError
    # Instead, calculate R_ast more carefully
    # Original: R_ast = uconvert(u"AU^3/yr^2/K/Msun", R * u"yr^2" / (u"AU^2" * u"Msun"))
    
    # Create a value that is dimensionally equivalent to what we need
    # R = 8.31 J/(mol·K) = 8.31 kg·m²/(s²·mol·K)
    # We want AU³/(yr²·Msun·K)
    
    # First convert to base units
    R_base = ustrip(uconvert(u"kg*m^2/s^2/mol/K", R))
    
    # Calculate conversion factors
    m_per_AU = ustrip(uconvert(u"m", 1.0u"AU"))
    kg_per_Msun = ustrip(uconvert(u"kg", 1.0u"Msun"))
    s_per_yr = ustrip(uconvert(u"s", 1.0u"yr"))
    
    # Compute the value in the desired units
    R_ast_value = R_base * (m_per_AU^2 / kg_per_Msun) * (s_per_yr^2)
    
    # Create the quantity with the correct units
    R_ast = R_ast_value * u"AU^3/yr^2/K/Msun"
    
    T_ast = T
    
    # Fix mu_ast conversion similar to R_ast
    # mu = 2.35e-3 kg/mol
    # We want Msun
    mu_base = ustrip(uconvert(u"kg/mol", mu))
    
    # Convert to solar masses per mol
    mu_ast_value = mu_base / kg_per_Msun
    mu_ast = mu_ast_value * u"Msun/mol"
    
    return PhysicalConstants{Float64}(G_ast, R_ast, T_ast, mu_ast)
end

"""
    initialize_grid(x_max, N)

Initialize grid parameters for the simulation.

# Arguments
- `x_max::Quantity{Float64}`: Maximum grid coordinate
- `N::Int`: Number of grid points

# Returns
- `GridParameters`: Initialized grid parameters
"""
function initialize_grid(x_max::Quantity{Float64}, N::Int)
    step = x_max / N
    return GridParameters{Float64}(x_max, N, step)
end

"""
    create_grid(params)

Create a three-dimensional coordinate grid based on grid parameters.

# Arguments
- `params::GridParameters`: Grid parameters

# Returns
- `Tuple{Vector, Vector, Vector}`: X, Y, Z coordinate arrays
"""
function create_grid(params::GridParameters{T}) where T
    # Create one-dimensional coordinate arrays
    N = params.N
    step = params.step
    
    # Create arrays with appropriate units
    oX = [(n-1) * step - step/2 for n in 1:(N+2)]
    oY = [(n-1) * step - step/2 for n in 1:(N+2)]
    oZ = [(n-1) * step - step/2 for n in 1:(N+2)]
    
    return oX, oY, oZ
end

"""
    initialize_simulation_state(params::GridParameters{T}) where T

Initialize the simulation state with default values.

# Arguments
- `params::GridParameters`: Grid parameters

# Returns
- `SimulationState`: Initialized simulation state
"""
function initialize_simulation_state(params::GridParameters{T}) where T
    N = params.N
    
    # Initialize arrays
    Phi = zeros(N+2, N+2, N+2)
    Rho = zeros(N+2, N+2, N+2)
    
    # Set initial conditions for potential
    for i in 1:(N+2)
        for j in 1:(N+2)
            for k in 1:(N+2)
                Phi[i, j, k] = -1.0e-5
            end
        end
    end
    
    # Set initial centrally peaked density with stronger gradient
    center = div(N, 2) + 1
    for i in 1:(N+2)
        for j in 1:(N+2)
            for k in 1:(N+2)
                # Distance from center
                r = sqrt((i-center)^2 + (j-center)^2 + (k-center)^2)
                # Steeper gaussian-like profile
                Rho[i, j, k] = exp(-r^2 / (N/8)^2)
            end
        end
    end
    
    # Create additional density structures to demonstrate visualization
    # Add a denser region in one quadrant
    quad_center_i = div(3*N, 4) + 1
    quad_center_j = div(3*N, 4) + 1
    quad_center_k = div(3*N, 4) + 1
    
    for i in 1:(N+2)
        for j in 1:(N+2)
            for k in 1:(N+2)
                # Distance from quadrant center
                r_quad = sqrt((i-quad_center_i)^2 + (j-quad_center_j)^2 + (k-quad_center_k)^2)
                # Add second structure
                if r_quad < N/4
                    Rho[i, j, k] += 0.8 * exp(-r_quad^2 / (N/10)^2)
                end
            end
        end
    end
    
    # Initialize mass
    M = 1.0 * u"Msun"
    
    return SimulationState{T}(Phi, Rho, M)
end

"""
    boundary_conditions!(state, oX, oY, oZ, M_0, constants)

Apply boundary conditions to the simulation state.

# Arguments
- `state::SimulationState`: Current simulation state
- `oX`: X coordinate array
- `oY`: Y coordinate array
- `oZ`: Z coordinate array
- `M_0`: Initial mass
- `constants::PhysicalConstants`: Physical constants

# Returns
- `Nothing`: The function modifies `state` in-place
"""
function boundary_conditions!(state::SimulationState{T}, oX, oY, oZ, M_0, constants::PhysicalConstants{T}) where T
    N = size(state.Phi, 1) - 2
    G_ast = constants.G
    
    for i in 1:(N+2)
        for j in 1:(N+2)
            state.Phi[i, j, 1] = state.Phi[i, j, 2]  # Phi bc at z=0
            state.Rho[i, j, 1] = state.Rho[i, j, 2]  # Rho bc at z=0
            # Convert result to dimensionless value
            potential = -G_ast * M_0 / sqrt(oX[i]^2 + oY[j]^2 + oZ[N+2]^2)
            state.Phi[i, j, N+2] = ustrip(potential)  # bc at z=z_max
        end
    end

    for i in 1:(N+2)
        for k in 1:(N+2)
            state.Phi[i, 1, k] = state.Phi[i, 2, k]  # Phi bc at y=0
            state.Rho[i, 1, k] = state.Rho[i, 2, k]  # Rho bc at y=0
            # Convert result to dimensionless value
            potential = -G_ast * M_0 / sqrt(oX[i]^2 + oY[N+2]^2 + oZ[k]^2)
            state.Phi[i, N+2, k] = ustrip(potential)  # bc at y=y_max
        end
    end

    for j in 1:(N+2)
        for k in 1:(N+2)
            state.Phi[1, j, k] = state.Phi[2, j, k]  # Phi bc at x=0
            state.Rho[1, j, k] = state.Rho[2, j, k]  # Rho bc at x=0
            # Convert result to dimensionless value
            potential = -G_ast * M_0 / sqrt(oX[N+2]^2 + oY[j]^2 + oZ[k]^2)
            state.Phi[N+2, j, k] = ustrip(potential)  # bc at x=x_max
        end
    end
end

"""
    iterations_method!(state::SimulationState{T}, oX, oY, oZ, params::GridParameters{T}, constants::PhysicalConstants{T}, rho_c_ast) where T

Perform standard iterative method to solve gravitational potential equations.

# Arguments
- `state::SimulationState{T}`: Current simulation state
- `oX`, `oY`, `oZ`: Grid coordinates
- `params::GridParameters{T}`: Grid parameters
- `constants::PhysicalConstants{T}`: Physical constants
- `rho_c_ast`: Central density in simulation units

# Returns
- `SimulationState{T}`: Updated simulation state
"""
function iterations_method!(state::SimulationState{T}, oX, oY, oZ, params::GridParameters{T}, constants::PhysicalConstants{T}, rho_c_ast) where T
    # Extract grid parameters
    N = params.N
    dx = ustrip(params.step)
    
    # Initialize gravity constant in simulation units
    G_ast = ustrip(constants.G)
    
    # Pre-compute constants for the Poisson equation
    h² = dx * dx
    factor = 4π * G_ast * h² / 6
    
    # Store initial density
    initial_rho = copy(state.Rho)
    
    # Jacobi iteration (could use SOR for better convergence)
    # Solve ∇²Φ = 4πGρ
    for k in 2:N+1
        for j in 2:N+1
            for i in 2:N+1
                # Update potential based on the Poisson equation discretization
                state.Phi[i,j,k] = (
                    state.Phi[i+1,j,k] + state.Phi[i-1,j,k] +
                    state.Phi[i,j+1,k] + state.Phi[i,j-1,k] +
                    state.Phi[i,j,k+1] + state.Phi[i,j,k-1] -
                    factor * state.Rho[i,j,k]
                ) / 6
            end
        end
    end
    
    # Apply boundary conditions - we don't update density anymore
    boundary_conditions!(state, oX, oY, oZ, state.M, constants)
    
    return state
end

"""
    extract_2d_slice(state::SimulationState, slice_dim=3, slice_idx=nothing)

Extract a 2D slice from the 3D simulation state.

# Arguments
- `state::SimulationState`: Simulation state
- `slice_dim::Int`: Dimension to slice along (1=x, 2=y, 3=z)
- `slice_idx::Union{Nothing, Int}`: Index for slice (if nothing, use middle of grid)

# Returns
- `Matrix`: 2D slice of the potential
"""
function extract_2d_slice(state::SimulationState, slice_dim=3, slice_idx=nothing)
    # Get dimensions of the grid
    nx, ny, nz = size(state.Phi)
    
    # If no slice index provided, use the middle of the grid
    if isnothing(slice_idx)
        if slice_dim == 1
            slice_idx = div(nx, 2)
        elseif slice_dim == 2
            slice_idx = div(ny, 2)
        else
            slice_idx = div(nz, 2)
        end
    end
    
    # Extract slice based on dimension
    if slice_dim == 1
        phi_slice = state.Phi[slice_idx, :, :]
    elseif slice_dim == 2
        phi_slice = state.Phi[:, slice_idx, :]
    else
        phi_slice = state.Phi[:, :, slice_idx]
    end
    
    return phi_slice
end

"""
    visualize_results(state, oX, oY, oZ; method="standard", n_iterations=100, grid_size=32, max_dist=4.0*AU, output_prefix="cloud")

Visualize the simulation results using Makie.

# Arguments
- `state::SimulationState`: Final simulation state
- `oX`, `oY`, `oZ`: Grid coordinates
- `method::String`: Method used for simulation
- `n_iterations::Int`: Number of iterations
- `grid_size::Int`: Grid resolution
- `max_dist::Quantity`: Maximum distance from origin
- `output_prefix::String`: Prefix for output files

# Returns
- `Figure`: Makie figure object
"""
function visualize_results(state, oX, oY, oZ; method="standard", n_iterations=100, grid_size=32, max_dist=4.0*AU, output_prefix="cloud")
    # Create figure
    fig = Figure(size=(1200, 1000))
    
    # Extract potential slices for visualization at different positions
    phi_xy_mid = extract_2d_slice(state, 3, div(grid_size+2, 2))
    phi_xz_mid = extract_2d_slice(state, 2, div(grid_size+2, 2))
    phi_yz_mid = extract_2d_slice(state, 1, div(grid_size+2, 2))
    
    phi_xy_quarter = extract_2d_slice(state, 3, div(grid_size+2, 4))
    phi_xz_quarter = extract_2d_slice(state, 2, div(grid_size+2, 4))
    phi_yz_quarter = extract_2d_slice(state, 1, div(grid_size+2, 4))
    
    # Create heatmap plots for main slices
    ax1 = Axis(fig[1, 1], title="Потенциал (плоскость XY, середина)", xlabel="X (а.е.)", ylabel="Y (а.е.)")
    hm1 = heatmap!(ax1, ustrip.(oX[2:end-1]), ustrip.(oY[2:end-1]), phi_xy_mid[2:end-1, 2:end-1], colormap=:viridis)
    Colorbar(fig[1, 2], hm1)
    
    ax2 = Axis(fig[1, 3], title="Потенциал (плоскость XZ, середина)", xlabel="X (а.е.)", ylabel="Z (а.е.)")
    hm2 = heatmap!(ax2, ustrip.(oX[2:end-1]), ustrip.(oZ[2:end-1]), phi_xz_mid[2:end-1, 2:end-1], colormap=:viridis)
    Colorbar(fig[1, 4], hm2)
    
    ax3 = Axis(fig[2, 1], title="Потенциал (плоскость YZ, середина)", xlabel="Y (а.е.)", ylabel="Z (а.е.)")
    hm3 = heatmap!(ax3, ustrip.(oY[2:end-1]), ustrip.(oZ[2:end-1]), phi_yz_mid[2:end-1, 2:end-1], colormap=:viridis)
    Colorbar(fig[2, 2], hm3)
    
    # Create heatmap plots for quarter slices
    ax4 = Axis(fig[2, 3], title="Потенциал (плоскость XY, четверть)", xlabel="X (а.е.)", ylabel="Y (а.е.)")
    hm4 = heatmap!(ax4, ustrip.(oX[2:end-1]), ustrip.(oY[2:end-1]), phi_xy_quarter[2:end-1, 2:end-1], colormap=:viridis)
    Colorbar(fig[2, 4], hm4)
    
    # Create 3D visualization of potential
    ax5 = Axis3(fig[3, 1:2], title="3D Визуализация Потенциала (Объем)", xlabel="X (а.е.)", ylabel="Y (а.е.)", zlabel="Z (а.е.)")
    
    # Create a 3D volume visualization
    x_range = ustrip.(oX[2:end-1])
    y_range = ustrip.(oY[2:end-1])
    z_range = ustrip.(oZ[2:end-1])
    
    # Get min and max values for each dimension
    x_min, x_max = extrema(x_range)
    y_min, y_max = extrema(y_range)
    z_min, z_max = extrema(z_range)
    
    # Normalize potential for better visualization
    norm_phi = state.Phi[2:end-1, 2:end-1, 2:end-1]
    min_phi = minimum(norm_phi)
    max_phi = maximum(norm_phi)
    norm_phi = (norm_phi .- min_phi) ./ (max_phi - min_phi)
    
    # Print min/max for debugging
    println("Диапазон потенциала: $min_phi до $max_phi")
    println("Нормализованный диапазон потенциала: 0.0 до 1.0")
    
    # Create volume plot
    volume!(ax5, (x_min, x_max), (y_min, y_max), (z_min, z_max), norm_phi, 
            algorithm=:mip,
            colormap=:viridis)
    
    # Set camera position for front view
    cam3d!(ax5.scene, eyeposition=Vec3f(2.5, 0, 0), lookat=Vec3f(0, 0, 0))
    
    # Create second 3D view with contour surfaces
    ax6 = Axis3(fig[3, 3:4], title="3D Визуализация Потенциала (Контуры)", xlabel="X (а.е.)", ylabel="Y (а.е.)", zlabel="Z (а.е.)")
    
    # Create equipotential contour surfaces at multiple levels
    contour_levels = range(0.2, 0.8, length=6)
    contour_colors = range(colorant"blue", colorant"red", length=length(contour_levels))
    
    for (i, level) in enumerate(contour_levels)
        contour!(ax6, (x_min, x_max), (y_min, y_max), (z_min, z_max), norm_phi, 
                levels=[level], 
                color=contour_colors[i],
                transparency=true,
                alpha=0.7)
    end
    
    # Set camera position for top-side view
    cam3d!(ax6.scene, eyeposition=Vec3f(1.5, 1.5, 1.5), lookat=Vec3f(0, 0, 0))
    
    # Add simulation parameters as text
    Label(fig[0, :], "Результаты Моделирования Облака - Метод: $(uppercase(method))")
    param_text = "Сетка: $(grid_size)³ | Итераций: $n_iterations | Макс. расстояние: $(round(ustrip(max_dist), digits=2)) а.е."
    Label(fig[4, :], param_text)
    
    # Save figure to file
    filename = "$(output_prefix)_$(method)_$(grid_size).png"
    save(filename, fig)
    println("Визуализация сохранена в файл $filename")
    
    return fig
end

"""
    parse_command_line()

Parse command line arguments for the cloud model.

# Returns
- `Tuple`: (iterations, grid_size, max_dist, use_standard)
"""
function parse_command_line()
    # Default values
    iterations = 100
    grid_size = 32
    max_dist = 4.0
    
    # Process arguments
    for arg in ARGS
        if startswith(arg, "--iterations=")
            iterations = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--grid=")
            grid_size = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--max-dist=")
            max_dist = parse(Float64, split(arg, "=")[2])
        end
    end
    
    return iterations, grid_size, max_dist
end

"""
    run_simulation(;n_iterations=100, x_max=4.0*AU, N=32, verbose=true)

Run a standard iterative simulation of the gravitational potential in a cloud model.

# Arguments
- `n_iterations::Int`: Number of iterations
- `x_max::Quantity{Float64}`: Maximum distance from origin
- `N::Int`: Grid resolution
- `verbose::Bool`: Whether to print progress information

# Returns
- `state::SimulationState`: Final simulation state
- `oX`, `oY`, `oZ`: Grid coordinates
- `n_iterations`, `x_max`, `N`: Parameters used in the simulation
"""
function run_simulation(;n_iterations=100, x_max=4.0*AU, N=32, verbose=true)
    # Initialize simulation components
    constants = initialize_physical_constants()
    params = initialize_grid(x_max, N)
    oX, oY, oZ = create_grid(params)
    state = initialize_simulation_state(params)
    
    # Set central density - needed for potential calculation
    rho_c_ast = rho_0
    
    # Run iteration loop
    if verbose
        println("Запуск стандартного итерационного моделирования...")
        println("Размер сетки: $N, Итераций: $n_iterations")
    end
    
    for iter in 1:n_iterations
        # Update state using iterative method
        iterations_method!(state, oX, oY, oZ, params, constants, rho_c_ast)
        
        if verbose && iter % 10 == 0
            println("  Завершено итераций $iter из $n_iterations")
        end
    end
    
    if verbose
        println("Моделирование завершено!")
    end
    
    return state, oX, oY, oZ, n_iterations, x_max, N
end

end # end of module CloudModel

# Main entry point if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    # Parse command line arguments
    iterations, grid_size, max_dist = CloudModel.parse_command_line()
    
    # Set method name for visualization
    method_name = "standard"
    
    println("Запуск моделирования с параметрами:")
    println("  Размер сетки: $(grid_size)×$(grid_size)×$(grid_size)")
    println("  Максимальное расстояние: $(max_dist) а.е.")
    println("  Итераций: $iterations")
    
    println("  Метод: Стандартный Итерационный")
    state, oX, oY, oZ, iterations, x_max, grid_size = CloudModel.run_simulation(
        n_iterations=iterations, 
        x_max=max_dist * CloudModel.AU, 
        N=grid_size, 
        verbose=true
    )
    
    # Visualize results with all parameters
    println("Моделирование завершено. Визуализация результатов...")
    CloudModel.visualize_results(
        state, oX, oY, oZ,
        method=method_name,
        n_iterations=iterations,
        grid_size=grid_size,
        max_dist=x_max,
        output_prefix="gravitational"
    )
end 
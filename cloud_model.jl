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
- `run_simulation_mt`: Run simulation with multithreaded computation
- `visualize_results`: Visualize simulation results

## Examples

```julia
# Run with standard method
state, oX, oY, oZ = run_simulation()

# Run with multithreading
state, oX, oY, oZ = run_simulation_mt(threads=Threads.nthreads())

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
using Base.Threads: @threads, nthreads, threadid

# Fundamental constants
const G = ustrip(6.67e-11u"m^3/kg/s^2" |> upreferred) # Gravitational constant
const R = ustrip(8.31u"J/mol/K" |> upreferred) # Gas constant


# Export constants and types
export G, R
export SimulationConstants, SimulationState
export run_simulation, visualize_results, extract_2d_slice
export parse_command_line


"""
    SimulationConstants{T}

Structure for storing physical constants used in the simulation.

# Fields
- `T::Float64`: Temperature of the cloud (K)
- `rho_0::Float64`: Density of the cloud (kg/m^3)
- `mu::Float64`: Average molecular mass (Msun/mol)
- `Step::Float64`: Grid step
- `N::Int`: Number of grid points
"""
struct SimulationConstants
    T::Float64          # Temperature of the cloud (K)
    rho_0::Float64      # Density of the cloud (kg/m^3)
    mu::Float64         # Average molecular mass (Msun/mol)
    Step::Float64       # Grid step
    N::Int              # Number of grid points
    distance_to_center::Array{Float64, 3} # Distance to the center of the cloud
end

function SimulationConstants(T::Float64, rho_0::Float64, mu::Float64, Step::Float64, N::Int)
    distance_to_center = [sqrt((i - 1.5)^2 + (j-1.5)^2 + (k-1.5)^2) * Step for i in 1:(N+2), j in 1:(N+2), k in 1:(N+2)]
    return SimulationConstants(T, rho_0, mu, Step, N, distance_to_center)
end

"""
    SimulationState{T}

Structure for storing the current state of the simulation.

This structure holds the gravitational potential and density fields, along with grid parameters.
It is used throughout the simulation to track the evolving state of the cloud model.
We will update the state of the simulation for 2:N+1 indices, while the first and last indices are the boundary conditions.

# Fields
- `Phi::Array{Float64, 3}`: Gravitational potential
- `Rho::Array{Float64, 3}`: Density
- `M::Float64`: Mass of the cloud (kg)
"""
mutable struct SimulationState
    Phi::Array{Float64, 3}  # Gravitational potential
    Rho::Array{Float64, 3}  # Density
    M::Float64              # Mass of the cloud (kg)
end

"""
    initialize_simulation_state(size::Float64, N::Int, constants::SimulationConstants)

Initialize the simulation state with default values.

# Arguments
- `params::GridParameters`: Grid parameters

# Returns
- `SimulationState`: Initialized simulation state
"""
function initialize_simulation_state(size::Float64, constants::SimulationConstants)
    # Initialize arrays
    Step = size / constants.N # size is the size of the grid, N is the number of grid points. There is a point to the left and right of the grid points, which are the boundary conditions.
    Phi = fill(0.0, constants.N+2, constants.N+2, constants.N+2)
    Rho = fill(constants.rho_0, constants.N+2, constants.N+2, constants.N+2)

    M = constants.rho_0 * size^3 * 8 # 8 is the number of octants in the grid
    Rho[1:end-1, 1:end-1, 1:end-1] .= constants.rho_0 # Set the density of the cloud to the average density
    for i in 1:(constants.N+2)
        for j in 1:(constants.N+2)
            for k in 1:(constants.N+2)
                Phi[i, j, k] = -G * M / constants.distance_to_center[i, j, k]
            end
        end
    end
    
    return SimulationState(Phi, Rho, M)
end

"""
    boundary_conditions!(state)

Apply boundary conditions to the potential field.

# Arguments
- `state::SimulationState`: Current simulation state

# Returns
- `Nothing`: The function modifies `state` in-place
"""
function boundary_conditions(state::SimulationState, constants::SimulationConstants)
    state.Phi[1, :, :] .= state.Phi[2, :, :]
    state.Phi[:, 1, :] .= state.Phi[:, 2, :]
    state.Phi[:, :, 1] .= state.Phi[:, :, 2]

    state.Rho[1, :, :] .= state.Rho[2, :, :]
    state.Rho[:, 1, :] .= state.Rho[:, 2, :]
    state.Rho[:, :, 1] .= state.Rho[:, :, 2]

    for i in 1:(constants.N+2)
        for j in 1:(constants.N+2)
            x = -G * state.M / constants.distance_to_center[i, j, constants.N+2]
            state.Phi[i, j, constants.N+2] = x
            state.Phi[i, constants.N+2, j] = x
            state.Phi[constants.N+2, i, j] = x
        end
    end
end

"""
    iterations_method!(state::SimulationState{T}, constants::SimulationConstants{T}) where T

Perform standard iterative method to solve gravitational potential equations.

# Arguments
- `state::SimulationState{T}`: Current simulation state
- `constants::SimulationConstants{T}`: Physical constants

# Returns
- `SimulationState{T}`: Updated simulation state
"""
function iterations_method!(state::SimulationState, constants::SimulationConstants)
    boundary_conditions(state, constants)

    factor = 4π * G * constants.Step^2 / 6
    
    # Calculate the B constant for the barometric equation
    B = constants.mu / R / constants.T
    
    # Jacobi iteration (could use SOR for better convergence)
    # Solve ∇²Φ = 4πGρ
    Phi2 = copy(state.Phi)
    local_sums = zeros(Float64, Threads.nthreads())
    @threads for k in 2:constants.N+1
        local_sum = 0.0
        for j in 2:constants.N+1
            for i in 2:constants.N+1
                # Update potential based on the Poisson equation discretization
                Phi2[i,j,k] = (
                    state.Phi[i+1,j,k] + state.Phi[i-1,j,k] +
                    state.Phi[i,j+1,k] + state.Phi[i,j-1,k] +
                    state.Phi[i,j,k+1] + state.Phi[i,j,k-1] -
                    factor * state.Rho[i,j,k]
                ) / 6
                state.Rho[i,j,k] = constants.rho_0 * exp(-Phi2[i,j,k] * B)
                local_sum += 8 * state.Rho[i,j,k] * constants.Step^3
            end
        end
        local_sums[Threads.threadid()] += local_sum
    end
    state.Phi .= Phi2
    state.M = sum(local_sums)
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
    visualize_results(state::SimulationState; n_iterations=1000, method="mt", output_prefix="cloud_model", size=40u"AU")

Visualize simulation results with 1D, 2D slices and 3D contours for both density and potential.

# Arguments
- `state::SimulationState`: Final state of the simulation
- `n_iterations::Int`: Number of iterations performed (default: 1000)
- `method::String`: Simulation method used (default: "mt" for multithreaded)
- `output_prefix::String`: Prefix for output filename (default: "cloud_model")
- `size::Float64`: Physical size of the simulation domain (default: 40.0)

# Returns
- `Figure`: The created figure object
"""
function visualize_results(state::SimulationState, constants::SimulationConstants; 
                           n_iterations=1000, 
                           output_prefix="cloud_model",
)
    # Create figure with 3 rows, 2 columns
    fig = Figure(size=(1600, 1300))
    size = constants.Step * (constants.N)
    # Create coordinate arrays
    N = constants.N
    state.Rho = state.Rho ./ constants.rho_0
    # Create coordinate ranges for visualization
    x_coords = range(0, ustrip(Float64, u"AU", size * u"m") , length=N)
    y_coords = range(0, ustrip(Float64, u"AU", size * u"m") , length=N)
    z_coords = range(0, ustrip(Float64, u"AU", size * u"m") , length=N)
    
    # Extract 1D slices for density and potential along X-axis where Y=0, Z=0
    x_values = collect(x_coords)
    rho_1d = state.Rho[2:end-1, 1, 1]
    phi_1d = state.Phi[2:end-1, 1, 1]
    
    # Extract 2D slices for Z=0 plane
    rho_xy = state.Rho[2:end-1, 2:end-1, 1]
    phi_xy = state.Phi[2:end-1, 2:end-1, 1]
    
    # 1D density plot (top-left)
    ax1 = Axis(fig[1, 1], title="Density Distribution (ρ(X) for Y=0, Z=0)", 
              xlabel="X (AU)", ylabel="Density")
    lines!(ax1, x_values, rho_1d, color=:blue, linewidth=2)

    
    # 1D potential plot (top-right)
    ax2 = Axis(fig[1, 2], title="Potential Distribution (Φ(X) for Y=0, Z=0)", 
              xlabel="X (AU)", ylabel="Potential")
    lines!(ax2, x_values, phi_1d, color=:red, linewidth=2)

    # 2D density colormap (middle-left)
    ax3 = Axis(fig[2, 1], title="Density Distribution (ρ(X,Y) for Z=0)", 
              xlabel="X (AU)", ylabel="Y (AU)")
    y_values = collect(y_coords)
    hm1 = heatmap!(ax3, x_values, y_values, rho_xy, colormap=:inferno)
    Colorbar(fig[2, 1, Right()], hm1, label="Density")

    
    # 2D potential colormap (middle-right)
    ax4 = Axis(fig[2, 2], title="Potential Distribution (Φ(X,Y) for Z=0)", 
              xlabel="X (AU)", ylabel="Y (AU)")
    hm2 = heatmap!(ax4, x_values, y_values, phi_xy, colormap=:viridis)
    Colorbar(fig[2, 2, Right()], hm2, label="Potential")
    # Set manual ticks to avoid warnings
    ax4.xticks = LinearTicks(5)
    ax4.yticks = LinearTicks(5)
    
    # Prepare 3D data
    # Get ranges for 3D visualization
    x_range = x_values
    y_range = y_values
    z_range = collect(z_coords)
    
    # Extract 3D data and normalize for visualization
    norm_rho = state.Rho[2:end-1, 2:end-1, 2:end-1]
    min_rho = minimum(norm_rho)
    max_rho = maximum(norm_rho)
    norm_rho = (norm_rho .- min_rho) ./ (max_rho - min_rho)
    
    norm_phi = state.Phi[2:end-1, 2:end-1, 2:end-1]
    min_phi = minimum(norm_phi)
    max_phi = maximum(norm_phi)
    norm_phi = (norm_phi .- min_phi) ./ (max_phi - min_phi)
    
    # Get min and max values for each dimension
    x_min, x_max = extrema(x_range)
    y_min, y_max = extrema(y_range)
    z_min, z_max = extrema(z_range)
    
    # Define contour levels and colors
    density_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    potential_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    contour_colors = [:orange, :red, :purple, :blue, :cyan]
    
    # 3D density contour plot (bottom-left)
    ax5 = Axis3(fig[3, 1], title="Density Contours (3D)", 
               xlabel="X (AU)", ylabel="Y (AU)", zlabel="Z (AU)")
    
    for (i, level) in enumerate(density_levels)
        contour!(ax5, (x_min, x_max), (y_min, y_max), (z_min, z_max), norm_rho,
                levels=[level],
                color=contour_colors[i],
                transparency=true,
                alpha=0.6,
                linewidth=0.5)
    end
    
    # Set camera position and manual ticks
    cam3d!(ax5.scene, eyeposition=Vec3f(2.5, 2.5, 2.5), lookat=Vec3f(0, 0, 0))
    
    # 3D potential contour plot (bottom-right)
    ax6 = Axis3(fig[3, 2], title="Potential Contours (3D)",
               xlabel="X (AU)", ylabel="Y (AU)", zlabel="Z (AU)")
    
    for (i, level) in enumerate(potential_levels)
        contour!(ax6, (x_min, x_max), (y_min, y_max), (z_min, z_max), norm_phi,
                levels=[level],
                color=contour_colors[i],
                transparency=true,
                alpha=0.6,
                linewidth=0.5)
    end
    
    # Set camera position and manual ticks
    cam3d!(ax6.scene, eyeposition=Vec3f(2.5, 2.5, 2.5), lookat=Vec3f(0, 0, 0))
    
    # Add simulation parameters as text
    Label(fig[0, :], "Cloud Model Simulation Results",
          fontsize=20)
    param_text = "Grid: $(N)³ | Iterations: $n_iterations | Max Distance: $(round(ustrip(Float64, u"AU", size * u"m"), digits=2)) AU | Mass: $(round(ustrip(Float64, u"Msun", 1000 * state.M * u"kg"), digits=2)) 1e-3 Msun"
    Label(fig[4, :], param_text, fontsize=16)
    
    # Save figure to file
    filename = "$(output_prefix)_$(N)_$(n_iterations)_$(ustrip(Float64, u"AU", size * u"m")).png"
    save(filename, fig)
    println("Visualization saved to file $filename")
    
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
    iterations = 1000
    N = 50
    size = 40.0
    
    # Process arguments
    for arg in ARGS
        if startswith(arg, "--iterations=")
            iterations = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--N=")
            N = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--size=")
            size = parse(Float64, split(arg, "=")[2])
        end
    end
    
    return iterations, N, size
end

"""
    run_simulation(; n_iterations=1000, size=40, N=50, verbose=true, M=1e-3, T=50, rho_0=6e-17, mu=2.35e-3)

Run the cloud model simulation with specified parameters.

# Arguments
- `n_iterations::Int`: Number of iterations to run (default: 1000)
- `size::Float64`: Size of the simulation domain in AU (default: 40)
- `N::Int`: Number of grid points per dimension (default: 50)
- `verbose::Bool`: Whether to print progress information (default: true)
- `M::Float64`: Mass of the cloud in solar masses (default: 1e-3)
- `T::Float64`: Temperature of the cloud in Kelvin (default: 50)
- `rho_0::Float64`: Initial density of the cloud in kg/m³ (default: 6e-17)
- `mu::Float64`: Average molecular mass in kg/mol (default: 2.35e-3)

# Returns
- `SimulationState`: Final state of the simulation after all iterations
"""
function run_simulation(;
        n_iterations=1000::Int, 
        size=40.0::Real, 
        N=50::Int, 
        verbose=true::Bool, 
        T=50.0,           #K
        rho_0=6e-17,    #kg/m^3
        mu=2.35e-3,     #kg/mol
        )
    constants = SimulationConstants(
        ustrip(T * u"K" |> upreferred),                     # Temperature in K
        ustrip(rho_0 * u"kg/m^3" |> upreferred),            # Density in kg/m^3
        ustrip(mu * u"kg/mol" |> upreferred),               # Molar mass in kg/mol
        ustrip(size * u"AU" / N |> upreferred),             # Step in meters
        N,                                                  # Number of grid points
    )
    # Initialize simulation components
    state = initialize_simulation_state(
        ustrip(Float64, u"m", size * u"AU"), #size in meters
        constants
    )
    
    # Run iteration loop
    if verbose
        println("Запуск многопоточного итерационного моделирования...")
        println("Размер сетки: $N, Итераций: $n_iterations, Потоков: $(nthreads())")
    end
    
    for iter in 1:n_iterations
        # Update state using iterative method with multithreading
        iterations_method!(state, constants)

        # Avoid division by zero for progress reporting
        if verbose && iter % max(1, n_iterations ÷ 20) == 0
            println("  Завершено итераций $iter из $n_iterations")
        end
    end
    
    if verbose
        println("Моделирование завершено!")
    end
    
    return state, constants
end

end 
# Main entry point if this file is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    # Parse command line arguments
    global iterations, N, size = CloudModel.parse_command_line()
    

    println("Запуск моделирования с параметрами:")
    println("  Размер сетки: $(N)×$(N)×$(N)")
    println("  Размер области: $(size)а.е.x$(size)а.е.x$(size)а.е.")
    println("  Итераций: $iterations")
    
    println("  Метод: Многопоточный Итерационный ($(Threads.nthreads()) потоков)")
    state, constants = CloudModel.run_simulation(
        n_iterations=iterations, 
        size=size, 
        N=N, 
        verbose=true
    )

    # Visualize results with all parameters
    println("Моделирование завершено. Визуализация результатов...")
    CloudModel.visualize_results(
        state, constants,
        n_iterations=iterations,
    )
end

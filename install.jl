#!/usr/bin/env julia

println("Installing CloudModel...")

# Add required packages
using Pkg

# Define dependencies with their UUIDs
dependencies = Dict(
    "Unitful" => "1986cc42-f94f-5a68-af5c-568840ba703d",
    "UnitfulAstro" => "6112ee07-acf9-5e0f-b108-d242c714bf9f",
    "GLMakie" => "e9467ef8-e4e7-5192-8a1a-b1aee30e663a",
    "StaticArrays" => "90137ffa-7385-5640-81b9-e52037218182",
    "ProgressBars" => "49802e3a-d2f1-5c88-81d8-b72133a6f568"
)

# Install dependencies
missing_deps = String[]
for (pkg, uuid) in dependencies
    if !haskey(Pkg.installed(), pkg)
        push!(missing_deps, pkg)
    end
end

if !isempty(missing_deps)
    println("Installing missing dependencies: $(join(missing_deps, ", "))")
    Pkg.add(missing_deps)
end

# Get the current directory of this script
script_dir = dirname(Base.source_path())

# Development installation (if running from cloned repo)
if isdir(joinpath(script_dir, "CloudModel"))
    # We're in the repo directly - develop the local package
    println("Installing CloudModel in development mode...")
    Pkg.develop(path=script_dir)
    
    # Create a command-line script
    bin_dir = joinpath(homedir(), ".julia", "bin")
    mkpath(bin_dir)
    
    script_path = joinpath(bin_dir, "cloud_model")
    
    open(script_path, "w") do io
        write(io, """
        #!/usr/bin/env julia
        
        using CloudModel
        
        if abspath(PROGRAM_FILE) == @__FILE__
            iterations, N, size = CloudModel.parse_command_line()
            
            println("Запуск моделирования газового облака с параметрами:")
            println("  Размер сетки: \$(N)×\$(N)×\$(N) точек")
            println("  Размер области: \$(size) а.е. × \$(size) а.е. × \$(size) а.е.")
            println("  Количество итераций: \$iterations")
            println("  Количество потоков: \$(Threads.nthreads())")
            
            state, constants = CloudModel.run_simulation(
                n_iterations=iterations, 
                size=size, 
                N=N, 
                verbose=true
            )
        
            println("Моделирование завершено. Выполняется визуализация результатов...")
            CloudModel.visualize_results(
                state, constants,
                n_iterations=iterations,
            )
            
            println("Визуализация завершена. Расчет выполнен с использованием \$(Threads.nthreads()) потоков.")
        end
        """)
    end
    
    # Make the script executable
    chmod(script_path, 0o755)
    
    # Add to PATH if not already there
    if !occursin(bin_dir, get(ENV, "PATH", ""))
        println("\nTo use CloudModel from anywhere, add the following to your ~/.bashrc or ~/.zshrc:")
        println("export PATH=\"\$PATH:$(bin_dir)\"")
        println("\nOr run this command now:")
        println("export PATH=\"\$PATH:$(bin_dir)\"")
    else
        println("\nCloudModel command installed to $(script_path)")
    end
else
    # Direct installation from GitHub
    println("Installing CloudModel from GitHub...")
    Pkg.add(url="https://github.com/AlarionCaelestine/cloud-model")
end

println("\nInstallation complete! You can now use CloudModel in your Julia code:")
println("using CloudModel")
println("state, constants = CloudModel.run_simulation()")
println("CloudModel.visualize_results(state, constants)")

if isdir(joinpath(script_dir, "CloudModel"))
    println("\nOr run from command line:")
    println("cloud_model --iterations=1000 --N=50 --size=40")
end
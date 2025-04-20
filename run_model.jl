#!/usr/bin/env julia

"""
run_model.jl

Скрипт для прямого запуска модели газового облака без установки пакета.

Это основной и рекомендуемый способ запуска симуляции. Скрипт автоматически:
1. Проверяет и устанавливает необходимые зависимости
2. Загружает исходный код модели
3. Запускает симуляцию с параметрами из командной строки
4. Визуализирует результаты

Использование:
    ./run_model.jl
    
Параметры:
    --iterations=N: количество итераций (по умолчанию 1000)
    --N=N: размер сетки (по умолчанию 50)
    --size=N: размер области в а.е. (по умолчанию 40)
    --threads=N: количество потоков для вычислений (по умолчанию: максимум доступных)

Примеры:
    ./run_model.jl --iterations=5000 --N=100 --size=50
    ./run_model.jl --threads=4 --iterations=1000
    
Также можно задать количество потоков через переменную окружения:
    JULIA_NUM_THREADS=8 ./run_model.jl
"""

# Add packages we need
using Pkg
pkg_list = ["Unitful", "UnitfulAstro", "GLMakie", "StaticArrays", "ProgressBars"]

# Check and install missing packages
for pkg in pkg_list
    try
        @eval using $(Symbol(pkg))
    catch
        println("Installing $pkg...")
        Pkg.add(pkg)
        @eval using $(Symbol(pkg))
    end
end

# Import other required packages
using LinearAlgebra

# Include original cloud_model.jl script directly
include("cloud_model.jl")

# Parse command line arguments
iterations, N, size = CloudModel.parse_command_line()

# Display startup information
println("Запуск моделирования газового облака с параметрами:")
println("  Размер сетки: $(N)×$(N)×$(N) точек")
println("  Размер области: $(size) а.е. × $(size) а.е. × $(size) а.е.")
println("  Количество итераций: $iterations")
println("  Количество потоков: $(Threads.nthreads())")

# Run simulation
state, constants = CloudModel.run_simulation(
    n_iterations=iterations, 
    size=size, 
    N=N, 
    verbose=true
)

# Visualize results
println("Моделирование завершено. Выполняется визуализация результатов...")
CloudModel.visualize_results(
    state, constants,
    n_iterations=iterations,
)

println("Визуализация завершена. Расчет выполнен с использованием $(Threads.nthreads()) потоков.") 
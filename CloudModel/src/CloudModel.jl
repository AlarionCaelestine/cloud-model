"""
CloudModel

Комплексная реализация гравитационной модели газовых облаков в астрофизике на языке Julia.
Эта модель рассчитывает гравитационный потенциал, плотность и распределение массы в газовом облаке
с использованием итеративных численных методов и решает систему уравнений Пуассона.

## Физическая основа
- Уравнение Пуассона для гравитационного потенциала: ∇²Φ = 4πGρ
- Барометрическое уравнение для плотности: ρ = ρ₀·exp(-Φμ/RT)
- Итеративное решение системы связанных уравнений

## Основные возможности

- Решение уравнения Пуассона для гравитационных потенциалов
- Барометрическое распределение плотности
- Работа с физическими единицами через Unitful.jl
- Инструменты визуализации результатов в 1D, 2D и 3D
- Многопоточные вычисления для ускорения расчетов

## Основные функции

- `run_simulation`: Запуск симуляции облака с многопоточным вычислением
- `visualize_results`: Визуализация результатов симуляции в виде графиков и карт
- `extract_2d_slice`: Извлечение 2D срезов из 3D моделей для анализа

## Примеры

```julia
# Запуск симуляции с параметрами по умолчанию
state, constants = run_simulation()

# Запуск с пользовательскими параметрами
state, constants = run_simulation(
    n_iterations=2000, 
    size=60.0,
    N=100, 
    T=30.0
)

# Визуализация результатов
visualize_results(state, constants)
```

Подробности смотрите в файле README.md.
"""
module CloudModel

# Загрузка зависимостей
using Unitful
using UnitfulAstro
using GLMakie
using LinearAlgebra
using StaticArrays
using Base.Threads: @threads, nthreads, threadid
using ProgressBars

# Включение файла с основной функциональностью
include("core.jl")

# Экспорт функций, которые должны быть доступны пользователям
export run_simulation, visualize_results, extract_2d_slice
export SimulationConstants, SimulationState
export G, R


# Функция для проверки и установки зависимостей
function check_dependencies()
    missing_deps = String[]
    
    required_pkgs = ["Unitful", "UnitfulAstro", "GLMakie", "StaticArrays", "ProgressBars"]
    
    for pkg in required_pkgs
        try
            # Проверяем, загружается ли пакет
            @eval import $(Symbol(pkg))
        catch e
            push!(missing_deps, pkg)
        end
    end
    
    if !isempty(missing_deps)
        @warn "Отсутствуют необходимые пакеты: $(join(missing_deps, ", "))"
        println("Устанавливаем недостающие пакеты...")
        
        using Pkg
        Pkg.add(missing_deps)
        
        # Предкомпиляция пакетов
        for pkg in missing_deps
            @eval using $(Symbol(pkg))
        end
        
        println("Все зависимости установлены!")
    end
    
    return true
end

# Автоматическая проверка зависимостей при загрузке пакета
function __init__()
    check_dependencies()
end

end # module

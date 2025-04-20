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

using Unitful
using UnitfulAstro
using GLMakie
using LinearAlgebra
using StaticArrays
using Base.Threads: @threads, nthreads, threadid
using ProgressBars  

# Фундаментальные физические константы
const G = ustrip(6.67e-11u"m^3/kg/s^2" |> upreferred) # Гравитационная постоянная
const R = ustrip(8.31u"J/mol/K" |> upreferred) # Газовая постоянная


# Экспорт констант, типов и функций
export G, R
export SimulationConstants, SimulationState
export run_simulation, visualize_results, extract_2d_slice
export parse_command_line


"""
    SimulationConstants

Структура для хранения физических констант и параметров сетки, используемых в симуляции.

# Поля
- `T::Float64`: Температура облака (K)
- `rho_0::Float64`: Начальная плотность облака (kg/m^3)
- `mu::Float64`: Средняя молекулярная масса газа (kg/mol)
- `Step::Float64`: Шаг пространственной сетки (m)
- `N::Int`: Количество точек сетки по одному измерению
- `distance_to_center::Array{Float64, 3}`: Предварительно рассчитанные расстояния от каждой точки сетки до центра (m)
"""
struct SimulationConstants
    T::Float64          # Температура облака (K)
    rho_0::Float64      # Начальная плотность облака (kg/m^3)
    mu::Float64         # Средняя молекулярная масса газа (kg/mol)
    Step::Float64       # Шаг пространственной сетки (m)
    N::Int              # Количество точек сетки по одному измерению
    distance_to_center::Array{Float64, 3} # Расстояние от каждой точки до центра (m)
end

"""
    SimulationConstants(T::Float64, rho_0::Float64, mu::Float64, Step::Float64, N::Int)

Конструктор для создания структуры SimulationConstants с предварительным расчетом расстояний до центра.

# Аргументы
- `T::Float64`: Температура облака (K)
- `rho_0::Float64`: Начальная плотность облака (kg/m^3)
- `mu::Float64`: Средняя молекулярная масса (kg/mol)
- `Step::Float64`: Шаг пространственной сетки (m)
- `N::Int`: Количество точек сетки по одному измерению

# Возвращает
- `SimulationConstants`: Структура с константами симуляции и рассчитанными расстояниями
"""
function SimulationConstants(T::Float64, rho_0::Float64, mu::Float64, Step::Float64, N::Int)
    # Расчет расстояний от каждой точки сетки до центра
    # Центр облака размещен в точке (1.5, 1.5, 1.5) в индексах сетки
    distance_to_center = [sqrt((i - 1.5)^2 + (j-1.5)^2 + (k-1.5)^2) * Step for i in 1:(N+2), j in 1:(N+2), k in 1:(N+2)]
    return SimulationConstants(T, rho_0, mu, Step, N, distance_to_center)
end

"""
    SimulationState

Структура для хранения текущего состояния симуляции газового облака.

Эта структура содержит основные физические характеристики моделируемого облака:
гравитационный потенциал, распределение плотности и общую массу. Структура используется
на протяжении всей симуляции для отслеживания эволюции модели облака.

Моделирование происходит на трехмерной сетке размером (N+2)×(N+2)×(N+2), где крайние элементы
(с индексами 1 и N+2) служат граничными условиями, а расчеты ведутся для внутренних ячеек
с индексами 2:N+1.

# Поля
- `Phi::Array{Float64, 3}`: Трехмерный массив значений гравитационного потенциала
- `Rho::Array{Float64, 3}`: Трехмерный массив значений плотности вещества
- `M::Float64`: Общая масса облака (kg)
"""
mutable struct SimulationState
    Phi::Array{Float64, 3}  # Гравитационный потенциал
    Rho::Array{Float64, 3}  # Плотность
    M::Float64              # Масса облака (kg)
end

"""
    initialize_simulation_state(size::Float64, constants::SimulationConstants)

Инициализация начального состояния симуляции с заданным размером области и константами.

Эта функция создает начальное распределение плотности и гравитационного потенциала
для газового облака. На начальном этапе облако имеет постоянную плотность,
а потенциал рассчитывается для сферически-симметричного распределения массы.

# Аргументы
- `size::Float64`: Физический размер области моделирования (m)
- `constants::SimulationConstants`: Физические константы, используемые в симуляции

# Возвращает
- `SimulationState`: Инициализированное начальное состояние симуляции
"""
function initialize_simulation_state(size::Float64, constants::SimulationConstants)
    # Инициализация трехмерных массивов для потенциала и плотности
    Phi = fill(0.0, constants.N+2, constants.N+2, constants.N+2)
    Rho = fill(0.0, constants.N+2, constants.N+2, constants.N+2)

    # Расчет общей массы облака (размер^3 * 8 октантов * плотность)
    M = constants.rho_0 * size^3 * 8
    
    # Установка постоянной начальной плотности во всем облаке
    Rho[1:end, 1:end, 1:end] .= constants.rho_0
    
    # Инициализация начального гравитационного потенциала
    # с использованием предварительно рассчитанных расстояний до центра
    for i in 1:(constants.N+2)
        for j in 1:(constants.N+2)
            for k in 1:(constants.N+2)
                # Потенциал сферически-симметричного тела: -GM/r
                Phi[i, j, k] = -G * M / constants.distance_to_center[i, j, k]
            end
        end
    end
    
    return SimulationState(Phi, Rho, M)
end

"""
    boundary_conditions(state::SimulationState, constants::SimulationConstants)

Применение физически корректных граничных условий к полям потенциала и плотности.

Эта функция устанавливает граничные условия для численного решения. Граничные условия
для потенциала включают условия непрерывности на ближних границах (зеркальные граничные условия)
и условия потенциала точечной массы на дальних границах.

# Аргументы
- `state::SimulationState`: Текущее состояние симуляции
- `constants::SimulationConstants`: Физические константы и параметры сетки

# Возвращает
Функция не возвращает значений, но модифицирует входную структуру `state`
"""
function boundary_conditions(state::SimulationState, constants::SimulationConstants)
    # Зеркальные граничные условия на ближних границах сетки (x=0, y=0, z=0)
    # для потенциала и плотности
    state.Phi[1, :, :] .= state.Phi[2, :, :]
    state.Phi[:, 1, :] .= state.Phi[:, 2, :]
    state.Phi[:, :, 1] .= state.Phi[:, :, 2]

    state.Rho[1, :, :] .= state.Rho[2, :, :]
    state.Rho[:, 1, :] .= state.Rho[:, 2, :]
    state.Rho[:, :, 1] .= state.Rho[:, :, 2]

    # Граничные условия на дальних границах (x=max, y=max, z=max)
    # устанавливаем потенциал точечной массы -GM/r
    for i in 1:(constants.N+2)
        for j in 1:(constants.N+2)
            # Расчет потенциала на основе текущей массы облака
            x = -G * state.M / constants.distance_to_center[i, j, constants.N+2]
            # Применение граничных условий для трех граничных плоскостей
            state.Phi[i, j, constants.N+2] = x
            state.Phi[i, constants.N+2, j] = x
            state.Phi[constants.N+2, i, j] = x
        end
    end
end

"""
    iterations_method!(state::SimulationState, constants::SimulationConstants)

Выполнение одной итерации метода для решения связанной системы уравнений гравитационного потенциала
и барометрического распределения плотности.

Эта функция реализует итерационный метод Якоби для решения уравнения Пуассона (∇²Φ = 4πGρ)
совместно с барометрическим уравнением плотности (ρ = ρ₀·exp(-Φμ/RT)). 
В каждой итерации обновляется потенциал, затем плотность, и пересчитывается общая масса облака.
Вычисления распараллеливаются по оси Z (индекс k) для повышения производительности.

# Аргументы
- `state::SimulationState`: Текущее состояние симуляции
- `constants::SimulationConstants`: Физические константы и параметры сетки

# Возвращает
- `SimulationState`: Обновленное состояние симуляции после одной итерации
"""
function iterations_method!(state::SimulationState, constants::SimulationConstants)
    # Сначала применяем граничные условия
    boundary_conditions(state, constants)

    # Коэффициент для уравнения Пуассона в дискретной форме
    # factor = 4πG(Δx)²/6, где 6 - коэффициент для трехмерного оператора Лапласа
    factor = 4π * G * constants.Step^2 / 6
    
    # Расчет константы B для барометрического уравнения: B = μ/(RT)
    # Эта константа определяет, насколько быстро падает плотность с ростом потенциала
    B = constants.mu / R / constants.T
    
    # Копируем текущий потенциал для расчета обновленных значений
    # (Метод Якоби требует использования значений с предыдущей итерации)
    Phi2 = copy(state.Phi)
    
    # Массив для хранения локальных сумм массы по потокам
    local_sums = zeros(Float64, Threads.nthreads())
    
    # Основной цикл итерации, распараллеленный по оси Z (индекс k)
    @threads for k in 2:constants.N+1
        local_sum = 0.0  # Локальная сумма массы для текущего потока
        for j in 2:constants.N+1
            for i in 2:constants.N+1
                # Дискретизация уравнения Пуассона (∇²Φ = 4πGρ) по 7-точечному шаблону
                # ∇²Φ ≈ (Φᵢ₊₁,ⱼ,ₖ + Φᵢ₋₁,ⱼ,ₖ + Φᵢ,ⱼ₊₁,ₖ + Φᵢ,ⱼ₋₁,ₖ + Φᵢ,ⱼ,ₖ₊₁ + Φᵢ,ⱼ,ₖ₋₁ - 6Φᵢ,ⱼ,ₖ)/(Δx)²
                Phi2[i,j,k] = (
                    state.Phi[i+1,j,k] + state.Phi[i-1,j,k] +
                    state.Phi[i,j+1,k] + state.Phi[i,j-1,k] +
                    state.Phi[i,j,k+1] + state.Phi[i,j,k-1] -
                    factor * state.Rho[i,j,k]
                ) / 6
                
                # Обновление плотности по барометрическому уравнению: ρ = ρ₀·exp(-Φμ/RT)
                state.Rho[i,j,k] = constants.rho_0 * exp(-Phi2[i,j,k] * B)
                
                # Накопление локальной массы (8 - коэффициент для расчета объема ячейки)
                local_sum += state.Rho[i,j,k] 
            end
        end
        # Сохраняем локальную сумму для текущего потока
        local_sums[Threads.threadid()] += local_sum
    end
    
    # Обновляем потенциал новыми рассчитанными значениями
    state.Phi .= Phi2
    
    # Обновляем общую массу облака суммированием локальных сумм
    state.M = sum(local_sums) * 8 * constants.Step^3
    
    return state
end


"""
    extract_2d_slice(state::SimulationState, slice_dim=3, slice_idx=nothing)

Извлечение двумерного среза из трехмерного массива состояния симуляции.

Эта функция извлекает 2D срез (плоскость) из 3D массивов потенциала или плотности,
что позволяет визуализировать и анализировать внутреннюю структуру моделируемого облака.
Можно выбрать плоскость среза (XY, XZ или YZ) и конкретный индекс.

# Аргументы
- `state::SimulationState`: Состояние симуляции, содержащее трехмерные массивы данных
- `slice_dim::Int`: Измерение для среза (1=x, 2=y, 3=z, по умолчанию=3)
- `slice_idx::Union{Nothing, Int}`: Индекс среза (если не указан, используется середина сетки)

# Возвращает
- `Matrix{Float64}`: Двумерный срез массива потенциала
"""
function extract_2d_slice(state::SimulationState, slice_dim=3, slice_idx=nothing)
    # Получение размеров сетки из массива потенциала
    nx, ny, nz = size(state.Phi)
    
    # Если индекс среза не указан, используем середину сетки
    if isnothing(slice_idx)
        if slice_dim == 1
            slice_idx = div(nx, 2)  # Срез по плоскости YZ (постоянное X)
        elseif slice_dim == 2
            slice_idx = div(ny, 2)  # Срез по плоскости XZ (постоянное Y)
        else
            slice_idx = div(nz, 2)  # Срез по плоскости XY (постоянное Z)
        end
    end
    
    # Извлечение среза в зависимости от выбранного измерения
    if slice_dim == 1
        phi_slice = state.Phi[slice_idx, :, :]  # Срез YZ
    elseif slice_dim == 2
        phi_slice = state.Phi[:, slice_idx, :]  # Срез XZ
    else
        phi_slice = state.Phi[:, :, slice_idx]  # Срез XY
    end
    
    return phi_slice
end


"""
    visualize_results(state::SimulationState, constants::SimulationConstants; 
                     n_iterations=1000, output_prefix="cloud_model")

Визуализация результатов симуляции в виде 1D/2D/3D графиков для плотности и потенциала.

Эта функция создает комплексную визуализацию результатов моделирования газового облака,
включая:
1. Графики одномерных срезов плотности и потенциала
2. Двумерные тепловые карты срезов облака
3. Трехмерные контурные поверхности для анализа структуры

Результат сохраняется в PNG-файл с названием, содержащим параметры симуляции.

# Аргументы
- `state::SimulationState`: Финальное состояние симуляции
- `constants::SimulationConstants`: Физические константы и параметры сетки
- `n_iterations::Int`: Количество выполненных итераций (по умолчанию: 1000)
- `output_prefix::String`: Префикс для имени выходного файла (по умолчанию: "cloud_model")

# Возвращает
- `Figure`: Созданный объект фигуры для дальнейшего использования
"""
function visualize_results(state::SimulationState, constants::SimulationConstants; 
                           n_iterations=1000, 
                           output_prefix="cloud_model",
)
    # Создание фигуры с 3 строками, 2 столбцами для размещения всех графиков
    fig = Figure(size=(1600, 1300))
    
    # Расчет физического размера области моделирования в метрах и нормализация плотности
    size = constants.Step * (constants.N)
    N = constants.N
    state.Rho = state.Rho ./ constants.rho_0  # Нормализация плотности для визуализации
    
    # Создание диапазонов координат в астрономических единицах для визуализации
    x_coords = range(0, ustrip(Float64, u"AU", size * u"m"), length=N + 1)
    y_coords = range(0, ustrip(Float64, u"AU", size * u"m"), length=N + 1)
    z_coords = range(0, ustrip(Float64, u"AU", size * u"m"), length=N + 1)
    
    # Извлечение одномерных срезов для плотности и потенциала вдоль оси X при Y=0, Z=0
    x_values = collect(x_coords)
    rho_1d = state.Rho[2:end, 1, 1]  # Плотность вдоль оси X
    phi_1d = state.Phi[2:end, 1, 1]  # Потенциал вдоль оси X
    
    # Извлечение двумерных срезов для плоскости Z=0
    rho_xy = state.Rho[2:end, 2:end, 1]  # Срез плотности в плоскости XY
    phi_xy = state.Phi[2:end, 2:end, 1]  # Срез потенциала в плоскости XY
    
    # 1D график плотности (в верхнем левом углу)
    ax1 = Axis(fig[1, 1], title="Распределение плотности (ρ(X) для Y=0, Z=0)", 
              xlabel="X (а.е.)", ylabel="Относительная плотность (ρ/ρ₀)")
    lines!(ax1, x_values, rho_1d, color=:blue, linewidth=2)
    
    # 1D график потенциала (в верхнем правом углу)
    ax2 = Axis(fig[1, 2], title="Распределение потенциала (Φ(X) для Y=0, Z=0)", 
              xlabel="X (а.е.)", ylabel="Гравитационный потенциал")
    lines!(ax2, x_values, phi_1d, color=:red, linewidth=2)

    # 2D тепловая карта плотности (в центре слева)
    ax3 = Axis(fig[2, 1], title="Распределение плотности (ρ(X,Y) для Z=0)", 
              xlabel="X (а.е.)", ylabel="Y (а.е.)")
    y_values = collect(y_coords)
    hm1 = heatmap!(ax3, x_values, y_values, rho_xy, colormap=:inferno)
    Colorbar(fig[2, 1, Right()], hm1, label="Относительная плотность")
    
    # 2D тепловая карта потенциала (в центре справа)
    ax4 = Axis(fig[2, 2], title="Распределение потенциала (Φ(X,Y) для Z=0)", 
              xlabel="X (а.е.)", ylabel="Y (а.е.)")
    hm2 = heatmap!(ax4, x_values, y_values, phi_xy, colormap=:viridis)
    Colorbar(fig[2, 2, Right()], hm2, label="Потенциал")
    ax4.xticks = LinearTicks(5)  # Улучшение отображения осей
    ax4.yticks = LinearTicks(5)
    
    # Подготовка данных для 3D визуализации
    z_range = collect(z_coords)
    x_range = x_values
    y_range = y_values
    
    # Извлечение и нормализация 3D данных для визуализации
    # Нормализуем значения от 0 до 1 для лучшего отображения контуров
    norm_rho = state.Rho[2:end, 2:end, 2:end]
    min_rho, max_rho = extrema(norm_rho)
    norm_rho = (norm_rho .- min_rho) ./ (max_rho - min_rho)
    
    norm_phi = state.Phi[2:end, 2:end, 2:end]
    min_phi, max_phi = extrema(norm_phi)
    norm_phi = (norm_phi .- min_phi) ./ (max_phi - min_phi)
    
    # Получение границ области для 3D визуализации
    x_min, x_max = extrema(x_range)
    y_min, y_max = extrema(y_range)
    z_min, z_max = extrema(z_range)
    
    # Определение уровней контуров и цветов для 3D визуализации
    density_levels = [0.1, 0.3, 0.5, 0.7, 0.9]  # Уровни контуров плотности
    potential_levels = [0.1, 0.3, 0.5, 0.7, 0.9]  # Уровни контуров потенциала
    contour_colors = [:orange, :red, :purple, :blue, :cyan]  # Цвета контуров
    
    # 3D контурный график плотности (внизу слева)
    ax5 = Axis3(fig[3, 1], title="Контуры плотности (3D)", 
               xlabel="X (а.е.)", ylabel="Y (а.е.)", zlabel="Z (а.е.)")
    
    # Построение контурных поверхностей плотности для разных уровней
    for (i, level) in enumerate(density_levels)
        contour!(ax5, (x_min, x_max), (y_min, y_max), (z_min, z_max), norm_rho,
                levels=[level],
                color=contour_colors[i],
                transparency=true,
                alpha=0.6,
                linewidth=0.5)
    end
    
    # Настройка камеры для 3D визуализации плотности
    cam3d!(ax5.scene, eyeposition=Vec3f(2.5, 2.5, 2.5), lookat=Vec3f(0, 0, 0))
    
    # 3D контурный график потенциала (внизу справа)
    ax6 = Axis3(fig[3, 2], title="Контуры потенциала (3D)",
               xlabel="X (а.е.)", ylabel="Y (а.е.)", zlabel="Z (а.е.)")
    
    # Построение контурных поверхностей потенциала для разных уровней
    for (i, level) in enumerate(potential_levels)
        contour!(ax6, (x_min, x_max), (y_min, y_max), (z_min, z_max), norm_phi,
                levels=[level],
                color=contour_colors[i],
                transparency=true,
                alpha=0.6,
                linewidth=0.5)
    end
    
    # Настройка камеры для 3D визуализации потенциала
    cam3d!(ax6.scene, eyeposition=Vec3f(2.5, 2.5, 2.5), lookat=Vec3f(0, 0, 0))
    
    # Добавление заголовка и подписи с параметрами симуляции
    Label(fig[0, :], "Результаты симуляции модели облака",
          fontsize=20)
    param_text = "Сетка: $(N)³ | Итерации: $n_iterations | " * 
                 "Макс. расстояние: $(ustrip(Float64, u"AU", size * u"m")) а.е. | " * 
                 "Масса: $(ustrip(Float64, u"Msun", 1000 * state.M * u"kg")) Msun"
    Label(fig[4, :], param_text, fontsize=16)
    
    # Сохранение фигуры в файл PNG
    # Создаем директорию visualizations, если она не существует
    isdir("visualizations") || mkdir("visualizations")
    
    # Формируем имя файла и сохраняем в директорию visualizations/
    filename = "visualizations/$(output_prefix)_$(N)_$(n_iterations)_$(ustrip(Float64, u"AU", size * u"m")).png"
    save(filename, fig)
    println("Визуализация сохранена в файл $filename")
    
    return fig
end

"""
    parse_command_line()

Разбор аргументов командной строки для настройки параметров симуляции облака.

Эта функция обрабатывает следующие аргументы командной строки:
- `--iterations=N`: Установка количества итераций симуляции
- `--N=N`: Установка размера сетки (количества точек по каждому измерению)
- `--size=N`: Установка физического размера области моделирования в астрономических единицах

# Возвращает
- `Tuple{Int, Int, Float64}`: Кортеж (iterations, N, size) с параметрами симуляции
"""
function parse_command_line()
    # Значения параметров по умолчанию
    iterations = 1000  # Количество итераций
    N = 50            # Размер сетки
    size = 40.0       # Размер области в а.е.
    
    # Обработка аргументов командной строки
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
    run_simulation(; n_iterations=1000, size=40.0, N=50, verbose=true, 
                  T=50.0, rho_0=6e-17, mu=2.35e-3)

Запуск полной симуляции модели газового облака с указанными параметрами.

Эта функция выполняет полный цикл симуляции газового облака, включая инициализацию
начального состояния и итеративное решение связанной системы уравнений для 
гравитационного потенциала и плотности. Симуляция выполняется в многопоточном режиме,
что значительно ускоряет вычисления на многоядерных системах.

# Аргументы
- `n_iterations::Int`: Количество итераций для выполнения (по умолчанию: 1000)
- `size::Real`: Размер области симуляции в астрономических единицах (по умолчанию: 40.0)
- `N::Int`: Количество точек сетки по каждому измерению (по умолчанию: 50)
- `verbose::Bool`: Выводить ли информацию о прогрессе симуляции (по умолчанию: true)
- `T::Float64`: Температура облака в Кельвинах (по умолчанию: 50.0)
- `rho_0::Float64`: Начальная плотность облака в кг/м³ (по умолчанию: 6e-17)
- `mu::Float64`: Средняя молекулярная масса газа в кг/моль (по умолчанию: 2.35e-3)

# Возвращает
- `Tuple{SimulationState, SimulationConstants}`: Финальное состояние симуляции и константы
"""
function run_simulation(;
        n_iterations::Int=1000, 
        size::Real=40.0, 
        N::Int=50, 
        verbose::Bool=true, 
        T::Float64=50.0,        # Температура облака (K)
        rho_0::Float64=6e-17,   # Плотность облака (kg/m^3)
        mu::Float64=2.35e-3,    # Молекулярная масса (kg/mol)
        )
    
    # Инициализация структуры констант симуляции с преобразованием единиц
    constants = SimulationConstants(
        ustrip(T * u"K" |> upreferred),                    # Температура в K
        ustrip(rho_0 * u"kg/m^3" |> upreferred),           # Плотность в kg/m^3
        ustrip(mu * u"kg/mol" |> upreferred),              # Молярная масса в kg/mol
        ustrip(size * u"AU" / N |> upreferred),            # Шаг сетки в метрах
        N,                                                 # Количество точек сетки
    )
    
    # Инициализация начального состояния симуляции
    state = initialize_simulation_state(
        ustrip(Float64, u"m", size * u"AU"),  # Размер области в метрах
        constants
    )
    
    # Запуск цикла итераций
    if verbose
        println("Запуск многопоточного итерационного моделирования...")
        println("Размер сетки: $N, Итераций: $n_iterations, Потоков: $(nthreads())")
    end
    
    # Основной цикл итераций симуляции с индикатором прогресса
    for iter in ProgressBar(1:n_iterations)
        # Выполняем одну итерацию метода для обновления потенциала и плотности
        iterations_method!(state, constants)
        
        # В режиме verbose не нужны дополнительные сообщения, так как ProgressBar уже показывает прогресс
    end
    
    if verbose
        println("Моделирование завершено! Общая масса облака: $(round(ustrip(Float64, u"Msun", state.M * u"kg"), digits=6)) Msun")
    end
    
    return state, constants
end

end 
# Точка входа для запуска скрипта напрямую из командной строки
if abspath(PROGRAM_FILE) == @__FILE__
    # Разбор аргументов командной строки для получения параметров симуляции
    global iterations, N, size = CloudModel.parse_command_line()
    
    # Вывод информации о параметрах запуска
    println("Запуск моделирования газового облака с параметрами:")
    println("  Размер сетки: $(N)×$(N)×$(N) точек")
    println("  Размер области: $(size) а.е. × $(size) а.е. × $(size) а.е.")
    println("  Количество итераций: $iterations")
    println("  Количество потоков: $(Threads.nthreads())")
    
    # Запуск симуляции с указанными параметрами и визуальным индикатором прогресса
    state, constants = CloudModel.run_simulation(
        n_iterations=iterations, 
        size=size, 
        N=N, 
        verbose=true
    )

    # Визуализация результатов симуляции
    println("Моделирование завершено. Выполняется визуализация результатов...")
    CloudModel.visualize_results(
        state, constants,
        n_iterations=iterations,
    )
    
    println("Визуализация завершена. Расчет выполнен с использованием $(Threads.nthreads()) потоков.")
end

module CloudModel

using Unitful
using UnitfulAstro
using GLMakie
using LinearAlgebra
using StaticArrays

# Константы для часто используемых единиц измерения
const AU = u"AU"
const MSUN = u"Msun"
const KELVIN = u"K"
const YEAR = u"yr"

# Экспорт констант
export AU, MSUN, KELVIN, YEAR

# Определения типов для лучшей организации кода
"""
    PhysicalConstants

Структура для хранения физических констант, используемых в симуляции.
"""
struct PhysicalConstants
    G::Quantity{Float64}      # Гравитационная постоянная
    R::Quantity{Float64}      # Газовая постоянная
    T::Quantity{Float64}      # Температура
    mu::Quantity{Float64}     # Средняя молекулярная масса
end

"""
    GridParameters

Структура для хранения параметров сетки, используемых в симуляции.
"""
struct GridParameters
    x_max::Quantity{Float64}  # Максимальная x-координата
    N::Int                   # Количество точек сетки
    step::Quantity{Float64}   # Шаг сетки
end

"""
    SimulationState

Структура для хранения состояния симуляции.
"""
mutable struct SimulationState
    Phi::Array{Float64, 3}    # Гравитационный потенциал
    Rho::Array{Float64, 3}    # Плотность
    M::Quantity{Float64}      # Масса
end

# Функции первого шага (методы численного интегрирования)
"""
    F(X, Y, Z)

Вычисление производных для дифференциального уравнения.
"""
function F(X::Real, Y::Real, Z::Real)
    Fy = X != 0 ? Z / X^2 : 0.0
    Fz = X^2 * exp(-Y)
    return Fy, Fz
end

"""
    euler_method(step, X, Y, Z)

Интегрирование методом Эйлера.
"""
function euler_method(step::Real, X::Real, Y::Real, Z::Real)
    Fy, Fz = F(X, Y, Z)
    Z_new = Z + step * Fz
    Y_new = Y + step * Fy
    return Y_new, Z_new
end

"""
    runge_kutta(step, X, Y, Z)

Интегрирование методом Рунге-Кутта (4-го порядка).
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

# Функции второго шага
"""
    initialize_physical_constants()

Инициализация физических констант для симуляции.
"""
function initialize_physical_constants()
    G_ast = 4π^2 * u"AU^3/yr^2/Msun"  # Гравитационная постоянная в астрономических единицах
    R = 8.31446261815324u"J/K/mol"    # Газовая постоянная
    T = 20.0u"K"                      # Температура (предполагается из third_step)
    mu = 2.0u"g/mol"                  # Средняя молекулярная масса (предполагается из third_step)
    
    return PhysicalConstants(G_ast, R, T, mu)
end

"""
    initialize_grid(x_max, N)

Инициализация параметров сетки.
"""
function initialize_grid(x_max::Quantity{Float64}, N::Int)
    step = x_max / N
    return GridParameters(x_max, N, step)
end

"""
    create_grid(params::GridParameters)

Создание трехмерной сетки на основе параметров сетки.
"""
function create_grid(params::GridParameters)
    # Создание одномерных массивов координат
    N = params.N
    step = params.step
    
    # Создание массивов с соответствующими единицами измерения
    oX = [(n-1) * step - step/2 for n in 1:(N+2)]
    oY = [(n-1) * step - step/2 for n in 1:(N+2)]
    oZ = [(n-1) * step - step/2 for n in 1:(N+2)]
    
    return oX, oY, oZ
end

"""
    initialize_simulation_state(params::GridParameters, rho_c::Quantity{Float64})

Инициализация состояния симуляции со значениями по умолчанию.
"""
function initialize_simulation_state(params::GridParameters, rho_c::Quantity{Float64})
    N = params.N
    x_max = params.x_max
    
    # Инициализация массивов
    Phi = zeros(N+2, N+2, N+2)
    Rho = zeros(N+2, N+2, N+2)
    
    # Установка начальных условий
    for i in 1:(N+2)
        for j in 1:(N+2)
            for k in 1:(N+2)
                Phi[i, j, k] = -1.0e-5
            end
        end
    end
    
    # Инициализация массы
    M_0_value = 4 * ustrip(rho_c) * (ustrip(x_max) / 10)^2
    M = M_0_value * u"Msun"
    
    return SimulationState(Phi, Rho, M)
end

"""
    boundary_conditions!(state::SimulationState, oX, oY, oZ, M_0, constants::PhysicalConstants)

Применение граничных условий к состоянию симуляции.
"""
function boundary_conditions!(state::SimulationState, oX, oY, oZ, M_0, constants::PhysicalConstants)
    N = size(state.Phi, 1) - 2
    G_ast = constants.G
    
    for i in 1:(N+2)
        for j in 1:(N+2)
            state.Phi[i, j, 1] = state.Phi[i, j, 2]  # гр. усл. Phi на z=0
            state.Rho[i, j, 1] = state.Rho[i, j, 2]  # гр. усл. Rho на z=0
            # Преобразуем результат в безразмерную величину
            potential = -G_ast * M_0 / sqrt(oX[i]^2 + oY[j]^2 + oZ[N+2]^2)
            state.Phi[i, j, N+2] = ustrip(potential)  # гр. усл. на z=z_max
        end
    end

    for i in 1:(N+2)
        for k in 1:(N+2)
            state.Phi[i, 1, k] = state.Phi[i, 2, k]  # гр. усл. Phi на y=0
            state.Rho[i, 1, k] = state.Rho[i, 2, k]  # гр. усл. Phi на y=0
            # Преобразуем результат в безразмерную величину
            potential = -G_ast * M_0 / sqrt(oX[i]^2 + oY[N+2]^2 + oZ[k]^2)
            state.Phi[i, N+2, k] = ustrip(potential)  # гр. усл. на y=y_max
        end
    end

    for j in 1:(N+2)
        for k in 1:(N+2)
            state.Phi[1, j, k] = state.Phi[2, j, k]  # гр. усл. Phi на x=0
            state.Rho[1, j, k] = state.Rho[2, j, k]  # гр. усл. Phi на x=0
            # Преобразуем результат в безразмерную величину
            potential = -G_ast * M_0 / sqrt(oX[N+2]^2 + oY[j]^2 + oZ[k]^2)
            state.Phi[N+2, j, k] = ustrip(potential)  # гр. усл. на x=x_max
        end
    end
end

"""
    iterations_method!(state::SimulationState, oX, oY, oZ, params::GridParameters, 
                      constants::PhysicalConstants, rho_c_ast)

Выполнение итераций для решения гравитационного потенциала.
"""
function iterations_method!(state::SimulationState, oX, oY, oZ, params::GridParameters, 
                           constants::PhysicalConstants, rho_c_ast)
    N = params.N
    step = params.step
    T = constants.T
    G_ast = constants.G
    R_ast = constants.R
    mu_ast = constants.mu
    
    # Вычисление коэффициентов
    A = 4.0 * π * G_ast * rho_c_ast
    B = mu_ast / R_ast
    
    # Инициализация аккумулятора массы
    M = 0.0u"Msun"
    
    # Применение граничных условий
    boundary_conditions!(state, oX, oY, oZ, state.M, constants)
    
    # Временный массив для расчетов
    f = zeros(N+2, N+2, N+2)
    
    # Основной цикл итераций
    for i0 in 1:N
        i = i0 + 1  # смещение от границы
        for j0 in 1:N
            j = j0 + 1
            for k0 in 1:N
                k = k0 + 1
                # Внутренние итерации для решения трансцендентного уравнения
                for _ in 1:5
                    # Удаляем единицы измерения для совместимости с массивом f
                    B_T = ustrip(B / T)
                    f[i, j, k] = ustrip(A) * exp(-state.Phi[i, j, k] * B_T)
                    
                    # Вычисляем новое значение потенциала
                    laplacian = (state.Phi[i+1, j, k] + state.Phi[i-1, j, k]
                               + state.Phi[i, j+1, k] + state.Phi[i, j-1, k]
                               + state.Phi[i, j, k+1] + state.Phi[i, j, k-1])
                    step_squared = ustrip(step^2)
                    state.Phi[i, j, k] = (laplacian - step_squared * f[i, j, k]) / 6.0
                end
                
                # Обновление плотности и массы
                state.Rho[i, j, k] = ustrip(rho_c_ast) * exp(-state.Phi[i, j, k] * ustrip(B / T))
                M += 8.0 * (state.Rho[i, j, k] * u"Msun/AU^3") * step^3
            end
        end
    end
    
    # Обновление массы состояния
    state.M = M
    
    return state
end

"""
    extract_2d_slice(state::SimulationState, slice_dim::Int=1, slice_idx::Int=1)

Извлечение двумерного среза из трехмерного поля потенциала.
"""
function extract_2d_slice(state::SimulationState, slice_dim::Int=3, slice_idx::Int=1)
    if slice_dim == 1
        return state.Phi[slice_idx, :, :]
    elseif slice_dim == 2
        return state.Phi[:, slice_idx, :]
    else
        return state.Phi[:, :, slice_idx]
    end
end

"""
    run_simulation(n_iterations::Int=1000, x_max::Quantity{Float64}=4.0u"AU", N::Int=15)

Запуск симуляции облачной модели.
"""
function run_simulation(n_iterations::Int=1000, x_max::Quantity{Float64}=4.0u"AU", N::Int=15)
    # Инициализация констант и параметров
    constants = initialize_physical_constants()
    grid_params = initialize_grid(x_max, N)
    
    # Вычисление производных констант
    M_sun = 1u"Msun"
    au = 1u"AU"
    one_year = 365.2425u"d"
    
    # Центральная плотность и преобразования
    rho_c = 1.3e-9u"kg/m^3"  # Центральная плотность
    rho_c_ast = uconvert(u"Msun/AU^3", rho_c)
    
    # Создание сетки
    oX, oY, oZ = create_grid(grid_params)
    
    # Инициализация состояния симуляции
    state = initialize_simulation_state(grid_params, rho_c_ast)
    
    # Запуск итераций
    @info "Начало симуляции с $n_iterations итераций..."
    for q in 1:n_iterations
        iterations_method!(state, oX, oY, oZ, grid_params, constants, rho_c_ast)
        if q % 100 == 0
            @info "Итерация $q: Масса = $(state.M); Phi[3,4,5]= $(state.Phi[3,4,5]), Rho[3,4,5]= $(state.Rho[3,4,5])"
        end
    end
    
    return state, oX, oY, oZ
end

"""
    visualize_results(state::SimulationState, oX, oY, oZ; slice_dim::Int=3, slice_idx::Int=1)

Визуализация результатов симуляции.
"""
function visualize_results(state::SimulationState, oX, oY, oZ; slice_dim::Int=3, slice_idx::Int=1)
    # Извлечение двумерного среза для визуализации
    Phi_2d = extract_2d_slice(state, slice_dim, slice_idx)
    
    # Определение координат для использования в зависимости от размерности среза
    if slice_dim == 1
        coords_x = oY
        coords_y = oZ
        xlabel = "y (AU)"
        ylabel = "z (AU)"
    elseif slice_dim == 2
        coords_x = oX
        coords_y = oZ
        xlabel = "x (AU)"
        ylabel = "z (AU)"
    else
        coords_x = oX
        coords_y = oY
        xlabel = "x (AU)"
        ylabel = "y (AU)"
    end
    
    # Преобразование значений сетки для построения графика
    X_plot = [ustrip(coords_x[i]) for i in 1:length(coords_x)]
    Y_plot = [ustrip(coords_y[j]) for j in 1:length(coords_y)]
    
    # Создание фигуры
    fig = Figure(size = (900, 700), fontsize = 14)
    
    # Трехмерный поверхностный график
    ax1 = Axis3(fig[1, 1:2], 
               xlabel = xlabel, 
               ylabel = ylabel, 
               zlabel = "Потенциал",
               title = "Гравитационный потенциал")
    
    # Используем surface с векторами X_plot, Y_plot вместо матриц
    surface!(ax1, X_plot, Y_plot, Phi_2d,
             colormap = :viridis, 
             shading = FastShading,
             alpha = 0.9)
    
    # Двумерная тепловая карта
    ax2 = Axis(fig[2, 1],
               xlabel = xlabel,
               ylabel = ylabel,
               title = "Потенциал (Вид сверху)")
    
    # Heatmap работает с векторами для координат
    hm = heatmap!(ax2, X_plot, Y_plot, Phi_2d,
                 colormap = :viridis)
    
    Colorbar(fig[2, 2], hm, label = "Потенциал")
    
    # Добавление контурного графика
    ax3 = Axis(fig[2, 3],
               xlabel = xlabel,
               ylabel = ylabel,
               title = "Контуры потенциала")
    
    # Contour также работает с векторами для координат
    contour!(ax3, X_plot, Y_plot, Phi_2d,
             levels = 15,
             colormap = :viridis)
    
    # Отображение фигуры
    display(fig)
    
    # Сохранение фигуры
    save("gravitational_potential.png", fig)
    
    return fig
end

end # модуль CloudModel

# Пример использования (вне модуля)
using .CloudModel

# Запуск симуляции
state, oX, oY, oZ = CloudModel.run_simulation(10000)

# Визуализация результатов
CloudModel.visualize_results(state, oX, oY, oZ)

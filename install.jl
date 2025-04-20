#!/usr/bin/env julia

# Проверяем наличие пакета
if !isdir(joinpath(homedir(), ".julia", "dev", "CloudModel"))
    # Клонируем репозиторий
    run(`git clone https://github.com/your-username/CloudModel.jl.git $(joinpath(homedir(), ".julia", "dev", "CloudModel"))`)
end

# Устанавливаем пакет в режиме разработки
using Pkg
Pkg.develop(path=joinpath(homedir(), ".julia", "dev", "CloudModel"))

# Устанавливаем все зависимости
Pkg.instantiate()

# Компилируем пакет
using CloudModel

# Создаем симлинк в /usr/local/bin или другой директории в PATH
bin_path = "/usr/local/bin/cloud_model"
if !isfile(bin_path)
    script_path = joinpath(homedir(), ".julia", "dev", "CloudModel", "bin", "cloud_model")
    try
        run(`sudo ln -s $script_path $bin_path`)
        println("Создан симлинк в $bin_path")
    catch
        println("Запустите следующую команду вручную:")
        println("sudo ln -s $script_path $bin_path")
    end
end

println("Установка завершена! Теперь вы можете запустить модель командой:")
println("cloud_model")
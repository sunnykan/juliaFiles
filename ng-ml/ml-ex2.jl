using DataFrames
using CSV
using Gadfly
using GLM
using Optim
using Polynomials

function read_data(file)
    df = CSV.read(file, datarow = 1)
    first(df, 5)
    names(df)
    #categorical!(df, :admit)
    #describe(df, stats = [:eltype])
    #plot(df, x = :exam1, y = :exam2, color = :admit)
    return(df)
end

function sigmoidx(x)
    # input: scalar, vector or matrix of values
    # ouput: value(s) returned by sigmoid function
    1 ./(1 .+ map(exp, -x))
end

function costx(thetavals, datamat, outcome)

    thetaX = thetavals'datamat'
    pos = -map(log, sigmoidx(thetaX)) * outcome
    neg = map(log, (1 .- sigmoidx(thetaX))) * (1 .- outcome)
    cost_θ = (pos - neg)/length(target)
    return first(cost_θ)
#    cost_θ = (pos - neg)/length(target)
#    gradient_θ = (datamat' * (sigmoidx(thetaX)' - outcome))/length(outcome)
#    return cost_θ, gradient_θ
#    return cost_θ
end

function grd(thetavals, datamat, outcome)
    thetaX = thetavals'datamat'
    partials = (datamat' * (sigmoidx(thetaX)' - outcome))/length(outcome)
    return partials
end

function designmat(df)
    # input: dataframe with features (end - 1) and outcome (end)
    # output:
            # matrix X: constant + feature vectors (examples * features)
            # vector y: outcome (examples * 1)
    X = hcat(ones(size(df[1])), convert(Matrix, df))[:,1:end-1]
    y = df[:,end]
    #return X, y
    return convert(Array{Float64, 2}, X), convert(Array{Float64, 1}, y)
end

# Fit using GLM library
function fit_glm(df)
    glm(@formula(admit ~ exam1 + exam2), df, Binomial(), LogitLink())
end

##
file_url = "./machine-learning-ex2/ex2/ex2data1.txt"
data = read_data(file_url)
describe(data, stats = [:eltype])
rename!(data, :Column1 => :exam1, :Column2 => :exam2, :Column3 => :admit)
glm_mod = fit_glm(data)

thetas = [-24, 0.2, 0.2]
features, target = designmat(data)
costx(thetas, features, target)
grd(thetas, features, target)

# https://discourse.julialang.org/t/sanity-check-with-optimization-function/7836/3
function ltx(X, y)
    costf(θ) = costx(θ, X, y)
    θ_initial = zeros(Float64, 3)
    optimize(costf, θ_initial)
end

#create logistic object
ltx_model = ltx(features, target)
Optim.minimizer(ltx_model)
Optim.minimum(ltx_model)
Optim.iterations(ltx_model)
##

## Using the gradient function
# https://stackoverflow.com/questions/32703119/logistic-regression-in-julia-using-optim-jl
function ltx(X, y)
    return (θ_val::Array) -> begin
        costx(θ_val, X, y)
    end, (θ_val::Array) -> begin
        grd(θ_val, X, y)
    end
end

cost, grdx = ltx(features, target)
cost(thetas)
grdx(thetas)

thetas = [0.0, 0.0, 0.0]
res = optimize(cost, grdx, thetas; inplace = false)
Optim.minimizer(res)
Optim.minimum(res)
Optim.iterations(res)

function ltx(X, y)
    costf(θ_val::Array) = costx(θ_val, X, y)
    gf(θ_val::Array) = grd(θ_val, X, y)
    θ_initial = zeros(Float64, 3)
    optimize(costf, gf, θ_initial; inplace = false)
end

res = ltx(features, target)
Optim.minimizer(res)
Optim.minimum(res)
Optim.iterations(res)

test_data = [1.0 45.0 85.0]
sigmoidx(test_data * Optim.minimizer(res))

### REGULARIZATION ###
file_url = "./machine-learning-ex2/ex2/ex2data2.txt"
data = read_data(file_url)
describe(data, stats = [:eltype])
rename!(data, :Column1 => :test1, :Column2 => :test2, :Column3 => :accept)

decision = categorical(data[:accept])

plot(data, x = :test1, y = :test2, color = decision)

function mapfeature(X1, X2)::Array
    degree = 6;
    out = ones(length(X1))
        for i = 1:degree
            for j = 0:i
                out = hcat(out, (X1.^(i-j)).*(X2.^j))
            end
        end
    out
end

xmat = convert(Matrix, data)
ymat = data[:accept]
polymat = mapfeature(x[:,1], x[:,2])

function costx_reg(thetavals, datamat, outcome, λ)
    nobs = length(outcome)
    thetaX = thetavals'datamat'
    pos = -map(log, sigmoidx(thetaX)) * outcome
    neg = map(log, (1 .- sigmoidx(thetaX))) * (1 .- outcome)
    cost_θ = (pos - neg)/nobs
    penalty_θ = (λ/(2 * nobs)) * (thetavals[2:end]'thetavals[2:end])
    return first(cost_θ) + penalty_θ
#    return first(cost_θ + penalty_θ)
#    cost_θ = (pos - neg)/length(target)
#    gradient_θ = (datamat' * (sigmoidx(thetaX)' - outcome))/length(outcome)
#    return cost_θ, gradient_θ
#    return cost_θ
end

thetas = zeros(size(polymat)[2])
lambda = 0.0001

costx_reg(thetas, polymat, ymat, lambda)

function grd_reg(thetavals, datamat, outcome, λ)
    nobs = length(outcome)
    thetaX = thetavals'datamat'
    partials = (datamat' * (sigmoidx(thetaX)' - outcome))/nobs
    penalty_θ = vcat(0, (λ/nobs) * thetavals[2:end])
    return partials + penalty_θ
end

grd_reg(thetas, polymat, ymat, lambda)

res = ltx_reg(polymat, ymat, lambda)

thetas = ones(size(polymat)[2])
lambda = 10
costx_reg(thetas, polymat, ymat, lambda)
grd_reg(thetas, polymat, ymat, lambda)

function ltx_reg(X, y, λ)
    costf(θ_val::Array) = costx_reg(θ_val, X, y, λ)
    gf(θ_val::Array) = grd_reg(θ_val, X, y, λ)
    θ_initial = zeros(Float64, size(X)[2])
    optimize(costf, gf, θ_initial; inplace = false)
end

res = ltx_reg(polymat, ymat, lambda)
Optim.minimizer(res)
Optim.minimum(res)
Optim.iterations(res)

res = map(lambda -> ltx_reg(polymat, ymat, lambda), [0.00001, 1, 10, 100])
# no convergence with 0. use very small value instead
Optim.minimizer(res[1])
Optim.minimum(res[1])
Optim.iterations(res[1])

Optim.minimizer(res[2])
Optim.minimum(res[2])
Optim.iterations(res[2])

Optim.minimizer(res[3])
Optim.minimum(res[3])
Optim.iterations(res[3])

Optim.minimizer(res[4])
Optim.minimum(res[4])
Optim.iterations(res[4])

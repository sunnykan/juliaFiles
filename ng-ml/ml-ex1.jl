using DataFrames
using CSV
using Gadfly
using GLM
using Statistics
using LinearAlgebra

file ="./machine-learning-ex1/ex1/ex1data1.txt"
df = CSV.read(file, datarow = 1)
rename!(df, :Column1 => :X, :Column2 => :Y)
head(df)

#df = DataFrame(X=[1,2,3], Y=[2,4,7])
#ols = lm(@formula(Y ~ X), df)

#calculate ols estimates using GLM library
#fit_lm(dataset) = lm(@formula(Y ~ X), dataset)

plt1 = plot(df, x = :X, y = :Y, Geom.point,
    Guide.xlabel("Population of city in 10,000s"),
    Guide.ylabel("Profit in 10,000s"))

function designmat(data)
    X = hcat(ones(size(df[1])), convert(Matrix, data))[:,1:2]
    y = data[:, end]
    return X, y
end

function residuals(feature_mat, weights, output)
    predictions = weights' * feature_mat
    predictions - output'
end

function slr_gd(feature_mat, output, α, initial_wgts, max_iter = 1500)
    weights = initial_wgts
    nobs = length(output)
    jtheta = Array{Float64}(undef, max_iter)

    for iter = 1:max_iter
        errors = residuals(feature_mat, weights, output)
        weights = weights - (α/nobs * ((errors * feature_mat')'))
        jtheta[iter] = (mean(errors.^2)/2) #broadcasting: dot operator
    end
    return weights, jtheta
end

ols = fit_lm(df) # fit ols using GLM
neqs_est = pinv(x'x)x'y #estimates using Normal eqns

features, target = designmat(df)

thetas = zeros(size(features)[2])
step_size = 0.01
max_iterations = 1500
params, cost = slr_gd(features', target, step_size, thetas, max_iterations)

x = features
y = target
neqs_est = pinv(x'x)x'y #estimates using Normal eqns

plot(df, x = :X, y = :Y, Geom.point,
    intercept = [params[1]], slope = [params[2]], Geom.abline(color = "red"),
    Guide.xlabel("Population of city in 10,000s"),
    Guide.ylabel("Profit in 10,000s"))

# predictions
[1, 3.5]' * params
[1, 7]' * params

## Multiple Linear Regression
file ="./machine-learning-ex1/ex1/ex1data2.txt"
df1 = CSV.read(file, datarow = 1)
rename!(df1, :Column1 => :size_sqft, :Column2 => :nbedrooms, :Column3 => :price)

plot(df1, x = :size_sqft, y = :price, Geom.point)
plot(df1, x = :nbedrooms, y = :price, Geom.point)

function normalize_features(input_features)
    # X = Array{Float64}(undef, size(input_features))
    X = ones(size(input_features)[1]) # add vector of constants
    stats = Dict{Symbol,Tuple{Float64, Float64}}() # save means and sds

    for name in names(input_features)

        mean_name = mean(input_features[:, name])
        sd_name = std(input_features[:, name])
        X_name = (input_features[:,name] .- mean_name) ./ std(input_features[:,name])
        X = hcat(X, X_name)
        stats[name] = (mean_name, sd_name)
    end
    X, stats
end

# obtain normalized features
norm_features, feature_stats = normalize_features(df1[:,1:2])
target = convert(Array{Float64,1}, df1[end])

#neqs_est = pinv(norm_features'norm_features)norm_features'target
neqs_est = norm_features \ target
norm_df = hcat(norm_features[:, 2:3], target)
ols = lm(norm_features, target)

#gradient descent
thetas = zeros(size(norm_features)[2])
step_size = 0.9
max_iterations = 2
params, cost = slr_gd(norm_features', target, step_size, thetas, max_iterations)

# cost as a function of iterations
plot(x = collect(1:1:max_iterations), y = cost, Geom.line)

# try different learning rates
max_iterations = 50
step_size = 0.9
models = Array{Tuple{String, Array{Float64, 1}, Array{Float64, 1}}}(undef, 0)

while step_size > 0.001
    params, cost = slr_gd(norm_features', target, step_size, thetas, max_iterations)
    push!(models, (string(step_size), params, cost))
    global step_size = round(step_size / 3; digits = 3)
end

function create_df()
    costs = Array{Any}(undef, 0, 3)
    for n in 1:length(models)
        costs = vcat(costs, hcat(collect(1:1:max_iterations),
                    fill(models[n][1], max_iterations),models[n][3]))
    end
    convert(DataFrame, costs)
end

#create a dataframe with the models for different learning rates with the
#corresponding costs
costs_df = create_df()
rename!(costs_df,
        :x1 => :iter_num,
        :x2 => :learning_rate,
        :x3 => :cost)
plot(costs_df, x = :iter_num,
                y = :cost, color = :learning_rate, Geom.line)

#predictions

test_data = [1, 1650, 3]
test_norm = [1, (test_data[2] - feature_stats[:size_sqft][1])/feature_stats[:size_sqft][2],
            (test_data[3] - feature_stats[:nbedrooms][1])/feature_stats[:nbedrooms][2]]

# use learning rate = 0.9
test_norm' * models[1][2]
test_norm' * neqs_est

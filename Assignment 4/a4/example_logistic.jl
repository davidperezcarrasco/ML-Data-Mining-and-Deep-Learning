using Statistics

# Load X and y variable
using JLD
data = load("logisticData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Standardize columns and add bias
n = size(X,1)
include("misc.jl")
(X,mu,sigma) = standardizeCols(X)
X = [ones(n,1) X]

# Standardize columns of test data, using mean/std from train data
t = size(Xtest,1)
Xtest = standardizeCols(Xtest,mu=mu,sigma=sigma)
Xtest = [ones(t,1) Xtest]

# Fit logistic regression model
include("leastSquares.jl")
model = binaryLeastSquares(X,y)

# Count number of non-zeroes in model
numberOfNonZero = sum(model.w .!= 0)
@show(numberOfNonZero)

# Compute training and validation error
yhat = model.predict(X)
trainError = mean(yhat .!= y)
@show(trainError)
yhat = model.predict(Xtest)
validError = mean(yhat .!= ytest)
@show(validError)
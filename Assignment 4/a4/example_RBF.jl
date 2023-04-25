using Printf
using Random
using LinearAlgebra

# Load X and y variable
using JLD
data = load("basisData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])

# Data is sorted, so *randomly* split into train and validation:
n = size(X,1)
perm = randperm(n)
validStart = Int64(n/2+1) # Start of validation indices
validEnd = Int64(n) # End of validation incides
validNdx = perm[validStart:validEnd] # Indices of validation examples
trainNdx = perm[setdiff(1:n,validStart:validEnd)] # Indices of training examples
Xtrain = X[trainNdx,:]
ytrain = y[trainNdx]
Xvalid = X[validNdx,:]
yvalid = y[validNdx]

@show validStart
@show validEnd 
@show size(X)
@show size(Xvalid)
@show size(Xtrain)
@show perm

# Find best value of RBF variance parameter,
#	training on the train set and validating on the test set
include("leastSquares.jl")
minErr = Inf
bestSigma = []
for sigma in 2.0.^(-15:15)

	# Train on the training set
	model_sigma = leastSquaresRBF(Xtrain,ytrain,sigma, 10^-12)

	# Compute the error on the validation set
	yhat_sigma = model_sigma.predict(Xvalid)
	validError = sum((yhat_sigma - yvalid).^2)/(n/2)
	@printf("With sigma = %.3f, validError = %.2f\n",sigma,validError)

	# Keep track of the lowest validation error
	if validError < minErr
		global minErr = validError
		global bestSigma = sigma
	end

end

# Now fit the model based on the full dataset
model = leastSquaresRBF(X,y,1, 10^-12)

# Report the error on the test set
t = size(Xtest,1)
yhat = model.predict(Xtest)
testError = sum((yhat - ytest).^2)/t
@printf("With best sigma of %.3f, testError = %.2f\n",bestSigma,testError)

# Plot model
using Plots
scatter(X,y,legend=false,linestyle=:dot)
scatter!(Xtest,ytest,legend=false,linestyle=:dot)
Xhat = minimum(X):.1:maximum(X)
Xhat = reshape(Xhat,length(Xhat),1) # Make into an n by 1 matrix
yhat = model.predict(Xhat)
plot!(Xhat,yhat,legend=false)
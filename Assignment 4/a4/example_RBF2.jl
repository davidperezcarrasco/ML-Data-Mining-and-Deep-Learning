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


# Find best value of RBF variance parameter,
#	training on the train set and validating on the test set
include("leastSquares.jl")
minErr = Inf
bestSigma = []
for sigma in 2.0.^(-15:15)

	# Apply 10-fold CV
    FoldErr = 0
    for fold in 0:9
    
        local validStart = Int64(fold*n/10+1) # Start of fold indices
        local validEnd = Int64((fold+1)*n/10) # End of fold incides
        local validNdx = perm[validStart:validEnd] # Indices of fold examples
        local trainNdx = perm[setdiff(1:n,validStart:validEnd)] # Indices of training examples
        local Xtrain = X[trainNdx,:]
        local ytrain = y[trainNdx]
        local Xvalid = X[validNdx,:]
        local yvalid = y[validNdx]


	    model_sigma = leastSquaresRBF(Xtrain,ytrain,sigma, 10^-12)

	    # Compute the error on the validation set
	    yhat_sigma = model_sigma.predict(Xvalid)
	    FoldvalidError = sum((yhat_sigma - yvalid).^2)/(n/2)

        FoldErr+=FoldvalidError

    end

    validError = FoldErr/10

    @printf("With sigma = %.3f, Average validError with 10-fold CV = %.2f\n",sigma,validError)

	# Keep track of the lowest validation error
	if validError < minErr
		global minErr = validError
		global bestSigma = sigma
	end

end

# Now fit the model based on the full dataset
model = leastSquaresRBF(X,y,bestSigma, 10^-12)

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
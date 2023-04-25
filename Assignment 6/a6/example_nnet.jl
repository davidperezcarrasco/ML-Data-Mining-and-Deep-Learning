using Printf

# Load X and y variable
using JLD
using Plots
data = load("basisData.jld")
(X,y) = (data["X"],data["y"])
(n,d) = size(X)
X2 = ones(n,d+1)
X2[:,2:end] = X
X=X2
d+=1
# Choose network structure and randomly initialize weights
include("NeuralNet.jl")
nHidden = [7 7]
nParams = NeuralNet_nParams(d,nHidden)
w = randn(nParams,1)

# Train with stochastic gradient
maxIter = 30000
stepSize = 1e-3
for t in 1:maxIter

	# The stochastic gradient update:
	i = rand(1:n)
	(f,g) = NeuralNet_backprop(w,X[i,:],y[i],nHidden)
	global w = w - stepSize*g

	# Every few iterations, plot the data/model:
	if (mod(t-1,round(maxIter/50)) == 0)
		@printf("Training iteration = %d\n",t-1)
		xVals = -10:.05:10
		Xhat = ones(length(xVals),2)
		Xhat[:,2] .= xVals
		yhat = NeuralNet_predict(w,Xhat,nHidden)
		scatter(X[:,2],y,legend=false,linestyle=:dot)
		plot!(Xhat[:,2],yhat,legend=false)
		gui()
		sleep(.1)
	end
end
plot!()

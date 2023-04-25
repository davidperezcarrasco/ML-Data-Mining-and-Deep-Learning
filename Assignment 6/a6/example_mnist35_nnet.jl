# Load X and y variable
using JLD, Printf
data = load("mnist35.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])
y[y.==2] .= -1
ytest[ytest.==2] .= -1
(n,d) = size(X)
X2 = ones(n,d+1)
X2[:,2:end] = X
X=X2

(t,d) = size(Xtest)
Xtest2 = ones(t,d+1)
Xtest2[:,2:end] = Xtest
Xtest=Xtest2
d+=1
# Choose network structure and randomly initialize weights
include("NeuralNet.jl")
nHidden = [10 10]
nParams = NeuralNet_nParams(d,nHidden)
w = randn(nParams,1)

# Train with stochastic gradient
maxIter = 1500000
stepSize = 1e-2
for t in 1:maxIter

	# The stochastic gradient update:
	i = rand(1:n)
	(f,g) = NeuralNet_backprop(w,X[i,:],y[i],nHidden)
	gW = reshape(g[1:3*d],3,d) # Gradient with respect to weights in first layer
	gv = g[3*d+1:end] # Gradient with respect to weights in seond layer
	global w = w - stepSize*g

	# Every few iterations, plot the data/model:
	if (mod(t-1,round(maxIter/50)) == 0)
		yh = sign.(NeuralNet_predict(w,Xtest,nHidden))
		errorRate = sum(yh .!= ytest)/size(Xtest,1)
		@printf("Training iteration = %d, error rate = %.2f\n",t-1,errorRate)
	end
end

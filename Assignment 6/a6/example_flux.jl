# Load X and y variable
using JLD
data = load("mnist35.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])
y[y.==2] .= -1
ytest[ytest.==2] .= -1
(n,d) = size(X)

# Pick a random training example where we will compute NLL and gradient
i = rand(1:n)

# Compute the squared loss for a neural network with 1 hidden layer of 3 units
W = randn(3,d)
v = randn(3)
loss(x,y,W,v) = (1/2)*(v'*tanh.(W*x)-y)^2
f = loss(X[i,:],y[i],W,v)

# Compute loss and gradient using manually-written code from previous assignment
include("NeuralNet.jl")
nHidden  = [3]
w = [W[:];v[:]]
(f_manual,g) = NeuralNet_backprop(w,X[i,:],y[i],nHidden)
gW_manual = reshape(g[1:3*d],3,d) # Gradient with respect to weights in first layer
gv_manual = g[3*d+1:end] # Gradient with respect to weights in seond layer

# Compute the gradient using Flux's automatic differentiation
using Flux
g_AD = gradient(loss,X[i,:],y[i],W,v) # Returns gradient of function 'loss' with respect to each argument
gW_AD = g_AD[3]
gv_AD = g_AD[4]

# Re-writing the objective using Flux's "Chain" and "Flux.params" functions
model = Chain(x -> W*x,z -> tanh.(z), a -> v'a)
loss2(x,y) = (1/2)*(model(x)-y)^2
f_chain = loss2(X[i,:],y[i])
g_chain = gradient(()->loss2(X[i,:],y[i]),Flux.params([W,v])) # Using the "no argument" function ()->loss2(...) "delays" executing the loss so AD can do its work
gW_chain = g_chain[W]
gv_chain = g_chain[v]

# An alternate syntax that Flux supports
g_chain2 = gradient(Flux.params([W,v])) do
	loss2(X[i,:],y[i])
end
gW_chain2 = g_chain2[W]
gv_chain2 = g_chain2[v]

# Re-writing the objective using Flux's pre-defined layer function
vt = reshape(v,1,3)
model2 = Chain(Dense(W,false,tanh),Dense(vt,false,identity))
loss3(x,y) = (1/2)*(model2(x)[1]-y)^2
f_layer = loss3(X[i,:],y[i])
g_layer = gradient(Flux.params(model2)) do
	loss3(X[i,:],y[i])
end
gW_layer = g_layer[W]
gv_layer = g_layer[vt]



###########################
# Different ways to write gradient descent update

stepSize = 1e-4

# Concatenating all weights into a big vector
w - stepSize*g

# Updating parameters in their original shape
W - stepSize*gW_manual
v - stepSize*gv_manual

# Using named gradient values
W - stepSize*g_chain[W]
v - stepSize*g_chain[v]

# For loop over tunable parameters
for p in Flux.params([W,v])
	p - stepSize*g_chain[p]
end

# For loop over layers
for p in Flux.params(model2)
	p - stepSize*g_layer[p]
end

# You can also import and use the "update!" function,
# to remove the "for" loop above and update all parameters with
# a selected optimizer (and without reshape to a vector)
# see here: https://fluxml.ai/Flux.jl/stable/training/optimisers/

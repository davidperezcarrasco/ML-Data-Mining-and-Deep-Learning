# Load faces data
using JLD
X = load("faces.jld")["X"]
(n,d) = size(X)

# Places the faces in a random order
using Random
X = X[randperm(n),:]

# Shows examples of faces
using Plots
plot_array = []
for i in 1:16
	push!(plot_array,heatmap(reshape(X[i,:],32,32),yflip=true,c=:grays,legend=false))
end
facesOriginal = plot(plot_array...)

# Run PCA
include("PCA.jl")
k = 16
model = NMF(X,k)
#model = PCA_gradient(X,k)

# Make low-dimensional representation
Z = model.compress(X)

# Look at low-dimensional representation in original space
Xhat = model.expand(Z)

# Show reconstruction of original faces
plot_array = []
for i in 1:16
	push!(plot_array,heatmap(reshape(Xhat[i,:],32,32),yflip=true,c=:grays,legend=false))
end
facesCompressed = plot(plot_array...)

# Show mean face
faceMu = heatmap(reshape(mean(X,dims=1)
,32,32),yflip=true,c=:grays,legend=false)

# Show mean face and eigenfaces
plot_array = []
for c in 1:k
	push!(plot_array,heatmap(reshape(model.W[c,:],32,32),yflip=true,c=:grays,legend=false))
end
eigenFaces = plot(plot_array...)






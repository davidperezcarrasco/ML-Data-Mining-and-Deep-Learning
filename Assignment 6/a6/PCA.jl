using Printf
using Statistics
using LinearAlgebra
include("misc.jl")
include("findMin.jl")

function PCA(X,k)
    (n,d) = size(X)

    # Subtract mean
    mu = mean(X,dims=1)
    X -= repeat(mu,n,1)

    (U,S,V) = svd(X)
    W = V[:,1:k]'

    compress(Xhat) = compressFunc(Xhat,W,mu)
    expand(Z) = expandFunc(Z,W,mu)

    return CompressModel(compress,expand,W)
end

function compressFunc(Xhat,W,mu)
    (t,d) = size(Xhat)
    Xcentered = Xhat - repeat(mu,t,1)
    return Xcentered*W' # Assumes W has orthogonal rows
end

function expandFunc(Z,W,mu)
    (t,k) = size(Z)
    return Z*W + repeat(mu,t,1)
end

function PCA_gradient(X,k)
    (n,d) = size(X)

    # Subtract mean
    mu = mean(X,dims=1)
    X -= repeat(mu,n,1)

    # Initialize W and Z
    W = randn(k,d)
    Z = randn(n,k)

    R = Z*W - X
    f = sum(R.^2)
    funObjZ(z) = pcaObjZ(z,X,W)
    funObjW(w) = pcaObjW(w,X,Z)
    for iter in 1:50
        fOld = f

        # Update Z
        Z[:] = findMin(funObjZ,Z[:],verbose=false,maxIter=10)

        # Update W
        W[:] = findMin(funObjW,W[:],verbose=false,maxIter=10)

        R = Z*W - X
        f = (1/2)sum(R.^2)
        @printf("Iteration %d, loss = %f\n",iter,f/length(X))

        if (fOld - f)/length(X) < 1e-2
            break
        end
    end


    # We didn't enforce that W was orthogonal so we need to optimize to find Z
    compress(Xhat) = compress_gradientDescent(Xhat,W,mu)
    expand(Z) = expandFunc(Z,W,mu)

    return CompressModel(compress,expand,W)
end

function compress_gradientDescent(Xhat,W,mu)
    (t,d) = size(Xhat)
    k = size(W,1)
    Xcentered = Xhat - repeat(mu,t,1)
    Z = zeros(t,k)

    funObj(z) = pcaObjZ(z,Xcentered,W)
    Z[:] = findMin(funObj,Z[:],verbose=false)
    return Z
end


function pcaObjZ(z,X,W)
    # Rezie vector of parameters into matrix
    n = size(X,1)
    k = size(W,1)
    Z = reshape(z,n,k)

    # Comptue function value
    R = Z*W - X
    f = (1/2)sum(R.^2)

    # Comptue derivative with respect to each residual
    dR = R

    # Multiply by W' to get elements of gradient
    G = dR*W'

    # Return function and gradient vector
    return (f,G[:])
end

function pcaObjW(w,X,Z)
    # Rezie vector of parameters into matrix
    d = size(X,2)
    k = size(Z,2)
    W = reshape(w,k,d)

    # Comptue function value
    R = Z*W - X
    f = (1/2)sum(R.^2)

    # Comptue derivative with respect to each residual
    dR = R

    # Multiply by Z' to get elements of gradient
    G = Z'dR

    # Return function and gradient vector
    return (f,G[:])
end


function NMF(X,k)
    (n,d) = size(X)

    # Initialize W and Z
    W = randn(k,d)
    Z = randn(n,k)

    for i in 1:k
        for j in 1:d
            W[i,j] = max(W[i,j],0)
        end
    end

    for i in 1:n
        for j in 1:k
            Z[i,j] = max(Z[i,j],0)
        end
    end

    R = Z*W - X
    f = sum(R.^2)
    funObjZ(z) = pcaObjZ(z,X,W)
    funObjW(w) = pcaObjW(w,X,Z)
    for iter in 1:50
        fOld = f

        # Update Z
        Z[:] = findMinNN(funObjZ,Z[:],verbose=false,maxIter=10)

        # Update W
        W[:] = findMinNN(funObjW,W[:],verbose=false,maxIter=10)
        R = Z*W - X
        f = (1/2)sum(R.^2)
        @printf("Iteration %d, loss = %f\n",iter,f/length(X))

        if (fOld - f)/length(X) < 1e-2
            break
        end
    end

    # We didn't enforce that W was orthogonal so we need to optimize to find Z
    compress(Xhat) = compress_gradientDescentNMF(Xhat,W)
    expand(Z) = expandFuncNMF(Z,W)

    return CompressModel(compress,expand,W)
end
function compress_gradientDescentNMF(Xhat,W)
    (t,d) = size(Xhat)
    k = size(W,1)
    Z = zeros(t,k)

    funObj(z) = pcaObjZ(z,Xhat,W)
    Z[:] = findMinNN(funObj,Z[:],verbose=false)
    return Z
end
function expandFuncNMF(Z,W)
    return Z*W 
end
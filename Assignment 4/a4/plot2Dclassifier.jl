using Plots

function plot2Dclassifier(X,y,model;Xtest=[],ytest=[],biasIncluded=false,k=2)

	if biasIncluded
		f1 = 2
		f2 = 3
	else
		f1 = 1
		f2 = 2
	end

	# Make predictions across range of plot
	increment = 200
	xmin = minimum(X[:,f1])
	xmax = maximum(X[:,f1])
	xDomain = range(xmin,stop=xmax,length=increment)
	xValues = repeat(xDomain',length(xDomain),1)

	ymin = minimum(X[:,f2])
	ymax = maximum(X[:,f2])
	yDomain = range(ymin,stop=ymax,length=increment)
	yValues = repeat(yDomain,1,length(yDomain))

	if biasIncluded
		t = length(xValues)
		z = model.predict([ones(t,1) xValues[:] yValues[:]])
	else
		z = model.predict([xValues[:] yValues[:]])
	end

	@assert(length(z) == length(xValues),"Size of model function's output is wrong");

	zValues = reshape(z,size(xValues))

	# Pick some colors/makers for the classes
	colours = [	0 0 1
			1 0 0
			0 1 0
			1 1 1
			1 0 1
			0 1 1
			1 1 0
			.1 .1 .1
			1 .5 0
			0 .5 0
			.5 .5 .5
			.5 .25 0
			.5 0 .5
			0 .5 1]

	markers = [:circle
		:rect
		:star5
		:diamond
		:hexagon
		:cross
		:xcross
		:utriangle
		:dtriangle
		:rtriangle
		:ltriangle
		:pentagon
		:octagon
		:star4
		:star6
		:star7
		:star8
		:vline
		:hline
		:+
		:-]

	# Plot decision surface
	usedColours = Array{RGB{Float64},1}(undef,0)
	usedValues = zeros(0,1)
	for c in 1:k
		if any(zValues[:] .== c)
			push!(usedColours,RGB(.5*colours[c,1],.5*colours[c,2],.5*colours[c,3]))
			usedValues = [usedValues;c]
		end
	end
	contour(xDomain,yDomain,zValues,fill=true,seriescolor=cgrad(usedColours,usedValues),legend=false)

	# Overlay data points
	for c in 1:k
		scatter!(X[y.==c,f1],X[y.==c,f2],legend=false,markercolor=RGB(colours[c,1],colours[c,2],colours[c,3]),markershape=markers[c])
	end
	scatter!()
end

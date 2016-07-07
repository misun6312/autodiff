##############################################################################################
Automatic Differentiation
##############################################################################################

Goal
------------
Adapting Bing’s model to compute it using Algorithmic Differentiation.
[Bing's paper] (http://brodylab.org/publications-2/brunton-et-al-2013)

Automatic Differentiation is a technology for automatically augmenting computer programs, including arbitrarily complex simulations, with statements for the computation of derivatives, also known as sensitivities.

Status
-------
[Julia_autodiff_JupyterNotebook](https://github.com/misun6312/autodiff/blob/master/Julia_autodiff.ipynb)

In Julia, now it produces the same Loglikelihood values with the bing's matlab code.  
And it can calculate the gradients automatically using ForwardDiff.  
From the original version of the code in Julia, I updated the code for click adaptation part using inter-click interval instead of dt to consider all clicks and fixed some bugs. 
There is a little difference between the graident values of 9 parameters with error mean, std().

After updating the Julia packages with Pkg.update(), ForwardDiff was updated from v0.1 to v0.2. It turns out few API changes have occured between ForwardDiff v0.1 and v0.2.  

Instead of ForwardDiff.GradientNumber and ForwardDiff.HessianNumber, we will use ForwardDiff.Dual for the covert function.

    convert(::Type{Float64}, x::ForwardDiff.Dual) = Float64(x.value)
    function convert(::Array{Float64}, x::Array{ForwardDiff.Dual}) 
        y = zeros(size(x)); 
        for i in 1:prod(size(x)) 
            y[i] = convert(Float64, x[i]) 
        end
        return y
    end

ForwardDiff.Dual is now supporting ceil, floor function like below. 

    binN = ceil(Int, B/dx) 
    binBias = floor(Int, bias/dx) + binN+1  

To retrieve all the lower-order calculations along with the normal result of v0.2 API function, pass an instance of the appropriate ForwardDiffResult type to the in-place version of the function.

    # old way
    answer, results = ForwardDiff.hessian(f, x, AllResults)
    v = ForwardDiff.value(results)
    g = ForwardDiff.gradient(results)
    h = ForwardDiff.hessian(results) 
    # new way
    out = HessianResult(x)
    ForwardDiff.hessian!(out, f, x)
    v = ForwardDiff.value(out)
    g = ForwardDiff.gradient(out)
    h = ForwardDiff.hessian(out)




```
[LL dLL likey output] = single_trial35(param, mydata)
```

Setup Environment
-------
Julia 0.4.5  
FowardDiff v0.2

Implementing in Python (Theano / Tensorflow) 
-------


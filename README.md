##############################################################################################
Automatic Differentiation
##############################################################################################

Goal
------------
Adapting Bing’s model to compute it using Algorithmic Differentiation.
[(Brunton et al. 2013 Science)] (http://brodylab.org/publications-2/brunton-et-al-2013)

Automatic Differentiation is a technology for automatically augmenting computer programs, including arbitrarily complex simulations, with statements for the computation of derivatives, also known as sensitivities. Automatic differentiation is important because you don't want to have to hand-code a variation of gradient computation every time you're experimenting with a new arrangement of model. AD exploits the fact that every computer program, no matter how complicated, executes a sequence of elementary arithmetic operations (addition, subtraction, multiplication, division, etc.) and elementary functions (exp, log, sin, cos, etc.). By applying the chain rule repeatedly to these operations, derivatives of arbitrary order can be computed automatically, accurately to working precision, and using at most a small constant factor more arithmetic operations than the original program.

Status
-------
[Julia_autodiff_JupyterNotebook](https://github.com/misun6312/autodiff/blob/master/Julia_autodiff.ipynb)

In Julia, now it produces exact same Loglikelihood values as the LL values from the bing's matlab code.  
And it can compute the gradients for each of the parameters automatically using ForwardDiff (http://www.juliadiff.org/ForwardDiff.jl/index.html).  
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
* Julia 0.4.5  
* FowardDiff v0.2

Install Julia

    sudo add-apt-repository ppa:staticfloat/juliareleases
    sudo add-apt-repository ppa:staticfloat/julia-deps
    sudo apt-get update
    sudo apt-get install julia

Install ForwardDiff, simply use Julia’s package manager:

    julia> Pkg.add("ForwardDiff")


Implementing in Python (Theano / Tensorflow) 
-------
## Theano 
* [Theano Numerical Differentiation JupyterNotebook (it's working)](https://github.com/misun6312/autodiff/blob/master/Theano_Manualdiff.ipynb)   
This follows the numerical approach to compute the derivative for each parameters from Bing's paper [(Section 3.2 the Supplementary Information)](http://science.sciencemag.org/content/suppl/2013/04/04/340.6128.95.DC1).  

* [Theano Automatic Differentiation JupyterNotebook (incomplete)](https://github.com/misun6312/autodiff/blob/master/Theano_autodiff.ipynb)   
It can only compute gradients of bias and lapse automatically. For the rest of parameters, it produces nan value.

Theano is a Python library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays both on CPU and GPU efficiently.   

- Scan = for loop
- set_subtensor = array element assignment
- Shared Variable
- Construct Model(Graph) -> Compile
- configuration (fast_run mode, debug mode)
- 

## Tensorflow 
* [Tensorflow Automatic Differentiation JupyterNotebook (incomplete)](https://github.com/misun6312/autodiff/blob/master/Tensorflow_autodiff3.ipynb)  

TensorFlow is a Google-developed Python and C++ library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays both on CPU and GPU efficiently.

- Very inconvenient to use array index.. (not supporting negative index, have to use tf.slice to get the subset of array)
- Type casting is complicated and limited
- 
- Variable
- Placeholder : a variable that we will assign the data later. It allows to create computation graph without data we then feed data into the graph through placeholders.
- Construct Model(Graph) -> Launch the Graph


## Conclusion
Both libraries are using the concept of Tensor, which is the element generating a computational flow graph. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays(tensors) communicated between them. A Tensor object is a symbolic handle to the result of an operation, but does not actually hold the values of the operation's output. The strength of Tensor is once you build up complicated expressions as a dataflow graph, then you can offload the computation of the entire graph to a Tensorflow Session, which is able to execute the whole computation much more efficiently than executing the operations one-by-one. 
The downside of both libraries is if you want to use the symbolic gradient, They can only compute the symbolic gradient inside computation graph done in Theano/Tensorflow. So you will need to convert that part completely to Theano/Tensorflow functions. And it's hard to debug. The error message is not intuitive because they are following define-and-run method. 

We still can use the numerical method to compute gradients and expect to speed up with GPU. 

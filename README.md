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
* Automatic Differentiation in Julia [(notebook)](https://github.com/misun6312/autodiff/blob/master/Julia_autodiff.ipynb) [(code)](https://github.com/misun6312/autodiff/blob/master/julia/opt_Julia_autodiff_script.jl)

In Julia, now it produces exact same Loglikelihood values as the LL values from the bing's matlab code.  
And it can compute the gradients for each of the parameters automatically using ForwardDiff (http://www.juliadiff.org/ForwardDiff.jl/index.html). For optimizing the log likelihood function with respect to parameters, we will use Optim(https://github.com/JuliaOpt/Optim.jl). 

From the original version of the code in Julia, I updated the code for click adaptation part using inter-click interval instead of dt to consider all clicks and fixed some bugs. ** And Optimized it! There was one line in the code which required huge memory allocation.
```julia
lp = find(bin_centers .<= sbins[k])[end]
```
After fixing this line it's way faster than before! And I tried to avoid using global variables. There are some more tips to make the code run as fast as possible in [this link](http://docs.julialang.org/en/release-0.4/manual/performance-tips/?highlight=performance). 

After updating the Julia packages with Pkg.update(), ForwardDiff was updated from v0.1 to v0.2. It turns out few API changes have occured between ForwardDiff v0.1 and v0.2.  

Instead of ForwardDiff.GradientNumber and ForwardDiff.HessianNumber, we will use ForwardDiff.Dual for the covert function.
```julia
convert(::Type{Float64}, x::ForwardDiff.Dual) = Float64(x.value)
function convert(::Array{Float64}, x::Array{ForwardDiff.Dual}) 
    y = zeros(size(x)); 
    for i in 1:prod(size(x)) 
        y[i] = convert(Float64, x[i]) 
    end
    return y
end
```
ForwardDiff.Dual is now supporting *ceil, floor* function like below. 
```julia
binN = ceil(Int, B/dx) 
binBias = floor(Int, bias/dx) + binN+1  
```
To retrieve all the lower-order calculations along with the normal result of v0.2 API function, pass an instance of the appropriate ForwardDiffResult type to the in-place version of the function.
```julia
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


Setup Environment
-------
* Julia 0.4.5  
* FowardDiff v0.2 : for Automatic Differentiation
* Optim v0.5.0 : for functional optimization
* MAT v0.2 : to read .mat file

Install Julia
```bash
sudo add-apt-repository ppa:staticfloat/juliareleases
sudo add-apt-repository ppa:staticfloat/julia-deps
sudo apt-get update
sudo apt-get install julia
```
Install ForwardDiff and other available packages, simply use Julia’s package manager:
```bash
julia> Pkg.add("ForwardDiff")
julia> Pkg.add("Optim")
```

Next step
-------
* Fit the model and compare the result with bing's matlab code. 
* Tweak the model and test the Automatic Differentiation
* Upload the code to Amazon AWS and run it on the cloud :) 

Amazon AWS is a setting where you can easily load a given image on a computer that can be chosen from a range of different memory and processing characteristics (you can find a list of available instances here). 

Tips
-------
### Jupyter Notebook
The Jupyter Notebook is a web application that allows you to create and share documents that contain live code, equations, visualizations and explanatory text. 

### Profiling In Julia
Profiling allows you to measure the performance of your running code and identify lines that serve as bottlenecks.

The most useful tool for measuring performance in Julia is the @time macro.
```
@time f(x)
```
On the first call function, f gets compiled. You should not take the results of this run seriously. For the second run, note that in addition to reporting the time, it also indicated that a large amount of memory was allocated.

@profile runs your expression while taking periodic backtraces. These are appended to an internal buffer of backtraces.
```
@profile f(x)
Profile.print()
Profile.clear_malloc_data() 
```


******
### Attempt implementing in Python (Theano / Tensorflow) 

### Theano 
* Theano Numerical Differentiation(it's working) [(notebook)](https://github.com/misun6312/autodiff/blob/master/Theano_NumericDiff.ipynb)   
This follows the numerical approach to compute the derivative for each parameters from Bing's paper [(Section 3.2 the Supplementary Information)](http://science.sciencemag.org/content/suppl/2013/04/04/340.6128.95.DC1).  

* Theano Automatic Differentiation(incomplete) [(notebook)](https://github.com/misun6312/autodiff/blob/master/Theano_autodiff.ipynb)   
It can only compute gradients of 'bias' and 'lapse' automatically. For the rest of parameters, it produces nan value.
I dig into this problem for a long time but couldn't figure it out the exact reason(http://deeplearning.net/software/theano/tutorial/nan_tutorial.html).
I guess while I was converting the code with tensors there must be something wrong. Compared to implementation in Julia, I got helped a lot with convert function in Julia, which makes the code more flexible to implement. But in Theano, they don't support that kind of function. eval() and get_value() were good candidates but it was not working. 

Theano is a Python library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays both on CPU and GPU efficiently. Theano offers a good amount of flexibility, but has some limitations too. You must answer for yourself the following question: How can my algorithm be cleverly written so as to make the most of what Theano can do?   

Here's some tips that you need to make sure for using Theano
- While- or for-Loops within an expression graph are supported, but only via the theano.scan() op (which puts restrictions on how the loop body can interact with the rest of the graph).
- set_subtensor = array element assignment
- Shared Variable
- Construct Model(Graph) -> Compile
- configuration (fast_run mode, debug mode)

### Tensorflow 
* Tensorflow Automatic Differentiation (incomplete) [(notebook)](https://github.com/misun6312/autodiff/blob/master/Tensorflow_autodiff3.ipynb)  

TensorFlow is a Google-developed Python and C++ library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays both on CPU and GPU efficiently.

- Very inconvenient to use array index.. (not supporting negative index, have to use tf.slice to get the subset of array)
- Type casting is complicated and limited
- Variable
- Placeholder : a variable that we will assign the data later. It allows to create computation graph without data we then feed data into the graph through placeholders.
- Construct Model(Graph) -> Launch the Graph

There is a summary to compare some libraries in [this link](http://deeplearning4j.org/compare-dl4j-torch7-pylearn.html)

### Conclusion
Both libraries are using the concept of Tensor, which is the element generating a computational flow graph. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays(tensors) communicated between them. A Tensor object is a symbolic handle to the result of an operation, but does not actually hold the values of the operation's output. 
The strength of Tensor is once you build up complicated expressions as a dataflow graph, then you can offload the computation of the entire graph to a Tensorflow Session, which is able to execute the whole computation much more efficiently than executing the operations one-by-one. 
The downside of both libraries is if you want to use the symbolic gradient, They can only compute the symbolic gradient inside computation graph done in Theano/Tensorflow. So you will need to convert that part completely to Theano/Tensorflow functions. And it's hard to debug. The error message is not intuitive because they are following define-and-run method. 

With Theano, I implemented the code to compute gradients numerically and we can use this later to speed up with GPU. 

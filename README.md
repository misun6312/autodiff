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

In Julia, it produces the same Loglikelihood values with the bing's matlab code. 
And it can calculate the gradients automatically.

```
[LL dLL likey output] = single_trial35(param, mydata)
```

Setup
-------
Julia 0.4.5  
FowardDiff v0.2


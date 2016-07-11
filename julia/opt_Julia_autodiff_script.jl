import ForwardDiff
using ForwardDiff
import PyPlot
using PyPlot
import Base.convert
using MAT

import Optim
using Optim

# Global variables 
const epsilon = 10.0^(-10);
const dx = 0.25;
const dt = 0.02;
const total_rate = 40;

# === Upgrading from ForwardDiff v0.1 to v0.2
# instead of ForwardDiff.GradientNumber and ForwardDiff.HessianNumber, 
# we will use ForwardDiff.Dual

convert(::Type{Float64}, x::ForwardDiff.Dual) = Float64(x.value)
function convert(::Array{Float64}, x::Array{ForwardDiff.Dual}) 
    y = zeros(size(x)); 
    for i in 1:prod(size(x)) 
        y[i] = convert(Float64, x[i]) 
    end
    return y
end

## Data Import
# get trial data
function trialdata(ratdata, trial)
    if ratdata["rawdata"]["pokedR"][trial] > 0
        rat_choice = 1;  # "R"
    else
        rat_choice = -1; # "L"
    end;
    
    if typeof(ratdata["rawdata"]["rightbups"][trial]) <: Array
        rvec = vec(ratdata["rawdata"]["rightbups"][trial]);
    else
        rvec = []
    end
    if typeof(ratdata["rawdata"]["leftbups"][trial]) <: Array
        lvec = vec(ratdata["rawdata"]["leftbups"][trial]);
    else
        lvec = []
    end
    
    return rvec, lvec, 
    ratdata["rawdata"]["T"][trial], rat_choice
end

"""
function bin_centers = make_bins(B, dx, binN)

Makes a series of points that will indicate bin centers. The first and
last points will indicate sticky bins. No "bin edges" are made-- the edge
between two bins is always implicity at the halfway point between their
corresponding centers. The center bin is always at x=0; bin spacing
(except for last and first bins) is always dx; and the position
of the first and last bins is chosen so that |B| lies exactly at the
midpoint between 1st (sticky) and 2nd (first real) bins, as well as
exactly at the midpoint between last but one (last real) and last
(sticky) bins.

Playing nice with ForwardDiff means that the *number* of bins must be predetermined.
So this function will not actually set the number of bins; what it'll do is determine their
locations. To accomplish this separation, the function uses as a third parameter binN,
which should be equal to the number of bins with bin centers > 0, as follows: 
   binN = ceil(B/dx)
and then the total number of bins will be 2*binN+1, with the center one always corresponding
to position zero. Use non-differentiable types for B and dx for this to work.
"""
function make_bins(B, dx, binN)
    bins = collect(1.0:binN)*B
    bins = dx*bins/B

    if bins[end] == B
        bins[end] = B + dx
    else
        bins[end] = 2*B - bins[end-1]
    end

    bins = [-bins[end:-1:1]; 0; bins]
    return bins
end;


"""
function F = Fmatrix([sigma, lambda, c], bin_centers)

Uses globals
    dt
    dx
    epsilon       (=10.0^-10)

Returns a square Markov matrix of transition probabilities. 
Plays nice with ForwardDiff-- that is why bin_centers is a global vector (so that the rem
operations that go into defining the bins, which ForwardDiff doesn't know how to deal with,
stay outside of this differentiable function)

sigma  should be in (accumulator units) per (second^(1/2))
lambda should be in s^-1
c      should be in accumulator units per second
bin_centers should be a vector of the centers of all the bins. Edges will be at midpoints
       between the centers, and the first and last bin will be sticky.

dx is not used inside Fmatrix, because bin_centers specifies all we need to know.
dt *is* used inside Fmatrix, to convert sigma, lambda, and c into timestep units
"""
function Fmatrix(params::Vector, bin_centers)
    sigma2 = params[1];
    lam   = params[2];
    c     = params[3];
    
    sigma2_sbin = convert(Float64, sigma2)
  
    F = zeros(typeof(sigma2),length(bin_centers),length(bin_centers))    
#     F = collect(1.0:length(bin_centers))*collect(1.0:length(bin_centers))';
#     F = 0.0*sigma2*F; # Multiplying by that sigma is needed, 
#                      # for type casting reasons I do not understand...

    # added condition if lambda=0 
    if lam == 0
        mus = bin_centers*exp(lam*dt)
    else
        mus = (bin_centers + c/lam)*exp(lam*dt) - c/lam
    end

    n_sbins = max(70, ceil(10*sqrt(sigma2_sbin)/dx))
    
    swidth = 5*sqrt(sigma2_sbin)
    sbinsize = swidth/n_sbins;#sbins[2] - sbins[1]
    sbins    = collect(-swidth:sbinsize:swidth)

    ps       = exp(-sbins.^2/(2*sigma2))#exp(-sbins.^2/(2*sigma^2)) / sqrt(2*sigma^2)
    ps       = ps/sum(ps);

    base_sbins = sbins;
        
    for j in 2:length(bin_centers)
        sbins = collect(0:(length(base_sbins)-1))*sbinsize
        sbins = sbins + mus[j]-swidth

        for k in 1:length(sbins)
            if sbins[k] < bin_centers[1] #(bin_centers[1] + bin_centers[2])/2
                F[1,j] = F[1,j] + ps[k]
            elseif bin_centers[end] <= sbins[k]#(bin_centers[end]+bin_centers[end-1])/2 <= sbins[k]
                F[end,j] = F[end,j] + ps[k]
            else # more condition
                if (sbins[k] > bin_centers[1] && sbins[k] < bin_centers[2])
                    lp = 1; hp = 2;
                elseif (sbins[k] > bin_centers[end-1] && sbins[k] < bin_centers[end])
                    lp = length(bin_centers)-1; hp = length(bin_centers);
                else 
                    lp = floor(Int,((sbins[k]-bin_centers[2])/dx) + 2)#find(bin_centers .<= sbins[k])[end]#Int(floor((sbins[k]-bin_centers[2])/dx) + 1);
                    hp = lp+1#Int(ceil((sbins[k]-bin_centers[2])/dx) + 1);
                end

                if lp < 1 
                    lp = 1; 
                end
                if hp < 1 
                    hp = 1;
                end

                if lp == hp
                    F[lp,j] = F[lp,j] + ps[k]
                else
                    F[hp,j] = F[hp,j] + ps[k]*(sbins[k] - bin_centers[lp])/(bin_centers[hp] - bin_centers[lp])
                    F[lp,j] = F[lp,j] + ps[k]*(bin_centers[hp] - sbins[k])/(bin_centers[hp] - bin_centers[lp])
                end                   
            end
        end
    end
    F[:,1] = 0; F[:,end] = 0; F[1,1] = 1; F[end,end] = 1;
    return F
end

"""
version with inter-click interval(ici) for c_eff_net / c_eff_tot (followed the matlab code) 
(which was using dt for c_eff)

function logProbRight(params::Vector)

    RightClickTimes   vector with elements indicating times of right clicks
    LeftClickTimes    vector with elements indicating times of left clicks
    Nsteps number of timesteps to simulate

Takes params
    sigma_a = params[1]; sigma_s = params[2]; sigma_i = params[3]; 
    lambda = params[4]; B = params[5]; bias = params[6]; 
    phi = params[7]; tau_phi = params[8]; lapse = params[9]

Returns the log of the probability that the agent chose Right. 
"""

function logProbRight(params::Vector, RightClickTimes::Vector, LeftClickTimes::Vector, Nsteps::Int)
    sigma_a = params[1]; sigma_s = params[2]; sigma_i = params[3]; 
    lambda = params[4]; B = params[5]; bias = params[6]; 
    phi = params[7]; tau_phi = params[8]; lapse = params[9]
        
    LeftClicks  = zeros(1, Nsteps); if isempty(RightClickTimes) RightClickTimes = zeros(0) end;
    RightClicks = zeros(1, Nsteps); if isempty(LeftClickTimes ) LeftClickTimes  = zeros(0) end;
    for i in ceil((LeftClickTimes+epsilon)/dt)  LeftClicks[Int(i)]  = LeftClicks[Int(i)] + 1 end
    for i in ceil((RightClickTimes+epsilon)/dt) RightClicks[Int(i)] = RightClicks[Int(i)] + 1 end
    
    # === Upgrading from ForwardDiff v0.1 to v0.2
    # instead of using convert we can use floor(Int, ForwardDiff.Dual) and
    # ceil(Int, ForwardDiff.Dual)

#     my_B = convert(Float64, B) # my_B won't be differentiated; ForwardDiff can't do ceil()
#     my_bias = convert(Float64, bias)  # my_bias won't be differentiated' FD can't do floor()
    binN = ceil(Int, B/dx)#Int(ceil(my_B/dx))  
    binBias = floor(Int, bias/dx) + binN+1  
    bin_centers = make_bins(B, dx, binN) 

    a_trace = zeros(length(bin_centers), Nsteps+1); 
    c_trace = zeros(1, Nsteps+1)
    
    a0 = zeros(length(bin_centers),1)*sigma_a*0.0; # That weirdo inexact error thing
    a0[binN+1] = 1-lapse; a0[1] = lapse/2; a0[end] = lapse/2;
    
    c_eff_r = 0
    c_eff_l = 0
    cnt_r = 0
    cnt_l = 0
    
    Fi = Fmatrix([sigma_i, 0, 0.0], bin_centers); 
    a = Fi*a0;
    a_trace[:,1] = a;

    F0 = Fmatrix([sigma_a*dt, lambda, 0.0], bin_centers)
    for i in 2:Nsteps 
        c_eff_tot = 0
        c_eff_net = 0

        tmp_r = RightClicks[i-1]
        tmp_l = LeftClicks[i-1]
        if tmp_r+tmp_l == 0.
            c_eff_tot = 0
            c_eff_net = 0

            a = F0*a
        else
            for j in 1:RightClicks[i-1]
                if cnt_r != 0 || j != 1
                    ici = RightClickTimes[cnt_r+j]-RightClickTimes[cnt_r+j-1]
                    c_eff_r = 1 + (c_eff_r*phi - 1)*exp(-ici/tau_phi)
                    c_eff_tot = c_eff_tot + c_eff_r
                    c_eff_net = c_eff_net + c_eff_r
                end
                if j == RightClicks[i-1]
                    cnt_r = cnt_r+j
                end
            end
            for j in 1:LeftClicks[i-1]
                if cnt_l != 0 || j != 1
                    ici = LeftClickTimes[cnt_l+j]-LeftClickTimes[cnt_l+j-1]
                    c_eff_l = 1 + (c_eff_l*phi - 1)*exp(-ici/tau_phi)
                    c_eff_tot = c_eff_tot + c_eff_l
                    c_eff_net = c_eff_net - c_eff_l
                end
                if j == LeftClicks[i-1]
                    cnt_l = cnt_l+j
                end
            end
            net_sigma = sigma_a*dt + (sigma_s*c_eff_tot)/total_rate
            F = Fmatrix(collect([net_sigma; lambda; c_eff_net/dt]), bin_centers)
            a = F*a
        end
        
        c_trace[i]   = convert(Float64, c_eff_tot)
        a_trace[:,i] = convert(Array{Float64}, a)
    end;
#     plot(1:Nsteps+1,c_trace[:])    
#     imshow(a_trace, interpolation="none")
    pright = sum(a[binBias+2:end]) + 
    a[binBias]*((bin_centers[binBias+1] - bias)/dx/2) +
    a[binBias+1]*(0.5 + (bin_centers[binBias+1] - bias)/dx/2)
    
    return log(pright)
end



function logLike(params::Vector, RightClickTimes::Vector, LeftClickTimes::Vector, Nsteps::Int, rat_choice::Int)
    if rat_choice > 0
        # println("Right")
        return logProbRight(params, RightClickTimes, LeftClickTimes, Nsteps)
    elseif rat_choice < 0
        # println("Left")
        return log(1 - exp(logProbRight(params, RightClickTimes, LeftClickTimes, Nsteps)))
    else
        error("Rat did what?? It was neither R nor L")
    end
end

""" 
function (LL, LLgrad, LLhessian, bin_centers, bin_times, a_trace) = 
    llikey(params, rat_choice, maxT=1, RightPulseTimes=[], LeftPulseTimes=[], dx=0.25, dt=0.02)

Computes the log likelihood according to Bing's model, and returns log likelihood, gradient, and hessian

params is a vector whose elements, in order, are
    sigma_a    square root of accumulator variance per unit time sqrt(click units^2 per second)
    sigma_s    standard deviation introduced with each click (will get scaled by click adaptation)
    sigma_i    square root of initial accumulator variance sqrt(click units^2)
    lambda     1/accumulator time constant (sec^-1). Positive means unstable, neg means stable
    B          sticky bound height (click units)
    bias       where the decision boundary lies (click units)
    phi        click adaptation/facilitation multiplication parameter
    tau_phi    time constant for recovery from click adaptation (sec)
    lapse      2*lapse fraction of trials are decided randomly

rat_choice     should be either "R" or "L"


RETURNS:


"""

function single_trial(params::Vector, RightClickTimes::Vector, LeftClickTimes::Vector, Nsteps::Int, rat_choice::Int)
    function llikey(params::Vector)
        logLike(params, RightClickTimes, LeftClickTimes, Nsteps, rat_choice)
    end

    result =  GradientResult(params)
    
    ForwardDiff.gradient!(result, llikey, params);
    
    LL     = ForwardDiff.value(result)
    LLgrad = ForwardDiff.gradient(result)
   
    return LL, LLgrad
end

function SumLikey_LL(params::Vector, ratdata, ntrials::Int)
    LL        = 0
        
    for i in 1:ntrials
        RightClickTimes, LeftClickTimes, maxT, rat_choice = trialdata(ratdata, i)
        Nsteps = Int(ceil(maxT/dt))

        LLi = logLike(params, RightClickTimes, LeftClickTimes, Nsteps, rat_choice)
        LL        = LL + LLi;
    end
    
    LL = -LL
    return LL
end

function SumLikey(params::Vector, ratdata, ntrials::Int)
    LL        = float(0)
    LLgrad    = zeros(size(params))
    
    for i in 1:ntrials
        if rem(i,1000)==0
            println("     sum_ll_all_trials: running trial ", i, "/", ntrials);
        end

        RightClickTimes, LeftClickTimes, maxT, rat_choice = trialdata(ratdata, i)
        Nsteps = Int(ceil(maxT/dt))

        LLi, LLgradi = single_trial(params, RightClickTimes, LeftClickTimes, Nsteps, rat_choice)
        LL        = LL + LLi;
        LLgrad    = LLgrad + LLgradi;
        
    end

    LL = -LL
    LLgrad = -LLgrad
    return LL, LLgrad
end

function main()

    ratname = "B069"
    # data import
    ratdata = matread(*("chrono_",ratname,"_rawdata.mat"))
    println("rawdata of ", ratname, " imported" )

    # number of trials
    ntrials = Int(ratdata["total_trials"])

    # Parameters
    sigma_a = 1.; sigma_s = 0.1; sigma_i = 0.2; 
    sigma_a_sbin = sigma_a  # remember we need this copy for Fmatrix
    lam = -0.0005; B = 6.1; bias = 0.1; 
    phi = 0.3; tau_phi = 0.1; lapse = 0.05*2;
    params = [sigma_a, sigma_s, sigma_i, lam, B, bias, phi, tau_phi, lapse]

    l = [0., 0., 0., -5., 5., -5., 0.01, 0.005, 0.]
    u = [200., 200., 30., 5., 25., 5., 1.2, 0.7, 1.]

    # @code_warntype SumLikey(params, ratdata, ntrials)

    function LL_f(params::Vector)
        return SumLikey_LL(params, ratdata, ntrials)
    end

    function LL_g!(params::Vector, grads::Vector)
        LL, LLgrad = SumLikey(params, ratdata, ntrials)
        for i=1:length(params)
            grads[i] = LLgrad[i]
        end
    end

    function LL_fg!(params::Vector, grads)
        LL, LLgrad = SumLikey(params, ratdata, ntrials)
        for i=1:length(params)
            grads[i] = LLgrad[i]
        end
        return LL
    end

    d4 = DifferentiableFunction(LL_f,
                                LL_g!,
                                LL_fg!)

    res = optimize(d4, params, l, u, Fminbox(); 
             optimizer = GradientDescent, optimizer_o = OptimizationOptions(g_tol = 1e-12,
                                                                            iterations = 10,
                                                                            store_trace = true,
                                                                            show_trace = true))

    # @profile res = optimize(d4, params, l, u, Fminbox(); 
    #          optimizer = GradientDescent, optimizer_o = OptimizationOptions(g_tol = 1e-12,
    #                                                                         iterations = 1,
    #                                                                         store_trace = true,
    #                                                                         show_trace = true))

    # Profile.print()
    # Profile.clear_malloc_data() 


end

# @code_warntype main()
main()

# using ProfileView

# ProfileView.view()
include("AutodiffModule.jl")
import AutodiffModule
using MAT
using Optim

function main()

    ratname = "B069"
    # data import
    ratdata = matread(*("chrono_",ratname,"_rawdata.mat"))
    println("rawdata of ", ratname, " imported" )

    # number of trials
    ntrials = 10#Int(ratdata["total_trials"])

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
        return AutodiffModule.SumLikey_LL(params, ratdata, ntrials)
    end

    function LL_g!(params::Vector, grads::Vector)
        LL, LLgrad = AutodiffModule.SumLikey(params, ratdata, ntrials)
        for i=1:length(params)
            grads[i] = LLgrad[i]
        end
    end

    function LL_fg!(params::Vector, grads)
        LL, LLgrad = AutodiffModule.SumLikey(params, ratdata, ntrials)
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
@time main()
# Profile.print()
# Profile.clear_malloc_data() 


# using ProfileView
# ProfileView.view()
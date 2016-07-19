include("AutodiffModule.jl")
import AutodiffModule
using MAT
using Optim

function main()

    ratname = readline(STDIN)#<- $echo $ratname | julia t3.jl  #"B069"
    # while !eof(STDIN)
    #     x = read(STDIN, Char)
    #     println("Found: $x")
    # end

    ratname = ratname[1:end-1]    
    # data import
    mpath = "/mnt/bucket/people/amyoon/Data/PBupsModel_rawdata/"
    ratdata = matread(*(mpath,"chrono_",ratname,"_rawdata.mat"))

    println("rawdata of ", ratname, " imported" )

    saveto_filename = *("julia_out_",ratname,".mat")

    # number of trials
    ntrials = Int(ratdata["total_trials"])

    # Parameters
    sigma_a = rand()*4.; sigma_s = rand()*4.; sigma_i = rand()*30.; 
    lam = randn(); B = rand()*20.+5.; bias = randn(); 
    phi = rand()*1.19+0.01; tau_phi = 0.695*rand()+0.005; lapse = rand();

    # sigma_a = 1.; sigma_s = 0.1; sigma_i = 0.2; 
    # lam = -0.0005; B = 6.1; bias = 0.1; 
    # phi = 0.3; tau_phi = 0.1; lapse = 0.05*2;
    params = [sigma_a, sigma_s, sigma_i, lam, B, bias, phi, tau_phi, lapse]

    l = [0.,   0.,   0., -5., 5., -5., 0.01, 0.005, 0.]
    u = [200., 200., 30., 5., 25., 5., 1.2,  0.7,   1.]

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

    tic()
    history = optimize(d4, params, l, u, Fminbox(); 
             optimizer = GradientDescent, optimizer_o = OptimizationOptions(g_tol = 1e-12,
                                                                            x_tol = 1e-10,
                                                                            f_tol = 1e-6,
                                                                            iterations = 200,
                                                                            store_trace = true,
                                                                            ))
    fit_time = toc()
    println(history.minimum)
    println(history)

    ## do a single functional evaluation at best fit parameters and save likely for each trial
    likely_all = zeros(typeof(sigma_i),ntrials)
    x_bf = history.minimum
    Likely_all_trials(likely_all, x_bf, ratdata, ntrials)

    matwrite(saveto_filename, Dict([("ratname",ratname),
                                    ("x_init",params),
                                    ("trials",ntrials),
                                    ("history",history),
                                    ("f",history.f_minimum), 
                                    ("x_converged",history.x_converged),
                                    ("f_converged",history.f_converged),
                                    ("g_converged",history.g_converged),                                    
                                    ("fit_time",fit_time),
                                    ("x_bf",history.minimum),
                                    ("myfval", history.f_minimum),
                                    ("likely",likely_all)
                                    ]))
     # hessian?

end

# @code_warntype main()
@time main()
# Profile.print()
# Profile.clear_malloc_data() 


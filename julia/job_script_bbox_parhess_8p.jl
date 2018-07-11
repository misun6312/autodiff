using MAT

n_core = 8
if nworkers() < n_core
    addprocs(n_core-nworkers(); exeflags="--check-bounds=yes")
end
@assert nprocs() > n_core
@assert nworkers() >= n_core

println(workers())

@everywhere include("bbox_hessian_keyword_minimization.jl")
@everywhere using PBupsModel
@everywhere using GeneralUtils
@everywhere using JLD

@everywhere const dt = 0.02


function main()

    ratname = readline(STDIN)#<- $echo $ratname | julia t3.jl  #"B069"
    ratname = ratname[1:end]    

    # data import
    mpath = "/scratch/gpfs/koay/Data/musc_inact/"
    fname = *("chrono_",ratname,"_rawdata.mat")

	ratdata, ntrials = LoadData(mpath, fname)

    println("rawdata of ", ratname, " imported" )

    saveto_filename = *("parhess_julia_out_",ratname,"_rseed_8p_bbox.mat")

	# using TrialData load trial data
	RightClickTimes, LeftClickTimes, maxT, rat_choice = TrialData(ratdata["rawdata"], 1)

	Nsteps = Int(ceil(maxT/dt))


    # Random Parameters
    sigma_a = rand()*100.; sigma_s_R = rand()*100.; #sigma_s_L = rand()*100.; sigma_i = rand()*3.; 
    lam = randn(); B = rand()*20.+5.; bias = randn(); 
    phi = rand()*0.9+0.1; tau_phi = 0.68*rand()+0.02; lapse_R = rand(); #lapse_L = rand();
  #  input_gain_weight = rand();

    params = [sigma_a, sigma_s_R, lam, B, bias, phi, tau_phi, lapse_R]   


	# args_11p = ["sigma_a","sigma_s_R","sigma_s_L","lambda","B","bias","phi","tau_phi","lapse_R","lapse_L","input_gain_weight"]
	args_8p = ["sigma_a","sigma_s_R","lambda","B","bias","phi","tau_phi","lapse_R"]
	# params = [106.2984,   79.4005,    4.8972,   -3.4972,    8.5650,   1.0831,    0.2,    0.0679,    0.00001,    0.1344,    0.3615]

	bbox = Dict(:sigma_a=>[0, 200], :sigma_s_R=>[0, 200], # :sigma_s_L=>[0, 200], 
		:lambda=>[-5, 5], :B=>[5, 25], :bias=>[-5, 5], 
		:phi=>[0, 1.2], :tau_phi=>[0.001, 1.5], 
		:lapse_R=>[0, 1])

	LLs = SharedArray(Float64, ntrials)

	func = (;pars...) -> ComputeLL_bbox(LLs, ratdata["rawdata"], ntrials; pars...)

    tic()
    println("=== start fitting ===")


	opars, traj, cost, cpm_traj, ftraj, H = bbox_Hessian_keyword_minimization(params, args_8p, bbox, func,
	verbose=true, verbose_level=1, softbox=true, start_eta=1)

    fit_time = toc()
    println(opars)
    println(H)

    ## do a single functional evaluation at best fit parameters and save likely for each trial
    likely_all = zeros(typeof(sigma_a),ntrials)
    x_bf = opars

    TrialsLikelihood(likely_all, ratdata["rawdata"], ntrials
    	;make_dict(args_8p, x_bf)...)

    matwrite(saveto_filename, Dict([("ratname",ratname),
                                    ("x_init",params),
                                    ("trials",ntrials),
                                    ("f",cost), 
                                    ("fit_time",fit_time),
                                    ("grad_trace",cpm_traj),
                                    ("f_trace",ftraj),
                                    ("x_trace",traj),
                                    ("x_bf",x_bf),
                                    ("myfval", cost),
                                    ("hessian", H),
                                    ("likely",likely_all)
                                    ]))
end

# @code_warntype main()
@time main()
# Profile.print()
# Profile.clear_malloc_data() 
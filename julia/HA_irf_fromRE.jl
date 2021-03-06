include("HA_stationary.jl")
include("HA_TE.jl")



## Define the function that return data names saved as the code runs
function strs_data()
    idio_strs = ["a", "c", "n", "en", "nu", "nu_bar", "nu_bar_c", "psi", "s"]
    aggr_strs = ["r", "R_cov", "theta", "w", "x", "y", "I"]
    return idio_strs, aggr_strs
end



## Define the function that write as the code runs
function write_data(a, c, n, ν, ν̄, ν̄c, ψ, r, R, s, θ, w, x, y, en, I, t, path)
    idio_strs, aggr_strs = strs_data()
    idios = [a, c, n, en, ν, ν̄, ν̄c, ψ, s]
    aggrs = [r, R, θ, w, x, y, I]
    ## Save all the idiosyncratic data and their means
    for (i, idio) in enumerate(idios)
        str = idio_strs[i]
        writedlm("../data/$(path)/$(str)/$t.csv", idio, ',')
        writedlm("../data/$(path)/mean_$(str)/$t.csv", mean(idio, 1), ',')
    end
    ## Save all the aggreagate data
    for (i, aggr) in enumerate(aggrs)
        str = aggr_strs[i]
        writedlm("../data/$(path)/$(str)/$t.csv", aggr, ',')
    end
end



## Define the function that combine all of the data
function combine_data(T, path)
    idio_strs, aggr_strs = strs_data()
    mean_idio_strs = ["mean_$(idio_strs[i])" for i in 1:length(idio_strs)]
    for str in vcat(mean_idio_strs, aggr_strs)
        data = zeros(T, length(readdlm("../data/$(path)/$(str)/1.csv", ',')))
        for t in 1:T
            data[t, :] = readdlm("../data/$(path)/$(str)/$(t).csv", ',')
        end
        writedlm("../data/$(path)/$(str)/combined.csv", data, ',')
        for t in 1:T
            rm("../data/$(path)/$(str)/$(t).csv")
        end
    end
end



####################################################################################################
## Define a function that initialize the data of interest
## Initialze distributions for asset a, productivity s and beleifs ψ
## based on draws from the stationary distribution
## This initialization assumes homogeneity in beliefs - all initialized as ψ̄
## This function is used by simul_learning
function init_as(agent_num, bin_midpts, draws)
    ai_1 = zeros(agent_num)
    si_1 = zeros(Int64, agent_num)
    for (i, draw) in enumerate(draws)
        ai_1[i] = bin_midpts[draw[1]]
        si_1[i] = draw[2]
    end
    return ai_1, si_1
end



## Define a function that initialize the data of interest
function init_data_learning(agent_num, bin_midpts, draws, ψ_init, R̄, ā, θ_t)
    r, w, y = zeros(3)
    c, n, ν, ν̄, ν̄c = [zeros(agent_num) for i in 1:5]
    a, s = init_as(agent_num, bin_midpts, draws)
    ψ = ψ_init
    s′ = zeros(Int64, agent_num)
    a′ = zeros(agent_num)
    ψ′ = zeros(agent_num, 3)
    R = zeros(3, 3)
    R′ = zeros(3, 3)
    R = R̄
    R′ = R̄
    θ, θ′ = θ_t[1:2]
    x_ = [1.; log(mean(a) / ā); log(θ)]
    x = x_
    x′ = ones(3)
    return c, n, ν, ν̄, ν̄c, r, w, s, s′, a, a′, ψ, ψ′, R, R′, θ, θ′, x_, x, x′, y
end


## Simulate the economy from aggregate productivity shock vector θ_t
## Boolean het controls the heterogeneity of beliefs initialzed
## Return variables of interest: averge (consumptions, labors, assets, marginal utilitys, beliefs),
## and interest rates and wages.
function simul_irf(para, θ_t, prepost, π, path)
    @unpack N, a_min, a_max, agent_num, T, ā, ρ, σ_ϵ, γ_gain, ψ_init, R̄, lear_path, A, α, iid, iid_str, amin_str, δ = para
    T = length(θ_t)
    ## Initialize functions
    cf = get_cf(para)
    bin_midpts = get_bins(a_min, a_max,  N)
    draws = dimtrans1to2.(N, rand(DiscreteRV(π), agent_num))
    ## Initialize data of interest
    ψ_init = readdlm("../data/HA/yearly/$(iid_str)_$amin_str/rational/psi.csv")' .* ones(agent_num)
    c, n, ν, ν̄, ν̄c, r, w, s, s′, a, a′, ψ, ψ′, R, R′, θ, θ′, x_, x, x′, y = init_data_learning(agent_num, bin_midpts, draws, ψ_init, R̄, ā, θ_t)
    set_ϕ_range!(para, ψ, x)
    #= Loop through t = 1..T, each variable is index comtemporanously.
    Here the vector ψ[:,t] = {ψ_0,ψ_1,ψ_2,...} R[:,:,t] = {R_0,R_1,R_2,...}
    The belief subscript indicates the time when the belief is formed.
    The belief is being used the period after it is formed.
    For example ψ_1 is the belief formed at end time 1 but used at the beginning of time 2 =#
    for t in 1:T
        println(t)
        c, n, a′, ν, ν̄, ν̄c, r, w = TE(para, a, s, ψ, x, cf)
        en = A[s] .* n
        I = mean(a′) - (1 - δ) * mean(a)
        y = θ * mean(a) ^ α * mean(en) ^ (1 - α)
        write_data(a, c, n, ν, ν̄, ν̄c, ψ, r, R, s, θ, w, x, y, en, I, t, path)
        if t == T break end
        s′ = update_s(para, s, t)
        θ′ = θ_t[t + 1]
        x′ = [1; log(mean(a′) / ā); log(θ′)]
        R′ = update_R(R, x, t - 1, γ_gain)
        if t == 1
            ψ′ = ψ
        else
            ## ψi_t[:, :, t + 1] corresponds to the subscrpt of ψ_t which is used at time t + 1.
            ## This is the the belief formed at time t (used at time t + 1).
            ## At time t, the agent only knows about information "x" up to time t
            ## Since this is a forcasting model,
            ## The right hand side is the data of "x" up to time t - 1
            ## The left hand side is "log(ν)" up to time t
            ψ′ = update_ψ(ψ, R, x_, ν, t, γ_gain, agent_num, ν̄c)
            x_ = x
        end
        a, s, θ, x, R, ψ = a′, s′, θ′, x′, R′, ψ′
        set_ϕ_range!(para, ψ, x)
    end
end



## Simulate the impulse response functions from shock to θ of one standard devation
## Period of the simulation is Sim_T, the shock enters at t = shock_enter
## The shock decays with rate of para.ρ
function IRFs(para, Sim_T, π)
    @unpack pos, σ_ϵ, ρ, irf_path, yearly_str, iid_str, amin_str, seed = para
    θ_t′, θ_t = [ones(Sim_T) for _ in 1:2]
    shock_sign = (1. - !pos * 2.)
    θ_t′ = exp.([shock_sign * σ_ϵ * ρ ^ (t - 1) for t in 1:Sim_T])
    ## Case with initialization of heterogeneous beliefs
    for (i, prepost) in enumerate(["post", "pre"])
        path = "HA/$yearly_str/$(iid_str)_$(amin_str)/learning/seed$seed/IRFs/from_RE/$(prepost)_shock"
        @time simul_irf(para, [θ_t′, θ_t][i], prepost, π, path)
        combine_data(Sim_T, path)
    end
end


## get irf from the saved data
function get_irf(para, Sim_T)
    @unpack irf_path, yearly_str, iid_str, amin_str, seed = para
    vars = ["r", "w", "mean_a", "y", "I", "mean_n", "mean_en", "theta", "mean_c"]
    for var in vars
        var_pre = readdlm("../data/HA/$yearly_str/$(iid_str)_$(amin_str)/learning/seed$seed/IRFs/from_RE/pre_shock/$var/combined.csv", ',')
        var_post = readdlm("../data/HA/$yearly_str/$(iid_str)_$(amin_str)/learning/seed$seed/IRFs/from_RE/post_shock/$var/combined.csv", ',')
        if var == "r"
            var_irf = 100(var_post .- var_pre)
        else
            var_irf = 100(var_post .- var_pre) ./ var_pre
        end
        writedlm("../data/HA/$yearly_str/$(iid_str)_$(amin_str)/learning/seed$seed/IRFs/from_RE/irf/$var.csv", var_irf, ',')
    end

    for i in 1:3
        psi_pre_i = readdlm("../data/HA/$yearly_str/$(iid_str)_$(amin_str)/learning/seed$seed/IRFs/from_RE/pre_shock/mean_psi/combined.csv", ',')[:, i]
        psi_post_i = readdlm("../data/HA/$yearly_str/$(iid_str)_$(amin_str)/learning/seed$seed/IRFs/from_RE/post_shock/mean_psi/combined.csv", ',')[:, i]
        psi_irf = psi_post_i .- psi_pre_i
        writedlm("../data/HA/$yearly_str/$(iid_str)_$(amin_str)/learning/seed$seed/IRFs/from_RE/irf/mean_psi$i.csv", psi_irf, ',')
    end
end



## Plot irf with confidence intervels
function plot_irf(para, Sim_T)
    @unpack irf_path, yearly_str, iid_str, amin_str, seed, gain_str = para
    vars = ["r", "w", "mean_a", "y", "I", "mean_n", "mean_en"]
    for var in vars
        p = plot(grid = false, title = "$var")
        data_RE = -1. * readdlm("../data/HA/$yearly_str/$(iid_str)_$(amin_str)/rational/$var.csv")
        data_learning = readdlm("../data/HA/$yearly_str/$(iid_str)_$(amin_str)/learning/seed$seed/IRFs/from_RE/irf/$var.csv")
        plot!(p, data_RE, label = "RE")
        plot!(p, data_learning, label = "learning")
        savefig(p, "../figures/HA/$yearly_str/$(iid_str)_$(amin_str)/learning/seed$seed/IRFs/fromRE/$gain_str/$var.pdf")
    end
end



####################################################################################################
#                           IRF with confidence intervals
####################################################################################################
s = ArgParseSettings()
@add_arg_table s begin
    "i"
        arg_type = Int
        required = true
        help = "indx going from 0 to 1"
end
ps = parse_args(s)
indx = ps["i"]
iid = Bool(indx)
Sim_T = 150
para = HAmodel(iid = iid, pos = false)
para, π, k, ϵn_grid, n_grid, a_grid, c_grid = calibrate_stationary!(para)
IRFs(para, Sim_T, π)
get_irf(para, Sim_T)
plot_irf(para, Sim_T)

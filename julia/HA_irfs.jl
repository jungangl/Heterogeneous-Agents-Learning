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
function init_data_irf(agent_num, R, ā, a, θ_t)
    r, w = zeros(2)
    c, n, ν, ν̄, ν̄c, I, en = [zeros(agent_num) for i in 1:7]
    s′ = zeros(Int64, agent_num)
    a′ = zeros(agent_num)
    ψ′ = zeros(agent_num, 3)
    R′ = R
    θ, θ′  = θ_t[1:2]
    x_ = [1.; log(mean(a) / ā); log(θ)]
    x = x_
    x′ = ones(3)
    return c, n, ν, ν̄, ν̄c, r, w, s′, a′, ψ′, R, R′,θ, θ′, x_, x, x′, I, en
end



## Simulate the economy from aggregate productivity shock vector θ_t
## Boolean het controls the heterogeneity of beliefs initialzed
## Return variables of interest: averge (consumptions, labors, assets, marginal utilitys, beliefs),
## and interest rates and wages.
function simul_irf(para, θ_t, t_sample, prepost)
    @unpack α, A, δ, N, a_min, a_max, agent_num, ā, ρ, σ_ϵ, γ_gain, ψ_init, irf_path, yearly_str, gain_str, iid_str, amin_str, seed = para
    T = length(θ_t)
    ## Initialize functions
    cf = get_cf(para)
    data_str = "../data/HA/$yearly_str/$(iid_str)_$amin_str/learning/seed$seed/simulations/from_rational_RA/$(gain_str)"
    a = convert(Array{Float64, 2}, readdlm("$data_str/a/$t_sample.csv", ','))
    s = convert(Array{Int64, 2}, readdlm("$data_str/s/$t_sample.csv", ','))
    ψ = convert(Array{Float64, 2}, readdlm("$data_str/psi/$t_sample.csv", ','))
    R = reshape(convert(Array{Float64, 2}, readdlm("$data_str/R_cov/combined.csv", ','))[t_sample, :], (3, 3))
    c, n, ν, ν̄, ν̄c, r, w, s′, a′, ψ′, R, R′,θ, θ′, x_, x, x′, I, en = init_data_irf(agent_num, R, ā, a, θ_t)
    set_ϕ_range!(para, ψ, x)
    #=Loop through t = 1..T, each variable is index comtemporanously.
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
        path = "$irf_path/$t_sample/$(prepost)_shock"
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
function IRFs(para, t_sample, Sim_T)
    @unpack pos, σ_ϵ, ρ, irf_path = para
    θ_t′, θ_t = [ones(Sim_T) for _ in 1:2]
    shock_sign = (1. - !pos * 2.)
    θ_t′ = exp.([shock_sign * σ_ϵ * ρ ^ (t - 1) for t in 1:Sim_T])
    ## Case with initialization of heterogeneous beliefs
    for (i, prepost) in enumerate(["post", "pre"])
        path = "$irf_path/$t_sample/$(prepost)_shock"
        @time simul_irf(para, [θ_t′, θ_t][i], t_sample, prepost)
        combine_data(Sim_T, path)
    end
end


## get irf from the saved data
function get_irf(para, t_sample, Sim_T)
    @unpack irf_path = para
    vars = ["r", "w", "mean_a", "y", "I", "mean_n", "mean_en", "theta", "mean_c"]
    for var in vars
        var_pre = readdlm("../data/$irf_path/$t_sample/pre_shock/$var/combined.csv", ',')
        var_post = readdlm("../data/$irf_path/$t_sample/post_shock/$var/combined.csv", ',')
        if var == "r"
            var_irf = 100(var_post .- var_pre)
        else
            var_irf = 100(var_post .- var_pre) ./ var_pre
        end
        writedlm("../data/$irf_path/$t_sample/irf/$var.csv", var_irf, ',')
    end

    for i in 1:3
        psi_pre_i = readdlm("../data/$irf_path/$t_sample/pre_shock/mean_psi/combined.csv", ',')[:, i]
        psi_post_i = readdlm("../data/$irf_path/$t_sample/post_shock/mean_psi/combined.csv", ',')[:, i]
        psi_irf = psi_post_i .- psi_pre_i
        writedlm("../data/$irf_path/$t_sample/irf/mean_psi$i.csv", psi_irf, ',')
    end
end



## combine irfs
function combine_irfs(para, Sim_T, samples)
    @unpack irf_path = para
    vars = ["r", "w", "mean_a", "y", "I", "mean_n", "mean_en", "theta", "mean_c", "mean_psi1", "mean_psi2", "mean_psi3"]
    data = zeros(Sim_T, length(samples))
    for var in vars
        println("var = $var")
        for (t, t_sample) in enumerate(samples)
            data[:, t] = readdlm("../data/$irf_path/$t_sample/irf/$var.csv", ',')
        end
        writedlm("../data/$irf_path/combined/$var.csv", data, ',')
    end
end



## Plot irf with confidence intervels
function plot_irf_confidence(para, α, Sim_T)
    @unpack irf_path, yearly_str, iid_str, amin_str, pos = para
    vars = ["r", "w", "mean_a", "y", "I", "mean_n", "mean_en", "theta", "mean_c", "mean_psi1", "mean_psi2", "mean_psi3"]
    p_low = (1 - α) / 2
    p_high = 1 - p_low
    for var in vars
        data_low, data_median, data_high = [zeros(Sim_T) for i in 1:3]
        data = readdlm("../data/$irf_path/combined/$var.csv", ',')[1:Sim_T, :]
        for t in 1:Sim_T
            data_low[t] = quantile(data[t, :], p_low)
            data_median[t] = quantile(data[t, :], 0.5)
            data_high[t] = quantile(data[t, :], p_high)
        end
        p = plot(grid = false, title = "$var")
        plot!(p, data_low, ls = :dash, color = :black, label = "")
        plot!(p, data_median, color = :black, label = "")
        plot!(p, data_high, ls = :dash, color = :black, label = "")
        if var ∈ ["r", "w", "mean_a", "y", "I", "mean_n", "mean_en"]
            data_RE = (1. - 2. * !pos) * readdlm("../data/HA/$yearly_str/$(iid_str)_$(amin_str)/rational/$var.csv")
            plot!(p, data_RE, ls = :dash, color = :red, label = "RE")
        end
        savefig(p, "../figures/$irf_path/$var.pdf")
    end
end


## Plot irf with confidence intervels
function check_symmetry(para, Sim_T)
    @unpack irf_path, yearly_str, iid_str, amin_str, seed, gain_str = para
    vars = ["r", "w", "mean_a", "y", "I", "mean_n", "mean_en", "theta", "mean_c", "mean_psi1", "mean_psi2", "mean_psi3"]
    for var in vars
        p = plot(grid = false, title = "$var")
        for pos in [true; false]
            label = if pos "positive" else "negative" end
            data = (1. - 2. * !pos) * readdlm("../data/HA/$yearly_str/$(iid_str)_$(amin_str)/learning/seed$seed/IRFs/$label/$gain_str/combined/$var.csv", ',')[1:Sim_T, :]
            data_median = zeros(Sim_T)
            for t in 1:Sim_T
                data_median[t] = quantile(data[t, :], 0.5)
            end
            plot!(p, data_median, label = label)
        end
        savefig(p, "../figures/HA/$yearly_str/$(iid_str)_$(amin_str)/learning/seed$seed/IRFs/compare/$gain_str/$var.pdf")
    end
end


#=
####################################################################################################
#                           IRF with confidence intervals
####################################################################################################
s = ArgParseSettings()
@add_arg_table s begin
    "i"
        arg_type = Int
        required = true
        help = "indx going from 1 to 2000"
end
ps = parse_args(s)
indx = ps["i"]
## Initialize
srand(1)
samples = sort(sample(5001:10000, 500, replace = false))
Sim_T = 200
## Divide the indx into iid/noiid and pos/nagetive
pos = indx <= 1000
indx = indx - (indx > 1000) * 1000
iid = indx <= 500
indx = indx - (indx > 500) * 500
t_sample = samples[indx]
para = HAmodel(iid = iid, pos = pos)
para, π, k, ϵn_grid, n_grid, a_grid = calibrate_stationary!(para)
IRFs(para, t_sample, Sim_T)
get_irf(para, t_sample, Sim_T)
=#



#=
## Combine all of the irfs
srand(1)
samples = sort(sample(5001:10000, 500, replace = false))
Sim_T = 200
for iid in [true; false]
    for pos in [true; false]
        println("iid = $iid, pos = $pos")
        para = HAmodel(iid = iid, pos = pos)
        combine_irfs(para, Sim_T, samples)
    end
end
=#




## Plot all of irfs with confidence intervals, compared with RE
α = 0.95
Sim_T = 150
for iid in [true; false]
    for pos in [true; false]
        para = HAmodel(iid = iid, pos = pos)
        plot_irf_confidence(para, α, Sim_T)
    end
end
## Plot symmetry
for iid in [true; false]
    para = HAmodel(iid = iid)
    check_symmetry(para, Sim_T)
end

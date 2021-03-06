include("HA_stationary.jl")
#include("HA_TE.jl")
include("HA_TE.jl")
using ArgParse



## Define the function that return data names saved as the code runs
function strs_data()
    idio_strs = ["a", "c", "n", "en", "nu", "nu_bar", "nu_bar_c", "psi", "s"]
    aggr_strs = ["r", "theta", "w", "x", "y", "I"]
    return idio_strs, aggr_strs
end



## Define the function that write as the code runs
function write_data(a, c, n, ν, ν̄, ν̄c, ψ, r, R, s, θ, w, x, y, en, Inv, t, path)
    idio_strs, aggr_strs = strs_data()
    idios = [a, c, n, en, ν, ν̄, ν̄c, ψ, s]
    aggrs = [r, θ, w, x, y, Inv]
    ## Save all the idiosyncratic data and their means
    for (i, idio) in enumerate(idios)
        str = idio_strs[i]
        writedlm("../data/$(path)/mean_$(str)/$t.csv", mean(idio, dims = 1), ',')
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




## Define a function that initialize the data of interest
function init_data_irf(agent_num, R, ā, a, ψ, θ_t)
    r, w, y = zeros(3)
    c, n, ν, ν̄, ν̄c, Inv, en = [zeros(agent_num) for i in 1:7]
    s′ = zeros(Int64, agent_num)
    a′ = zeros(agent_num)
    ψ′ = zeros(agent_num, 3)
    R′ = R
    θ, θ′  = θ_t[1:2]
    x_ = [1.; log(mean(a) / ā); log(θ)]
    x = x_
    x′ = ones(3)
    ϕ = compute_ϕ(ψ, x)
    return c, n, ν, ν̄, ν̄c, r, w, s′, a′, ψ′, R′,θ, θ′, x_, x, x′, y, ϕ, Inv, en
end


function construct_x(agent_num, K, ā, θ, a, s, A)
    lnθ = log(θ)
    lnε = log.(A[s])
    x = ones(agent_num, 7)
    x[:, 2] .= log.(K / ā)
    x[:, 3] .= lnθ
    x[:, 4] .= log.(K / ā) .* (a .- ā)
    x[:, 5] .= lnθ .* (a .- ā)
    x[:, 6] .= log.(K / ā) .* lnε
    x[:, 7] .= lnθ .* lnε
    return x
end


function init_data_irf_expanded(agent_num, ā, a, ψ, θ_t, s, A)
    r, w, y = zeros(3)
    c, n, ν, ν̄, ν̄c, Inv, en = [zeros(agent_num) for i in 1:7]

    s′ = zeros(Int64, agent_num)
    a′ = zeros(agent_num)
    ψ′ = zeros(agent_num, 7)
    θ, θ′  = θ_t[1:2]
    x_ = construct_x(agent_num, mean(a), ā, θ, a, s, A)
    x = x_
    x′ = ones(agent_num, 7)
    ϕ = compute_ϕ(ψ, x)
    return c, n, ν, ν̄, ν̄c, r, w, s′, a′, ψ′, θ, θ′, x_, x, x′, y
end
## Simulate the economy from aggregate productivity shock vector θ_t
#  Boolean het controls the heterogeneity of beliefs initialzed
#  Return variables of interest: averge (consumptions, labors, assets, marginal utilitys, beliefs),
#  and interest rates and wages.
function simul_irf(para, θ_t, t_sample, prepost)
    @unpack α, A, δ, N, a_min, a_max, agent_num, ā, ρ, σ_ϵ, γ_gain, ψ_init, irf_path, yearly_str, gain_str, iid_str, R̄ = para
    T = length(θ_t)
    ## Initialize functions
    cf1 = get_cf(para)
    #cf2_ss = get_cf2(para, cf, r̄, w̄)
    data_str = "../data/HA/$yearly_str/$(iid_str)/learning/simulation/from_rational_RA/$(gain_str)"
    a = vec(convert(Array{Float64, 2}, readdlm("$data_str/a/$t_sample.csv", ',')))
    s = vec(convert(Array{Int64, 2}, readdlm("$data_str/s/$t_sample.csv", ',')))
    ψ = convert(Array{Float64, 2}, readdlm("$data_str/psi/$t_sample.csv", ','))
    R = convert(Array{Float64, 2}, readdlm("$data_str/R_cov/$t_sample.csv", ','))
    init_data = init_data_irf(agent_num, R, ā, a, ψ, θ_t)
    c, n, ν, ν̄, ν̄c, r, w, s′, a′, ψ′, R′, θ, θ′, x_, x, x′, y, ϕ, Inv, en = init_data
    set_ϕ_range!(para, ϕ)
    n̂_func, ĉ_func, â′_func = compute_indi_grids(cf1, para)
    #=Loop through t = 1..T, each variable is index comtemporanously.
    Here the vector ψ[:,t] = {ψ_0,ψ_1,ψ_2,...} R[:,:,t] = {R_0,R_1,R_2,...}
    The belief subscript indicates the time when the belief is formed.
    The belief is being used the period after it is formed.
    For example ψ_1 is the belief formed at end time 1 but used at the beginning of time 2 =#
    for t in 1:T
        #println(t)
        c, n, a′, ν, ν̄, ν̄c, r, w = TE(para, θ, a, s, ϕ, cf1, n̂_func, ĉ_func, â′_func)
        en = A[s] .* n
        Inv = mean(a′) - (1 - δ) * mean(a)
        y = θ * mean(a) ^ α * mean(en) ^ (1 - α)
        path = "$irf_path/$t_sample/$(prepost)_shock"
        # save the following variables for IRFs ["r", "w", "mean_a", "y", "I", "mean_n", "mean_en", "theta", "mean_c"]
        write_data(a, c, n, ν, ν̄, ν̄c, ψ, r, R, s, θ, w, x, y, en, Inv, t, path)
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
        ϕ = compute_ϕ(ψ, x)
        set_ϕ_range!(para, ϕ)
    end
end



function simul_irf_new_expanded(para, θ_t, t_sample, prepost)
    @unpack α, A, δ, N, a_min, a_max, agent_num, ā, ρ, σ_ϵ, γ_gain, ψ_init_expanded, irf_path, yearly_str, gain_str, iid_str, R̄_expanded = para
    ψ_init = ψ_init_expanded
    R̄ = R̄_expanded
    T = length(θ_t)
    ## Initialize functions
    cf1 = get_cf(para)
    #cf2_ss = get_cf2(para, cf, r̄, w̄)
    data_str = "../data/HA/$yearly_str/$(iid_str)/learning/simulation/from_rational_RA_new_expanded/$(gain_str)"
    a = vec(convert(Array{Float64, 2}, readdlm("$data_str/a/$t_sample.csv", ',')))
    s = vec(convert(Array{Int64, 2}, readdlm("$data_str/s/$t_sample.csv", ',')))
    ψ = convert(Array{Float64, 2}, readdlm("$data_str/psi/$t_sample.csv", ','))
    R = zeros(agent_num, 7, 7)
    R_cov = vec(convert(Array{Float64, 2}, readdlm("$data_str/R_cov/$t_sample.csv", ',')))
    for foo in 1:49
        pos2, pos3 = 0, 0
        if mod(foo, 7) != 0
            pos2, pos3 = mod(foo, 7), (foo ÷ 7) + 1
        else
            pos2, pos3 = 7, (foo ÷ 7)
        end
        #println(pos2, pos3)
        R[:, pos2, pos3] = R_cov[((foo - 1) * agent_num + 1):(foo * agent_num)]
    end
    R′ = R
    init_data = init_data_irf_expanded(agent_num, ā, a, ψ, θ_t, s, A)
    c, n, ν, ν̄, ν̄c, r, w, s′, a′, ψ′, θ, θ′, x_, x, x′, y = init_data
    ϕ = compute_ϕ(ψ, x)
    set_ϕ_range!(para, ϕ)
    n̂_func, ĉ_func, â′_func = compute_indi_grids(cf1, para)
    #=Loop through t = 1..T, each variable is index comtemporanously.
    Here the vector ψ[:,t] = {ψ_0,ψ_1,ψ_2,...} R[:,:,t] = {R_0,R_1,R_2,...}
    The belief subscript indicates the time when the belief is formed.
    The belief is being used the period after it is formed.
    For example ψ_1 is the belief formed at end time 1 but used at the beginning of time 2 =#
    for t in 1:T
        #println(t)
        c, n, a′, ν, ν̄, ν̄c, r, w = TE(para, θ, a, s, ϕ, cf1, n̂_func, ĉ_func, â′_func)
        en = A[s] .* n
        Inv = mean(a′) - (1 - δ) * mean(a)
        y = θ * mean(a) ^ α * mean(en) ^ (1 - α)
        path = "$irf_path/$t_sample/$(prepost)_shock"
        # save the following variables for IRFs ["r", "w", "mean_a", "y", "I", "mean_n", "mean_en", "theta", "mean_c"]
        write_data(a, c, n, ν, ν̄, ν̄c, ψ, r, R, s, θ, w, x, y, en, Inv, t, path)
        if t == T break end
        s′ = update_s(para, s, t)
        θ′ = θ_t[t + 1]
        x′ = construct_x(agent_num, mean(a′), ā, θ′, a′, s′, A)
        R′ = update_R_expanded(agent_num, R, x, t - 1, γ_gain)
        if t == 1
            ψ′ = ψ
        else
            ## ψi_t[:, :, t + 1] corresponds to the subscrpt of ψ_t which is used at time t + 1.
            ## This is the the belief formed at time t (used at time t + 1).
            ## At time t, the agent only knows about information "x" up to time t
            ## Since this is a forcasting model,
            ## The right hand side is the data of "x" up to time t - 1
            ## The left hand side is "log(ν)" up to time t
            ψ′ = update_ψ_expanded(ψ, R, x_, ν, t, γ_gain, agent_num, ν̄c)
            x_ = x
        end
        a, s, θ, x, R, ψ = a′, s′, θ′, x′, R′, ψ′
        ϕ = compute_ϕ(ψ, x)
        set_ϕ_range!(para, ϕ)
    end
end



## Simulate the impulse response functions from shock to θ of one standard devation
#  Period of the simulation is Sim_T, the shock enters at t = shock_enter
#  The shock decays with rate of para.ρ
function IRFs(para, t_sample, Sim_T)
    @unpack σ_ϵ, ρ, irf_path = para
    θ_t′, θ_t = [ones(Sim_T) for _ in 1:2]
    θ_t′ = exp.([-1. * σ_ϵ * ρ ^ (t - 1.) for t in 1:Sim_T])
    ## Case with initialization of heterogeneous beliefs
    for (i, prepost) in enumerate(["post", "pre"])
        path = "$irf_path/$t_sample/$(prepost)_shock"
        #simul_irf(para, [θ_t′, θ_t][i], t_sample, prepost)
        simul_irf_new_expanded(para, [θ_t′, θ_t][i], t_sample, prepost)
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
        writedlm("../data/$irf_path/combined_new/$var.csv", data, ',')
    end
end



## Plot irf with confidence intervels
function plot_irf_confidence(para, α, Sim_T)
    @unpack irf_path, yearly_str, iid_str = para
    vars = ["r", "w", "mean_a", "y", "I", "mean_n", "mean_en", "mean_c"]
    p_low = (1 - α) / 2
    p_high = 1 - p_low
    for v in vars
        print(v)
        data_low, data_median, data_high = [zeros(Sim_T) for i in 1:3]
        data = readdlm("../data/HA/yearly/iid/learning/IRFs_newTE_expanded/gain_0.05/combined_new/$v.csv", ',')[1:Sim_T, :]
        for t in 1:Sim_T
            data_low[t] = quantile(data[t, :], p_low)
            data_median[t] = quantile(data[t, :], 0.5)
            data_high[t] = quantile(data[t, :], p_high)
        end
        p = plot(grid = false, title = "$v")
        plot!(p, data_low, ls = :dash, color = :black, label = "")
        plot!(p, data_median, color = :black, label = "")
        plot!(p, data_high, ls = :dash, color = :black, label = "")

        rational = -1 .* readdlm("../data/HA/yearly/iid/rational/$v.csv", ',')[1:Sim_T, :]
        plot!(p, rational, ls = :dash, color = :red, label = "")


        savefig(p, "new_$v.pdf")
    end
end

## IRF with confidence intervals

s = ArgParseSettings()
@add_arg_table s begin
    "i"
        arg_type = Int
        required = true
        help = "indx going from 1 to 2000"
end
ps = parse_args(s)
i = ps["i"]

# Initialize
samples = convert.(Int, readdlm("../data/HA/periods.csv", ','))
Sim_T = 200

# Divide the indx into gain 1, 2, 3, 4 with 500 periods each case
function get_gain_indx(indx, sample_T)
    gain_i = (indx ÷ sample_T) + 1
    indx_t = (indx % sample_T)
    if indx_t == 0
        gain_i -= 1
        indx_t = sample_T
    end
    return gain_i, indx_t
end

gain_i, indx_t = get_gain_indx(i, length(samples))
t_sample = samples[indx_t]
para = HAmodel(gain = gain_i)
new_path = "$(para.irf_path[1:23])IRFs_newTE_expanded$(para.irf_path[28:end])"
para.irf_path = new_path
println("index = $i, $(para.gain_str), t_sample = $t_sample")
para, πval, k, ϵn_grid, n_grid, a_grid, c_grid = calibrate_stationary!(para)
IRFs(para, t_sample, Sim_T)
get_irf(para, t_sample, Sim_T)






#=
## Combine all of the irfs
samples = convert.(Int, readdlm("../data/HA/periods.csv", ','))
Sim_T = 200
for gain in 1:1
    para = HAmodel(gain = gain)
    new_path = "$(para.irf_path[1:23])IRFs_newTE_expanded$(para.irf_path[28:end])"
    para.irf_path = new_path
    combine_irfs(para, Sim_T, samples)
end
=#




## Plot all of irfs with confidence intervals, compared with RE
α = 0.95
Sim_T = 150
for gain in 3:3
    para = HAmodel(gain = gain)
    new_path = "$(para.irf_path[1:23])IRFs_newTE_expanded$(para.irf_path[28:end])"
    para.irf_path = new_path
    plot_irf_confidence(para, α, Sim_T)
end




#=
####################################################################################################
#                           IRF with confidence intervals high gain
####################################################################################################
s = ArgParseSettings()
@add_arg_table s begin
    "i"
        arg_type = Int
        required = true
        help = "indx going from 1 to 1000"
end
ps = parse_args(s)
indx = ps["i"]
## Initialize
srand(1)
samples = sort(sample(1000:3850, 500, replace = false))
Sim_T = 150
## Divide the indx into iid/noiid and pos/nagetive
pos = false
iid = indx <= 500
indx = indx - (indx > 500) * 500
t_sample = samples[indx]
para = HAmodel(iid = iid, pos = pos)
para, π, k, ϵn_grid, n_grid, a_grid, c_grid = calibrate_stationary!(para)
IRFs(para, t_sample, Sim_T)
get_irf(para, t_sample, Sim_T)
=#


#=
## Combine all of the irfs
srand(1)
samples = setdiff(sort(sample(1000:3850, 500, replace = false)), [1006, 2157, 2158, 2828, 2847, 2985, 3107])
Sim_T = 150
for iid in [true; false]
    for pos in [false]
        println("iid = $iid, pos = $pos")
        para = HAmodel(iid = iid, pos = pos)
        combine_irfs(para, Sim_T, samples)
    end
end
=#

#=
## Plot all of irfs with confidence intervals, compared with RE
α = 0.95
Sim_T = 150
para = HAmodel(gain = 3, iid = true)
plot_irf_confidence(para, α, Sim_T)
=#

include("HA_stationary.jl")
include("HA_TE.jl")



## Define the function that return data names saved as the code runs
function strs_data()
    idio_strs = ["a", "c", "n", "nu", "nu_bar", "nu_bar_c", "psi", "s"]
    aggr_strs = ["r", "R_cov", "theta", "w", "x", "y"]
    return idio_strs, aggr_strs
end



## Define the function that write as the code runs
function write_data(a, c, n, ν, ν̄, ν̄c, ψ, r, R, s, θ, w, x, y, t, lear_path)
    idio_strs, aggr_strs = strs_data()
    idios = [a, c, n, ν, ν̄, ν̄c, ψ, s]
    aggrs = [r, R, θ, w, x, y]
    ## Save all the idiosyncratic data and their means
    for (i, idio) in enumerate(idios)
        str = idio_strs[i]
        writedlm("../data/$(lear_path)/$(str)/$t.csv", idio, ',')
        writedlm("../data/$(lear_path)/mean_$(str)/$t.csv", mean(idio, 1), ',')
    end
    ## Save all the aggreagate data
    for (i, aggr) in enumerate(aggrs)
        str = aggr_strs[i]
        writedlm("../data/$(lear_path)/$(str)/$t.csv", aggr, ',')
    end
end



## Define the function that combine all of the data
function combine_data(T, lear_path)
    idio_strs, aggr_strs = strs_data()
    mean_idio_strs = ["mean_$(idio_strs[i])" for i in 1:length(idio_strs)]
    for str in vcat(mean_idio_strs, aggr_strs)
        data = zeros(T, length(readdlm("../data/$(lear_path)/$(str)/1.csv", ',')))
        for t in 1:T
            data[t, :] = readdlm("../data/$(lear_path)/$(str)/$(t).csv", ',')
        end
        writedlm("../data/$(lear_path)/$(str)/combined.csv", data, ',')
        for t in 1:T
            rm("../data/$(lear_path)/$(str)/$(t).csv")
        end
    end
end



## Define the function that plot all the combined data
function plot_data(lear_path)
    idio_strs, aggr_strs = strs_data()
    mean_idio_strs = ["mean_$(idio_strs[i])" for i in 1:length(idio_strs)]
    for str in vcat(mean_idio_strs, aggr_strs)
        data = readdlm("../data/$(lear_path)/$(str)/combined.csv", ',')
        p = plot(data, title = "$str", grid = false, xaxis = "Time", yaxis = "", label = "")
        savefig(p, "../figures/$(lear_path)/$str.png")
    end
end



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
function init_data_learning(agent_num, bin_midpts, draws, ψ_init, R̄, ā)
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
    θ, θ′  = ones(2)
    x_ = [1.; log(mean(a) / ā); log(θ)]
    x = x_
    x′ = ones(3)
    return c, n, ν, ν̄, ν̄c, r, w, s, s′, a, a′, ψ, ψ′, R, R′, θ, θ′, x_, x, x′, y
end



## Simulate the economy based on learning
## 1. νi_t - current marginal utility ν,
## 2. ν̄i_t - expected future marginal utility ν̄,
## 3. ν̄ci_t - current marginal utility in steady state ν̄c
## 4. x_t - vector of aggregate states [1; log(mean(a) / ā); log(θ_t[t])]
function simul_learning(para, π)
    @unpack N, a_min, a_max, agent_num, T, ā, ρ, σ_ϵ, γ_gain, ψ_init, R̄, lear_path, A, α = para
    ## Initialize functions
    cf = get_cf(para)
    bin_midpts = get_bins(a_min, a_max, N)
    draws = dimtrans1to2.(N, rand(DiscreteRV(π), agent_num))
    ## Initialize data of interest
    c, n, ν, ν̄, ν̄c, r, w, s, s′, a, a′, ψ, ψ′, R, R′, θ, θ′, x_, x, x′, y = init_data_learning(agent_num, bin_midpts, draws, ψ_init, R̄, ā)
    set_ϕ_range!(para, ψ, x)
    #= Loop through t = 1..T, each variable is index comtemporanously.
    Here the vector ψ[:,t] = {ψ_0,ψ_1,ψ_2,...} R[:,:,t] = {R_0,R_1,R_2,...}
    The belief subscript indicates the time when the belief is formed.
    The belief is being used the period after it is formed.
    For example ψ_1 is the belief formed at end time 1 but used at the beginning of time 2 =#
    for t in 1:T
        println(t)
        c, n, a′, ν, ν̄, ν̄c, r, w = TE(para, a, s, ψ, x, cf)
        y = θ * mean(a) ^ α * mean(A[s] .* n) ^ (1 - α)
        write_data(a, c, n, ν, ν̄, ν̄c, ψ, r, R, s, θ, w, x, y, t, lear_path)
        if t == T break end
        s′ = update_s(para, s, t)
        θ′ = drawθ(θ, σ_ϵ, ρ)
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



####################################################################################################
#                            Long simulations with learning
####################################################################################################
s = ArgParseSettings()
@add_arg_table s begin
    "i"
        arg_type = Int
        required = true
        help = "indx going from 0 to 7, only consider: yearly, high_amin, seed1
                Switching iid, from_zero, and gain_low"
end
ps = parse_args(s)
indx = ps["i"]



int_vec = Bool.([parse.(Int, bin(indx, 3)[i]) for i in 1:3])
iid, from_zero, gain_low = int_vec
println("i=$indx, iid=$iid, from_zero=$from_zero, gain_low=$gain_low")
para = HAmodel(yearly = true, high_amin = true, seed = 1,
               iid = iid, from_zero = from_zero, gain_low = gain_low)
para, π, k, ϵn_grid, n_grid, a_grid = calibrate_stationary!(para)
simul_learning(para, π)
combine_data(para.T, para.lear_path)
plot_data(para.lear_path)

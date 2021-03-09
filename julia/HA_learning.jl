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
        writedlm("../data/$(lear_path)/mean_$(str)/$t.csv", mean(idio, dims = 1), ',')
    end
    ## Save all the aggreagate data
    for (i, aggr) in enumerate(aggrs)
        str = aggr_strs[i]
        writedlm("../data/$(lear_path)/$(str)/$t.csv", aggr, ',')
    end
end



## Save the variables
function write_datatest(a, c, n, ν, ν̄, ν̄c, ψ, r, R, s, θ, w, x, y, ν̂, r̂, ŵ, ψ̂, t, lear_path)
    idio_strs, aggr_strs = strs_datatest()
    idios = [a, c, n, ν, ν̄, ν̄c, ψ, s, ψ̂, ν̂]
    aggrs = [r, R, θ, w, x, y, r̂, ŵ]
    ## Save all the idiosyncratic data and their means
    for (i, idio) in enumerate(idios)
        str = idio_strs[i]
        writedlm("../data/$(lear_path)/$(str)/$t.csv", idio, ',')
        writedlm("../data/$(lear_path)/mean_$(str)/$t.csv", mean(idio, dims = 1), ',')
    end
    ## Save all the aggreagate data
    for (i, aggr) in enumerate(aggrs)
        str = aggr_strs[i]
        writedlm("../data/$(lear_path)/$(str)/$t.csv", aggr, ',')
    end
end



## Define the function that combine all of the data
function combine_data(T, lear_path)
    idio_strs, aggr_strs = strs_datatest()
    mean_idio_strs = ["mean_$(idio_strs[i])" for i in 1:length(idio_strs)]
    for str in vcat(mean_idio_strs, aggr_strs)
        data = zeros(T, length(readdlm("../data/$(lear_path)/$(str)/1.csv", ',')))
        for t in 1:T
            data[t, :] = readdlm("../data/$(lear_path)/$(str)/$(t).csv", ',')
        end
        writedlm("../data/$(lear_path)/$(str)/testing.csv", data, ',')
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
# based on draws from the stationary distribution
# This initialization assumes homogeneity in beliefs - all initialized as ψ̄
# This function is used by simul_learning
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
    ϕ = compute_ϕ(ψ, x)
    return c, n, ν, ν̄, ν̄c, r, w, s, s′, a, a′, ψ, ψ′, R, R′, θ, θ′, x_, x, x′, y, ϕ
end


## Define the function that assembles information set
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


## Define a function that initialize the data of interest
function init_data_learning_expanded(agent_num, bin_midpts, draws, ψ_init, R̄, ā, A)
    r, w, y = zeros(3)
    c, n, ν, ν̄, ν̄c = [zeros(agent_num) for i in 1:5]
    a, s = init_as(agent_num, bin_midpts, draws)
    ψ = ψ_init
    s′ = zeros(Int64, agent_num)
    a′ = zeros(agent_num)
    ψ′ = zeros(agent_num, 7)
    R = zeros(agent_num, 7, 7)
    R′ = zeros(agent_num, 7, 7)
    for i in 1:agent_num
        R[i, :, :] = R̄
        R′[i, :, :] = R̄
    end
    θ, θ′  = ones(2)
    x_ = construct_x(agent_num, mean(a), ā, θ, a, s, A)
    x = x_
    x′ = ones(agent_num, 7)
    ϕ = compute_ϕ(ψ, x)
    return c, n, ν, ν̄, ν̄c, r, w, s, s′, a, a′, ψ, ψ′, R, R′, θ, θ′, x_, x, x′, y, ϕ
end

## Simulate the economy based on learning
# 1. νi_t - current marginal utility ν,
# 2. ν̄i_t - expected future marginal utility ν̄,
# 3. ν̄ci_t - current marginal utility in steady state ν̄c
# 4. x_t - vector of aggregate states [1; log(mean(a) / ā); log(θ_t[t])]
function simul_learning(para, πval)
    @unpack N, a_min, a_max, agent_num, T, ā, ρ, σ_ϵ, γ_gain, ψ_init, R̄, lear_path, A, α, irf_path, yearly_str, gain_str, iid_str = para
    ## Initialize functions
    cf1 = get_cf(para)
    bin_midpts = get_bins(a_min, a_max, N)
    draws = dimtrans1to2.(N, rand(DiscreteRV(πval), agent_num))
    ## Initialize data of interest
    init_data = init_data_learning(agent_num, bin_midpts, draws, ψ_init, R̄, ā)
    c, n, ν, ν̄, ν̄c, r, w, s, s′, a, a′, ψ, ψ′, R, R′, θ, θ′, x_, x, x′, y, ϕ = init_data
    set_ϕ_range!(para, ϕ)
    n̂_func, ĉ_func, â′_func = compute_indi_grids(cf1, para)
    #periods = convert.(Int, readdlm("../data/HA/periods.csv", ','))
    #data_str = "../data/HA/$yearly_str/$(iid_str)/learning/simulations/from_rational_RA/$(gain_str)"

    #Loop through t = 1..T, each variable is index comtemporanously.
    #Here the vector ψ[:,t] = {ψ_0,ψ_1,ψ_2,...} R[:,:,t] = {R_0,R_1,R_2,...}
    #The belief subscript indicates the time when the belief is formed.
    #The belief is being used the period after it is formed.
    #For example ψ_1 is the belief formed at end time 1 but used at the beginning of time 2
    mean_psi = zeros(T, 3)
    for t in 1:T
        println(t)
        c, n, a′, ν, ν̄, ν̄c, r, w = TE(para, θ, a, s, ϕ, cf1, n̂_func, ĉ_func, â′_func)
        y = θ * mean(a) ^ α * mean(A[s] .* n) ^ (1 - α)
        #write_data(a, c, n, ν, ν̄, ν̄c, ψ, r, R, s, θ, w, x, y, t, lear_path)
        #=
        if t in periods
            writedlm("$data_str/a/$t.csv", a, ',')
            writedlm("$data_str/s/$t.csv", s, ',')
            writedlm("$data_str/psi/$t.csv", ψ, ',')
            writedlm("$data_str/R_cov/$t.csv", R, ',')
        end
        =#
        mean_psi[t, :] = mean(ψ, dims = 1)
        #print(mean_psi[t, :])
        if t == T break end
        s′ = update_s(para, s, t)
        θ′ = drawθ(θ, σ_ϵ, ρ)
        x′ = [1; log(mean(a′) / ā); log(θ′)]
        R′ = update_R(R, x, t - 1, γ_gain)
        if t == 1
            ψ′ = ψ
        else
            #ψi_t[:, :, t + 1] corresponds to the subscrpt of ψ_t which is used at time t + 1.
            #This is the the belief formed at time t (used at time t + 1).
            #At time t, the agent only knows about information "x" up to time t
            #Since this is a forcasting model,
            #The right hand side is the data of "x" up to time t - 1
            #The left hand side is "log(ν)" up to time t
            ψ′ = update_ψ(ψ, R, x_, ν, t, γ_gain, agent_num, ν̄c)
            x_ = x
        end
        a, s, θ, x, R, ψ = a′, s′, θ′, x′, R′, ψ′
        ϕ = compute_ϕ(ψ, x)
        set_ϕ_range!(para, ϕ)
    end
    writedlm("mean_beliefs_0.001.csv", mean_psi, ',')
end



function simul_learning_new_expanded(para, πval)
    @unpack N, a_min, a_max, agent_num, T, ā, ρ, σ_ϵ, γ_gain, ψ_init_expanded, R̄_expanded, lear_path, A, α, yearly_str, iid_str, gain_str = para
    ψ_init = ψ_init_expanded
    R̄ = R̄_expanded
    ## Initialize functions
    cf1 = get_cf(para)
    bin_midpts = get_bins(a_min, a_max, N)
    draws = dimtrans1to2.(N, rand(DiscreteRV(πval), agent_num))
    ## Initialize data of interest
    init_data = init_data_learning_expanded(agent_num, bin_midpts, draws, ψ_init, R̄, ā, A)
    c, n, ν, ν̄, ν̄c, r, w, s, s′, a, a′, ψ, ψ′, R, R′, θ, θ′, x_, x, x′, y, ϕ = init_data
    set_ϕ_range!(para, ϕ)
    #periods = 25_000 .+ convert.(Int, readdlm("../data/HA/periods.csv", ','))
    periods = convert.(Int, readdlm("../data/HA/periods.csv", ','))
    data_str = "../data/HA/$yearly_str/$(iid_str)/learning/simulation/from_rational_RA_new_expanded/$(gain_str)"
    n̂_func, ĉ_func, â′_func = compute_indi_grids(cf1, para)
    #= Loop through t = 1..T, each variable is index comtemporanously.
    Here the vector ψ[:,t] = {ψ_0,ψ_1,ψ_2,...} R[:,:,t] = {R_0,R_1,R_2,...}
    The belief subscript indicates the time when the belief is formed.
    The belief is being used the period after it is formed.
    For example ψ_1 is the belief formed at end time 1 but used at the beginning of time 2 =#
    for t in 1:T
        println(t)
        c, n, a′, ν, ν̄, ν̄c, r, w = TE(para, θ, a, s, ϕ, cf1, n̂_func, ĉ_func, â′_func)
        y = θ * mean(a) ^ α * mean(A[s] .* n) ^ (1 - α)
        if t in periods
            writedlm("$data_str/a/$(t).csv", convert(Vector{Float32}, a), ',')
            writedlm("$data_str/s/$(t).csv", convert(Vector{Int8}, s), ',')
            writedlm("$data_str/psi/$(t).csv", convert(Array{Float32, 2}, ψ), ',')
            writedlm("$data_str/R_cov/$(t).csv", convert(Array{Float32, 3}, R), ',')
        end
        if t == T break end
        s′ = update_s(para, s, t)
        θ′ = drawθ(θ, σ_ϵ, ρ)
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



function simul_learning!(para, πval, simulation_df)
    @unpack N, a_min, a_max, agent_num, T, ā, ρ, σ_ϵ, γ_gain, ψ_init, R̄, lear_path, A, α = para
    ## Initialize functions
    cf1 = get_cf(para)
    bin_midpts = get_bins(a_min, a_max, N)
    draws = dimtrans1to2.(N, rand(DiscreteRV(πval), agent_num))
    ## Initialize data of interest
    init_data = init_data_learning(agent_num, bin_midpts, draws, ψ_init, R̄, ā)
    c, n, ν, ν̄, ν̄c, r, w, s, s′, a, a′, ψ, ψ′, R, R′, θ, θ′, x_, x, x′, y, ϕ = init_data
    set_ϕ_range!(para, ϕ)
    if para.yearly
        n̂_func, ĉ_func, â′_func = compute_indi_grids(cf1, para)
    end
    #Loop through t = 1..T, each variable is index comtemporanously.
    #Here the vector ψ[:,t] = {ψ_0,ψ_1,ψ_2,...} R[:,:,t] = {R_0,R_1,R_2,...}
    #The belief subscript indicates the time when the belief is formed.
    #The belief is being used the period after it is formed.
    #For example ψ_1 is the belief formed at end time 1 but used at the beginning of time 2
    for t in 1:T
        println(t)
        if para.yearly
            c, n, a′, ν, ν̄, ν̄c, r, w = TE(para, θ, a, s, ϕ, cf1, n̂_func, ĉ_func, â′_func)
        else
            c, n, a′, ν, ν̄, ν̄c, r, w = TE_old(para, a, s, ψ, x, θ, cf1)
        end

        y = θ * mean(a) ^ α * mean(A[s] .* n) ^ (1 - α)
        push!(simulation_df, [mean(a), mean(c), mean(n), y, θ])
        if t == T break end
        s′ = update_s(para, s, t)
        θ′ = drawθ(θ, σ_ϵ, ρ)
        x′ = [1; log(mean(a′) / ā); log(θ′)]
        R′ = update_R(R, x, t - 1, γ_gain)
        if t == 1
            ψ′ = ψ
        else
            #ψi_t[:, :, t + 1] corresponds to the subscrpt of ψ_t which is used at time t + 1.
            #This is the the belief formed at time t (used at time t + 1).
            #At time t, the agent only knows about information "x" up to time t
            #Since this is a forcasting model,
            #The right hand side is the data of "x" up to time t - 1
            #The left hand side is "log(ν)" up to time t
            ψ′ = update_ψ(ψ, R, x_, ν, t, γ_gain, agent_num, ν̄c)
            x_ = x
        end
        a, s, θ, x, R, ψ = a′, s′, θ′, x′, R′, ψ′
        ϕ = compute_ϕ(ψ, x)
        set_ϕ_range!(para, ϕ)
    end
end




function simul_learning_hold!(para, πval, left_matrix, right_matrix, last_asset)
    @unpack N, a_min, a_max, agent_num, T, ā, ρ, σ_ϵ, γ_gain, ψ_init, R̄, lear_path, A, α = para
    ## Initialize functions
    cf1 = get_cf(para)
    bin_midpts = get_bins(a_min, a_max, N)
    draws = dimtrans1to2.(N, rand(DiscreteRV(πval), agent_num))
    ## Initialize data of interest
    init_data = init_data_learning(agent_num, bin_midpts, draws, ψ_init, R̄, ā)
    c, n, ν, ν̄, ν̄c, r, w, s, s′, a, a′, ψ, ψ′, R, R′, θ, θ′, x_, x, x′, y, ϕ = init_data
    set_ϕ_range!(para, ϕ)
    n̂_func, ĉ_func, â′_func = compute_indi_grids(cf1, para)
    psi = readdlm("mean_beliefs_0.001.csv", ',')
    psi_hold = mean(psi[25001:end, :], dims = 1) .* ones(agent_num)
    ψ = psi_hold

    #Loop through t = 1..T, each variable is index comtemporanously.
    #Here the vector ψ[:,t] = {ψ_0,ψ_1,ψ_2,...} R[:,:,t] = {R_0,R_1,R_2,...}
    #The belief subscript indicates the time when the belief is formed.
    #The belief is being used the period after it is formed.
    #For example ψ_1 is the belief formed at end time 1 but used at the beginning of time 2
    for t in 1:T
        println(t)
        c, n, a′, ν, ν̄, ν̄c, r, w = TE(para, θ, a, s, ϕ, cf1, n̂_func, ĉ_func, â′_func)
        left_matrix[t, :] = log.(ν ./ ν̄c)
        right_matrix[t, :] = x
        if t == T
            last_asset = a
        end
        if t == T break end
        s′ = update_s(para, s, t)
        θ′ = drawθ(θ, σ_ϵ, ρ)
        x′ = [1; log(mean(a′) / ā); log(θ′)]
        R′ = update_R(R, x, t - 1, γ_gain)
        if t == 1
            ψ′ = psi_hold
        else
            #ψi_t[:, :, t + 1] corresponds to the subscrpt of ψ_t which is used at time t + 1.
            #This is the the belief formed at time t (used at time t + 1).
            #At time t, the agent only knows about information "x" up to time t
            #Since this is a forcasting model,
            #The right hand side is the data of "x" up to time t - 1
            #The left hand side is "log(ν)" up to time t
            ψ′ = psi_hold
            x_ = x
        end
        a, s, θ, x, R, ψ = a′, s′, θ′, x′, R′, ψ′
        ϕ = compute_ϕ(ψ, x)
        set_ϕ_range!(para, ϕ)
    end
    writedlm("left.csv", left_matrix, ',')
    writedlm("right.csv", right_matrix, ',')
    writedlm("last_asset.csv", last_asset, ',')
end



function simul_learning_expanded!(para, πval, simulation_df)
    @unpack N, a_min, a_max, agent_num, T, ā, ρ, σ_ϵ, γ_gain, ψ_init_expanded, R̄_expanded, lear_path, A, α = para
    ψ_init = ψ_init_expanded
    R̄ = R̄_expanded
    ## Initialize functions
    cf1 = get_cf(para)
    bin_midpts = get_bins(a_min, a_max, N)
    draws = dimtrans1to2.(N, rand(DiscreteRV(πval), agent_num))
    ## Initialize data of interest
    init_data = init_data_learning_expanded(agent_num, bin_midpts, draws, ψ_init, R̄, ā, A)
    c, n, ν, ν̄, ν̄c, r, w, s, s′, a, a′, ψ, ψ′, R, R′, θ, θ′, x_, x, x′, y, ϕ = init_data
    set_ϕ_range!(para, ϕ)
    if para.yearly
        n̂_func, ĉ_func, â′_func = compute_indi_grids(cf1, para)
    end
    #= Loop through t = 1..T, each variable is index comtemporanously.
    Here the vector ψ[:,t] = {ψ_0,ψ_1,ψ_2,...} R[:,:,t] = {R_0,R_1,R_2,...}
    The belief subscript indicates the time when the belief is formed.
    The belief is being used the period after it is formed.
    For example ψ_1 is the belief formed at end time 1 but used at the beginning of time 2 =#
    ψ_sim = zeros(T, 7)
    for t in 1:T
        print(t)
        if para.yearly
            c, n, a′, ν, ν̄, ν̄c, r, w = TE(para, θ, a, s, ϕ, cf1, n̂_func, ĉ_func, â′_func)
        else
            c, n, a′, ν, ν̄, ν̄c, r, w = TE_old(para, a, s, ψ, x, θ, cf1)
        end
        y = θ * mean(a) ^ α * mean(A[s] .* n) ^ (1 - α)
        ψ_sim[t, :] = mean(ψ, dims = 1)
        push!(simulation_df, [mean(a), mean(c), mean(n), y, θ])
        if t == T break end
        s′ = update_s(para, s, t)
        θ′ = drawθ(θ, σ_ϵ, ρ)
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
    writedlm("psi_0.001.csv", ψ_sim, ',')
end




function simul_learning_expanded(para, πval)
    @unpack N, a_min, a_max, agent_num, T, ā, ρ, σ_ϵ, γ_gain, ψ_init_expanded, R̄_expanded, lear_path, A, α = para
    ψ_init = ψ_init_expanded
    R̄ = R̄_expanded
    ## Initialize functions
    cf1 = get_cf(para)
    bin_midpts = get_bins(a_min, a_max, N)
    draws = dimtrans1to2.(N, rand(DiscreteRV(πval), agent_num))
    ## Initialize data of interest
    init_data = init_data_learning_expanded(agent_num, bin_midpts, draws, ψ_init, R̄, ā, A)
    c, n, ν, ν̄, ν̄c, r, w, s, s′, a, a′, ψ, ψ′, R, R′, θ, θ′, x_, x, x′, y, ϕ = init_data
    set_ϕ_range!(para, ϕ)
    n̂_func, ĉ_func, â′_func = compute_indi_grids(cf1, para)
    #= Loop through t = 1..T, each variable is index comtemporanously.
    Here the vector ψ[:,t] = {ψ_0,ψ_1,ψ_2,...} R[:,:,t] = {R_0,R_1,R_2,...}
    The belief subscript indicates the time when the belief is formed.
    The belief is being used the period after it is formed.
    For example ψ_1 is the belief formed at end time 1 but used at the beginning of time 2 =#
    ψ_sim = zeros(T, 7)
    for t in 1:T
        println(t)

        c, n, a′, ν, ν̄, ν̄c, r, w = TE(para, θ, a, s, ϕ, cf1, n̂_func, ĉ_func, â′_func)
        y = θ * mean(a) ^ α * mean(A[s] .* n) ^ (1 - α)

        ψ_sim[t, :] = mean(ψ, dims = 1)
        print(ψ_sim[t, :])
        if t == T break end
        s′ = update_s(para, s, t)
        θ′ = drawθ(θ, σ_ϵ, ρ)
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
    writedlm("psi_0.001.csv", ψ_sim, ',')
end



#=
s = ArgParseSettings()
@add_arg_table! s begin
    "i"
        arg_type = Int
        required = true
        help = "indx going from 0 to 15, only consider: iid, from_zero, gains (4)"
end
ps = parse_args(s)
indx = ps["i"]


para = HAmodel(gain = indx, iid = true)
print("$(para.gain_str)")
para, πval, k, ϵn_grid, n_grid, a_grid, c_grid = calibrate_stationary!(para)
simul_learning_new_expanded(para, πval)
=#



#=
simul_learning(para, πval)

psi = readdlm("mean_beliefs_0.001.csv", ',')
plot(psi)
mean(psi[25001:end, :], dims = 1)


left_matrix = zeros(10_000, 100_000)
right_matrix = zeros(10_000, 3)
last_asset = readdlm("last_asset.csv", ',')

writedlm("coeffs.csv", coeffs, ',')
scatter(last_asset, coeffs[:, 1], alpha = 0.1, layout = (3, 1), legend = false)
scatter!(last_asset, coeffs[:, 2], alpha = 0.1, subplot = 2, legend = false)
scatter!(last_asset, coeffs[:, 3], alpha = 0.1, subplot = 3, legend = false, xlabel = "Asset Holding")
savefig("Distribution_with_Asset.png")


cov(last_asset, coeffs[:, 1])/(std(last_asset) * std(coeffs[:, 1]))
cov(last_asset, coeffs[:, 2])/(std(last_asset) * std(coeffs[:, 2]))
cov(last_asset, coeffs[:, 3])/(std(last_asset) * std(coeffs[:, 3]))

left = readdlm("left.csv", ',', dims = (10_000, 100_000))
left = CSV.read("left.csv")
right = readdlm("right.csv", ',', dims = (10_000, 4))


coeffs = ones(100_000, 3)
X = right_matrix[1:end-1, :]
for i in 1:100_000
    Y = left_matrix[2:end, i]
    coeffs[i, :] = inv(X' * X) * X' * Y
end
last_asset
histogram(coeffs[:, 1], grid = false, alpha = 0.15, lw = 0.5, legend = false, title = "RPE Histogram", layout = (3, 1))
histogram!(coeffs[:, 2], alpha = 0.15, lw = 0.5, subplot = 2, legend = false)
histogram!(coeffs[:, 3], alpha = 0.15, lw = 0.5, subplot = 3, legend = false)
vline!([0.000522421], subplot = 1)
vline!([-0.540497], subplot = 2)
vline!([-0.849366], subplot = 3)
savefig("RPE.png")
#simul_learning_old_expanded(para, πval)

#vec = readdlm("psi_0.001.csv", ',')
#plot(vec)
=#

## Long simulations with learning
#=
s = ArgParseSettings()
@add_arg_table s begin
    "i"
        arg_type = Int
        required = true
        help = "indx going from 0 to 15, only consider: iid, from_zero, gains (4)"
end
ps = parse_args(s)
indx = ps["i"]

int_vec = Bool.([parse.(Int, bin(indx ÷ 4, 2)[i]) for i in 1:2])
iid, from_zero = int_vec
gain = mod(indx, 4) + 1
println("yearly, i = $indx, iid = $iid, from_zero = $from_zero")
=#


#=
para = HAmodel(yearly = true, iid = true, from_zero = true, gain = 2)
para, πval, k, ϵn_grid, n_grid, a_grid, c_grid = calibrate_stationary!(para)
simul_learning(para, πval)
combine_data(para.T, para.lear_path)
=#



#=
plot_data(para.lear_path)
para = HAmodel(yearly = false, iid = true, from_zero = true, gain = 1)
para, πval, k, ϵn_grid, n_grid, a_grid, c_grid = calibrate_stationary!(para)
simul_learning(para, πval)



for indx in 0:15
    int_vec = Bool.([parse.(Int, bin(indx ÷ 4, 2)[i]) for i in 1:2])
    iid, from_zero = int_vec
    gain = mod(indx, 4) + 1
    if (gain == 1 || gain == 2)
        para = HAmodel(yearly = false, iid = iid, from_zero = from_zero, gain = gain)
        plot_data(para.lear_path)
    end
end
=#



#=
a = vec(readdlm("a5000.csv", ','))
s = vec(Int.(readdlm("s5000.csv", ',')))
ψ = readdlm("psi5000.csv", ',')
diff_r = zeros(10)
diff_w = zeros(10)
for (i, θ) in enumerate(LinRange(0.95, 1.05, 10))
    x = [1; log(mean(a) / ā); log(θ)]
    ϕ = compute_ϕ(ψ, x)
    c, n, a′, ν, ν̄, ν̄c, r, w = TE(para, a, s, ψ, x, θ, cf1)
    c_new, n_new, a′_new, ν_new, ν̄_new, ν̄c_new, r_new, w_new = TE(para, θ, a, s, ϕ, cf1, n̂_func, ĉ_func, â′_func)
    diff_r[i] = r - r_new
    diff_w[i] = w - w_new
end



θ = 1.05
plot(LinRange(0.95, 1.05, 10), diff_w, label = "w_diff", layout = (2, 1))
plot!(LinRange(0.95, 1.05, 10), diff_r, label = "r_diff", subplot = 2)
p = plot(layout = (3, 1), legend = false)
histogram!(p, c .- c_new, grid = false, title = "\\Delta c", subplot = 1)
histogram!(p, n .- n_new, grid = false, title = "\\Delta n", subplot = 2)
histogram!(p, a′ .- a′_new, grid = false, title = "\\Delta aprime", subplot = 3)
r - r_new
w  - w_new
savefig(p, "TE_diff_old_new.pdf")
=#

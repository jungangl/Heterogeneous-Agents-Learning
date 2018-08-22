include("HA_stationary.jl")




## Define the function that write as the code runs
function write_data(a, c, n, ν, ν̄, ν̄c, ψ, r, R, s, θ, w, x, t, str)
    writedlm("../data/HA_learning/simulations/$str/a/$t.csv", a, ',')
    writedlm("../data/HA_learning/simulations/$str/mean_a/$t.csv", mean(a), ',')
    writedlm("../data/HA_learning/simulations/$str/c/$t.csv", c, ',')
    writedlm("../data/HA_learning/simulations/$str/mean_c/$t.csv", mean(c), ',')
    writedlm("../data/HA_learning/simulations/$str/n/$t.csv", n, ',')
    writedlm("../data/HA_learning/simulations/$str/mean_n/$t.csv", mean(n), ',')
    writedlm("../data/HA_learning/simulations/$str/nu/$t.csv", ν, ',')
    writedlm("../data/HA_learning/simulations/$str/mean_nu/$t.csv", mean(ν), ',')
    writedlm("../data/HA_learning/simulations/$str/nu_bar/$t.csv", ν̄, ',')
    writedlm("../data/HA_learning/simulations/$str/mean_nu_bar/$t.csv", mean(ν̄), ',')
    writedlm("../data/HA_learning/simulations/$str/nu_bar_c/$t.csv", ν̄c, ',')
    writedlm("../data/HA_learning/simulations/$str/mean_nu_bar_c/$t.csv", mean(ν̄c), ',')
    writedlm("../data/HA_learning/simulations/$str/psi/$t.csv", ψ, ',')
    writedlm("../data/HA_learning/simulations/$str/mean_psi/$t.csv", mean(ψ, 1), ',')
    writedlm("../data/HA_learning/simulations/$str/r/$t.csv", r, ',')
    writedlm("../data/HA_learning/simulations/$str/R_cov/$t.csv", R, ',')
    writedlm("../data/HA_learning/simulations/$str/s/$t.csv", s, ',')
    writedlm("../data/HA_learning/simulations/$str/mean_s/$t.csv", mean(s), ',')
    writedlm("../data/HA_learning/simulations/$str/theta/$t.csv", θ, ',')
    writedlm("../data/HA_learning/simulations/$str/w/$t.csv", w, ',')
    writedlm("../data/HA_learning/simulations/$str/x/$t.csv", x, ',')
end



#------------------------------------------------------------------------#
## Let c̄(a, s) be the consumption function in the steady state
## Let ψ be the beliefs of the agents and x be the aggregate states
## Define ϕ = ψ'⋅x. For given r, we want to find
## c(a, ϕ, s), a′(a, ϕ, s), and n(a, ϕ, s) that solve
## 1. c(a, ϕ, s)^{-σ} ≥ ∑_{s'} π(s′|s)⋅β⋅(1 + r)⋅c̄(a′(a, ϕ, s), s′)^{-σ}⋅exp(ϕ)
## 2. c(a, ϕ, s) + a′(a, ϕ, s) = (1 + r)⋅a + A(s)⋅w⋅n(a, ϕ, s)
## 3. A(s)⋅w⋅c(a, ϕ, s)^{-σ} = χ⋅(1 - n(a, ϕ, s))^γ
#------------------------------------------------------------------------#



## Given r, w, for each s
## compute the vector of consumptions chosen by the agent if a′ = a = a_min.
function compute_cmin2!(para, r, w, cf)
  @unpack σ, γ, χ, w̄, r̄, A, S, a_min = para
  for s in 1:S
      function f(logc)
          c = exp.(logc)
          n = 1 - ((w * A[s] * c .^ (-σ)) / χ) .^ (-1 / γ)
          return c - A[s] * w * n - r * a_min
      end
      res = nlsolve(f, [0.]; inplace = false)
      para.c_min2[s] = exp.(res.zero[1])
  end
end



## Given r, w and the steady state consumption function
## Use 2D-interpolation to approximate the consumption function cf2[s](a, ϕ)
## The grid for the 2D-interpolation is
## a_vec =  linspace(a_min, a_max, N) and ϕ_vec
function get_cf2(para, cf, r, w, n_con = 10)
    compute_cmin2!(para, r, w, cf)
    @unpack σ, γ, β, χ, r̄, w̄, P, A, S, c_min2, k_spline = para
    @unpack a_min, a_max, ϕ_min, ϕ_max, N_ϕ = para

    ϕ_vec = linspace(ϕ_min, ϕ_max, N_ϕ)
    cf2 = Spline2D[]
    a′grid = vcat(linspace(a_min, a_min + 2, 20),
                  linspace(a_min + 2, a_max, 80)[2:end])
    N_a = length(a′grid)
    #preallocate for speed
    a_grid = zeros(S, N_ϕ, N_a + n_con)
    c_grid = zeros(S, N_ϕ, N_a + n_con)
    Uc′ = zeros(S)
    for (i_a,a′) in enumerate(a′grid)
        for s′ in 1:S
            Uc′[s′] = (cf[s′](a′)) ^ (-σ)
        end
        for (i_ϕ, ϕ) in enumerate(ϕ_vec)
            for s in 1:S

                c = (exp(ϕ) * β * (1 + r̄) * dot(P[s,:],Uc′) ) ^ (-1 / σ)
                n = max(1 - ((w * A[s] * c ^ (-σ)) / χ) ^ (-1 / γ), 0.)
                a = (a′ + c - A[s] * w * n) / (1 + r)
                a_grid[s, i_ϕ, i_a + n_con] = a
                c_grid[s, i_ϕ, i_a + n_con] = c
            end
        end
    end
    for s in 1:S
        for (i_ϕ, ϕ) in enumerate(ϕ_vec)
            if a_grid[s, i_ϕ, n_con + 1] > a_min
                for (i_c, ĉ) in enumerate(linspace(c_min2[s], c_grid[s, i_ϕ, n_con + 1], n_con + 1)[1:n_con])
                    n̂ = max(1 - ((w * A[s] * ĉ ^ (-σ)) / χ) ^ (-1 / γ), 0)
                    â = (a_min + ĉ - A[s] * w * n̂) / (1 + r)
                    a_grid[s, i_ϕ, i_c] = â
                    c_grid[s, i_ϕ, i_c] = ĉ
                end
            else
                a_grid[s, i_ϕ, 1:n_con] = -Inf
                c_grid[s, i_ϕ, 1:n_con] = -Inf
            end
        end
    end
    #Now interpolate
    cf2 = Vector{Spline2D}(S)
    for s in 1:S
        cvec = zeros(N_a, N_ϕ)
        for i_ϕ in 1:N_ϕ
            #If the constraint never binds don't need extra grid points
            if c_grid[s, i_ϕ, 1] == -Inf
                cvec[:, i_ϕ] .= Spline1D(a_grid[s, i_ϕ, n_con + 1:end], c_grid[s , i_ϕ, n_con + 1:end]; k = k_spline)(a′grid)
            #If the constraint binds binds need to use all the grid points
            else
                cvec[:, i_ϕ] .= Spline1D(a_grid[s, i_ϕ, :], c_grid[s, i_ϕ, :]; k = k_spline)(a′grid)
            end
        end
        cf2[s] = Spline2D(a′grid, ϕ_vec, cvec)
    end
    return cf2
end




## Compute the expected future marginal utility in steady state
## The expetation is taken over all future states
function ν̄_f(ai, si, cf, para)
    @unpack A, σ, χ, γ, P, S, r̄, w̄ = para
    ci = cf[si](ai)
    ni = max(1 - (w̄ * A[si] * ci ^ (-σ) / χ) ^ (-1 / γ), 0.0)
    ai′ = A[si] * w̄ * ni + (1 + r̄) * ai - ci
    ν̄ = (1 + r̄) *
        dot(
            P[si,:],
            [(cf[s](ai′)) ^ (-σ) for s in 1:S]
        )
    return ν̄
end



## Compute the current marginal utility in the steady state
function ν̄c_f(ai, si, cf, para)
    @unpack σ, r̄ = para
    ci = cf[si](ai)
    ν̄ = (1 + r̄) * ci ^ (-σ)
    return ν̄
end



## Compute the labor supply based on learning
## Note that solving for agent's labor supply with endogenous grid cf2
function indi_labor_supply(para, r, w, ai, si, ψi, x, cf2)
    @unpack σ, γ, β, χ, A, S, P, a_min,a_max = para
    ϕi = dot(ψi, x)
    ci = cf2[si](ai, ϕi)
    ni = max(1 - ((w * A[si] * ci ^ (-σ)) / χ) ^ (-1 / γ), 0.0)
    ai′ = (1 + r) * ai + A[si] * w * ni - ci
    if ai′ < a_min
        Δ = a_min - ai′
        ci -= Δ
        ai′ = a_min
    end
    if ai′ > a_max
        Δ = a_max - ai′
        ci -= Δ
        ai′ = a_max
    end
    return A[si] * ni, ci, ni, ai′
end



## Compute the average labor supply across all agents at time t
## Use parallelization
## ∑(A[si] ⋅ ni) / N
function avg_labor_supply(para, r, w, a, s, ψ, x, cf2)
    @unpack agent_num = para
    N = 0.
    for i in 1:agent_num
        N = N + indi_labor_supply(para, r, w, a[i], s[i], ψ[i, :], x, cf2)[1]
    end
    return N::Float64 / agent_num
end



## Given the the prices r and w, and individual states a, s,
## iterate through all the agents, return individual agent decisions
## and return expected future marginal utility ν̄
## and current marginal utility in steady state ν̄c
function iterate_agents(para, r, w, a, s, ψ, x, cf, cf2)
    @unpack σ, γ, β, χ, A, agent_num = para
    c, n, a′, ν̄, ν̄c = [zeros(agent_num) for _ in 1:5]
    for i in 1:agent_num
        ai, si, ψi = a[i], s[i], ψ[i,:]
        ci, ni, ai′ = indi_labor_supply(para, r, w, ai, si, ψi, x, cf2)[2:end]
        ν̄i = ν̄_f(ai, si, cf, para)
        ν̄ci = ν̄c_f(ai, si, cf, para)
        c[i], n[i], a′[i], ν̄[i], ν̄c[i] = ci, ni, ai′, ν̄i, ν̄ci
    end
    return c, n, a′, ν̄, ν̄c
end



## Define F function for rootsolving
function F(lnw, α, θ, δ, para, a, s, ψ, x, cf)
    w = exp.(lnw[1])
    k = (w / ((1 - α) * θ)) ^ (1 / α)
    r = α * θ * (k) ^ (α - 1) - δ
    cf2 = get_cf2(para, cf, r, w)
    N = avg_labor_supply(para, r, w, a, s, ψ, x, cf2)
    K = mean(a)
    #println("K = $K, N = $N")
    w_diff = w - (1 - α) * θ * (K / N) ^ α
    return w_diff
end



## Given individual states distribution (a, s, ψ) and information x,
## compute the temporary equilibrium TE
## return :
## 1. inididual agents deicisions c, n, a′
## 2. current marginal utility ν,
## 3. expected future marginal utility ν̄,
## 4. current marginal utility in steady state ν̄c
## 5. prices r, w
function TE(para, a, s, ψ, x, cf)
  @unpack σ, β, α, δ, P, A, agent_num, w̄ = para
  θ = exp.(x[3])
  lnw = nlsolve(lnw -> F(lnw, α, θ, δ, para, a, s, ψ, x, cf), [log(w̄)]; inplace = false).zero::Vector{Float64}
  w = exp.(lnw[1])
  k = (w / ((1 - α) * θ)) ^ (1 / α)
  r = α * θ * (k) ^ (α - 1) - δ
  cf2 = get_cf2(para, cf, r, w)
  c, n, a′, ν̄, ν̄c = iterate_agents(para, r, w, a, s, ψ, x, cf, cf2)
  ν = (1. + r) * c .^ (-σ)
  return c, n, a′, ν, ν̄, ν̄c, r, w
end



## Update the individual productivity state
function update_s(para::HAmodel, s, t)
  @unpack agent_num, P = para
  s′ = zeros(Int64, agent_num)
  srand(t)
  for i in 1:agent_num
      si = s[i]
      prob = P[si, :]
      s′[i] = rand(DiscreteRV(prob))
  end
  return s′
end



## Update the aggregate productivity state
function drawθ(θ, σ_ϵ, ρ)
  dist = Normal(0, σ_ϵ)
  return exp.(ρ * log(θ) + rand(dist))
end



## Update the variance-covariance matrix for least square learning
function update_R(R, x, t, γ_gain, keep_const)
    R′ = zeros(3, 3)
    if keep_const == false
        R′ = R + γ_gain(t) .* (x * x' - R)
        @assert det(R′) != 0 "det(R′) = 0!"
    else
        R′ = R
    end
    return R′
end



## Update the beliefs for all of the agents based on least square learning
function update_ψ(ψ, R′, x, ν, t, γ_gain, agent_num, ν̄c, keep_const)
    ψ′ = zeros(agent_num, 3)
    if keep_const == false
        for i in 1:agent_num
            ψ′[i, :] = ψ[i, :] + γ_gain(t) .* inv(R′) * x .* (log(ν[i] / ν̄c[i]) - ψ[i,:]' * x)[1]
        end
    else
        ψ′ = ψ
    end
    return ψ′
end



## Define the function that sets the range of ϕ = ψ ⋅ x
function set_ϕ_range!(para, ψ, x)
    ϕ_vec = zeros(para.agent_num)
    ## Find the min and max of ϕ = x ⋅ ψ
    for i in 1:para.agent_num
        ϕ_vec[i] = dot(ψ[i, :], x)
    end
    ϕ_min = minimum(ϕ_vec)
    ϕ_max = maximum(ϕ_vec)
    if ϕ_min == ϕ_max
        ϕ_min = ϕ_min - 0.001
        ϕ_max = ϕ_max + 0.001
    end
    para.ϕ_min = ϕ_min
    para.ϕ_max = ϕ_max
end



## Initialze distributions for asset a, productivity s and beleifs ψ
## based on draws from the stationary distribution
## This initialization assumes homogeneity in beliefs - all initialized as ψ̄
## This function is used by simul_learning
function init_asψ(agent_num, bin_midpts, draws, ψ̄)
    ai_1 = zeros(agent_num)
    si_1 = zeros(Int64, agent_num)
    ψi_0 = zeros(agent_num,3)
    for (i, draw) in enumerate(draws)
        ai_1[i] = bin_midpts[draw[1]]
        si_1[i] = draw[2]
        ψi_0[i,:] = ψ̄
    end
    return ai_1, si_1, ψi_0
end



####################################################################################################
## Define a function that initialize the data of interest
function init_data_learning(agent_num, bin_midpts, draws, ψ̄, R̄, ā)
    r, w = zeros(2)
    c, n, ν, ν̄, ν̄c = [zeros(agent_num) for i in 1:5]
    a, s, ψ = init_asψ(agent_num, bin_midpts, draws, ψ̄)
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
    return c, n, ν, ν̄, ν̄c, r, w, s, s′, a, a′, ψ, ψ′, R, R′, θ, θ′, x_, x, x′
end


a_v = readdlm("../data/HA_learning/simulations/from_zeros/gain_0.005/a/34.csv")
histogram(a_v)
find(a_v .< 0.)
minimum(a_v)
## Simulate the economy based on learning
## 1. νi_t - current marginal utility ν,
## 2. ν̄i_t - expected future marginal utility ν̄,
## 3. ν̄ci_t - current marginal utility in steady state ν̄c
## 4. x_t - vector of aggregate states [1; log(mean(a) / ā); log(θ_t[t])]
function simul_learning(para, π, str)
    @unpack N, a_min, a_max, agent_num, T, ā, ρ, σ_ϵ, γ_gain, ψ̄, R̄ = para
    ## Initialize functions
    cf = get_cf(para)
    bin_midpts = get_bins(a_min, a_max, N)
    draws = dimtrans1to2.(N, rand(DiscreteRV(π), agent_num))
    ## Initialize data of interest
    c, n, ν, ν̄, ν̄c, r, w, s, s′, a, a′, ψ, ψ′, R, R′, θ, θ′, x_, x, x′ = init_data_learning(agent_num, bin_midpts, draws, ψ̄, R̄, ā)
    set_ϕ_range!(para, ψ, x)
    #= Loop through t = 1..T, each variable is index comtemporanously.
    Here the vector ψ[:,t] = {ψ_0,ψ_1,ψ_2,...} R[:,:,t] = {R_0,R_1,R_2,...}
    The belief subscript indicates the time when the belief is formed.
    The belief is being used the period after it is formed.
    For example ψ_1 is the belief formed at end time 1 but used at the beginning of time 2 =#
    for t in 1:T
        println(t)
        c, n, a′, ν, ν̄, ν̄c, r, w = TE(para, a, s, ψ, x, cf)
        write_data(a, c, n, ν, ν̄, ν̄c, ψ, r, R, s, θ, w, x, t, str)
        #if t == T break end
        s′ = update_s(para, s, t)
        θ′ = drawθ(θ, σ_ϵ, ρ)
        x′ = [1; log(mean(a′) / ā); log(θ′)]
        R′ = update_R(R, x, t - 1, γ_gain, false)
        if t == 1
            ψ′ = ψ
        else
            ## ψi_t[:, :, t + 1] corresponds to the subscrpt of ψ_t which is used at time t + 1.
            ## This is the the belief formed at time t (used at time t + 1).
            ## At time t, the agent only knows about information "x" up to time t
            ## Since this is a forcasting model,
            ## The right hand side is the data of "x" up to time t - 1
            ## The left hand side is "log(ν)" up to time t
            ψ′ = update_ψ(ψ, R, x_, ν, t, γ_gain, agent_num, ν̄c, false)
            x_ = x
        end
        a, s, θ, x, R, ψ = a′, s′, θ′, x′, R′, ψ′
        set_ϕ_range!(para, ψ, x)
    end
end
####################################################################################################


####################################################################################################
## Define a function that initialize the data of interest
function init_data_irf(agent_num, ψ̄, R̄, ā, a, θ_t)
    r, w = zeros(2)
    c, n, ν, ν̄, ν̄c = [zeros(agent_num) for i in 1:5]
    s′ = zeros(Int64, agent_num)
    a′ = zeros(agent_num)
    ψ′ = zeros(agent_num, 3)
    R = zeros(3, 3)
    R′ = zeros(3, 3)
    R = R̄
    R′ = R̄
    θ, θ′  = θ_t[1:2]
    x_ = [1.; log(mean(a) / ā); log(θ)]
    x = x_
    x′ = ones(3)
    return c, n, ν, ν̄, ν̄c, r, w, s′, a′, ψ′, R, R′,θ, θ′, x_, x, x′
end



## Simulate the economy from aggregate productivity shock vector θ_t
## Boolean het controls the heterogeneity of beliefs initialzed
## Boolean keep_const controls if the beliefs update over time
## Return variables of interest: averge (consumptions, labors, assets, marginal utilitys, beliefs),
## and interest rates and wages.
function simul_irf(para, θ_t, gain, t_sample, prepost, keep_const)
    @unpack N, a_min, a_max, agent_num, ā, ρ, σ_ϵ, γ_gain, ψ̄, R̄ = para
    str = "irf_$gain/$t_sample/$(prepost)_shock"
    T = length(θ_t)
    ## Initialize functions
    cf = get_cf(para)
    a = convert(Array{Float64, 2}, readdlm("../data/steadystates_$gain/a/$t_sample.csv", ','))
    s = convert(Array{Int64, 2}, readdlm("../data/steadystates_$gain/s/$t_sample.csv", ','))
    ψ = convert(Array{Float64, 2}, readdlm("../data/steadystates_$gain/psi/$t_sample.csv", ','))
    c, n, ν, ν̄, ν̄c, r, w, s′, a′, ψ′, R, R′,θ, θ′, x_, x, x′ = init_data_irf(agent_num, ψ̄, R̄, ā, a, θ_t)
    set_ϕ_range!(para, ψ, x)
    #=Loop through t = 1..T, each variable is index comtemporanously.
    Here the vector ψ[:,t] = {ψ_0,ψ_1,ψ_2,...} R[:,:,t] = {R_0,R_1,R_2,...}
    The belief subscript indicates the time when the belief is formed.
    The belief is being used the period after it is formed.
    For example ψ_1 is the belief formed at end time 1 but used at the beginning of time 2 =#
    for t in 1:T
        println(t)
        c, n, a′, ν, ν̄, ν̄c, r, w = TE(para, a, s, ψ, x, cf)
        write_data(a, c, n, ν, ν̄, ν̄c, ψ, r, R, s, θ, w, x, t, str)
        if t == T break end
        s′ = update_s(para, s, t)
        θ′ = θ_t[t + 1]
        x′ = [1; log(mean(a′) / ā); log(θ′)]
        R′ = update_R(R, x, t - 1, γ_gain, keep_const)
        if t == 1
            ψ′ = ψ
        else
            ## ψi_t[:, :, t + 1] corresponds to the subscrpt of ψ_t which is used at time t + 1.
            ## This is the the belief formed at time t (used at time t + 1).
            ## At time t, the agent only knows about information "x" up to time t
            ## Since this is a forcasting model,
            ## The right hand side is the data of "x" up to time t - 1
            ## The left hand side is "log(ν)" up to time t
            ψ′ = update_ψ(ψ, R, x_, ν, t, γ_gain, agent_num, ν̄c, keep_const)
            x_ = x
        end
        a, s, θ, x, R, ψ = a′, s′, θ′, x′, R′, ψ′
        set_ϕ_range!(para, ψ, x)
    end
end



## Simulate the impulse response functions from shock to θ of one standard devation
## Period of the simulation is Sim_T, the shock enters at t = shock_enter
## The shock decays with rate of para.ρ
function IRFs(para, gain, t_sample, Sim_T)
    keep_const = false
    shock_enter = 1
    θ_t′, θ_t = [ones(Sim_T) for _ in 1:2]
    θ_t′[shock_enter:end] = exp.([para.σ_ϵ * para.ρ ^ (t - shock_enter) for t in shock_enter:Sim_T])
    ## Case with initialization of heterogeneous beliefs
    simul_irf(para, θ_t, gain, t_sample, "post", keep_const)
    simul_irf(para, θ_t, gain, t_sample, "pre", keep_const)
end



## get irf from the saved data
function get_irf(gain, t_sample, Sim_T)
    c_post, n_post, a_post, ν_post, r_post, w_post, θ_post = [zeros(Sim_T) for i in 1:7]
    c_pre, n_pre, a_pre, ν_pre, r_pre, w_pre, θ_pre = [zeros(Sim_T) for i in 1:7]
    ψ_post = zeros(Sim_T, 3)
    ψ_pre = zeros(Sim_T, 3)
    for t in 1:Sim_T
        c_post[t] = readdlm("../data/irf_$gain/post_shock/mean_c/$t.csv", ',')[1]
        n_post[t] = readdlm("../data/irf_$gain/post_shock/mean_n/$t.csv", ',')[1]
        a_post[t] = readdlm("../data/irf_$gain/post_shock/mean_a/$t.csv", ',')[1]
        ν_post[t] = readdlm("../data/irf_$gain/post_shock/mean_nu/$t.csv", ',')[1]
        r_post[t] = readdlm("../data/irf_$gain/post_shock/r/$t.csv", ',')[1]
        w_post[t] = readdlm("../data/irf_$gain/post_shock/w/$t.csv", ',')[1]
        ψ_post[t, :] = readdlm("../data/irf_$gain/post_shock/mean_psi/$t.csv", ',')
        θ_post[t] = readdlm("../data/irf_$gain/post_shock/theta/$t.csv", ',')[1]
        ############################################################################
        c_pre[t] = readdlm("../data/irf_$gain/pre_shock/mean_c/$t.csv", ',')[1]
        n_pre[t] = readdlm("../data/irf_$gain/pre_shock/mean_n/$t.csv", ',')[1]
        a_pre[t] = readdlm("../data/irf_$gain/pre_shock/mean_a/$t.csv", ',')[1]
        ν_pre[t] = readdlm("../data/irf_$gain/pre_shock/mean_nu/$t.csv", ',')[1]
        r_pre[t] = readdlm("../data/irf_$gain/pre_shock/r/$t.csv", ',')[1]
        w_pre[t] = readdlm("../data/irf_$gain/pre_shock/w/$t.csv", ',')[1]
        ψ_pre[t, :] = readdlm("../data/irf_$gain/pre_shock/mean_psi/$t.csv", ',')
        θ_pre[t] = readdlm("../data/irf_$gain/pre_shock/theta/$t.csv", ',')[1]
    end
    c_irf = c_post - c_pre
    n_irf = n_post - n_pre
    a_irf = a_post - a_pre
    ν_irf = ν_post - ν_pre
    r_irf = r_post - r_pre
    w_irf = w_post - w_pre
    ψ_irf = ψ_post - ψ_pre
    θ_irf = θ_post - θ_pre
    writedlm("../data/irf_$gain/$t_sample/irf/c.csv", c_irf, ',')
    writedlm("../data/irf_$gain/$t_sample/irf/n.csv", n_irf, ',')
    writedlm("../data/irf_$gain/$t_sample/irf/a.csv", a_irf, ',')
    writedlm("../data/irf_$gain/$t_sample/irf/nu.csv", ν_irf, ',')
    writedlm("../data/irf_$gain/$t_sample/irf/r.csv", r_irf, ',')
    writedlm("../data/irf_$gain/$t_sample/irf/w.csv", w_irf, ',')
    writedlm("../data/irf_$gain/$t_sample/irf/psi.csv", ψ_irf, ',')
    writedlm("../data/irf_$gain/$t_sample/irf/theta.csv", θ_irf, ',')
end



## Plot the impulse response functions
function plot_IRF(irf_het, irf_hom_update, irf_hom_const, titles, plotsize)
    pyplot()
    p1 = plot(layout = (3, 3), title = titles, size = plotsize, grid = false)
    for indx in 1:size(irf_het, 2)
        plot!(p1, irf_het[:, indx], label = "", subplot = indx)
        plot!(p1, irf_hom_update[:, indx], label = "", subplot = indx)
        plot!(p1, irf_hom_const[:, indx], label = "", subplot = indx)
    end
    p2 = plot(layout = (3, 3), title = titles, size = plotsize, grid = false)
    for indx in 1:size(irf_het, 2)
        plot!(p2, (irf_het[:, indx] - irf_hom_update[:, indx]), label = "", subplot = indx)
        plot!(p2, (irf_het[:, indx] - irf_hom_const[:, indx]), label = "", subplot = indx)
    end
    return p1, p2
end



## Save the impulse response functions as csv files
function write_irf(irf_het, irf_hom_update, irf_hom_const, irfnames)
    irfs = [irf_het, irf_hom_update, irf_hom_const]
    for i in 1:length(irfnames)
        writedlm("../data/$(irfnames[i]).csv", irfs[i], ',')
    end
end
####################################################################################################



## Housekeeping
s = ArgParseSettings()
@add_arg_table s begin
    "i"
        arg_type = Int
        required = true
        help = "index"
end
ps = parse_args(s)
indx = ps["i"]



para = HAmodel()
para, π, k, ϵn_grid, n_grid, a_grid = calibrate_stationary(para)
para.T = 10_000
para.agent_num = 100_000
str = ""
if indx == 1
    str = "from_zeros/gain_0.005"
    para.ψ̄ = zeros(3)
    para.γ_gain = t -> 0.005
elseif indx == 2
    str = "from_zeros/gain_0.01"
    para.ψ̄ = zeros(3)
    para.γ_gain = t -> 0.01
elseif indx == 3
    str = "from_RA/gain_0.005"
    para.ψ̄ = [-0.00131466; -0.765091; -0.655608]
    para.γ_gain = t -> 0.005
elseif indx == 4
    str = "from_RA/gain_0.01"
    para.ψ̄ = [-0.00131466; -0.765091; -0.655608]
    para.γ_gain = t -> 0.01
elseif indx == 5
    str = "from_HA/gain_0.005"
    para.ψ̄ = [6.32e-07; -0.618232182; -0.852232561]
    para.γ_gain = t -> 0.005
elseif indx == 6
    str = "from_HA/gain_0.01"
    para.ψ̄ = [6.32e-07; -0.618232182; -0.852232561]
    para.γ_gain = t -> 0.01
end
simul_learning(para, π, str)



#= Save the distribution of individual states from the last 20% time periods
filenames = ["asset", "indi_prod", "belief1", "belief2", "belief3"]
save_post_time_paths(para, π, filenames; perc = 20)
=#



#= Save the IRFs to csv files
irf_het, irf_hom_update, irf_hom_const = IRFs(para, filenames)
irfnames = ["irf_het", "irf_hom_update", "irf_hom_const"]
write_irf(irf_het, irf_hom_update, irf_hom_const, irfnames)
=#



#= Read the IRFs from the csv files
irf_het = readdlm("../data/irf_het.csv", ',')
irf_hom_update = readdlm("../data/irf_hom_update.csv", ',')
irf_hom_const = readdlm("../data/irf_hom_const.csv", ',')
=#



#= Plot the IRFs
titles = hcat("Consumption", "Labor", "Asset",
              "Belief 1", "Belief 2", "Belief3",
              "R-Weighted MU", "Interest Rate", "Wage Rate")
p1, p2 = plot_IRF(irf_het, irf_hom_update, irf_hom_const, titles, (1500, 1000))
savefig(p1, "../figures/irfs.pdf")
savefig(p2, "../figures/diffs.pdf")
=#



#=
psi = readdlm("../data/zeros_0.005/mean_psi/combined.csv", ',')
plot(psi, label = "", title = "Gain 0.005", xlabel = "Time", ylabel = "Belief Coefficient")
plot!(ones(size(psi, 1), 1) .* [3.866160387819722e-5 -0.5747956126764134 -0.4432431327901116], label = "", ls = :dash)
savefig("../figures/zeros/long_simulation_0.005.pdf")
=#



#=
s = ArgParseSettings()
@add_arg_table s begin
    "i"
        arg_type = Int
        required = true
        help = "index"
end
ps = parse_args(s)
indx = ps["i"]



srand(1)
samples = sort(sample(5001:10000, 500, replace = false))
para = HAmodel()
para.agent_num = 100_000
Sim_T = 150
gain = ""
t_sample = 1
if indx <= 500
    gain = "0.005"
    t_sample = samples[indx]
    para.γ_gain = t -> .005
else
    gain = "0.01"
    t_sample = samples[indx - 500]
    para.γ_gain = t -> .01
end
IRFs(para, gain, t_sample, Sim_T)
get_irf(gain, t_sample, Sim_T)
=#



#=
delete_periods = setdiff(5001:10000, samples)
for period in delete_periods
    rm("../data/irf_0.005/$period"; recursive = true)
    rm("../data/irf_0.01/$period"; recursive = true)
end
=#

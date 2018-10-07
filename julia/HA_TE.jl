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
      res = nlsolve(f, [log(1.)]; inplace = false)
      para.c_min2[s] = exp.(res.zero[1])
  end
end



## Given r, w and the steady state consumption function
## Use 2D-interpolation to approximate the consumption function cf2[s](a, ϕ)
## The grid for the 2D-interpolation is
## a_vec =  linspace(a_min, a_max, N) and ϕ_vec
function get_cf2(para, cf, r, w)
    n_con = 10
    compute_cmin2!(para, r, w, cf)
    @unpack σ, γ, β, χ, r̄, w̄, P, A, S, c_min2, k_spline = para
    @unpack a_min, a_max, Na, ϕ_min, ϕ_max, N_ϕ, lear_path = para
    ϕ_vec = linspace(ϕ_min, ϕ_max, N_ϕ)
    cf2 = Spline2D[]
    a′grid = construct_agrid(a_min,a_max,Na)

    N_a = length(a′grid)
    #preallocate for speed
    a_grid = zeros(S, N_ϕ, N_a + n_con)
    c_grid = zeros(S, N_ϕ, N_a + n_con)
    Uc′ = zeros(S)
    for (i_a, a′) in enumerate(a′grid)
        for s′ in 1:S
            Uc′[s′] = (cf[s′](a′)) ^ (-σ)
        end
        for (i_ϕ, ϕ) in enumerate(ϕ_vec)
            for s in 1:S

                c = (exp(ϕ) * β * (1 + r̄) * dot(P[s,:], Uc′) ) ^ (-1 / σ)
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
                start_i = find(a_grid[s, i_ϕ, :] .< a_min)[end]
                cvec[:, i_ϕ] .= Spline1D(a_grid[s, i_ϕ, start_i:end], c_grid[s , i_ϕ, start_i:end]; k = k_spline)(a′grid)
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
  @unpack agent_num, P, seed, T = para
  s′ = zeros(Int64, agent_num)
  srand((seed - 1) * T + t)
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
function update_R(R, x, t, γ_gain)
    R′ = zeros(3, 3)
    R′ = R + γ_gain(t) .* (x * x' - R)
    @assert det(R′) != 0 "det(R′) = 0!"
    return R′
end



## Update the beliefs for all of the agents based on least square learning
function update_ψ(ψ, R′, x, ν, t, γ_gain, agent_num, ν̄c)
    ψ′ = zeros(agent_num, 3)
    for i in 1:agent_num
        ψ′[i, :] = ψ[i, :] + γ_gain(t) .* inv(R′) * x .* (log(ν[i] / ν̄c[i]) - ψ[i,:]' * x)[1]
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

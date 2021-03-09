## Let c̄(a, s) be the consumption function in the steady state
# Let ψ be the beliefs of the agents and x be the aggregate states
# Define ϕ = ψ'⋅x. For given r, we want to find
# c(a, ϕ, s), a′(a, ϕ, s), and n(a, ϕ, s) that solve
# 1. c(a, ϕ, s)^{-σ} ≥ ∑_{s'} π(s′|s)⋅β⋅(1 + r)⋅c̄(a′(a, ϕ, s), s′)^{-σ}⋅exp(ϕ)
# 2. c(a, ϕ, s) + a′(a, ϕ, s) = (1 + r)⋅a + A(s)⋅w⋅n(a, ϕ, s)
# 3. A(s)⋅w⋅c(a, ϕ, s)^{-σ} = χ⋅(1 - n(a, ϕ, s))^γ
# Given r, w, for each s
# compute the vector of consumptions chosen by the agent if a′ = a = a_min.
function compute_cmin2!(para, r, w, cf)
  @unpack σ, γ, χ, w̄, r̄, A, S, a_min = para
  for s in 1:S
      function f(c)
          n = get_n(w, A[s], c, σ, χ, γ)
          return c - A[s] * w * n - r * a_min
      end
      para.c_min2[s] = find_zero(f, (1e-100, 150.))
  end
end



## Given r, w and the steady state consumption function
# Use 2D-interpolation to approximate the consumption function cf2[s](a, ϕ)
# The grid for the 2D-interpolation is
# a_vec =  linspace(a_min, a_max, N) and ϕ_vec
function get_cf2(para, cf, r, w)
    n_con = 10
    compute_cmin2!(para, r, w, cf)
    @unpack σ, γ, β, χ, r̄, w̄, P, A, S, c_min2, k_spline = para
    @unpack a_min, a_max, Na, ϕ_min, ϕ_max, N_ϕ, lear_path = para
    ϕ_vec = LinRange(ϕ_min, ϕ_max, N_ϕ)
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
                n = get_n(w, A[s], c, σ, χ, γ)
                a = (a′ + c - A[s] * w * n) / (1 + r)
                a_grid[s, i_ϕ, i_a + n_con] = a
                c_grid[s, i_ϕ, i_a + n_con] = c
            end
        end
    end

    for s in 1:S
        for (i_ϕ, ϕ) in enumerate(ϕ_vec)
            if a_grid[s, i_ϕ, n_con + 1] > a_min
                for (i_c, ĉ) in enumerate(LinRange(c_min2[s], c_grid[s, i_ϕ, n_con + 1], n_con + 1)[1:n_con])
                    n̂ = get_n(w, A[s], ĉ, σ, χ, γ)
                    â = (a_min + ĉ - A[s] * w * n̂) / (1 + r)
                    a_grid[s, i_ϕ, i_c] = â
                    c_grid[s, i_ϕ, i_c] = ĉ
                end
            else
                a_grid[s, i_ϕ, 1:n_con] .= -Inf
                c_grid[s, i_ϕ, 1:n_con] .= -Inf
            end
        end
    end
    #Now interpolate
    cf2 = Array{Spline2D}(undef, S)
    for s in 1:S
        c_vec = zeros(N_a, N_ϕ)
        for i_ϕ in 1:N_ϕ
            #If the constraint never binds don't need extra grid points
            if c_grid[s, i_ϕ, 1] == -Inf
                start_i = findall(a_grid[s, i_ϕ, :] .< a_min)[end]
                ## c_vec[:, i_ϕ] .= Spline1D(a_grid[s, i_ϕ, start_i:end], c_grid[s , i_ϕ, start_i:end]; k = k_spline)(a′grid)
                try
                    c_vec[:, i_ϕ] .= Spline1D(a_grid[s, i_ϕ, start_i:end], c_grid[s , i_ϕ, start_i:end]; k = k_spline)(a′grid)
                catch err
                    if isa(err, LoadError)
                        sort_indx = sortperm(a_grid[s, i_ϕ, start_i:end])
                        x_vec = (a_grid[s, i_ϕ, start_i:end])[sort_indx]
                        y_vec = (c_grid[s , i_ϕ, start_i:end])[sort_indx]
                        c_vec[:, i_ϕ] .= Spline1D(x_vec, y_vec; k = k_spline)(a′grid)
                        output = vcat([r w], [s ϕ_vec[i_ϕ]], hcat(a_grid[s, i_ϕ, start_i:end], c_grid[s , i_ϕ, start_i:end]))
                        writedlm("../data/$(lear_path)/spline/$(now()).csv", output, ',')
                    end
                end
            #If the constraint binds binds need to use all the grid points
            else
                ## c_vec[:, i_ϕ] .= Spline1D(a_grid[s, i_ϕ, :], c_grid[s, i_ϕ, :]; k = k_spline)(a′grid)
                try
                    c_vec[:, i_ϕ] .= Spline1D(a_grid[s, i_ϕ, :], c_grid[s, i_ϕ, :]; k = k_spline)(a′grid)
                catch err
                    if isa(err, LoadError)
                        c_vec[:, i_ϕ] .= Spline1D(sort(a_grid[s, i_ϕ, :]), sort(c_grid[s, i_ϕ, :]); k = k_spline)(a′grid)
                    end
                end
            end
        end
        cf2[s] = Spline2D(a′grid, ϕ_vec, c_vec)
    end
    return cf2
end



## Compute the expected future marginal utility in steady state
# The expetation is taken over all future states
function ν̄_f(ai, si, cf, para)
    @unpack A, σ, χ, γ, P, S, r̄, w̄ = para
    ci = cf[si](ai)
    ni = get_n(w̄, A[si], ci, σ, χ, γ)
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
# Note that solving for agent's labor supply with endogenous grid cf2
function indi_labor_supply(para, r, w, ai, si, ϕi, cf2)
    @unpack σ, γ, β, χ, A, S, P, a_min,a_max = para
    ci = cf2[si](ai, ϕi)
    ni = get_n(w, A[si], ci, σ, χ, γ)
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
# ∑(A[si] ⋅ ni) / N
function avg_labor_supply(para, r, w, a, s, ϕ, cf2)
    @unpack agent_num = para
    N = 0.
    for i in 1:agent_num
        N = N + indi_labor_supply(para, r, w, a[i], s[i], ϕ[i], cf2)[1]
    end
    return N::Float64 / agent_num
end




## Projection Method:
# Given the the prices r and w, and individual states a, s,
# iterate through all the agents, return individual agent decisions
# and return expected future marginal utility ν̄
# and current marginal utility in steady state ν̄c
function iterate_agents(A, s, a, ϕ, ĉ_proj, n̂_proj, â′_proj, cf)
    c, n, a′, ν̄, ν̄c = [zeros(length(s)) for _ in 1:5]
    @time for i in 1:length(s)
        si, ai, ϕi = s[i], a[i], ϕ[i]
        c[i] = ĉ_proj[si]([ai, ϕi])
        n[i] = n̂_proj[si]([ai, ϕi]) / A[s[i]]
        a′[i] = â′_proj[si]([ai, ϕi])
        ν̄[i] = ν̄_f(ai, si, cf, para)
        ν̄c[i] = ν̄c_f(ai, si, cf, para)
    end
    return c, n, a′, ν̄, ν̄c
end




## Old Method:
# Given the the prices r and w, and individual states a, s,
# iterate through all the agents, return individual agent decisions
# and return expected future marginal utility ν̄
# and current marginal utility in steady state ν̄c
function iterate_agents(para, r, w, a, s, ϕ, cf, cf2)
    @unpack σ, γ, β, χ, A = para
    c, n, a′, ν̄, ν̄c = [zeros(length(s)) for _ in 1:5]
    for i in 1:length(s)
        ai, si, ϕi = a[i], s[i], ϕ[i]
        ci, ni, ai′ = indi_labor_supply(para, r, w, ai, si, ϕi, cf2)[2:end]
        ν̄i = ν̄_f(ai, si, cf, para)
        ν̄ci = ν̄c_f(ai, si, cf, para)
        c[i], n[i], a′[i], ν̄[i], ν̄c[i] = ci, ni, ai′, ν̄i, ν̄ci
    end
    return c, n, a′, ν̄, ν̄c
end




function iterate_agents(para, r, w, a, s, ϕ, cf1, cf2)
    @unpack σ, γ, β, χ, A, agent_num, S, a_min, a_max = para
    c, n, a′, ν̄, ν̄c = [zeros(agent_num) for _ in 1:5]
    for j in 1:S
      mask = (j.== s)
      c[mask] .= [cf2[j](a[indx], ϕ[indx]) for indx in findall(mask)]
    end
    for i in 1:agent_num
        ai, si, ci = a[i], s[i], c[i]
        n[i] = get_n(w, A[si], ci, σ, χ, γ)
        a′[i] = (1 + r) * ai + A[si] * w * n[i] - ci
        if a′[i] < a_min
          Δ = a_min - a′[i]
          c[i] -= Δ
          a′[i] = a_min
        end
        if a′[i] > a_max
          Δ = a_max - a′[i]
          c[i] -= Δ
          a′[i] = a_max
        end
        ν̄[i] = ν̄_f(ai, si, cf1, para)
        ν̄c[i] = ν̄c_f(ai, si, cf1, para)
    end
    return c, n, a′, ν̄, ν̄c
end



## Define F function for rootsolving
function F_old(lnw, α, θ, δ, para, a, s, ϕ, cf)
    w = exp.(lnw[1])
    k = (w / ((1 - α) * θ)) ^ (1 / α)
    r = α * θ * (k) ^ (α - 1) - δ
    cf2 = get_cf2(para, cf, r, w)
    N = avg_labor_supply(para, r, w, a, s, ϕ, cf2)
    K = mean(a)
    #println("K = $K, N = $N")
    w_diff = w - (1 - α) * θ * (K / N) ^ α
    return w_diff
end



## Define F function for rootsolving
function F(lnw, α, θ, δ, a, s, N̂_func)
    w = exp.(lnw[1])
    k = (w / ((1 - α) * θ)) ^ (1 / α)
    r = α * θ * (k) ^ (α - 1) - δ
    N = N̂_func([r, w])
    K = mean(a)
    if K / N < 0.
        return 10.
    end
    w_diff = w - (1 - α) * θ * (K / N) ^ α
    return w_diff
end



## Given individual states distribution (a, s, ψ) and information x,
# compute the temporary equilibrium TE
# return :
# 1. inididual agents deicisions c, n, a′
# 2. current marginal utility ν,
# 3. expected future marginal utility ν̄,
# 4. current marginal utility in steady state ν̄c
# 5. prices r, w
function TE_old(para, a, s, ψ, x, θ, cf)
  @unpack σ, β, α, δ, P, A, agent_num, w̄ = para
  ϕ = compute_ϕ(ψ, x)
  lnw = nlsolve(lnw -> F_old(lnw, α, θ, δ, para, a, s, ϕ, cf), [log(w̄)]; inplace = false, iterations = 100_000).zero::Vector{Float64}
  w = exp.(lnw[1])
  k = (w / ((1 - α) * θ)) ^ (1 / α)
  r = α * θ * (k) ^ (α - 1) - δ
  cf2 = get_cf2(para, cf, r, w)
  c, n, a′, ν̄, ν̄c = iterate_agents(para, r, w, a, s, ϕ, cf, cf2)
  ν = (1. + r) * c .^ (-σ)
  return c, n, a′, ν, ν̄, ν̄c, r, w
end



function TE_oldtest(para, a, s, ψ, x, θ, cf, n̂_func, ĉ_func, â′_func)
  @unpack σ, β, α, δ, P, A, agent_num, w̄ = para
  ϕ = compute_ϕ(ψ, x)
  #First solve with new method
  N̂_func = integrateFunction(n̂_func, a, ϕ, s)
  ## Compute the Temporary Equilibrium w and r
  lnw = nlsolve(lnw -> F(lnw, α, θ, δ, a, s, N̂_func), [log(w̄)]; inplace = false).zero::Vector{Float64}
  w = exp.(lnw[1])
  k = (w / ((1 - α) * θ)) ^ (1 / α)
  r = α * θ * (k) ^ (α - 1) - δ
  cf2 = get_cf2(para, cf, r, w)
  c, n, a′, ν̄, ν̄c = iterate_agents(para, r, w, a, s, ϕ, cf, cf2)
  ν = (1. + r) * c .^ (-σ)
  #store variables to be saved
  ν̂,r̂,ŵ = ν,r,w
  ## Compute Marginal Utility
  #now solve with the old method
  lnw = nlsolve(lnw -> F_old(lnw, α, θ, δ, para, a, s, ϕ, cf), [log(w̄)]; inplace = false, iterations = 100_000).zero::Vector{Float64}
  w = exp.(lnw[1])
  k = (w / ((1 - α) * θ)) ^ (1 / α)
  r = α * θ * (k) ^ (α - 1) - δ
  cf2 = get_cf2(para, cf, r, w)
  c, n, a′, ν̄, ν̄c = iterate_agents(para, r, w, a, s, ϕ, cf, cf2)
  ν = (1. + r) * c .^ (-σ)
  return c, n, a′, ν, ν̄, ν̄c, r, w, ν̂, r̂, ŵ
end



## Given individual states distribution (a, s, ψ) and information x,
# compute the temporary equilibrium TE
# return :
# 1. inididual agents deicisions c, n, a′
# 2. current marginal utility ν,
# 3. expected future marginal utility ν̄,
# 4. current marginal utility in steady state ν̄c
# 5. prices r, w
function TE(para, θ, a, s, ϕ, cf1, n̂_func, ĉ_func, â′_func)
  @unpack σ, β, α, δ, P, A, agent_num, w̄ = para
  N̂_func = integrateFunction(n̂_func, a, ϕ, s)
  ## Compute the Temporary Equilibrium w and r
  lnw = nlsolve(lnw -> F(lnw, α, θ, δ, a, s, N̂_func), [log(w̄)]; inplace = false).zero::Vector{Float64}
  w = exp.(lnw[1])
  # k: capital-hour-ratio
  k = (w / ((1 - α) * θ)) ^ (1 / α)
  r = α * θ * (k) ^ (α - 1) - δ
  ## Old method
  cf2 = get_cf2(para, cf1, r, w)
  c, n, a′, ν̄, ν̄c = iterate_agents(para, r, w, a, s, ϕ, cf1, cf2)
  ## Compute Marginal Utility
  ν = (1. + r) * c .^ (-σ)
  return c, n, a′, ν, ν̄, ν̄c, r, w
end



## Update the individual productivity state
function update_s(para::HAmodel, s, t)
  @unpack agent_num, P, T = para
  s′ = zeros(Int64, agent_num)
  #srand(t)
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

function update_R_expanded(agent_num, R, x, t, γ_gain)
    R′ = zeros(agent_num, 7, 7)
    for i in 1:agent_num
        R′[i, :, :] = R[i, :, :] + γ_gain(t) .* (x[i, :] * x[i, :]' - R[i, :, :])
    end
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


function update_ψ_expanded(ψ, R′, x, ν, t, γ_gain, agent_num, ν̄c)
    ψ′ = zeros(agent_num, 7)
    for i in 1:agent_num
        ψ′[i, :] = ψ[i, :] + γ_gain(t) .* inv(R′[i, :, :]) * x[i, :] .* (log(ν[i] / ν̄c[i]) - ψ[i, :]' * x[i, :])[1]
    end
    return ψ′
end



function compute_ϕ(ψ, x)
    N = size(ψ, 1)
    ϕ = zeros(N)

    for i in 1:N

        if length(x) == size(ψ, 2)
            ϕ[i] = dot(ψ[i, :], x)
        else
            ϕ[i] = dot(ψ[i, :], x[i, :])
        end

        if ϕ[i] < -5.0
            ϕ[i] = -5.0
            println("ϕ < -5.0")
        elseif ϕ[i] > 5.0
            ϕ[i] = 5.0
            println("ϕ > 5.0")
        end

    end

    return ϕ
end




## Define the function that sets the range of ϕ = ψ ⋅ x
function set_ϕ_range!(para, ϕ)
    ## Find the min and max of ϕ = x ⋅ ψ
    ϕ_min = minimum(ϕ)
    ϕ_max = maximum(ϕ)
    if ϕ_min == ϕ_max
        ϕ_min = ϕ_min - 0.001
        ϕ_max = ϕ_max + 0.001
    end
    para.ϕ_min = ϕ_min
    para.ϕ_max = ϕ_max
end




## Interpolation Method for Aggregating the Labor Supply
function compute_indi_grids(cf, para)
    @unpack a_min, a_max, ϕ_min, ϕ_max, S, r̄, w̄, Na = para
    para.ϕ_min = -5.0
    para.ϕ_max = 5.0
    avec = construct_agrid(a_min, a_max, Na)
    abasis = SplineParams(avec, 0, 3)
    ϕbasis = ChebParams(30, para.ϕ_min, para.ϕ_max)
    rbasis = ChebParams(6, 0.5r̄, 2r̄)
    wbasis = ChebParams(6, 0.5w̄, 1.5w̄)
    basis = Basis(abasis, ϕbasis, rbasis, wbasis)
    X = nodes(basis)[1]
    N_nodes = size(X, 1)
    nbasis_vec = zeros(N_nodes, S)
    cbasis_vec = zeros(N_nodes, S)
    a′basis_vec = zeros(N_nodes, S)
    cf2_dict = Dict{Tuple{Float64,Float64},Vector{Spline2D}}()
    for si in 1:S
        for i in 1:N_nodes
            ai, ϕi, r, w = X[i, :]
            if !haskey(cf2_dict,(r,w))
                cf2_dict[r,w] = get_cf2(para, cf, r, w)
            end
            nbasis_vec[i, si], cbasis_vec[i, si], a′basis_vec[i, si] = indi_labor_supply(para, r, w, ai, si, ϕi, cf2_dict[r,w])[[1,2,4]]
        end
    end
    n̂_func = [Interpoland(basis, nbasis_vec[:, s]) for s in 1:S]
    ĉ_func = [Interpoland(basis, cbasis_vec[:, s]) for s in 1:S]
    â′_func = [Interpoland(basis, a′basis_vec[:, s]) for s in 1:S]
    return n̂_func, ĉ_func, â′_func
end



## Function: integrateFunction(fhat, a, ϕ, s, [ω])
# Integrates the first two dimensions of fhat over (a, ϕ), and
# returns a function of the last two dimensions over prices (r, w)
function integrateFunction(f̂, a::Vector{Float64}, ϕ::Vector{Float64},
    s::Vector{Int}, ω::Vector{Float64}= ones(length(a)) / length(a))
    S = length(f̂)
    basis = f̂[1].basis #We assume that all elements of F have the same basis

    #Construct basis over variables we will integrate over
    basisΩ = Basis(basis.params[1], basis.params[2])
    basisrw = Basis(basis.params[3], basis.params[4])

    Nrw = size(nodes(basisrw)[1])[1] #get the number of coefficients needed to
    chat = zeros(Nrw)
    #now integrate over all agents
    for j in 1:S
        mask = s.==j
        Ω̂ = hcat(a[mask], ϕ[mask])
        ω̂ = ω[mask]
        Φ1Φ2basis  = BasisMatrix(basisΩ, Expanded(), Ω̂).vals[1]
        Φ1Φ2 = Φ1Φ2basis' * ω̂
        NΦ1Φ2 = length(Φ1Φ2)

        c = f̂[j].coefs
        for i in 1:Nrw
            chat[i] += dot(c[1 + (i - 1) * NΦ1Φ2:i * NΦ1Φ2], Φ1Φ2)
        end
    end

    bsrw = BasisMatrix(basisrw, Tensor())
    return Interpoland(basisrw, chat, bsrw)
end



## projectFunction(fhat, r, w)
# Integrates the last two dimensions of fhat over (r, w), and
# returns a function of the first two dimensions over prices (a, ϕ)
# Basically: Given r and w, projects the function down to a function of a and ϕ
function projectFunction(f̂, r, w)
    S = length(f̂)
    basis = f̂[1].basis #We assume that all elements of F have the same basis

    #Construct basis over variables we will integrate over
    basisΩ = Basis(basis.params[1], basis.params[2])
    basisrw = Basis(basis.params[3], basis.params[4])
    bsΩ = BasisMatrix(basisΩ, Tensor())

    NΩ = size(nodes(basisΩ)[1])[1]
    Nrw = size(nodes(basisrw)[1])[1]

    f̂_proj = Vector{typeof(Interpoland(basisΩ,zeros(NΩ),bsΩ))}()
    for s in 1:S
        c = f̂[s].coefs
        chat = zeros(NΩ)
        Φrw = BasisMatrix(basisrw, Expanded(), [r, w]).vals[1]
        for i in 1:Nrw
            chat .+= c[1 + (i - 1) * NΩ:i * NΩ] .* Φrw[i]
        end
        push!(f̂_proj, Interpoland(basisΩ, chat, bsΩ))
    end
    return f̂_proj
end



#=
## Testing if the compute_indi_grids function works well
n̂_func, ĉ_func, â′_func = compute_indi_grids(cf, para)
r = 0.7para.r̄
w = 0.9para.w̄
n = zeros(para.agent_num)
n_old = zeros(para.agent_num)
cf2 = get_cf2(para, cf, r, w)
for indx in 1:para.agent_num
    si = s[indx]
    ai = a[indx]
    ϕi = dot(ψ[indx, :], x)
    n[indx] = n̂_func[si]([ai, ϕi, r, w])
    n_old[indx] = indi_labor_supply(para, r, w, ai, si, ϕi, cf2)[1]
end
labor_old = mean(n_old)
labor_new = mean(n)
=#

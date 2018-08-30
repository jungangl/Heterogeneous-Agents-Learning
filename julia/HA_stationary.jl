using QuantEcon, Parameters, CSV, StatsBase
using NLsolve, Dierckx, Plots, Distributions, ArgParse
using FastGaussQuadrature

function computeProductivityProcess(ρ_p,σ_p,σ_e,Np,Nt)
    """
    Computes productivity process by approximating AR(1) + iid shock
    """
    mc = rouwenhorst(Np, ρ_p, σ_p) #From
    P1 = mc.p
    e1 = mc.state_values

    nodes,weights = gausshermite(Nt)

    P2 = repmat(weights'/sqrt(π),Nt) #adjust weights by sqrt(π)
    e2 = sqrt(2)*σ_e*nodes

    P = kron(P1,P2) #kron combines matrixies multiplicatively
    e = kron(e1,ones(Nt)) + kron(ones(Np),e2) # e is log productivity
    return MarkovChain(P,e)
end

function construct_agrid(a_min,a_max,Na,curv=1.7)
    """
    Computes agrid with Na points with more mass near borrowing constraint.
    """
    a_grid = zeros(Na)
    a_grid[1] = a_min
    for i in 2:Na
        a_grid[i] = a_min  + (a_max-a_min)*((i-1.0)/(Na-1.0)) ^ curv
    end
    return a_grid
end


@with_kw type HAmodel
    ## Fundamental paramters
    σ::Float64 = 2.
    γ::Float64 = 2.
    β::Float64 = 0.9817232017253373#0.9819149759880317
    ρ::Float64 = 0.95
    σ_ϵ::Float64 = 0.007
    K2Y::Float64 = 10.26
    α::Float64 = 0.36
    δ::Float64 = 0.025
    χ::Float64 = 1.0933191294158184#1.293514964597886
    γ_gain::Function  = t -> 0.02
    ## Steady state values
    ā::Float64 = 14.16447244048578
    r̄::Float64 = α * inv(K2Y) - δ
    w̄::Float64 = (1 - α) * (K2Y) ^ (α / (1 - α))
    n̄::Float64 = 1/3
    ## MarkovChain for the state variable s
    mc::MarkovChain = computeProductivityProcess(0.9923,0.0983,sqrt(0.053),7,3)#rouwenhorst(11, 0.9923, 0.0983)
    P::Matrix{Float64} = mc.p
    A::Vector{Float64} = exp.(mc.state_values)
    S::Int64 = length(A)
    N_ϕ::Int64 = 50
    ## Environment variables
    a_min::Float64 = 0.
    a_max::Float64 = 300.
    Na::Int64 = 150 #number of asset grid points for spline
    ϕ_min::Float64 = 0.
    ϕ_max::Float64 = 0.
    N::Int64 = 1500
    k_spline::Int64 = 3
    c_min::Vector{Float64} = similar(A)
    c_min2::Vector{Float64} = similar(A)
    ## Simulation paramters
    T::Int64 = 150
    agent_num::Int64 = 10000
    ψ̄::Vector{Float64} = [ -0.001314661;  -0.765090668;   -0.655607579]
    R̄::Matrix{Float64} = [ 1.0000000000   -0.000916309    -0.000362956;
                          -0.000916309     0.00120064      0.000473709;
                          -0.000362956     0.000473709     0.000482302]
    path::String = "simulations/from_zeros/gain_0.005"
end



#------------------------------------------------------------------------#
## Let r̄ and w̄ be the prices in the steady state
## We want to find the consumption function in the steady state  - cf(a, s)
## We need to find c̄(a, s), a′(a, s), and n(a, s) that solve
## 1. c̄(a, s)^{-σ} ≥ β⋅(1 + r̄)⋅(∑_{s'} π(s′|s)⋅c̄(a′(a, s), s′)^{-σ})
## 2. c̄(a, s) + a′(a, s) = (1 + r̄)⋅a + A(s)⋅w̄⋅n(a, s)
## 3. A(s)⋅w̄⋅c̄(a, s)^{-σ} = χ⋅(1 - n(a, s))^γ
## The fixed point for c̄(a, s) is the consumption function in the steady state
#------------------------------------------------------------------------#



## For each s, compute the vector of consumptions
## chosen by the agent if a′ = a = a_min.
function compute_cmin!(para)
  @unpack σ, γ, χ, w̄, r̄, A, S, a_min = para
  for s in 1:S
    function f(logc)
      c = exp.(logc)
      n = 1 - ((w̄ * A[s] * c .^ (-σ)) / χ) .^ (-1 / γ)
      return c - A[s] * w̄ * n - r̄ * a_min
    end
    res = nlsolve(f, [0.]; inplace = false)
    para.c_min[s] = exp.(res.zero[1])
  end
end



## Given the previous cf (consumption function), approximate the new cf.
## Use interpolation to approximate the consumption function
## n_con is the number of grid points to interpolate the consumption vector from compute_cmin!
function approximate_c(cf, a′grid::Vector, para::HAmodel)
    @unpack σ, γ, β, χ, r̄, w̄, P, A, S, a_min, k_spline, c_min = para
    n_con = 10
    N_a = length(a′grid)
    #preallocate for speed
    a_grid = zeros(S, n_con + N_a)
    c_grid = zeros(S, n_con + N_a)
    Uc′ = zeros(S)
    ## For each element in a′grid, compute the correspoding c level
    for (i_a′, a′) in enumerate(a′grid)
        for s′ in 1:S
            Uc′[s′] = (cf[s′](a′)) ^ (-σ)
        end
        for s in 1:S
            c = (β * (1 + r̄) * dot(P[s, :], Uc′)) .^ (-1 / σ)
            n = max(1 - ((w̄ * A[s] * c ^ (-σ)) / χ) ^ (-1 / γ), 0)
            a = (a′ + c - A[s] * w̄ * n) / (1 + r̄)
            a_grid[s, i_a′ + n_con] = a
            c_grid[s, i_a′ + n_con] = c
        end
    end
    for s in 1:S
        if a_grid[s, 1 + n_con] > a_min
            for (i_ĉ, ĉ) in enumerate(linspace(c_min[s], c_grid[s, n_con + 1], n_con + 1)[1:n_con])
                n̂ = max(1 - ((w̄ * A[s] * ĉ ^ (-σ)) / χ) ^ (-1 / γ), 0)
                â = (a_min + ĉ - A[s] * w̄ * n̂) / (1 + r̄)
                a_grid[s, i_ĉ] = â
                c_grid[s, i_ĉ] = ĉ
            end
        else
            a_grid[s, 1:n_con] = -Inf
            c_grid[s, 1:n_con] = -Inf
        end
    end
    #Now interpolate
    cf′ = Vector{Spline1D}(S)
    for s in 1:S
        if c_grid[s, 1] == -Inf
            indx = find(a_grid[s, :] .< a_min)[end]
            cf′[s] = Spline1D(a_grid[s, indx:end], c_grid[s, indx:end]; k = k_spline)
        #If the constraint binds, we need to use all the grid points
        else
            cf′[s] = Spline1D(a_grid[s, :], c_grid[s, :]; k = k_spline)
        end
    end
    return cf′
end



## Use a contraction mapping to find the converged consumption function
## Tolerance to be set a 1e-5
function solve_c(cf, a′grid::Vector, para::HAmodel, tol = 1e-5)
    @unpack S = para
    compute_cmin!(para)
    diff = 1.0
    cf′ = Vector{Spline1D}(S)
    diffs = zeros(S)
    while diff > tol
        cf′ = approximate_c(cf, a′grid, para)
        for s in 1:S
            diffs[s] = norm(cf′[s].(a′grid) - cf[s].(a′grid), Inf)
        end
        diff = maximum(diffs)
        cf = cf′
    end
    return cf
end



## Function that computes the fixed point for the consumption function
function get_cf(para)
    compute_cmin!(para)
    @unpack r̄, w̄, a_min, a_max,Na, S, A, k_spline = para
    ## Set more curvature when in the lower range of a′grid
    a′grid = construct_agrid(a_min,a_max,Na)
    N = length(a′grid)
    ## Initialize the consumption function
    a_mat = zeros(S, N)
    c_mat = zeros(S, N)
    cf = Vector{Spline1D}(S)
    for s in 1:S
        a_vec = collect(linspace(a_min, a_max, N))
        a_mat[s, :] = a_vec
        for (i_a, a) in enumerate(a_vec)
            c_mat[s, i_a] = A[s] * w̄ + r̄ * a
        end
    end
    for s in 1:S
        cf[s] = Spline1D(a_mat[s, :], c_mat[s, :]; k = k_spline)
    end
    ## Solve for the fixed point of the consumption function
    oldk = para.k_spline
    para.k_spline   = 1
    cf = solve_c(cf, a′grid, para)
    para.k_spline   = oldk
    cf = solve_c(cf, a′grid, para)
    return cf
end



## plot the policy functions
function plot_policies(para)
    @unpack σ, γ, β, χ, r̄, w̄, P, A, S, a_min, a_max, k_spline, c_min = para
    cf = get_cf(para)
    p_c = plot(grid = false, xlabel = "a", ylabel = "c", title = "consumption policy")
    for s in 1:para.S
        plot!(p_c, a -> cf[s](a), a_min, a_max, label = "s = $s")
    end
    p_n = plot(grid = false, xlabel = "a", ylabel = "n", title = "labor policy")
    for s in 1:para.S
        plot!(p_n, a -> max(1 - ((w̄ * A[s] * cf[s](a) ^ (-σ)) / χ) ^ (-1 / γ), 0), a_min, a_max, label = "s = $s")
    end
    p_aprime = plot(grid = false, xlabel = "a", ylabel = "aprime", title = "aprime policy")
    for s in 1:para.S
        plot!(p_aprime, a -> (1 + r̄) * a + A[s] * w̄ * max(1 - ((w̄ * A[s] * cf[s](a) ^ (-σ)) / χ) ^ (-1 / γ), 0) - cf[s](a),
              a_min, a_max, label = "s = $s")
    end
    return p_c, p_n, p_aprime
end



## Mapping from two-dimensional indexing to one-dimensional indexing
function dimtrans2to1(N::Int64, i::Int64, s::Int64)
    return (i + (s - 1) * (N + 2))
end



## Mapping from tone-dimensional indexing to two-dimensional indexing
function dimtrans1to2(N::Int64, k::Int)
    i = mod(k, N + 2)
    s = div(k, N + 2) + 1
    if i == 0
        i = N + 2
        s = s - 1
    end
    return i, s
end



## Create bins for asset holding
## Default is that
function get_bins(a_min::Float64, a_max::Float64, N::Int64)
    increment = (a_max - a_min) / N
    bin_midpts = [a_min]
    append!(bin_midpts, [a_min + (increment / 2) + (i - 1) * increment for i in 1:N])
    push!(bin_midpts, a_max)
    return bin_midpts
end



## Construct the transition matrix for the states
function construct_H(para::HAmodel, cf)
    @unpack σ, γ, χ, r̄, w̄, P, A, S, N, a_min, a_max = para
    bin_midpts = get_bins(a_min, a_max, N)
    ϵn_grid = zeros((N + 2) * S)
    n_grid = zeros((N + 2) * S)
    a_grid = zeros((N + 2) * S)
    H = spzeros((N + 2) * S, (N + 2) * S)
    for k in 1:(N + 2) * S
        ## Transition to i′ with prob ω
        ## Transition to i′ + 1 with prob 1 - ω
        i′ = 0
        ω = 0.
        # Given i, s
        i, s = dimtrans1to2(N, k)
        a = bin_midpts[i]
        c = cf[s](a)
        n = max(1 - (w̄ * A[s] * c ^ (-σ) / χ) ^ (-1 / γ), 0.0)
        a′ = A[s] * w̄ * n + (1 + r̄) * a - c
        ϵn_grid[k] = n * A[s]
        n_grid[k] = n
        a_grid[k] = a
        ## Check if a′ falls into the very first or very last bin
        if a′ <= 0.0
            i′ = 1
            ω = 1.0
        ## Check if a′ falls into the very last bin
        elseif a′ >= bin_midpts[N + 2]
            i′ = N + 1
            ω = 0.0
        else
            i′ = findfirst(a′ .<= bin_midpts) - 1
            ## calculate ω
            ω = (bin_midpts[i′ + 1] - a′) /
                (bin_midpts[i′ + 1] - bin_midpts[i′])
        end
        #transition to i′ with prob ω
        #transition to i′ + 1 with prob 1-ω
        #transition to sprime ∈ {1, ..., S} with the fowlling probabilities
        for (iprime, prob) in zip([i′ i′ + 1], [ω 1 - ω])
            for sprime in 1:S
                k′ = dimtrans2to1(N, iprime, sprime)
                H[k, k′] = prob * P[s, sprime]
            end
        end
    end
    return H, ϵn_grid, n_grid, a_grid
end



## Compute the stationary distribution from the transition matrix for the states
function stat_dist(para::HAmodel, k::Float64)
    tol = 1e-10
    #k as the KN ratio
    @unpack α, δ, S, N = para
    para.r̄ = α * k ^ (α - 1) - δ
    para.w̄ = (1 - α) * k ^ α
    cf = get_cf(para)
    H, ϵn_grid, n_grid, a_grid = construct_H(para, cf)
    π = ones(S * (N + 2)) / (S * (N + 2))
    diff = 1.0
    while diff > tol
        π_new = (π' * H)'
        diff = norm(π_new - π, Inf)
        π = π_new
        println(diff)
    end
    return π, ϵn_grid, n_grid, a_grid
end



## Construct the residual function for calibrating χ and β
function stationary_resid(x, α, K2Y, n̄, K2EN, para)
    para.β, para.χ = x
    π, ϵn_grid, n_grid, a_grid = stat_dist(para, K2EN)
    ϵn = dot(ϵn_grid, π)
    n = dot(n_grid, π)
    K2EN = dot(a_grid, π) / ϵn
    diff_K2Y = K2EN ^ (1 - α) - K2Y
    diff_n̄ = n - n̄
    println("diff_K2Y = $diff_K2Y, diff_n̄ = $diff_n̄")
    return diff_K2Y, diff_n̄, π, K2EN, ϵn_grid, n_grid, a_grid
end



## Calibrating for χ and β, targeting K2Y ratio and average working hours
function calibrate_stationary(para)
    @unpack α, K2Y, n̄ = para
    K2EN = (K2Y) ^ (1 / (1 - α))
    #res = nlsolve(x -> stationary_resid(x, α, K2Y, n̄, K2EN, para)[1:2], [para.β; para.χ]; inplace = false)
    #para.β, para.χ = res.zero
    diff_K2Y, diff_n̄, π, K2EN, ϵn_grid, n_grid, a_grid = stationary_resid([para.β, para.χ], α, K2Y, n̄, K2EN, para)
    return para, π, K2EN, ϵn_grid, n_grid, a_grid
end



## Plot the wealth distribution over a, make sure there is no bunching at a_max
function wealth_dist(para, π)
    agrid = collect(get_bins(para.a_min, para.a_max, para.N))
    πgrid = zeros(length(agrid))
    for k in 1:length(π)
        i, s = dimtrans1to2(para.N, k)
        πgrid[i] = πgrid[i] + π[k]
    end
    p1 = scatter(agrid[1:5], πgrid[1:5], label = "a", grid = false, title = "wealth distribtion for a in ($(agrid[1]), $(round(agrid[5], 2)))")
    p2 = scatter(agrid[6:end], πgrid[6:end], label = "a", title = "wealth distribution from a = $(agrid[6])", grid = false)
    writedlm("../data/HA_stationary/dist_over_a/a.csv", agrid, ',')
    writedlm("../data/HA_stationary/dist_over_a/pi.csv", πgrid, ',')
    savefig(p1, "../figures/HA_stationary/dist_over_a/wealth_low.pdf")
    savefig(p2, "../figures/HA_stationary/dist_over_a/wealth_high.pdf")
end


#=
para = HAmodel()
para, π, k, ϵn_grid, n_grid, a_grid = calibrate_stationary(HAmodel())
=#



#=
filenames = ["pi", "en", "n", "a"]
data = π, ϵn_grid, n_grid, a_grid
for (i, filename) in enumerate(filenames)
    writedlm("../data/HA_stationary/dist_over_a_s/$(filename).csv", data[i], ',')
end
=#



#=
wealth_dist(para, π)
p_c, p_n, p_aprime = plot_policies(para)
savefig(p_c, "../figures/HA_stationary/policies/c.pdf")
savefig(p_n, "../figures/HA_stationary/policies/n.pdf")
savefig(p_aprime, "../figures/HA_stationary/policies/aprime.pdf")
=#

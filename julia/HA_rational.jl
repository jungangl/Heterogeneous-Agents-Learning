@everywhere include("HA_stationary.jl")



function OLSestimator(y, x)
    estimate = inv(x'* x) * (x' * y)
    R = inv(size(x, 1)) * x' * x
    return estimate, R
end



@everywhere function compute_cmin(para, w)
  @unpack σ, γ, χ, A, S = para
  cmin = zeros(S)
  for s in 1:S
    function f(logc)
      c = exp(logc)
      n = 1 - ((w * A[s] * c .^ (-σ)) / χ) .^ (-1 / γ)
      return c - A[s] * w * n
    end
    res = nlsolve(f, [0.]; inplace = false)
    cmin[s] = exp(res.zero[1])
  end
  return cmin
end



@everywhere function backward_cf(para, cf′, a′grid, r, r′, w)
    @unpack σ, γ, β, χ, P, A, S, a_min, k_spline = para
    c_min = compute_cmin(para, w)
    n_con = 10
    N_a = length(a′grid)
    #preallocate for speed
    a_grid = zeros(S, n_con + N_a)
    c_grid = zeros(S, n_con + N_a)
    Uc′ = zeros(S)
    ## For each element in a′grid, compute the correspoding c level
    for (i_a′, a′) in enumerate(a′grid)
        for s′ in 1:S
            Uc′[s′] = (cf′[s′](a′)) ^ (-σ)
        end
        for s in 1:S
            c = (β * (1 + r′) * dot(P[s, :], Uc′)) .^ (-1 / σ)
            n = max(1 - ((w * A[s] * c ^ (-σ)) / χ) ^ (-1 / γ), 0)
            a = (a′ + c - A[s] * w * n) / (1 + r)
            a_grid[s, i_a′ + n_con] = a
            c_grid[s, i_a′ + n_con] = c
        end
    end
    for s in 1:S
        if a_grid[s, 1 + n_con] > a_min
            for (i_ĉ, ĉ) in enumerate(linspace(c_min[s], c_grid[s, n_con + 1], n_con + 1)[1:n_con])
                n̂ = max(1 - ((w * A[s] * ĉ ^ (-σ)) / χ) ^ (-1 / γ), 0)
                â = (a_min + ĉ - A[s] * w * n̂) / (1 + r)
                a_grid[s, i_ĉ] = â
                c_grid[s, i_ĉ] = ĉ
            end
        else
            a_grid[s, 1:n_con] = -Inf
            c_grid[s, 1:n_con] = -Inf
        end
    end
    #Now interpolate
    cf = Vector{Dierckx.Spline1D}(S)
    for s in 1:S
        if c_grid[s, 1] == -Inf
            indx = find(a_grid[s, :] .< a_min)[end]
            cf[s] = Dierckx.Spline1D(a_grid[s, indx:end], c_grid[s, indx:end]; k = k_spline)
        #If the constraint binds, we need to use all the grid points
        else
            cf[s] = Dierckx.Spline1D(a_grid[s, :], c_grid[s, :]; k = k_spline)
        end
    end
    return cf
end



@everywhere function get_cft(para, T, rt, wt, cf_ss)
    @unpack S, a_min, a_max = para
    cft = Matrix{Dierckx.Spline1D}(S, T + 1) # running from time 0 to T
    cft[:, end] = cf_ss
    a′grid = vcat(linspace(a_min, a_min+ 2 , 20),
                  linspace(a_min + 2, a_max, 80)[2:end])
    # backwards compute the policies
    for τ in 0:T - 1
        t = T - τ
        r = rt[t]
        w = wt[t]
        r′ = rt[t + 1]
        cft[:, t] = backward_cf(para, cft[:, t + 1], a′grid, r, r′, w)
    end
    return cft
end



@everywhere function next_π(para, π, cf, r, w)
    @unpack N, S, α, a_min, a_max, A, χ, γ, σ, P = para
    bin_midpts = get_bins(a_min, a_max, N)
    ngrid = zeros((N + 2) * S)
    π′ = zeros(length(π))
    for indx in 1:length(π)
        #transition to iprime with prob ω
        #transition to iprime + 1 with prob 1-ω
        i′ = 0
        ω = 0.
        i, s = dimtrans1to2(N, indx)
        a = bin_midpts[i]
        c = cf[s](a)
        n = max(1 - ((w * A[s] * c ^ (-σ)) / χ) ^ (-1 / γ), 0)
        a′ = (1 + r) * a + A[s] * w * n - c
        ngrid[indx] = n * A[s]
        #check if aprime falls into the very first or very last bin
        if a′ <= 0.0
            i′ = 1
            ω = 1.0
        #check if aprime falls into the very last bin
        elseif a′ >= bin_midpts[N + 2]
            i′ = N + 1
            ω = 0.0
        else
            i′ = findfirst(a′.<= bin_midpts) - 1#find([aprime > bin_midpts[n] for n in 1:N+2])[end]
            #calculate ω
            ω = (bin_midpts[i′ + 1] - a′) / (bin_midpts[i′ + 1] - bin_midpts[i′])
        end
        #transition to i′ with prob ω
        #transition to i′ + 1 with prob 1-ω
        #transition to s′ϵ[1, .., S] with the fowlling probabilities
        for (i_prime, prob) in zip([i′ i′ + 1], [ω 1 - ω])
            for s_prime in 1:S
                k_prime = dimtrans2to1(N, i_prime, s_prime)
                π′[k_prime] = π′[k_prime] + π[indx] * prob * P[s, s_prime]
            end
        end
    end
    return π′, ngrid
end



@everywhere function get_πt(para, cft, rt, wt, π̄, n̄grid, T)
    @unpack S, N = para
    πt = zeros(length(π̄), T + 1)
    ngrid_t = zeros((N + 2) * S, T + 1)
    πt[:, 1] = π̄
    # forwards compute the evolution of distribution π
    for t in 1:T
        r = rt[t]
        w = wt[t]
        πt[:, t + 1], ngrid_t[:, t] = next_π(para, πt[:, t], cft[:, t], r, w)
    end
    ngrid_t[:, T + 1] = n̄grid
    return πt, ngrid_t
end



@everywhere function realized_kpath(πt, ngrid_t, agrid)
    T = length(πt[1, :]) - 1
    k̂ = zeros(T + 1)
    for t in 1:T + 1
        K = πt[:, t]' * agrid
        N = πt[:, t]' * ngrid_t[:, t]
        k̂[t] = K / N
    end
    return k̂
end


function solve_transition(para)
    @unpack a_min, a_max, ρ, α, δ, S = para
    T = 300
    lnθ₀ = para.σ_ϵ
    lnθt = [lnθ₀ * ρ .^ (t - 1) for t in 1:T + 1]
    θt = exp.(lnθt) # running from time 0 to T
    para, π̄, k̄, n̄grid,_, agrid = calibrate_stationary(para)
    cf_ss = get_cf(para)
    function f(k)
        k = vcat(k, k̄) #running from time 0 to T
        rt = α * θt .* k .^ (α - 1) - δ
        wt = (1 - α) * θt .* k .^ α
        cft = get_cft(para, T, rt, wt, cf_ss)
        πt, ngrid_t = get_πt(para, cft, rt, wt, π̄, n̄grid, T)
        k̂ = realized_kpath(πt, ngrid_t, agrid)
        diff = norm(k[1:T] - k̂[1:T], Inf)
        println(diff)
        return k[1:T] - k̂[1:T]
    end

    function f!(F,z)
        F[:] = f(z)
    end

    function j!(J,z)
        N = length(z)
        h = 1e-5
        points = hcat(z,z .+ h*eye(N))
        results = SharedArray{Float64}((N,N+1))
        @sync @parallel for i in 1:N+1
            results[:,i] = f(points[:,i])
        end
        for i in 1:N
            J[:,i] .= (results[:,i+1] .- results[:,1])./(h)
        end
    end

    function fj!(F,J,z)
        N = length(z)
        h = 1e-6
        points = hcat(z,z .+ h*eye(N))
        results = SharedArray{Float64}((N,N+1))
        @sync @parallel for i in 1:N+1
            results[:,i] = f(points[:,i])
        end
        for i in 1:N
            J[:,i] .= (results[:,i+1] .- results[:,1])./(h)
        end
        F[:] = results[:,1];
    end
    k_trans = readdlm("../data/HA_rational/k_trans.csv", ',')
    #=
    initial_k = ones(T) * k̄
    initial_F = zeros(T)
    df = OnceDifferentiable(f!,j!,fj!,initial_k,initial_F)
    res = nlsolve(df,initial_k)
    k_trans = res.zero::Vector{Float64}
    =#
    coeffs, R = compute_coeffs(para, k_trans, θt, agrid, k̄, cf_ss, T, n̄grid, π̄)
    Ct, Nt, Kt = compute_paths(para, k_trans, θt, agrid, k̄, cf_ss, T, n̄grid, π̄)
    return k_trans, coeffs, R, Ct, Nt, Kt
end


function compute_coeffs(para, k_trans, θt, agrid, k̄, cf_ss, T, n̄grid, π̄)
    @unpack α, δ, a_min, a_max, N, σ = para
    r̄ = α * k̄ ^ (α - 1) - δ
    k = vcat(k_trans, k̄) #running from time 0 to T
    rt = α * θt .* k .^ (α - 1) - δ
    wt = (1 - α) * θt .* k .^ α
    cft = get_cft(para, T, rt, wt, cf_ss)
    πt, ngrid_t = get_πt(para, cft, rt, wt, π̄, n̄grid, T)
    bin_midpts = get_bins(a_min, a_max, N)
    # Compute ln(K_t - \bar K) for each year
    Kt = [k[t] * ngrid_t[:, t]' * πt[:, t] for t in 1:T + 1]
    K̄ = k̄ * n̄grid' * π̄
    # Compute $\hat \nu^i_t$ for each bin for each period
    logν̂t = similar(πt)
    for t in 1:size(logν̂t, 2)
        for indx in 1:size(logν̂t, 1)
            i, s = dimtrans1to2(N, indx)
            a = bin_midpts[i]
            c = cft[s, t](a)
            logν̂t[indx, t] = log((1 + rt[t]) * c ^ (-σ)) - log((1 + r̄) * cf_ss[s](a) ^ (-σ))
        end
    end
    # Regressing for each bin
    coeffs_vec = zeros(3, size(logν̂t, 1))
    R_vec = zeros(3, 3, size(logν̂t, 1))
    for indx in 1:size(logν̂t, 1)
        LHS = logν̂t[indx, 2:end]
        RHS = [ones(T) log.((Kt / K̄))[1:T] log.(θt)[1:T]]
        coeffs_vec[:, indx], R_vec[:, :, indx] = OLSestimator(LHS, RHS)
    end
    coeffs = [coeffs_vec[i, :]' * π̄ for i in 1:3]
    R = zeros(3, 3)
    for i in 1:size(logν̂t, 1)
        R = R .+ R_vec[:, :, indx] * π̄[i]
    end
    return coeffs, R
end






function compute_paths(para, k_trans, θt, agrid, k̄, cf_ss, T, n̄grid, π̄)
    @unpack α, δ, a_min, a_max, N, σ = para
    r̄ = α * k̄ ^ (α - 1) - δ
    k = vcat(k_trans, k̄) #running from time 0 to T
    rt = α * θt .* k .^ (α - 1) - δ
    wt = (1 - α) * θt .* k .^ α
    cft = get_cft(para, T, rt, wt, cf_ss)
    πt, ngrid_t = get_πt(para, cft, rt, wt, π̄, n̄grid, T)
    bin_midpts = get_bins(a_min, a_max, N)
    # Compute ln(K_t - \bar K) for each year
    Kt = [k[t] * ngrid_t[:, t]' * πt[:, t] for t in 1:T + 1]
    Ct = zeros(T)
    Nt = zeros(T)
    for t in 1:T
        cgrid = similar(πt[:, t])
        for indx in 1:size(πt, 1)
            i, s = dimtrans1to2(N, indx)
            a = bin_midpts[i]
            cgrid[indx] = cft[s, t](a)
        end
        Ct[t] = dot(πt[:, t], cgrid)
        Nt[t] = dot(πt[:, t], ngrid_t[:, t])
    end
    return Ct, Nt, Kt
end



para = HAmodel()
k_trans, coeffs, R, Ct, Nt, Kt = solve_transition(para)
writedlm("../data/HA_rational/psi.csv", coeffs, ',')
writedlm("../data/HA_rational/R.csv", R, ',')
#writedlm("../data/HA_rational//k_trans.csv", k_trans, ',')

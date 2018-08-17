include("HA_stationary.jl")



function OLSestimator(y, x)
    estimate = inv(x'* x) * (x' * y)
    return estimate
end



function compute_cmin!(para, w)
  @unpack σ, γ, χ, A, S = para
  for s in 1:S
    function f(logc)
      c = exp(logc)
      n = 1 - ((w * A[s] * c .^ (-σ)) / χ) .^ (-1 / γ)
      return c - A[s] * w * n
    end
    res = nlsolve(f, [0.]; inplace = false)
    para.c_min[s] = exp(res.zero[1])
  end
end



function backward_cf(para, cf′, a′grid, r, r′, w)
    @unpack σ, γ, β, χ, P, A, S, a_min, k_spline = para
    compute_cmin!(para, w)
    n_con = 10
    #for each element in a′grid,  compute the correspoding c level
    cf = Spline1D[]
    for s in 1:S
        agrid = Float64[]
        cgrid = Float64[]
        Ucprime = zeros(S)
        for a′ in a′grid
            #For each value of a′,
            #back out the corresponding value of a
            for sprime in 1:S
                Ucprime[sprime] = cf′[sprime](a′) ^ (-σ)
            end
            E_Uc = dot(P[s, :], Ucprime)
            c = (β * (1 + r′) * E_Uc) ^ (-1 / σ)
            n = max(1 - ((w * A[s] * c ^ (-σ)) / χ) ^ (-1 / γ), 0)
            a = (a′ + c - A[s] * w * n) / (1 + r)
            #Next, the "if" condition implies agent is
            #borrowing constrained for any smaller a
            if (a′ == a_min) && (a > a_min)
              c_min = para.c_min[s]
              c_max = c
              for ĉ in linspace(c_min, c_max, n_con)
                n̂ = max(1 - ((w * A[s] * ĉ ^ (-σ)) / χ) ^ (-1 / γ), 0)
                â = (0. + ĉ - A[s] * w * n̂) / (1 + r)
                #add these new points to the endogenous grid,  note when
                #c = c_min we will have â = 0. by construction.
                push!(cgrid, ĉ)
                push!(agrid, â)
              end
            else
              push!(cgrid, c)
              push!(agrid, a)
            end
        end
        push!(cf, Spline1D(agrid, cgrid; k = k_spline))
    end
    return cf
end



function get_cft(para, T, rt, wt, cf_ss)
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



function next_π(para, π, cf, r, w)
    @unpack N, S, α, a_min, a_max, A, χ, γ, σ, P = para
    bin_midpts = get_bins(a_min, a_max, N)
    ϵngrid = zeros((N + 2) * S)
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
        ϵngrid[indx] = n * A[s]
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
    return π′, ϵngrid
end



function get_πt(para, cft, rt, wt, π̄, ϵn̄grid, T)
    @unpack S, N = para
    πt = zeros(length(π̄), T + 1)
    ϵngrid_t = zeros((N + 2) * S, T + 1)
    πt[:, 1] = π̄
    # forwards compute the evolution of distribution π
    for t in 1:T
        r = rt[t]
        w = wt[t]
        πt[:, t + 1], ϵngrid_t[:, t] = next_π(para, πt[:, t], cft[:, t], r, w)
    end
    ϵngrid_t[:, T + 1] = ϵn̄grid
    return πt, ϵngrid_t
end



function realized_kpath(πt, ϵngrid_t, agrid)
    T = length(πt[1, :]) - 1
    k̂ = zeros(T + 1)
    for t in 1:T + 1
        K = πt[:, t]' * agrid
        N = πt[:, t]' * ϵngrid_t[:, t]
        k̂[t] = K / N
    end
    return k̂
end



function compute_coeffs(para, k_trans, θt, agrid, k̄, cf_ss, T, ϵn̄grid, π̄)
    @unpack α, δ, a_min, a_max, N, σ = para
    r̄ = α * k̄ ^ (α - 1) - δ
    k = vcat(k_trans, k̄) #running from time 0 to T
    rt = α * θt .* k .^ (α - 1) - δ
    wt = (1 - α) * θt .* k .^ α
    cft = get_cft(para, T, rt, wt, cf_ss)
    πt, ϵngrid_t = get_πt(para, cft, rt, wt, π̄, ϵn̄grid, T)
    bin_midpts = get_bins(a_min, a_max, N)
    # Compute ln(K_t - \bar K) for each year
    Kt = [k[t] * ϵngrid_t[:, t]' * πt[:, t] for t in 1:T + 1]
    K̄ = k̄ * ϵn̄grid' * π̄
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
    for indx in 1:size(logν̂t, 1)
        LHS = logν̂t[indx, 2:end]
        RHS = [ones(T) log.((Kt / K̄))[1:T] log.(θt)[1:T]]
        coeffs_vec[:, indx] = OLSestimator(LHS, RHS)
    end
    coeffs = [coeffs_vec[i, :]' * π̄ for i in 1:3]
    return coeffs
end



function solve_transition(para)
    @unpack a_min, a_max, ρ, α, δ, S = para
    T = 100
    lnθ₀ = para.σ_ϵ
    lnθt = [lnθ₀ * ρ .^ (t - 1) for t in 1:T + 1]
    θt = exp.(lnθt) # running from time 0 to T
    para, π̄, k̄, ϵn̄_grid, n̄grid, agrid = calibrate_stationary(para)
    cf_ss = get_cf(para)
    function f(k)
        k = vcat(k, k̄) #running from time 0 to T
        rt = α * θt .* k .^ (α - 1) - δ
        wt = (1 - α) * θt .* k .^ α
        cft = get_cft(para, T, rt, wt, cf_ss)
        πt, ϵngrid_t = get_πt(para, cft, rt, wt, π̄, ϵn̄grid, T)
        k̂ = realized_kpath(πt, ϵngrid_t, agrid)
        diff = norm(k[1:T] - k̂[1:T], Inf)
        println(diff)
        return k[1:T] - k̂[1:T]
    end
    #k = readdlm("../data/k_trans.csv", ',')
    k = ones(100) * k̄
    res = nlsolve(f, k; inplace = false)
    k_trans = res.zero
    k_trans = k
    coeffs = compute_coeffs(para, k_trans, θt, agrid, k̄, cf_ss, T, ϵn̄grid, π̄)
    Ct, Nt, Kt = compute_paths(para, k_trans, θt, agrid, k̄, cf_ss, T, ϵn̄grid, π̄)
    return k_trans, coeffs, Ct, Nt, Kt
end



function compute_paths(para, k_trans, θt, agrid, k̄, cf_ss, T, ϵn̄grid, π̄)
    @unpack α, δ, a_min, a_max, N, σ = para
    r̄ = α * k̄ ^ (α - 1) - δ
    k = vcat(k_trans, k̄) #running from time 0 to T
    rt = α * θt .* k .^ (α - 1) - δ
    wt = (1 - α) * θt .* k .^ α
    cft = get_cft(para, T, rt, wt, cf_ss)
    πt, ϵngrid_t = get_πt(para, cft, rt, wt, π̄, ϵn̄grid, T)
    bin_midpts = get_bins(a_min, a_max, N)
    # Compute ln(K_t - \bar K) for each year
    Kt = [k[t] * ϵngrid_t[:, t]' * πt[:, t] for t in 1:T + 1]
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
        Nt[t] = dot(πt[:, t], ϵngrid_t[:, t])
    end
    return Ct, Nt, Kt
end



para = HAmodel()
k_trans, coeffs, Ct, Nt, Kt = solve_transition(para)
writedlm("../data/HA_rational/k_trans.csv", k_trans, ',')

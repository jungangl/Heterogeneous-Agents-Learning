addprocs(25)
@everywhere include("HA_stationary.jl")
using DataFrames



function OLSestimator(y, x)
  estimate = inv(x'* x) * (x' * y)
  R = inv(size(x, 1)) * x' * x
  return estimate, R
end



@everywhere function compute_cmin(para, w,r)
  @unpack σ, γ, χ, A, S, a_min = para
  cmin = zeros(S)
  for s in 1:S
    function f(logc)
      c = exp.(logc)
      n = 1 - ((w * A[s] * c .^ (-σ)) / χ) .^ (-1 / γ)
      return c - A[s] * w * n - r*a_min
    end
    res = nlsolve(f, [0.]; inplace = false)
    cmin[s] = exp.(res.zero[1])
  end
  return cmin
end



@everywhere function backward_cf(para, cf′, a′grid, r, r′, w)
  @unpack σ, γ, β, χ, P, A, S, a_min, k_spline = para
  c_min = compute_cmin(para, w,r)
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
  @unpack S, a_min, a_max, Na = para
  cft = Matrix{Dierckx.Spline1D}(S, T + 1) # running from time 0 to T
  cft[:, end] = cf_ss
  a′grid = construct_agrid(a_min,a_max,Na)
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
  hgrid = zeros((N + 2) * S)
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
    hgrid[indx] = n
    #check if aprime falls into the very first or very last bin
    if a′ <= a_min
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
  return π′,ngrid,hgrid
end



@everywhere function get_πt(para, cft, rt, wt, π0, n̄grid,h̄grid, T)
  @unpack S, N = para
  πt = zeros(length(π0), T + 1)
  ngrid_t = zeros((N + 2) * S, T + 1)
  hgrid_t = zeros((N + 2) * S, T + 1)
  πt[:, 1] = π0
  # forwards compute the evolution of distribution π
  for t in 1:T
    r = rt[t]
    w = wt[t]
    πt[:, t + 1],ngrid_t[:, t], hgrid_t[:,t] = next_π(para, πt[:, t], cft[:, t], r, w)
  end
  ngrid_t[:, T + 1] = n̄grid
  hgrid_t[:, T + 1] = h̄grid
  return πt, ngrid_t, hgrid_t
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


function solve_transition(para,π̄,k̄,n̄grid,h̄gid,agrid,π0,lnθt,k_trans=zeros(0))
  T = length(lnθt) - 1
  θt = exp.(lnθt) # running from time 0 to T
  cf_ss = get_cf(para)
  function f(k)
    k = vcat(k, k̄) #running from time 0 to T
    rt = α * θt .* k .^ (α - 1) - δ
    wt = (1 - α) * θt .* k .^ α
    cft = get_cft(para, T, rt, wt, cf_ss)
    πt, ngrid_t, hgrid_t = get_πt(para, cft, rt, wt, π0, n̄grid,h̄grid, T)
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
  #k_trans = readdlm("../data/HA_rational/k_trans1sd_yearly_noiid.csv", ',')
  #f(k_trans)
  if length(k_trans) != T
    initial_k = ones(T) * k̄
    initial_F = zeros(T)
    df = OnceDifferentiable(f!,j!,fj!,initial_k,initial_F)
    res = nlsolve(df,initial_k)
    k_trans = res.zero::Vector{Float64}
  end

  k = vcat(k_trans, k̄) #running from time 0 to T
  rt = α * θt .* k .^ (α - 1) - δ
  wt = (1 - α) * θt .* k .^ α
  cft = get_cft(para, T, rt, wt, cf_ss)
  πt, ngrid_t, hgrid_t = get_πt(para, cft, rt, wt, π0, n̄grid,h̄grid, T)
  k̂ = [dot(πt[:,t],agrid)/dot(πt[:,t],ngrid_t[:,t]) for t in 1:T]
  println(norm(k[1:T]-k̂))
  return  (πt, ngrid_t, hgrid_t, agrid, rt, wt, cft)
end

function computePsiFunction(para,k_trans=zeros(0),k_transK=zeros(0))
  @unpack a_min, a_max, ρ, α, δ, S, σ = para
  para, π̄, k̄, n̄grid, h̄grid, agrid = calibrate_stationary!(para)
  cfss = get_cf(para)
  T = 200
  lnθ₀ = -1*para.σ_ϵ
  lnθt = [lnθ₀ * ρ .^ (t - 1) for t in 1:T + 1]

  πt, ngrid_t, hgrid_t, agrid, rt, wt, cft = solve_transition(para,π̄,k̄,n̄grid,h̄gid,agrid,π̄,lnθt,k_trans)
  lnθ0 = zeros(T+1)
  πKt, ngridK_t, hgridK_t, agrid, rKt, wKt, cfKt = solve_transition(para,π̄,k̄,n̄grid,h̄grid,agrid,πt[:,2],lnθ0,k_transK)

  ψvec = zeros(length(agrid),3) #Now store ψ
  K = dot(πKt[:,1],agrid)
  K̄ = dot(π̄,agrid)
  for indx in 1:length(agrid)
    i, s = dimtrans1to2(para.N, indx)
    Uc = cfKt[s,2](agrid[indx])^(-σ)
    Ūc = cfss[s](agrid[indx])^(-σ)
    ψvec[indx,2] = log(Uc/Ūc) / log(K/K̄)

    Ucθ = cft[s,3](agrid[indx])^(-σ) #note timing convention
    ψvec[indx,3] = log(Ucθ/Uc) / lnθt[2]
  end

  return ((πt, ngrid_t, hgrid_t, agrid, rt, wt, cft),
          (πKt, ngridK_t, hgridK_t, agrid, rKt, wKt, cfKt),
          ψvec)
end


function compute_coeffs(para, k_trans, θt, agrid, k̄, cf_ss, T, n̄grid,h̄grid, π̄)
  @unpack α, δ, a_min, a_max, N, σ = para
  r̄ = α * k̄ ^ (α - 1) - δ
  k = vcat(k_trans, k̄) #running from time 0 to T
  rt = α * θt .* k .^ (α - 1) - δ
  wt = (1 - α) * θt .* k .^ α
  cft = get_cft(para, T, rt, wt, cf_ss)
  πt, ngrid_t, hgrid_t = get_πt(para, cft, rt, wt, π̄, n̄grid,h̄grid, T)
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
      R = R .+ R_vec[:, :, i] * π̄[i]
  end
  return coeffs, R
end






function compute_paths(para, k_trans, θt, agrid, k̄, cf_ss, T, n̄grid,h̄grid, π̄)
  @unpack α, δ, a_min, a_max, N, σ = para
  r̄ = α * k̄ ^ (α - 1) - δ
  w̄ = (1-α) * k̄ ^(α)
  K̄ = dot(π̄,agrid) #steady state capital
  N̄ = dot(π̄,n̄grid)
  H̄ = dot(π̄,h̄grid)

  k = vcat(k_trans, k̄) #running from time 0 to T
  rt = α * θt .* k .^ (α - 1) - δ
  wt = (1 - α) * θt .* k .^ α
  cft = get_cft(para, T, rt, wt, cf_ss)
  πt, ngrid_t, hgrid_t = get_πt(para, cft, rt, wt, π̄, n̄grid,h̄grid, T)
  bin_midpts = get_bins(a_min, a_max, N)
  # Compute ln(K_t - \bar K) for each year
  Kt = [k[t] * ngrid_t[:, t]' * πt[:, t] for t in 1:T + 1]
  Ct = zeros(T)
  Nt = zeros(T)
  Ht = zeros(T)
  for t in 1:T
      cgrid = similar(πt[:, t])
      for indx in 1:size(πt, 1)
          i, s = dimtrans1to2(N, indx)
          a = bin_midpts[i]
          cgrid[indx] = cft[s, t](a)
      end
      Ct[t] = dot(πt[:, t], cgrid)
      Nt[t] = dot(πt[:, t], ngrid_t[:, t])
      Ht[t] = dot(πt[:, t], hgrid_t[:, t])
  end
  Yt = θt[1:T] .* Kt[1:T].^α .* Nt.^(1-α)
  Ȳ = K̄^α * N̄ ^(1-α)
  It = Kt[2:T+1] - (1-δ)*Kt[1:T]
  Ī = K̄ - (1-δ) * K̄
  C̄ = Ȳ - Ī
  #compute impulse response all in % deviations
  df = DataFrame()
  df[:r] = 100*(rt[1:T] .- r̄)
  df[:w] = 100*(wt[1:T] ./ w̄ - 1.)
  df[:K] = 100*(Kt[1:T] ./ K̄ - 1.)
  df[:Y] = 100*(Yt[1:T] ./ Ȳ - 1.)
  df[:I] = 100*(It[1:T] ./ Ī - 1.)
  df[:H] = 100*(Ht[1:T] ./ H̄ - 1. )
  df[:N] = 100*(Nt[1:T] ./ N̄ - 1.)
  df[:C] = 100*(Ct[1:T] ./ C̄ - 1.)
  return df
end

## Compute the expected future marginal utility in steady state
## The expetation is taken over all future states
function ν_f(para,a, s, cf,cf′, r,r′,w)
    @unpack A, σ, χ, γ, P, S = para
    c = cf[s](a)
    n = max(1 - (w * A[s] * c ^ (-σ) / χ) ^ (-1 / γ), 0.0)
    a′ = A[s] * w * n + (1 + r) * a - c
    ν = (1 + r′) *
        dot(
            P[s,:],
            [(cf′[s](a′)) ^ (-σ) for s in 1:S]
        )
    return ν
end

## Compute the expected future marginal utility in steady state
## The expetation is taken over all future states
function EEerror_f(para,a, s, cf,cf′, r,r′,w)
    @unpack A, σ, χ, γ, P, S,β = para
    c = cf[s](a)
    n = max(1 - (w * A[s] * c ^ (-σ) / χ) ^ (-1 / γ), 0.0)
    a′ = A[s] * w * n + (1 + r) * a - c
    ν = (1 + r′) *
        dot(
            P[s,:],
            [(cf′[s](a′)) ^ (-σ) for s in 1:S]
        )
    error = log(β*ν/c^(-σ))/(-σ)
    if (a′ <0.001) && c^(-σ) >β*ν
      error = 0.
    end
    return error
end




para = HAmodel()
println("2500")
println(para.N)
@unpack a_min, a_max, ρ, α, δ, S, σ = para
para, π̄, k̄, n̄grid, h̄grid, agrid = calibrate_stationary!(para)
cfss = get_cf(para)
T = 200
lnθ₀ = -1*para.σ_ϵ
lnθt = [lnθ₀ * ρ .^ (t - 1) for t in 1:T + 1]

k_trans = readdlm("ktrans1.csv")
πt, ngrid_t, hgrid_t, agrid, rt, wt, cft = solve_transition(para,π̄,k̄,n̄grid,h̄grid,agrid,π̄,lnθt,k_trans)
df = compute_paths(para, k_trans, exp.(lnθt), agrid, k̄, cfss, T, n̄grid,h̄grid, π̄)


ψgrid = zeros(length(agrid),3) #Now store ψ
sgrid = zeros(Int,length(agrid))
νvec = zeros(length(agrid))
EEerror =zeros(length(agrid))

t2 = 2
K2 = dot(πt[:,t2],agrid)
K̄ = dot(π̄,agrid)
Kt = [dot(πt[:,t],agrid) for t in 1:T]
for indx in 1:length(agrid)
  i, s = dimtrans1to2(para.N, indx)
  ν = ν_f(para,agrid[indx],s,cft[:,1],cft[:,2],rt[1],rt[2],wt[1])
  ν̄ = ν_f(para,agrid[indx],s,cft[:,1],cfss,rt[1],para.r̄,wt[1])
  ψgrid[indx,3] = log( ν/ν̄ ) / lnθt[1]
  νvec[indx] = ν
  EEerror[indx] = EEerror_f(para,agrid[indx],s,cft[:,1],cft[:,2],rt[1],rt[2],wt[1])


  ν = ν_f(para,agrid[indx],s,cft[:,t2],cft[:,t2+1],rt[t2],rt[t2+1],wt[t2])
  ν̄ = ν_f(para,agrid[indx],s,cft[:,t2],cfss,rt[t2],para.r̄,wt[t2])
  ψgrid[indx,2] = (log(ν/ν̄) -ψgrid[indx,3] *lnθt[t2] ) / log(K2/K̄)
  sgrid[indx] = s
end

avec = unique(agrid);
ψf = Matrix{Spline1D}(S,3)
ψf2 = Matrix{Spline1D}(S,3)
ψf3 = Matrix{Spline1D}(S,3)
ψ̄ = ψgrid'*π̄
π̃ = π̄ .* agrid / K̄
ψ̄2 = ψgrid'*π̃
for s in 1:S
  ψvec = zeros(length(avec),3)
  ψvec2 = zeros(length(avec),3)
  ψvec3 = zeros(length(avec),3)
  for i in 1:length(avec)
    a = avec[i]
    ν = ν_f(para,a,s,cft[:,1],cft[:,2],rt[1],rt[2],wt[1])
    ν̄ = ν_f(para,a,s,cft[:,1],cfss,rt[1],para.r̄,wt[1])
    ψvec[i,3] = log( ν/ν̄ ) / lnθt[1]
    ψvec2[i,3] = ψ̄[3]
    ψvec3[i,3] = ψ̄2[3]

    ν = ν_f(para,a,s,cft[:,2],cft[:,3],rt[2],rt[3],wt[2])
    ν̄ = ν_f(para,a,s,cft[:,2],cfss,rt[2],para.r̄,wt[2])

    ψvec[i,2] = (log(ν/ν̄) -ψvec[i,3] *lnθt[2] ) / log(K2/K̄)
    ψvec2[i,2] = ψ̄[2]
    ψvec3[i,2] = ψ̄2[2]
  end
  ψf[s,1] = Spline1D(avec,ψvec[:,1],k=3)
  ψf[s,2] = Spline1D(avec,ψvec[:,2],k=3)
  ψf[s,3] = Spline1D(avec,ψvec[:,3],k=3)
  ψf2[s,1] = Spline1D(avec,ψvec2[:,1],k=3)
  ψf2[s,2] = Spline1D(avec,ψvec2[:,2],k=3)
  ψf2[s,3] = Spline1D(avec,ψvec2[:,3],k=3)
  ψf3[s,1] = Spline1D(avec,ψvec3[:,1],k=3)
  ψf3[s,2] = Spline1D(avec,ψvec3[:,2],k=3)
  ψf3[s,3] = Spline1D(avec,ψvec3[:,3],k=3)
end

test = [ψf[sgrid[i],2](agrid[i]) - ψgrid[i,2] for i in 1:length(agrid)]

include("HA_irfs.jl")
para.agent_num  = 1000000
para.N_ϕ = 200
a,s,ψ = drawinitial(π̄,agrid,ψgrid,sgrid,para.agent_num)
θt = exp(lnθt)[1:20]
df = simul_irf_df(para, θt,a,ψf,s)
df2 = simul_irf_df(para, θt,a,ψf,s,false)
df3 = simul_irf_df(para, θt,a,ψf2,s)
df4 = simul_irf_df(para, θt,a,ψf3,s)

plot((Kt[1:20]/K̄ - 1)*100,label="RE")
plot!((df[:K]/df[:K][1] - 1)*100,label="RE Beliefs")
#plot!((df2[:K]/df2[:K][1] - 1)*100)
plot!((df3[:K]/df3[:K][1] - 1)*100, label="Average RE Beliefs")
plot!((df4[:K]/df4[:K][1] - 1)*100, label="Capital Weighted Average Beliefs")
xlabel!("Time")
ylabel!("Percentage Deviation from Steady State Capital")
savefig("IRFComparison.png")

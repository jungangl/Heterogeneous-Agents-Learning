using NLsolve, Parameters, Distributions, Plots, Roots
@with_kw type RAmodel
  #fundamental paramters
  K2Y::Float64 = 10.26 #targeted
  σ::Float64 = 2.
  γ::Float64 = 2.
  β::Float64 = 0.9900130264871999 #this needs to be found by calibration
  ρ::Float64 = 0.95
  σ_ϵ::Float64 = 0.007
  α::Float64 = 0.36
  δ::Float64 = 0.025
  χ::Float64 = 1.2499931387511778 #this needs to be found by calibration
  γ_gain::Function  = t -> 0.02
  #steady state values
  r̄::Float64 = 0.01008771929824561
  w̄::Float64 = 2.3711026965018402
  ā::Float64 = 12.67058003443171 #aggregate capital
  c̄::Float64 = 0.9181848202339157
  n̄::Float64 = 1/3 #targeted
  ν̄::Float64 = 1.1981160163055604
  #simul_learningation parameters
  T::Int64 = 1000
  R̄::Matrix{Float64} =  [ 1.0000    0.0013    0.0001;
                          0.0013    0.0014    0.0004;
                          0.0001    0.0004    0.0007]
  ψ̄::Vector{Float64} =  [ 1.77e-5; -0.731; -0.4506]
end



## Define the steadystate residual function
function ss_resid(x, n̄, γ, σ, α, δ, K2Y)
  β, χ, r̄, w̄, ā, c̄, ν̄ = x
  resid = [β * (1 + r̄) - 1.0;
          χ * (1 - n̄) ^ (-γ) - w̄ * c̄ ^ (-σ);
          w̄ - (1 - α) * ā ^ (α) * n̄ ^ (-α);
          α * ā ^ (α - 1) * n̄ ^ (1 - α) - δ - r̄;
          ā ^ α * n̄ ^ (1 - α) - δ * ā - c̄;
          (1 + r̄) * c̄ ^ (-σ) - ν̄;
          ā / n̄ - (K2Y) ^ (1 / (1 - α))]
  return resid
end



## Define the calibration function
function calibrate_ss(para)
  @unpack σ, γ, ρ, α, δ, n̄, K2Y = para
  res = nlsolve(x -> ss_resid(x, n̄, γ, σ, α, δ, K2Y), ones(7);
                inplace = false)::NLsolve.SolverResults{Float64,Array{Float64,1},Array{Float64,1}}
  println(ss_resid(res.zero, n̄, γ, σ, α, δ, K2Y))
  β, χ, r̄, w̄, ā, c̄, ν̄ = res.zero
  @pack para = β, χ, r̄, w̄, ā, c̄, ν̄
  return para
end



## Construct function for wage rate residual, used in a root solver
function w_resid(w, c, σ, χ, γ, α, θ, a)
    n = max(1 - (w * c ^ (-σ) / χ) ^ (-1 / γ), 0.0)
    resid = w - θ * (1 - α) * a ^ α * n ^ (-α)
    return resid, n
end



## Temporary Equilibrium
function TE(para, a, ψ, x)
    @unpack ν̄, ā, σ, β, γ, ρ, α, δ, χ = para
    θ = exp(x[3])
    Eν = exp(log(ν̄) + dot(ψ,  x))
    c = (β * Eν) ^ (-1 / σ)
    w = fzero(w -> w_resid(w, c, σ, χ, γ, α, θ, a)[1], 0.01, 10.)::Float64
    n = w_resid(w, c, σ, χ, γ, α, θ, a)[2]
    r = α * θ * a ^ (α - 1) * n ^ (1 - α) - δ
    a′ = (1 + r) * a + w * n - c
    ν = (1 + r) * c ^ (-σ)
    return [r, w, c, n, a′, ν]
end



## Generate the next θ
function drawθ(θ, σ_ϵ, ρ)
  dist = Normal(0., σ_ϵ)
  return exp(ρ * log(θ) + rand(dist))
end



## Update the R matrix
function update_R(R, x, t, γ_gain)
    R′ = R + γ_gain(t) .* (x * x' - R)
    return R′
end



## Update the belief ψ
function update_ψ(ψ, R′, x, ν, t, γ_gain, ν̄)
    ψ′ = ψ + γ_gain(t) .* inv(R′) * x .* (log(ν / ν̄) - ψ' * x)[1]
    return ψ′
end



## simul_learningate representative agents with bounded rationality
function simul_learning(para)
  @unpack r̄, w̄, c̄, n̄, ν̄, ā, R̄, ψ̄, T, σ_ϵ, ρ, γ_gain = para
  c_t, r_t, w_t, n_t, ν_t, θ_t, a_t = [zeros(T) for _ in 1:7]
  ψ_t, x_t, R_t = zeros(3, T), zeros(3, T),zeros(3, 3, T)
  ## Initialize data from the steady state
  a_1 = ā
  θ_1 = 1.0
  x_1 = [1; log(a_1 / ā); log(θ_1)]
  ψ_0 = ψ̄
  R_0 = R̄
  a_t[1], θ_t[1], x_t[:, 1], ψ_t[:, 1], R_t[:, :, 1] = a_1, θ_1, x_1, ψ_0, R_0
  ## Loop through t = 1..T
  ## ψ[:, t] = {ψ_0, ψ_1, ...} R[:, :, t] = {R_0, R_1, ...}
  ## ψ_0 is generated at time 0 and will be used to forecast at time 1
  for t in 1:T
      a, ψ, x, θ, R = a_t[t], ψ_t[:, t], x_t[:, t], θ_t[t], R_t[:, :, t]
      r, w, c, n, a′, ν = TE(para, a, ψ, x)
      r_t[t], w_t[t], c_t[t], n_t[t], ν_t[t] = r, w, c, n, ν
      #update for the next period
      if t == T break end
      a_t[t + 1] = a′
      θ_t[t + 1] = drawθ(θ, σ_ϵ,ρ)
      x_t[:, t + 1] = [1; log(mean(a′) / ā); log(θ_t[t + 1])]
      R′ = update_R(R, x, t - 1, γ_gain)
      R_t[:, :, t + 1] = R′
      if t == 1
        ψ_t[:, 2] = ψ_t[:, 1]
      else
        ψ_t[:, t + 1] = update_ψ(ψ, R, x_t[:, t - 1], ν, t, γ_gain, ν̄)
      end
  end
  return c_t, r_t, w_t, n_t, ν_t, θ_t, a_t, ψ_t, x_t, R_t
end



## Write all of the results
function write_all(data, filenames)
    for (i, filename) in enumerate(filenames)
        writedlm("../data/RA_learning/$filename.csv", data[i], ',')
    end
end



## Plot all of the results
function plot_all(para, data, filenames)
    fig_vec = Vector{Plots.Plot{Plots.GRBackend}}(length(filenames))
    ss_vec = [para.c̄, para.r̄, para.w̄, para.n̄, para.ν̄, 1., para.ā]
    for i in 1:length(filenames)
        println(i)
        if i == 8
            fig_vec[i] = plot(grid = false, title = "Beliefs Evolution with Representative Agents")
            for j in 1:3
                plot!(fig_vec[i], data[i][j,:], label = "belief $j")
            end
            plot!(fig_vec[i], para.ψ̄' .* ones(para.T), label = "", ls = :dash)
        else
            fig_vec[i] = plot(grid = false, title = "Representative Agents with Learning: $(filenames[i])")
            plot!(fig_vec[i], data[i], label = "")
            plot!(fig_vec[i], ss_vec[i] * ones(para.T), label = "steady state level", ls = :dash)
        end
    end
    return fig_vec
end


para = RAmodel(data, filenames)
gain = .005
para.γ_gain = t -> gain
para = calibrate_ss(para)
para.T = 100_000
para.ψ̄ =  [0.00015195296492383106; -0.7562646342465844; -0.6417144194500661]
data = simul_learning(para)
c_t, r_t, w_t, n_t, ν_t, θ_t, a_t, ψ_t, x_t, R_t = data



#=
filenames = ["c", "r", "w", "n", "nu", "theta", "a", "psi", "x", "R"]
write_all(data)
fig_vec = plot_all(para, data, filenames[1:8])
for i in 1:8
    savefig(fig_vec[i], "../figures/RA_learning/$(filenames[i]).pdf")
end
=#

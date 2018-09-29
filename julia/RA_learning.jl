include("RAmodel.jl")



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
    y = θ * a ^ α * n ^ (1 - α)
    return [r, w, c, n, a′, ν, y]
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
  c_t, r_t, w_t, n_t, ν_t, θ_t, a_t, y_t = [zeros(T) for _ in 1:8]
  ψ_t, x_t, R_t = zeros(T, 3), zeros(T, 3), zeros(T, 3, 3)
  ## Initialize data from the steady state
  a_1 = ā
  θ_1 = 1.0
  x_1 = [1; log(a_1 / ā); log(θ_1)]
  ψ_0 = ψ̄
  R_0 = R̄
  a_t[1], θ_t[1], x_t[1, :], ψ_t[1, :], R_t[1, :, :] = a_1, θ_1, x_1, ψ_0, R_0
  ## Loop through t = 1..T
  ## ψ[:, t] = {ψ_0, ψ_1, ...} R[:, :, t] = {R_0, R_1, ...}
  ## ψ_0 is generated at time 0 and will be used to forecast at time 1
  for t in 1:T
      a, ψ, x, θ, R = a_t[t], ψ_t[t, :], x_t[t, :], θ_t[t], R_t[t, :, :]
      r, w, c, n, a′, ν, y = TE(para, a, ψ, x)
      r_t[t], w_t[t], c_t[t], n_t[t], ν_t[t], y_t[t] = r, w, c, n, ν, y
      #update for the next period
      if t == T break end
      a_t[t + 1] = a′
      θ_t[t + 1] = drawθ(θ, σ_ϵ, ρ)
      x_t[t + 1, :] = [1 log(mean(a′) / ā) log(θ_t[t + 1])]
      R′ = update_R(R, x, t - 1, γ_gain)
      R_t[t + 1, :, :] = R′
      if t == 1
        ψ_t[2, :] = ψ_t[1, :]
      else
        ψ_t[t + 1, :] = update_ψ(ψ, R, x_t[t - 1, :], ν, t, γ_gain, ν̄)
      end
  end
  return c_t, r_t, w_t, n_t, ν_t, θ_t, a_t, ψ_t, x_t, R_t, y_t
end



## Write all of the results
function write_all(data, filenames, str)
    for (i, filename) in enumerate(filenames)
        writedlm("$str/$filename.csv", data[i], ',')
    end
end



## Plot all of the results
function plot_all(para, data, filenames)
    @unpack c̄, r̄, w̄, n̄, ν̄, ā, α = para
    fig_vec = Vector{Plots.Plot{Plots.GRBackend}}(length(filenames))
    ss_vec = [c̄, r̄, w̄, n̄, ν̄, 1., ā, ā ^ α * n̄ ^ (1 - α)]
    for i in 1:length(filenames)
        println(i)
        if i == length(filenames)
            fig_vec[i] = plot(grid = false, title = "Beliefs Evolution with Representative Agents")
            for j in 1:3
                plot!(fig_vec[i], data[i][:, j], label = "belief $j")
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






filenames = ["c", "r", "w", "n", "nu", "theta", "a", "y", "psi"]
for yearly in [true; false]
    for (i, gain) in enumerate([t -> 0.005; t -> 0.01; t -> 1 / (t + 2)])
        str_yearly = (if yearly "yearly" else "quarterly" end)
        str_gain = if i == 1 "0.005" elseif i == 2 "0.01" else "decrease" end
        str = "../data/RA/$str_yearly/learning/gain_$(str_gain)"
        para = RAmodel(yearly = yearly, T = 100_000, γ_gain = gain)
        para = calibrate_ss(para)
        data = simul_learning(para)
        c_t, r_t, w_t, n_t, ν_t, θ_t, a_t, ψ_t, x_t, R_t, y_t = data
        data = [c_t, r_t, w_t, n_t, ν_t, θ_t, a_t, y_t, ψ_t]
        write_all(data, filenames, str)
        fig_vec = plot_all(para, data, filenames)
        for i in 1:9
            savefig(fig_vec[i], "../figures/RA/$str_yearly/learning/gain_$str_gain/$(filenames[i]).pdf")
        end
    end
end

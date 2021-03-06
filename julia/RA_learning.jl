include("RAmodel.jl")
#Define the residual function for solving the steady state: 7-variable system
#Preset Parameters: n̄, γ, σ, α, δ, K2Y
#Calibrate Parameters: β, χ, r̄, w̄, ā, c̄, ν̄ all together 7 variables
#Seven Equations:
#1) Household: Euler Equation
#2) Household: Labor Leisure
#3) Firm: Labor Demand
#4) Firm: Capital Demand
#5) Goods Market Clearing
#6) Marginal Utility of Wealth Definition
#7) Capital to Output Ratio Definition
function ss_resid(x, n̄, γ, σ, α, δ, K2Y)
  β, χ, r̄, w̄, ā, c̄, ν̄ = x
  resid = [β * (1 + r̄) - 1.0;
          χ * n̄ ^ γ - w̄ * c̄ ^ (-σ);
          w̄ - (1 - α) * ā ^ (α) * n̄ ^ (-α);
          α * ā ^ (α - 1) * n̄ ^ (1 - α) - δ - r̄;
          ā ^ α * n̄ ^ (1 - α) - δ * ā - c̄;
          (1 + r̄) * c̄ ^ (-σ) - ν̄;
          ā / n̄ - (K2Y) ^ (1 / (1 - α))]
  return resid
end



#Define the calibration function for the steady state
# Use nonlinear solver function "nlsolve" to solve for the system of equations
function calibrate_ss(para)
  @unpack σ, γ, ρ, α, δ, n̄, K2Y = para
  x_zero = nlsolve(x -> ss_resid(x, n̄, γ, σ, α, δ, K2Y), ones(7); inplace = false).zero::Vector{Float64}
  resid = ss_resid(x_zero, n̄, γ, σ, α, δ, K2Y)
  #println(resid)
  @assert(norm(resid, Inf) .< 1e-6, "Steady state not found")
  β, χ, r̄, w̄, ā, c̄, ν̄ = x_zero
  @pack! para = β, χ, r̄, w̄, ā, c̄, ν̄
  return para
end




## Define the residual function for solving
function νf_ss_resid(guess_vec, a_grid, para)
    @unpack β, χ, r̄, w̄, ā, c̄, ν̄, Na, σ, γ = para
    c̄_vec, n̄_vec, ν̄_vec = guess_vec[1:Na], guess_vec[Na + 1:2Na], guess_vec[2Na + 1:3Na]
    budget_constr_resid = c̄_vec .- (r̄ * a_grid .+ w̄ * n̄_vec)
    intra_temp_resid = w̄ * c̄_vec .^ (-σ) .- χ * n̄_vec .^ (γ)
    ν_def_resid = ν̄_vec .- (1. + r̄) * c̄_vec .^ (-σ)
    return vcat(budget_constr_resid, intra_temp_resid, ν_def_resid)
end



# Solve for the shadow price function ν(a)
function solve_νf(para)
    @unpack Na, c̄, n̄, ν̄, ā = para
    a_grid = LinRange(0.5ā, 1.5ā, Na)
    guess_vec = vcat(c̄ * ones(Na), n̄ * ones(Na), ν̄ * ones(Na))
    fixpt = nlsolve(guess_vec -> νf_ss_resid(guess_vec, a_grid, para), guess_vec; inplace = false)
    c̄_vec, n̄_vec, ν̄_vec = fixpt.zero[1:Na], fixpt.zero[Na + 1:2Na], fixpt.zero[2Na + 1:3Na]
    ν̄f = Spline1D(a_grid, ν̄_vec; k = 3)
    return ν̄f
end



## Construct function for wage rate residual, used in a root solver: 1-variable system
#1) Given [w] we can back out labor supply from agent's Labor Leisure Condition
#2) From labor supply, we can back out [w]
#3) We need to find the fixed point for [w]
function TE_resid(guess, para, a, ψ, x, ν̄f)
    @unpack ν̄, ā, σ, β, γ, ρ, α, δ, χ = para
    c_guess, n_guess = guess
    θ = exp(x[3])
    w = θ * (1 - α) * a ^ α * n_guess ^ (-α)
    n = (w * c_guess ^ (-σ) / χ) ^ (1. / γ)
    r = α * θ * a ^ (α - 1) * n ^ (1 - α) - δ
    a′ = (1 + r) * a + w * n - c_guess
    Eν = exp(log(ν̄f(a′)) + dot(ψ,  x))::Float64
    c = (β * Eν) ^ (-1 / σ)
    ν = (1 + r) * c ^ (-σ)
    y = θ * a ^ α * n ^ (1 - α)
    resid = [c - c_guess; n - n_guess]
    return resid, [r, w, c, n, a′, ν, y]
end


## Temporary equilibrium function takes:
#1) asset holding: a
#2) belief vector: ψ
#3) information set: x
#and spits out:
#1) interest rate: r
#2) wage rate: w
#3) consumption: c
#4) labor: n
#5) future asset holding: a′
#6) marginal utility of wealth: ν
#7) output: y
function TE(para, a, ψ, x, ν̄f)
    @unpack ν̄, ā, σ, β, γ, ρ, α, δ, χ, c̄, n̄ = para
    guess = [c̄; n̄]
    fixpt = nlsolve(guess -> TE_resid(guess, para, a, ψ, x, ν̄f)[1], guess)
    r, w, c, n, a′, ν, y = TE_resid(fixpt.zero, para, a, ψ, x, ν̄f)[2]
    return [r, w, c, n, a′, ν, y]
end



##Exogenous TFP shocks θ follow an AR(1) process in logs.
#log(θ′) = ρ⋅log(θ) + ϵ, where ϵ ∼ N(0, σ_ϵ).
function drawθ(θ, σ_ϵ, ρ)
  dist = Normal(0., σ_ϵ)
  return exp(ρ * log(θ) + rand(dist))
end



## Update the R matrix which is the sample second-moment matrix of the regressors.
function update_R(R̄, R, x, t, γ_gain)
    R′ = R + γ_gain(t) .* (x * x' - R)
    return R′
    #return R̄
end



## Update the belief vector ψ accoding to recursive least square.
function update_ψ(ψ, R′, x, ν, t, γ_gain, ν̄)
    ψ′ = ψ + γ_gain(t) .* inv(R′) * x .* (log(ν / ν̄) - ψ' * x)[1]
    return ψ′
end


# Simulate θ_t
function simul_θ(σ_ϵ, ρ, T)
    θ_t = ones(T)
    for t in 2:T
        θ_t[t] = drawθ(θ_t[t-1], σ_ϵ, ρ)
    end
    return θ_t
end



## simul_learning representative agents with bounded rationality
function simul_learning(para)
  para = calibrate_ss(para)
  @unpack r̄, w̄, c̄, n̄, ν̄, ā, R̄, ψ̄, T, σ_ϵ, ρ, γ_gain, yearly = para
  c_t, r_t, w_t, n_t, ν_t, θ_t, a_t, y_t = [zeros(T) for _ in 1:8]
  ψ_t, x_t, R_t = zeros(T, 3), zeros(T, 3), zeros(T, 3, 3)
  θ_t = simul_θ(σ_ϵ, ρ, T)
  ## Initialize data from the steady state
  a_1, θ_1 = ā, 1.0
  x_1 = [1; log(a_1 / ā); log(θ_1)]
  ψ_0, R_0 = ψ̄, R̄
  a_t[1], θ_t[1], x_t[1, :], ψ_t[1, :], R_t[1, :, :] = a_1, θ_1, x_1, ψ_0, R_0
  ν̄f = solve_νf(para)
  ## Loop through t = 1..T
  ## ψ[:, t] = {ψ_0, ψ_1, ...} R[:, :, t] = {R_0, R_1, ...}
  ## ψ_0 is generated at time 0 and will be used to forecast at time 1
  for t in 1:T
      #println(t)
      a, ψ, x, θ, R = a_t[t], ψ_t[t, :], x_t[t, :], θ_t[t], R_t[t, :, :]
      r, w, c, n, a′, ν, y = TE(para, a, ψ, x, ν̄f)
      r_t[t], w_t[t], c_t[t], n_t[t], ν_t[t], y_t[t] = r, w, c, n, ν, y
      #update for the next period
      if t < T
          a_t[t + 1] = a′
          #θ_t[t + 1] = drawθ(θ, σ_ϵ, ρ)
          x_t[t + 1, :] = [1 log(mean(a′) / ā) log(θ_t[t + 1])]
          R′ = update_R(R̄, R, x, t - 1, γ_gain)
          R_t[t + 1, :, :] = R′
          if t == 1
            ψ_t[2, :] = ψ_t[1, :]
          else
            ψ_t[t + 1, :] = update_ψ(ψ, R, x_t[t - 1, :], ν, t, γ_gain, ν̄)
          end
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
    fig_vec = Array{Plots.Plot{Plots.GRBackend}}(undef, length(filenames))
    ss_vec = [c̄, r̄, w̄, n̄, ν̄, 1., ā, ā ^ α * n̄ ^ (1 - α)]
    for i in 1:length(filenames)
        println(i)
        if i == length(filenames)
            fig_vec[i] = plot(grid = false, layout = (3, 1), size = (600, 600))
            for j in 1:3
                plot!(fig_vec[i], data[i][:, j], title = "\\psi$(j-1)", label = "", subplot = j)
            end
            plot!(fig_vec[i], (para.yearly * [0.; -0.7711960305969916; -0.7865505980462995] + !para.yearly * [0.;  -0.765090668;   -0.655607579])' .* ones(para.T), label = "", ls = :dash, lw = 2)
        else
            fig_vec[i] = plot(grid = false, title = "Representative Agents with Learning: $(filenames[i])")
            plot!(fig_vec[i], data[i], label = "")
            plot!(fig_vec[i], ss_vec[i] * ones(para.T), label = "steady state level", ls = :dash)
        end
    end
    return fig_vec
end


#=
filenames = ["c", "r", "w", "n", "nu", "theta", "a", "y", "psi"]
for yearly in [true]
    for from_RE in [true; false]
        for (gain_i, gain) in enumerate([t -> 0.005; t -> 0.01; t -> 0.05])
            print("$(yearly), $(from_RE)")
            str_yearly = (if yearly "yearly" else "quarterly" end)
            str_from_RE = (if from_RE "from_RE" else "from_minus1" end)
            str_gain = if gain_i == 1 "0.005" elseif gain_i == 2 "0.01" elseif gain_i == 3 "0.05" end
            str = "RA/$str_yearly/learning/$str_from_RE/gain_$(str_gain)"
            para = RAmodel(yearly = yearly, from_RE = from_RE, γ_gain = gain, T = 100_000)
            para = calibrate_ss(para)
            println("$gain_i")
            data = simul_learning(para)
            c_t, r_t, w_t, n_t, ν_t, θ_t, a_t, ψ_t, x_t, R_t, y_t = data
            data = [c_t, r_t, w_t, n_t, ν_t, θ_t, a_t, y_t, ψ_t]
            write_all(data, filenames, "../data/$str")
            fig_vec = plot_all(para, data, filenames)
            for i in 1:9
                savefig(fig_vec[i], "../figures/$str/$(filenames[i]).pdf")
            end
        end
    end
end
=#

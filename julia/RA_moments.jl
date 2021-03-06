include("RAmodel.jl")
include("RA_learning.jl")
using SparseArrays



function hp_filter(y::Vector{Float64}, λ::Float64)
    n = length(y)
    @assert n >= 4
    diag2 = λ * ones(n - 2)
    diag1 = [-2λ; -4λ * ones(n - 3); -2λ]
    diag0 = [1 + λ; 1 + 5λ; (1 + 6λ) * ones(n - 4); 1 + 5λ; 1 + λ]
    D = spdiagm(-2 => diag2, -1 => diag1, 0 => diag0, 1 => diag1, 2 => diag2)
    return D \ y
end


## Draw from LREE Simulations
function draw_simul(para, T_irf, repeat_time, ree)
    ## Compute the steady state level from the stationary distribution
    @unpack α, ā, w̄, r̄, n̄, T, yearly, σ, γ = para
    K̄ = ā
    N̄ = n̄
    Ȳ = K̄ ^ α * (N̄ ^ (1. - α))
    C̄ = r̄ * ā + w̄ * n̄
    Ī = Ȳ - C̄
    θ̄ = 1.
    ## Draw from the simulation from learning
    T_conv = 50000  # the series are converged startin from T_conv
    variable_vec = ["a", "c", "n", "y", "theta"]
    I_sim_vec, K_sim_vec, C_sim_vec, N_sim_vec, Y_sim_vec, θ_sim_vec = [zeros(T_irf, repeat_time) for i in 1:6]
    Random.seed!(1)
    t_start_vec = sort(sample(T_conv:T - T_irf, repeat_time; replace = false))
    yearly_str = if (yearly) "yearly" elseif (!yearly) "quarterly" end
    σ_str = if (σ == 2.0) "h" elseif (σ == 1.0) "m" elseif (σ == 0.5) "l" end
    γ_str = if (γ == 2.0) "h" elseif (γ == 1.0) "m" elseif (γ == 0.5) "l" end
    if ree
        simul_vec = [zeros(T_irf) for i in 1:length(variable_vec)]
        for (i, var) in enumerate(variable_vec)
            simul_vec[i] = vec(readdlm("../dynare/$(yearly_str)_$(σ_str)$(γ_str)/$(var).csv", ','))
        end
        K_sim, C_sim, N_sim, Y_sim, θ_sim = simul_vec
    else
        K_sim, C_sim, N_sim, Y_sim, θ_sim = simul_learning(para)[[7;1;4;11;6]] #["a", "c", "n", "y", "theta"]
    end



    I_sim = Y_sim .- C_sim
    for (r, t_start) in enumerate(t_start_vec)
        ## Convert to percentage deviation from steady state
        K_sim_vec[:, r] = 100(K_sim[t_start:t_start + T_irf - 1] .- K̄) ./ K̄
        C_sim_vec[:, r] = 100(C_sim[t_start:t_start + T_irf - 1] .- C̄) ./ C̄
        N_sim_vec[:, r] = 100(N_sim[t_start:t_start + T_irf - 1] .- N̄) ./ N̄
        Y_sim_vec[:, r] = 100(Y_sim[t_start:t_start + T_irf - 1] .- Ȳ) ./ Ȳ
        θ_sim_vec[:, r] = 100(θ_sim[t_start:t_start + T_irf - 1] .- θ̄) ./ θ̄
        I_sim_vec[:, r] = 100((Y_sim .- C_sim)[t_start:t_start + T_irf - 1] .- Ī) ./ Ī
    end
    return I_sim_vec, K_sim_vec, C_sim_vec, N_sim_vec, Y_sim_vec, θ_sim_vec
end



function compute_moments(para, T_irf, repeat_time, ree)
    I_sim_vec, K_sim_vec, C_sim_vec, N_sim_vec, Y_sim_vec, θ_sim_vec = draw_simul(para, T_irf, repeat_time, ree)
    ## Standard Deviations
    σ_I = mean(std(I_sim_vec, dims = 1))
    σ_K = mean(std(K_sim_vec, dims = 1))
    σ_C = mean(std(C_sim_vec, dims = 1))
    σ_N = mean(std(N_sim_vec, dims = 1))
    σ_Y = mean(std(Y_sim_vec, dims = 1))
    σ_Pr = mean(std(θ_sim_vec, dims = 1))
    ## Relative Standard Deviations
    σ_CY = σ_C ./ σ_Y
    σ_IY = σ_I ./ σ_Y
    σ_NY = σ_N ./ σ_Y
    σ_PrY = σ_Pr ./ σ_Y
    ## Correlations
    ρ_IY = mean([cor(I_sim_vec[:, r], Y_sim_vec[:, r]) for r in 1:repeat_time])
    ρ_KY = mean([cor(K_sim_vec[:, r], Y_sim_vec[:, r]) for r in 1:repeat_time])
    ρ_CY = mean([cor(C_sim_vec[:, r], Y_sim_vec[:, r]) for r in 1:repeat_time])
    ρ_NY = mean([cor(N_sim_vec[:, r], Y_sim_vec[:, r]) for r in 1:repeat_time])
    ρ_PrY = mean([cor(θ_sim_vec[:, r], Y_sim_vec[:, r]) for r in 1:repeat_time])
    ρ_PrN = mean([cor(θ_sim_vec[:, r], N_sim_vec[:, r]) for r in 1:repeat_time])
    return [σ_CY; σ_IY; σ_NY; σ_PrY; ρ_IY; ρ_KY; ρ_CY; ρ_NY; ρ_PrY; ρ_PrN]
end



## Simulate and Compute the Moments
function run_(yearly)
    letters = ["l", "m", "h"]
    paras = [0.5; 1.0; 2.0]
    gains = [0.001; 0.005; 0.01]
    T_irf = 140
    repeat_time = 5000
    for i in 1:3
        for j in 1:3
            println("i = $i, j = $j")
            corrs = zeros(10, 3)
            df = DataFrame()
            df.Moments = ["sigma_CY", "sigma_IY", "sigma_NY", "sigma_PrY", "rho_IY", "rho_KY", "rho_CY", "rho_NY", "rho_PrY", "rho_PrN"]
            yearly_str = if yearly "yearly" else "quarterly" end
            for (m, gain) in enumerate(gains)
                println("gain = $gain")
                para = calibrate_ss(RAmodel(yearly = yearly, T = 101000, σ = paras[i], γ = paras[j], γ_gain = t -> gain))
                para.ψ̄ = vec(readdlm("../data/RA/$(yearly_str)/rational/$(letters[i])$(letters[j])/psi.csv", ','))
                para.R̄ = readdlm("../data/RA/$(yearly_str)/rational/$(letters[i])$(letters[j])/R_cov.csv", ',')
                corrs[:, m] = compute_moments(para, T_irf, repeat_time, false)
            end
            para = calibrate_ss(RAmodel(yearly = yearly, T = 101000, σ = paras[i], γ = paras[j]))
            df.ree = compute_moments(para, T_irf, repeat_time, true)
            df.gain0_001, df.gain0_005, df.gain0_01 = corrs[:, 1], corrs[:, 2], corrs[:, 3]
            CSV.write("../data/moments/RA_$(yearly_str)/$(letters[i])$(letters[j]).csv", df)
        end
    end
end
run_(false)

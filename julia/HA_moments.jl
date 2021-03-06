## This script simulates paths of aggregate variables from REE and LREE
# and computes and compares the moments
include("HA_stationary.jl")
include("HA_TE.jl")
include("HA_learning.jl")
using Random, DataFrames



function hp_filter(y::Vector{Float64}, λ::Float64)
    n = length(y)
    @assert n >= 4
    diag2 = λ * ones(n - 2)
    diag1 = [-2λ; -4λ * ones(n - 3); -2λ]
    diag0 = [1 + λ; 1 + 5λ; (1 + 6λ) * ones(n - 4); 1 + 5λ; 1 + λ]
    D = spdiagm(-2 => diag2, -1 => diag1, 0 => diag0, 1 => diag1, 2 => diag2)
    return D \ y
end



function read_re_irf(para, T_irf)
    variable_vec = ["I", "mean_a", "mean_c", "mean_n", "y"]
    irf_vec = [zeros(T_irf) for i in 1:length(variable_vec)]
    for (i, var) in enumerate(variable_vec)
        irf_vec[i] = vec(readdlm("../data/HA/yearly/$(para.iid_str)/rational/$(var).csv", ',')[1:T_irf])
    end
    return irf_vec
end


function read_lre_irf(para, T_irf)
    @unpack irf_path = para
    variable_vec = ["I", "mean_a", "mean_c", "mean_n", "y"]
    irf_vec = [zeros(T_irf) for i in 1:length(variable_vec)]
    for (i, var) in enumerate(variable_vec)
        irf_vec[i] = vec(readdlm("../figures/$irf_path/$(var)_median.csv", ',')[1:T_irf])
    end
    return irf_vec
end



function initialize_εpath(para, T)
    @unpack σ_ϵ = para
    dist = Normal(0., σ_ϵ)
    ε_path = rand(dist, T)
    return ε_path
end



function simulate_θpath(para, ε_path, T_irf)
    θ_sim = ones(length(ε_path))
    for t in 2:2T_irf
        θ_sim[t] = exp(para.ρ * log(θ_sim[t - 1]) + ε_path[t])
    end
    return 100(θ_sim[end - T_irf + 1:end] .- 1)
end


# Simulate T_irf periods of aggregate varibales from the IRFs
function simulate_aggregates(para, T_irf)
    @unpack σ_ϵ = para
    T = 2T_irf

    #irfs = read_re_irf(para, T_irf)
    irfs = read_lre_irf(para, T_irf)

    num_irf = size(irfs, 1)
    I_irf, K_irf, C_irf, N_irf, Y_irf = irfs
    ε_path = initialize_εpath(para, T)
    θ_sim = simulate_θpath(para, ε_path, T_irf)
    I_sim, K_sim, C_sim, N_sim, Y_sim = [zeros(T_irf) for i in 1:num_irf]
    for (indx, t) in enumerate((T_irf + 1):T)
        #println("t = $t")
        for s in (t - T_irf + 1):t
            τ = t - s + 1
            #println("s = $s")
            #println("τ = $τ")
            shock_size = ε_path[s] / σ_ϵ
            ΔI, ΔK, ΔC, ΔN, ΔY = I_irf[τ], K_irf[τ], C_irf[τ], N_irf[τ], Y_irf[τ]
            I_sim[indx] += shock_size * ΔI
            K_sim[indx] += shock_size * ΔK
            C_sim[indx] += shock_size * ΔC
            N_sim[indx] += shock_size * ΔN
            Y_sim[indx] += shock_size * ΔY
        end
    end
    return I_sim, K_sim, C_sim, N_sim, Y_sim, θ_sim
end



function repeat_time_simulate_aggregates(para, T_irf, repeat_time)
    I_sim_vec, K_sim_vec, C_sim_vec, N_sim_vec, Y_sim_vec, θ_sim_vec = [zeros(T_irf, repeat_time) for i in 1:6]
    for r in 1:repeat_time
        I_sim_vec[:, r], K_sim_vec[:, r], C_sim_vec[:, r], N_sim_vec[:, r], Y_sim_vec[:, r], θ_sim_vec[:, r] = simulate_aggregates(para, T_irf)
    end
    return I_sim_vec, K_sim_vec, C_sim_vec, N_sim_vec, Y_sim_vec, θ_sim_vec
end



function draw_learning_simul(para, T_irf, repeat_time, expanded, T_conv)
    ## Compute the steady state level from the stationary distribution
    #λ = 6.25
    para, π, k, ϵn_grid, n_grid, a_grid, c_grid = calibrate_stationary!(para)
    K̄, N̄ = para.ā, para.n̄
    Ȳ = K̄ ^ para.α * (dot(π, ϵn_grid) ^ (1. - para.α))
    C̄ = dot(π, c_grid)
    Ī = Ȳ - C̄
    θ̄ = 1.
    ## Draw from the simulation from learning
    I_sim_vec, K_sim_vec, C_sim_vec, N_sim_vec, Y_sim_vec, θ_sim_vec = [zeros(T_irf, repeat_time) for i in 1:6]
    Random.seed!(1)
    t_start_vec = sort(sample(T_conv:para.T - T_irf, repeat_time; replace = false))
    variable_vec = ["mean_a", "mean_c", "mean_n", "y", "theta"]
    simul_vec = [zeros(T_irf) for i in 1:length(variable_vec)]
    simulation_df = DataFrame(mean_a = Float64[], mean_c = Float64[], mean_n = Float64[], y = Float64[], theta = Float64[])
    if expanded
        simul_learning_expanded!(para, π, simulation_df)
    else
        simul_learning!(para, π, simulation_df)
    end

    K_sim, C_sim, N_sim, Y_sim, θ_sim = [convert(Matrix, simulation_df)[:, i] for i in 1:5]
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
    return I_sim_vec, K_sim_vec, C_sim_vec, N_sim_vec, Y_sim_vec, θ_sim_vec, simulation_df
end



function compute_moments(para, T_irf, repeat_time, ree, expanded, T_conv)
    simulation_df = DataFrame(mean_a = Float64[], mean_c = Float64[], mean_n = Float64[], y = Float64[], theta = Float64[])
    I_sim_vec, K_sim_vec, C_sim_vec, N_sim_vec, Y_sim_vec, θ_sim_vec = [zeros(T_irf, repeat_time) for i in 1:6]
    if ree == true
        I_sim_vec, K_sim_vec, C_sim_vec, N_sim_vec, Y_sim_vec, θ_sim_vec = repeat_time_simulate_aggregates(para, T_irf, repeat_time)
    else
        I_sim_vec, K_sim_vec, C_sim_vec, N_sim_vec, Y_sim_vec, θ_sim_vec, simulation_df = draw_learning_simul(para, T_irf, repeat_time, expanded, T_conv)
    end
    ## Standard Deviations
    σ_I = mean(std(I_sim_vec, dims = 1))
    σ_K = mean(std(K_sim_vec, dims = 1))
    σ_C = mean(std(C_sim_vec, dims = 1))
    σ_N = mean(std(N_sim_vec, dims = 1))
    σ_Y = mean(std(Y_sim_vec, dims = 1))
    σ_Pr = mean(std(θ_sim_vec, dims = 1))
    ## Relative Standard Deviations
    σ_CY = σ_C / σ_Y
    σ_IY = σ_I / σ_Y
    σ_NY = σ_N / σ_Y
    σ_PrY = σ_Pr / σ_Y
    ## Correlations
    ρ_IY = mean([cor(I_sim_vec[:, r], Y_sim_vec[:, r]) for r in 1:repeat_time])
    ρ_KY = mean([cor(K_sim_vec[:, r], Y_sim_vec[:, r]) for r in 1:repeat_time])
    ρ_CY = mean([cor(C_sim_vec[:, r], Y_sim_vec[:, r]) for r in 1:repeat_time])
    ρ_NY = mean([cor(N_sim_vec[:, r], Y_sim_vec[:, r]) for r in 1:repeat_time])
    ρ_PrY = mean([cor(θ_sim_vec[:, r], Y_sim_vec[:, r]) for r in 1:repeat_time])
    ρ_PrN = mean([cor(θ_sim_vec[:, r], N_sim_vec[:, r]) for r in 1:repeat_time])
    return [σ_CY; σ_IY; σ_NY; σ_PrY; ρ_IY; ρ_KY; ρ_CY; ρ_NY; ρ_PrY; ρ_PrN], simulation_df
end


s = ArgParseSettings()
@add_arg_table! s begin
    "i"
        arg_type = Int
        required = true
        help = "indx going from 1 to 54, only consider: iid, from_zero, gains (4)"
end
ps = parse_args(s)
indx = ps["i"]

## gain (3X) ⋅ σ (3X) ⋅ γ (3X) ⋅ yearly(2X) ⋅ expanded(2X) = 108
println("indx = $indx")
gain_vec = [1, 2, 3]
σ_vec = ["h", "m", "l"]
γ_vec = ["h", "m", "l"]
yearly_vec = [true, false]
expanded_bool = [false, true]
expanded_str = ["unexpanded", "expanded"]
dim_vec = [3, 3, 3, 2, 2]
ans_indx = dim1toMultiple(dim_vec, indx)
para = HAmodel(gain = gain_vec[ans_indx[1]],
               σ_str = σ_vec[ans_indx[2]],
               γ_str = γ_vec[ans_indx[3]],
               yearly = yearly_vec[ans_indx[4]])
expanded = expanded_bool[ans_indx[5]]



T_irf, repeat_time = 140, 2000
ree = false
T_conv = 25_000
#print("$(para.gain_str), ")
#print("ree = $ree, expanded = $expanded")

try
res, simulation_df = compute_moments(para, T_irf, repeat_time, ree, expanded, T_conv)
catch err
    println(err)
    println(ans_indx)
end

df = DataFrame()
df.Moments = ["sigma_CY", "sigma_IY", "sigma_NY", "sigma_PrY", "rho_IY", "rho_KY", "rho_CY", "rho_NY", "rho_PrY", "rho_PrN"]
df.value = res
CSV.write("../data/moments/HA_$(para.yearly_str)/$(para.σ_str)$(para.γ_str)/$(expanded_str[ans_indx[5]])/$(para.gain_str).csv", df)
CSV.write("../data/moments/HA_$(para.yearly_str)/$(para.σ_str)$(para.γ_str)/$(expanded_str[ans_indx[5]])/simul_$(para.gain_str).csv", simulation_df)


#=
df = readtable("moments.csv")
res_ree = compute_moments(para, T_irf, repeat_time, true)
df.HA_REE = res_ree
=#

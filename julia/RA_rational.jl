using MATLAB
include("RAmodel.jl")
include("RA_learning.jl")
using ArgParse

s = ArgParseSettings()
@add_arg_table s begin
    "i"
        arg_type = Int
        required = true
        help = "indx going from 1 to 9"
end
ps = parse_args(s)
i = ps["i"]

if i == 1
    mat"""
        addpath /Applications/Dynare/4.6.3/matlab/
        cd('../dynare/quarterly_ll')
        dynare dynare.mod
    """
elseif i == 2
    mat"""
        addpath /Applications/Dynare/4.6.3/matlab/
        cd('../dynare/quarterly_lm')
        dynare dynare.mod
    """
elseif i == 3
    mat"""
        addpath /Applications/Dynare/4.6.3/matlab/
        cd('../dynare/quarterly_lh')
        dynare dynare.mod
    """
elseif i == 4
    mat"""
        addpath /Applications/Dynare/4.6.3/matlab/
        cd('../dynare/quarterly_ml')
        dynare dynare.mod
    """
elseif i == 5
    mat"""
        addpath /Applications/Dynare/4.6.3/matlab/
        cd('../dynare/quarterly_mm')
        dynare dynare.mod
    """
elseif i == 6
    mat"""
        addpath /Applications/Dynare/4.6.3/matlab/
        cd('../dynare/quarterly_mh')
        dynare dynare.mod
    """
elseif i == 7
    mat"""
        addpath /Applications/Dynare/4.6.3/matlab/
        cd('../dynare/quarterly_hl')
        dynare dynare.mod
    """
elseif i == 8
    mat"""
        addpath /Applications/Dynare/4.6.3/matlab/
        cd('../dynare/quarterly_hm')
        dynare dynare.mod
    """
elseif i == 9
    mat"""
        addpath /Applications/Dynare/4.6.3/matlab/
        cd('../dynare/quarterly_hh')
        dynare dynare.mod
    """
end




#=
mat"""
    addpath /Applications/Dynare/4.5.7/matlab/
    cd('../dynare')
    dynare rational_equilibrium_quarterly.mod
    csvwrite('../data/RA/quarterly/rational/a.csv', k)
    csvwrite('../data/RA/quarterly/rational/c.csv', c)
    csvwrite('../data/RA/quarterly/rational/n.csv', n)
    csvwrite('../data/RA/quarterly/rational/nu.csv', nu)
    csvwrite('../data/RA/quarterly/rational/r.csv', r)
    csvwrite('../data/RA/quarterly/rational/theta.csv', theta)
    csvwrite('../data/RA/quarterly/rational/w.csv', w)
    csvwrite('../data/RA/quarterly/rational/w.csv', w)
    csvwrite('../data/RA/quarterly/rational/y.csv', y)
"""
=#


function construct_ψR(para, ν, a, θ)
    T = length(a)
    x = ones(T - 2, 3)
    y = zeros(T - 2)
    for t in 1:T - 2
        x[t, 2:3] = [log(a[t] / para.ā); log(θ[t + 1])]
        y[t] = log.(ν[t + 2] / para.ν̄)
    end
    ψ = inv(x'* x) * (x' * y)
    R = inv(size(x, 1)) * x' * x
    return ψ, R
end



function plot_all(para, data, filenames)
    @unpack α, ā, n̄, ν̄, w̄, r̄, c̄ = para
    fig_vec = Array{Plots.Plot{Plots.GRBackend}}(undef, length(filenames))
    ss_vec = [c̄, r̄, w̄, n̄, ν̄, 1., ā, ā ^ α * n̄ ^ (1 - α)]
    for i in 1:length(filenames)
        fig_vec[i] = plot(grid = false, title = "Representative Agents with Learning: $(filenames[i])")
        plot!(fig_vec[i], data[i], label = "", lw = 0.2, alpha = 0.5)
        plot!(fig_vec[i], ss_vec[i] * ones(para.T), label = "steady state level", ls = :dash)
    end
    return fig_vec
end


str_yearly = "quarterly"
letters = ["l", "m", "h"]
paras = [0.5; 1.0; 2.0]
for i in 1:3
    for j in 1:3
        a = readdlm("../dynare/$(str_yearly)_$(letters[i])$(letters[j])/a.csv", ',')
        c = readdlm("../dynare/$(str_yearly)_$(letters[i])$(letters[j])/c.csv", ',')
        n = readdlm("../dynare/$(str_yearly)_$(letters[i])$(letters[j])/n.csv", ',')
        ν = readdlm("../dynare/$(str_yearly)_$(letters[i])$(letters[j])/nu.csv", ',')
        r = readdlm("../dynare/$(str_yearly)_$(letters[i])$(letters[j])/r.csv", ',')
        θ = readdlm("../dynare/$(str_yearly)_$(letters[i])$(letters[j])/theta.csv", ',')
        w = readdlm("../dynare/$(str_yearly)_$(letters[i])$(letters[j])/w.csv", ',')
        y = readdlm("../dynare/$(str_yearly)_$(letters[i])$(letters[j])/y.csv", ',')
        para = RAmodel(yearly = (str_yearly == "yearly"), T = 100_000, σ = paras[i], γ = paras[j])
        para = calibrate_ss(para)
        ψ, R = construct_ψR(para, ν, a, θ)
        writedlm("../data/RA/$str_yearly/rational/$(letters[i])$(letters[j])/psi.csv", ψ, ',')
        writedlm("../data/RA/$str_yearly/rational/$(letters[i])$(letters[j])/R_cov.csv", R, ',')
        filenames = ["c", "r", "w", "n", "nu", "theta", "a", "y"]
        data = [c, r, w, n, ν, θ, a, y]
        fig_vec = plot_all(para, data, filenames)
        for k in 1:8
            print(k)
            savefig(fig_vec[k], "../figures/RA/$str_yearly/rational/simulations/$(letters[i])$(letters[j])/$(filenames[k]).png")
        end
    end
end

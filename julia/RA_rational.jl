using MATLAB
include("RAmodel.jl")



mat"""
    addpath /Applications/Dynare/4.5.1/matlab/
    cd('../dynare')
    dynare rational_equilibrium_yearly.mod
    csvwrite('../data/RA/yearly/rational/a.csv', k)
    csvwrite('../data/RA/yearly/rational/c.csv', c)
    csvwrite('../data/RA/yearly/rational/n.csv', n)
    csvwrite('../data/RA/yearly/rational/nu.csv', nu)
    csvwrite('../data/RA/yearly/rational/r.csv', r)
    csvwrite('../data/RA/yearly/rational/theta.csv', theta)
    csvwrite('../data/RA/yearly/rational/w.csv', w)
    csvwrite('../data/RA/yearly/rational/w.csv', w)
    csvwrite('../data/RA/yearly/rational/y.csv', y)
"""



mat"""
    addpath /Applications/Dynare/4.5.1/matlab/
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
    fig_vec = Vector{Plots.Plot{Plots.GRBackend}}(length(filenames))
    ss_vec = [c̄, r̄, w̄, n̄, ν̄, 1., ā, ā ^ α * n̄ ^ (1 - α)]
    for i in 1:length(filenames)
        println(i)
        fig_vec[i] = plot(grid = false, title = "Representative Agents with Learning: $(filenames[i])")
        plot!(fig_vec[i], data[i], label = "", lw = 0.2, alpha = 0.5)
        plot!(fig_vec[i], ss_vec[i] * ones(para.T), label = "steady state level", ls = :dash)
    end
    return fig_vec
end



for str_yearly in ["yearly", "quarterly"]
    a = readdlm("../data/RA/$str_yearly/rational/a.csv", ',')
    c = readdlm("../data/RA/$str_yearly/rational/c.csv", ',')
    n = readdlm("../data/RA/$str_yearly/rational/n.csv", ',')
    ν = readdlm("../data/RA/$str_yearly/rational/nu.csv", ',')
    r = readdlm("../data/RA/$str_yearly/rational/r.csv", ',')
    θ = readdlm("../data/RA/$str_yearly/rational/theta.csv", ',')
    w = readdlm("../data/RA/$str_yearly/rational/w.csv", ',')
    y = readdlm("../data/RA/$str_yearly/rational/y.csv", ',')
    para = RAmodel(yearly = (str_yearly == "yearly"), T = 100_000)
    ψ, R = construct_ψR(para, ν, a, θ)
    writedlm("../data/RA/$str_yearly/rational/psi.csv", ψ, ',')
    writedlm("../data/RA/$str_yearly/rational/R_cov.csv", R, ',')

    filenames = ["c", "r", "w", "n", "nu", "theta", "a", "y"]
    data = [c, r, w, n, ν, θ, a, y]
    fig_vec = plot_all(para, data, filenames)
    for i in 1:8
        savefig(fig_vec[i], "../figures/RA/$str_yearly/rational/simulations/$(filenames[i]).pdf")
    end
end

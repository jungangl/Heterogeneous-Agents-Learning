using MATLAB
include("RAmodel.jl")




mat"""
    addpath /Applications/Dynare/4.5.1/matlab/
    cd('../dynare')
    dynare rational_equilibrium.mod
    csvwrite('../data/RA_rational/a.csv', k)
    csvwrite('../data/RA_rational/c.csv', c)
    csvwrite('../data/RA_rational/n.csv', n)
    csvwrite('../data/RA_rational/nu.csv', nu)
    csvwrite('../data/RA_rational/r.csv', r)
    csvwrite('../data/RA_rational/theta.csv', theta)
    csvwrite('../data/RA_rational/w.csv', w)
"""





function OLSestimator(y, x)
    estimate = inv(x'* x) * (x' * y)
    return estimate
end



function construct_ψ(para,ν,a, θ)
    T = length(a)
    x = ones(T-2, 3)
    y = zeros(T-2)
    for t in 1:T-2
        x[t, 2:3] = [log(a[t] / para.ā); log(θ[t+1])]
        y[t] = log.(ν[t+2] / para.ν̄)
    end
    ψ = OLSestimator(y, x)
    return ψ
end




function plot_all(para, data, filenames)
    fig_vec = Vector{Plots.Plot{Plots.GRBackend}}(length(filenames))
    ss_vec = [para.c̄, para.r̄, para.w̄, para.n̄, para.ν̄, 1., para.ā]
    for i in 1:length(filenames)
        println(i)
        fig_vec[i] = plot(grid = false, title = "Representative Agents with Learning: $(filenames[i])")
        plot!(fig_vec[i], data[i], label = "", lw = 0.2, alpha = 0.5)
        plot!(fig_vec[i], ss_vec[i] * ones(para.T), label = "steady state level", ls = :dash)
    end
    return fig_vec
end



a = readdlm("../data/RA_rational/a.csv", ',')
c = readdlm("../data/RA_rational/c.csv", ',')
n = readdlm("../data/RA_rational/n.csv", ',')
ν = readdlm("../data/RA_rational/nu.csv", ',')
r = readdlm("../data/RA_rational/r.csv", ',')
θ = readdlm("../data/RA_rational/theta.csv", ',')
w = readdlm("../data/RA_rational/w.csv", ',')


para = RAmodel(T = 100_000)
ψ = construct_ψ(para,ν,a,θ)
#writedlm("../data/RA_rational/psi.csv", ψ, ',')


#=
filenames = ["c", "r", "w", "n", "nu", "theta", "a"]
fig_vec = plot_all(para, data, filenames)
for i in 1:7
    savefig(fig_vec[i], "../figures/RA_rational/simuls/$(filenames[i]).pdf")
end
=#

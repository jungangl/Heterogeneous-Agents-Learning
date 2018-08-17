using MATLAB
include("RAmodel.jl")



#=
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
=#




function OLSestimator(y, x)
    estimate = inv(x'* x) * (x' * y)
    return estimate
end



function construct_x(para, a, θ)
    T = length(a)
    x = ones(T, 3)
    for t in 1:T
        x[t, 2:3] = [log(a[t] / para.ā); log(θ[t])]
    end
    return x
end



function compute_ψ(para, ν, x)
    LHS = log.(ν[2:end] / para.ν̄)
    RHS = x[1:end - 1, :]
    ψ = OLSestimator(RHS, LHS)
    return ψ
end



a = readdlm("../data/RA_rational/a.csv", ',')
c = readdlm("../data/RA_rational/c.csv", ',')
n = readdlm("../data/RA_rational/n.csv", ',')
ν = readdlm("../data/RA_rational/nu.csv", ',')
r = readdlm("../data/RA_rational/r.csv", ',')
θ = readdlm("../data/RA_rational/theta.csv", ',')
w = readdlm("../data/RA_rational/w.csv", ',')
x = construct_x(para, a, θ)


para = RAmodel()
ψ = compute_ψ(para, ν, x)
writedlm("../data/RA_rational/psi.csv", ψ, ',')

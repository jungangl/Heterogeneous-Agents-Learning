## Computes productivity process by approximating AR(1) + iid shock
Pkg.add("FastGaussQuadrature.jl")
using FastGaussQuadrature
function computeProductivityProcess(ρ_p, σ_p, σ_e, Np, Nt, Ns, with_iid)
    if with_iid
        mc = rouwenhorst(Np, ρ_p, σ_p)::QuantEcon.MarkovChain{Float64,Array{Float64,2},StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}}
        P1 = mc.p
        e1 = mc.state_values

        nodes, weights = gausshermite(Nt)::Tuple{Array{Float64,1},Array{Float64,1}}

        P2 = repmat(weights' / sqrt(pi), Nt) #adjust weights by sqrt(π)
        e2 = sqrt(2) * σ_e * nodes

        P = kron(P1, P2) #kron combines matrixies multiplicatively
        e = kron(e1, ones(Nt)) + kron(ones(Np), e2) # e is log productivity
        return MarkovChain(P, e)
    else
        return rouwenhorst(Ns, ρ_p, σ_p)::QuantEcon.MarkovChain{Float64,Array{Float64,2},StepRangeLen{Float64,Base.TwicePrecision{Float64},Base.TwicePrecision{Float64}}}
    end
end

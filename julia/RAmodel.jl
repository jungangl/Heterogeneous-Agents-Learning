using NLsolve, Parameters, Distributions, Plots, Roots
using LinearAlgebra
using DelimitedFiles
using Random, DataFrames
using CSV, Dierckx
@with_kw mutable struct RAmodel
  #fundamental paramters
  yearly::Bool = true
  from_RE::Bool = true
  K2Y::Float64 = yearly * (10.26 / 4) + !yearly * 10.26 #targeted
  σ::Float64 = 2.
  γ::Float64 = 0.5
  β::Float64 = yearly * 0.9612141652785352 + !yearly * 0.9900130264871999
  ρ::Float64 = yearly *  0.95 ^ 4 + !yearly * 0.95
  σ_ϵ::Float64 = yearly * 0.014 + !yearly * 0.007
  α::Float64 = 0.36
  δ::Float64 = yearly * 0.1 + !yearly * 0.025
  χ::Float64 = yearly * 10.624524926168915 + !yearly * 25.312361059711197
  γ_gain::Function  = t -> 0.02
  #steady state values
  Na::Int64 = 200
  r̄::Float64 = yearly * 0.04035087719298244 + !yearly * 0.01008771929824561
  w̄::Float64 = yearly * 1.087155379772843 + !yearly * 2.3711026965018402
  ā::Float64 = yearly * 1.4523716401652824 + !yearly * 12.67058003443171 #aggregate capital
  c̄::Float64 = yearly * 0.4209895962818275 + !yearly * 0.9181848202339157
  n̄::Float64 = yearly * (1 / 3) + !yearly * (1 / 3) #targeted
  ν̄::Float64 = yearly * 5.869986612756786 + !yearly * 1.1981160163055604
  #simul_learningation parameters
  T::Int64 = 10_000
  ψ̄::Vector{Float64} =  !from_RE * [0.; -1.; -1] + from_RE * (yearly * [0.; -0.7711960305969916; -0.7865505980462995] + !yearly * [0.;  -0.765090668;   -0.655607579])
  R̄::Matrix{Float64} =  yearly * [1.0000000000  -0.000837131    -0.000349309;
                                  -0.000837131   0.001363378	 0.000498454;
                                  -0.000349309	 0.000498454	 0.000574746] +
                       !yearly * [1.0000000000   -0.000916309    -0.000362956;
                                  -0.000916309    0.00120064      0.000473709;
                                  -0.000362956    0.000473709     0.000482302]
end

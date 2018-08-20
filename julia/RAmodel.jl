using NLsolve, Parameters, Distributions, Plots, Roots
@with_kw type RAmodel
  #fundamental paramters
  K2Y::Float64 = 10.26 #targeted
  σ::Float64 = 2.
  γ::Float64 = 2.
  β::Float64 = 0.9900130264871999 #this needs to be found by calibration
  ρ::Float64 = 0.95
  σ_ϵ::Float64 = 0.007
  α::Float64 = 0.36
  δ::Float64 = 0.025
  χ::Float64 = 1.2499931387511778 #this needs to be found by calibration
  γ_gain::Function  = t -> 0.02
  #steady state values
  r̄::Float64 = 0.01008771929824561
  w̄::Float64 = 2.3711026965018402
  ā::Float64 = 12.67058003443171 #aggregate capital
  c̄::Float64 = 0.9181848202339157
  n̄::Float64 = 1/3 #targeted
  ν̄::Float64 = 1.1981160163055604
  #simul_learningation parameters
  T::Int64 = 1000
  R̄::Matrix{Float64} =  [ 1.0000    0.0013    0.0001;
                          0.0013    0.0014    0.0004;
                          0.0001    0.0004    0.0007]
  ψ̄::Vector{Float64} =  [ 1.77e-5; -0.731; -0.4506]
end

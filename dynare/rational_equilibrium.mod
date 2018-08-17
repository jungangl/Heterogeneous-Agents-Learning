var r w c n l ltheta y nu k theta;
varexo eps;
parameters sigma gamma beta rho alpha delta sigma_eps X;


//Initialize our parameters.
sigma = 2.;
gamma = 2.;
beta = 0.9900130264871999;
rho = 0.95;
alpha = 0.36;
delta = 0.025;
sigma_eps = 0.007;
X = 1.2499931387511778; 

/*
Specify the model that characterizes the endo variables.
We have 8 endo variables, so we need 8 equations.
*/
model;
//(1) Euler Equation for consumption
c^(-sigma) = beta * nu(+1);
//(2) Labor consumption trade off
c^(-sigma) * w = X * l^(-gamma);
//(3) Firms capital choice
w = theta * (1-alpha) *( k(-1)^alpha) * (n^(-alpha));
//(4) Firms labor choice
r + delta = theta * alpha * (k(-1)^(alpha-1)) * (n)^(1-alpha);
//(6) Labor market clears
n = 1 - l;
//(7) Goods market clears
c + k = y + (1-delta)*k(-1);
//(8) Stochastic process
ltheta = rho*ltheta(-1) + eps;
theta = exp(ltheta);
nu = (1 + r) * c^(-sigma);
y = theta*(k(-1)^alpha)*(n)^(1-alpha);
end;


//Give initial values
initval;
  k = 12.67058003443171;
  c = 0.9181848202339157;
  l = 0.6666666666666667;
  ltheta = 0; 
  eps = 0;
  r = 0.01008771929824561;
  w = 2.3711026965018402;
  theta = 1;
  nu = 1.1981160163055604;
  n = 1-l;
end;

//Get the steady state
steady;
check;


//Specify stochastic shocks
shocks;
var eps =  sigma_eps^2;
end;

//Simulate the evolution of the variables
stoch_simul(periods=100000, drop=10000,order=3);
var r w c n l ltheta y nu k theta;
varexo eps;
parameters sigma gamma beta rho alpha delta sigma_eps X;


//Initialize our parameters.
sigma = 2.;
gamma = 2.;
beta = 0.98;
rho = 0.85;
alpha = 0.33;
delta = 0.023;
sigma_eps = 0.014;
X = 1.97; 

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
  k = 6.19;
  c = 0.0672;
  l = 0.7;
  ltheta = 0; 
  eps = 0;
  r = 1/beta -1;
  w = 1.82;
  theta = 1;
  nu = 1.;
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
stoch_simul(periods=10000, drop=1000,order=3);
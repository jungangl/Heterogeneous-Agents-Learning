function [residual, g1, g2, g3] = rational_equilibrium_static(y, x, params)
%
% Status : Computes static model for Dynare
%
% Inputs : 
%   y         [M_.endo_nbr by 1] double    vector of endogenous variables in declaration order
%   x         [M_.exo_nbr by 1] double     vector of exogenous variables in declaration order
%   params    [M_.param_nbr by 1] double   vector of parameter values in declaration order
%
% Outputs:
%   residual  [M_.endo_nbr by 1] double    vector of residuals of the static model equations 
%                                          in order of declaration of the equations.
%                                          Dynare may prepend or append auxiliary equations, see M_.aux_vars
%   g1        [M_.endo_nbr by M_.endo_nbr] double    Jacobian matrix of the static model equations;
%                                                       columns: variables in declaration order
%                                                       rows: equations in order of declaration
%   g2        [M_.endo_nbr by (M_.endo_nbr)^2] double   Hessian matrix of the static model equations;
%                                                       columns: variables in declaration order
%                                                       rows: equations in order of declaration
%   g3        [M_.endo_nbr by (M_.endo_nbr)^3] double   Third derivatives matrix of the static model equations;
%                                                       columns: variables in declaration order
%                                                       rows: equations in order of declaration
%
%
% Warning : this file is generated automatically by Dynare
%           from model file (.mod)

residual = zeros( 10, 1);

%
% Model equations
%

T11 = y(3)^(-params(1));
T30 = y(9)^params(5);
T34 = y(4)^(-params(5));
T42 = y(9)^(params(5)-1);
T43 = y(10)*params(5)*T42;
T44 = y(4)^(1-params(5));
lhs =T11;
rhs =params(3)*y(8);
residual(1)= lhs-rhs;
lhs =T11*y(2);
rhs =params(8)*y(5)^(-params(2));
residual(2)= lhs-rhs;
lhs =y(2);
rhs =y(10)*(1-params(5))*T30*T34;
residual(3)= lhs-rhs;
lhs =y(1)+params(6);
rhs =T43*T44;
residual(4)= lhs-rhs;
lhs =y(4);
rhs =1-y(5);
residual(5)= lhs-rhs;
lhs =y(3)+y(9);
rhs =y(7)+y(9)*(1-params(6));
residual(6)= lhs-rhs;
lhs =y(6);
rhs =y(6)*params(4)+x(1);
residual(7)= lhs-rhs;
lhs =y(10);
rhs =exp(y(6));
residual(8)= lhs-rhs;
lhs =y(8);
rhs =T11*(1+y(1));
residual(9)= lhs-rhs;
lhs =y(7);
rhs =T44*y(10)*T30;
residual(10)= lhs-rhs;
if ~isreal(residual)
  residual = real(residual)+imag(residual).^2;
end
if nargout >= 2,
  g1 = zeros(10, 10);

  %
  % Jacobian matrix
  %

T70 = getPowerDeriv(y(3),(-params(1)),1);
T77 = getPowerDeriv(y(4),1-params(5),1);
T88 = getPowerDeriv(y(9),params(5),1);
  g1(1,3)=T70;
  g1(1,8)=(-params(3));
  g1(2,2)=T11;
  g1(2,3)=y(2)*T70;
  g1(2,5)=(-(params(8)*getPowerDeriv(y(5),(-params(2)),1)));
  g1(3,2)=1;
  g1(3,4)=(-(y(10)*(1-params(5))*T30*getPowerDeriv(y(4),(-params(5)),1)));
  g1(3,9)=(-(T34*y(10)*(1-params(5))*T88));
  g1(3,10)=(-(T34*(1-params(5))*T30));
  g1(4,1)=1;
  g1(4,4)=(-(T43*T77));
  g1(4,9)=(-(T44*y(10)*params(5)*getPowerDeriv(y(9),params(5)-1,1)));
  g1(4,10)=(-(T44*params(5)*T42));
  g1(5,4)=1;
  g1(5,5)=1;
  g1(6,3)=1;
  g1(6,7)=(-1);
  g1(6,9)=1-(1-params(6));
  g1(7,6)=1-params(4);
  g1(8,6)=(-exp(y(6)));
  g1(8,10)=1;
  g1(9,1)=(-T11);
  g1(9,3)=(-((1+y(1))*T70));
  g1(9,8)=1;
  g1(10,4)=(-(y(10)*T30*T77));
  g1(10,7)=1;
  g1(10,9)=(-(T44*y(10)*T88));
  g1(10,10)=(-(T30*T44));
  if ~isreal(g1)
    g1 = real(g1)+2*imag(g1);
  end
if nargout >= 3,
  %
  % Hessian matrix
  %

  g2 = sparse([],[],[],10,100);
if nargout >= 4,
  %
  % Third order derivatives
  %

  g3 = sparse([],[],[],10,1000);
end
end
end
end

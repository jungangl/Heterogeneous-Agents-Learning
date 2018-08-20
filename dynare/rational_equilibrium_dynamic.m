function [residual, g1, g2, g3] = rational_equilibrium_dynamic(y, x, params, steady_state, it_)
%
% Status : Computes dynamic model for Dynare
%
% Inputs :
%   y         [#dynamic variables by 1] double    vector of endogenous variables in the order stored
%                                                 in M_.lead_lag_incidence; see the Manual
%   x         [nperiods by M_.exo_nbr] double     matrix of exogenous variables (in declaration order)
%                                                 for all simulation periods
%   steady_state  [M_.endo_nbr by 1] double       vector of steady state values
%   params    [M_.param_nbr by 1] double          vector of parameter values in declaration order
%   it_       scalar double                       time period for exogenous variables for which to evaluate the model
%
% Outputs:
%   residual  [M_.endo_nbr by 1] double    vector of residuals of the dynamic model equations in order of 
%                                          declaration of the equations.
%                                          Dynare may prepend auxiliary equations, see M_.aux_vars
%   g1        [M_.endo_nbr by #dynamic variables] double    Jacobian matrix of the dynamic model equations;
%                                                           rows: equations in order of declaration
%                                                           columns: variables in order stored in M_.lead_lag_incidence followed by the ones in M_.exo_names
%   g2        [M_.endo_nbr by (#dynamic variables)^2] double   Hessian matrix of the dynamic model equations;
%                                                              rows: equations in order of declaration
%                                                              columns: variables in order stored in M_.lead_lag_incidence followed by the ones in M_.exo_names
%   g3        [M_.endo_nbr by (#dynamic variables)^3] double   Third order derivative matrix of the dynamic model equations;
%                                                              rows: equations in order of declaration
%                                                              columns: variables in order stored in M_.lead_lag_incidence followed by the ones in M_.exo_names
%
%
% Warning : this file is generated automatically by Dynare
%           from model file (.mod)

%
% Model equations
%

residual = zeros(10, 1);
T11 = y(5)^(-params(1));
T30 = y(2)^params(5);
T34 = y(6)^(-params(5));
T42 = y(2)^(params(5)-1);
T43 = y(12)*params(5)*T42;
T44 = y(6)^(1-params(5));
lhs =T11;
rhs =params(3)*y(13);
residual(1)= lhs-rhs;
lhs =T11*y(4);
rhs =params(8)*y(7)^(-params(2));
residual(2)= lhs-rhs;
lhs =y(4);
rhs =y(12)*(1-params(5))*T30*T34;
residual(3)= lhs-rhs;
lhs =y(3)+params(6);
rhs =T43*T44;
residual(4)= lhs-rhs;
lhs =y(6);
rhs =1-y(7);
residual(5)= lhs-rhs;
lhs =y(5)+y(11);
rhs =y(9)+y(2)*(1-params(6));
residual(6)= lhs-rhs;
lhs =y(8);
rhs =params(4)*y(1)+x(it_, 1);
residual(7)= lhs-rhs;
lhs =y(12);
rhs =exp(y(8));
residual(8)= lhs-rhs;
lhs =y(10);
rhs =T11*(1+y(3));
residual(9)= lhs-rhs;
lhs =y(9);
rhs =T44*y(12)*T30;
residual(10)= lhs-rhs;
if nargout >= 2,
  g1 = zeros(10, 14);

  %
  % Jacobian matrix
  %

T73 = getPowerDeriv(y(5),(-params(1)),1);
T80 = getPowerDeriv(y(6),1-params(5),1);
T91 = getPowerDeriv(y(2),params(5),1);
  g1(1,5)=T73;
  g1(1,13)=(-params(3));
  g1(2,4)=T11;
  g1(2,5)=y(4)*T73;
  g1(2,7)=(-(params(8)*getPowerDeriv(y(7),(-params(2)),1)));
  g1(3,4)=1;
  g1(3,6)=(-(y(12)*(1-params(5))*T30*getPowerDeriv(y(6),(-params(5)),1)));
  g1(3,2)=(-(T34*y(12)*(1-params(5))*T91));
  g1(3,12)=(-(T34*(1-params(5))*T30));
  g1(4,3)=1;
  g1(4,6)=(-(T43*T80));
  g1(4,2)=(-(T44*y(12)*params(5)*getPowerDeriv(y(2),params(5)-1,1)));
  g1(4,12)=(-(T44*params(5)*T42));
  g1(5,6)=1;
  g1(5,7)=1;
  g1(6,5)=1;
  g1(6,9)=(-1);
  g1(6,2)=(-(1-params(6)));
  g1(6,11)=1;
  g1(7,1)=(-params(4));
  g1(7,8)=1;
  g1(7,14)=(-1);
  g1(8,8)=(-exp(y(8)));
  g1(8,12)=1;
  g1(9,3)=(-T11);
  g1(9,5)=(-((1+y(3))*T73));
  g1(9,10)=1;
  g1(10,6)=(-(y(12)*T30*T80));
  g1(10,9)=1;
  g1(10,2)=(-(T44*y(12)*T91));
  g1(10,12)=(-(T30*T44));

if nargout >= 3,
  %
  % Hessian matrix
  %

  g2 = sparse([],[],[],10,196);
if nargout >= 4,
  %
  % Third order derivatives
  %

  g3 = sparse([],[],[],10,2744);
end
end
end
end

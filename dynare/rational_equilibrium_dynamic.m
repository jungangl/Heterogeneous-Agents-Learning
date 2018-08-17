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
T77 = getPowerDeriv(y(6),(-params(5)),1);
T80 = getPowerDeriv(y(6),1-params(5),1);
T91 = getPowerDeriv(y(2),params(5),1);
T95 = getPowerDeriv(y(2),params(5)-1,1);
T96 = y(12)*params(5)*T95;
  g1(1,5)=T73;
  g1(1,13)=(-params(3));
  g1(2,4)=T11;
  g1(2,5)=y(4)*T73;
  g1(2,7)=(-(params(8)*getPowerDeriv(y(7),(-params(2)),1)));
  g1(3,4)=1;
  g1(3,6)=(-(y(12)*(1-params(5))*T30*T77));
  g1(3,2)=(-(T34*y(12)*(1-params(5))*T91));
  g1(3,12)=(-(T34*(1-params(5))*T30));
  g1(4,3)=1;
  g1(4,6)=(-(T43*T80));
  g1(4,2)=(-(T44*T96));
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

  v2 = zeros(33,3);
T111 = getPowerDeriv(y(5),(-params(1)),2);
T116 = getPowerDeriv(y(6),(-params(5)),2);
T121 = getPowerDeriv(y(2),params(5),2);
T130 = getPowerDeriv(y(6),1-params(5),2);
T135 = getPowerDeriv(y(2),params(5)-1,2);
T136 = y(12)*params(5)*T135;
  v2(1,1)=1;
  v2(1,2)=61;
  v2(1,3)=T111;
  v2(2,1)=2;
  v2(2,2)=60;
  v2(2,3)=T73;
  v2(3,1)=2;
  v2(3,2)=47;
  v2(3,3)=  v2(2,3);
  v2(4,1)=2;
  v2(4,2)=61;
  v2(4,3)=y(4)*T111;
  v2(5,1)=2;
  v2(5,2)=91;
  v2(5,3)=(-(params(8)*getPowerDeriv(y(7),(-params(2)),2)));
  v2(6,1)=3;
  v2(6,2)=76;
  v2(6,3)=(-(y(12)*(1-params(5))*T30*T116));
  v2(7,1)=3;
  v2(7,2)=20;
  v2(7,3)=(-(T77*y(12)*(1-params(5))*T91));
  v2(8,1)=3;
  v2(8,2)=72;
  v2(8,3)=  v2(7,3);
  v2(9,1)=3;
  v2(9,2)=16;
  v2(9,3)=(-(T34*y(12)*(1-params(5))*T121));
  v2(10,1)=3;
  v2(10,2)=160;
  v2(10,3)=(-(T77*(1-params(5))*T30));
  v2(11,1)=3;
  v2(11,2)=82;
  v2(11,3)=  v2(10,3);
  v2(12,1)=3;
  v2(12,2)=156;
  v2(12,3)=(-(T34*(1-params(5))*T91));
  v2(13,1)=3;
  v2(13,2)=26;
  v2(13,3)=  v2(12,3);
  v2(14,1)=4;
  v2(14,2)=76;
  v2(14,3)=(-(T43*T130));
  v2(15,1)=4;
  v2(15,2)=20;
  v2(15,3)=(-(T80*T96));
  v2(16,1)=4;
  v2(16,2)=72;
  v2(16,3)=  v2(15,3);
  v2(17,1)=4;
  v2(17,2)=16;
  v2(17,3)=(-(T44*T136));
  v2(18,1)=4;
  v2(18,2)=160;
  v2(18,3)=(-(T80*params(5)*T42));
  v2(19,1)=4;
  v2(19,2)=82;
  v2(19,3)=  v2(18,3);
  v2(20,1)=4;
  v2(20,2)=156;
  v2(20,3)=(-(T44*params(5)*T95));
  v2(21,1)=4;
  v2(21,2)=26;
  v2(21,3)=  v2(20,3);
  v2(22,1)=8;
  v2(22,2)=106;
  v2(22,3)=(-exp(y(8)));
  v2(23,1)=9;
  v2(23,2)=59;
  v2(23,3)=(-T73);
  v2(24,1)=9;
  v2(24,2)=33;
  v2(24,3)=  v2(23,3);
  v2(25,1)=9;
  v2(25,2)=61;
  v2(25,3)=(-((1+y(3))*T111));
  v2(26,1)=10;
  v2(26,2)=76;
  v2(26,3)=(-(y(12)*T30*T130));
  v2(27,1)=10;
  v2(27,2)=20;
  v2(27,3)=(-(T80*y(12)*T91));
  v2(28,1)=10;
  v2(28,2)=72;
  v2(28,3)=  v2(27,3);
  v2(29,1)=10;
  v2(29,2)=16;
  v2(29,3)=(-(T44*y(12)*T121));
  v2(30,1)=10;
  v2(30,2)=160;
  v2(30,3)=(-(T30*T80));
  v2(31,1)=10;
  v2(31,2)=82;
  v2(31,3)=  v2(30,3);
  v2(32,1)=10;
  v2(32,2)=156;
  v2(32,3)=(-(T44*T91));
  v2(33,1)=10;
  v2(33,2)=26;
  v2(33,3)=  v2(32,3);
  g2 = sparse(v2(:,1),v2(:,2),v2(:,3),10,196);
if nargout >= 4,
  %
  % Third order derivatives
  %

  v3 = zeros(71,3);
T158 = getPowerDeriv(y(5),(-params(1)),3);
T170 = getPowerDeriv(y(2),params(5),3);
T181 = getPowerDeriv(y(6),1-params(5),3);
  v3(1,1)=1;
  v3(1,2)=845;
  v3(1,3)=T158;
  v3(2,1)=2;
  v3(2,2)=844;
  v3(2,3)=T111;
  v3(3,1)=2;
  v3(3,2)=649;
  v3(3,3)=  v3(2,3);
  v3(4,1)=2;
  v3(4,2)=831;
  v3(4,3)=  v3(2,3);
  v3(5,1)=2;
  v3(5,2)=845;
  v3(5,3)=y(4)*T158;
  v3(6,1)=2;
  v3(6,2)=1267;
  v3(6,3)=(-(params(8)*getPowerDeriv(y(7),(-params(2)),3)));
  v3(7,1)=3;
  v3(7,2)=1056;
  v3(7,3)=(-(y(12)*(1-params(5))*T30*getPowerDeriv(y(6),(-params(5)),3)));
  v3(8,1)=3;
  v3(8,2)=272;
  v3(8,3)=(-(y(12)*(1-params(5))*T91*T116));
  v3(9,1)=3;
  v3(9,2)=1000;
  v3(9,3)=  v3(8,3);
  v3(10,1)=3;
  v3(10,2)=1052;
  v3(10,3)=  v3(8,3);
  v3(11,1)=3;
  v3(11,2)=216;
  v3(11,3)=(-(T77*y(12)*(1-params(5))*T121));
  v3(12,1)=3;
  v3(12,2)=268;
  v3(12,3)=  v3(11,3);
  v3(13,1)=3;
  v3(13,2)=996;
  v3(13,3)=  v3(11,3);
  v3(14,1)=3;
  v3(14,2)=212;
  v3(14,3)=(-(T34*y(12)*(1-params(5))*T170));
  v3(15,1)=3;
  v3(15,2)=2232;
  v3(15,3)=(-((1-params(5))*T30*T116));
  v3(16,1)=3;
  v3(16,2)=1062;
  v3(16,3)=  v3(15,3);
  v3(17,1)=3;
  v3(17,2)=1140;
  v3(17,3)=  v3(15,3);
  v3(18,1)=3;
  v3(18,2)=2176;
  v3(18,3)=(-(T77*(1-params(5))*T91));
  v3(19,1)=3;
  v3(19,2)=278;
  v3(19,3)=  v3(18,3);
  v3(20,1)=3;
  v3(20,2)=356;
  v3(20,3)=  v3(18,3);
  v3(21,1)=3;
  v3(21,2)=1006;
  v3(21,3)=  v3(18,3);
  v3(22,1)=3;
  v3(22,2)=1136;
  v3(22,3)=  v3(18,3);
  v3(23,1)=3;
  v3(23,2)=2228;
  v3(23,3)=  v3(18,3);
  v3(24,1)=3;
  v3(24,2)=2172;
  v3(24,3)=(-(T34*(1-params(5))*T121));
  v3(25,1)=3;
  v3(25,2)=222;
  v3(25,3)=  v3(24,3);
  v3(26,1)=3;
  v3(26,2)=352;
  v3(26,3)=  v3(24,3);
  v3(27,1)=4;
  v3(27,2)=1056;
  v3(27,3)=(-(T43*T181));
  v3(28,1)=4;
  v3(28,2)=272;
  v3(28,3)=(-(T96*T130));
  v3(29,1)=4;
  v3(29,2)=1000;
  v3(29,3)=  v3(28,3);
  v3(30,1)=4;
  v3(30,2)=1052;
  v3(30,3)=  v3(28,3);
  v3(31,1)=4;
  v3(31,2)=216;
  v3(31,3)=(-(T80*T136));
  v3(32,1)=4;
  v3(32,2)=268;
  v3(32,3)=  v3(31,3);
  v3(33,1)=4;
  v3(33,2)=996;
  v3(33,3)=  v3(31,3);
  v3(34,1)=4;
  v3(34,2)=212;
  v3(34,3)=(-(T44*y(12)*params(5)*getPowerDeriv(y(2),params(5)-1,3)));
  v3(35,1)=4;
  v3(35,2)=2232;
  v3(35,3)=(-(params(5)*T42*T130));
  v3(36,1)=4;
  v3(36,2)=1062;
  v3(36,3)=  v3(35,3);
  v3(37,1)=4;
  v3(37,2)=1140;
  v3(37,3)=  v3(35,3);
  v3(38,1)=4;
  v3(38,2)=2176;
  v3(38,3)=(-(T80*params(5)*T95));
  v3(39,1)=4;
  v3(39,2)=278;
  v3(39,3)=  v3(38,3);
  v3(40,1)=4;
  v3(40,2)=356;
  v3(40,3)=  v3(38,3);
  v3(41,1)=4;
  v3(41,2)=1006;
  v3(41,3)=  v3(38,3);
  v3(42,1)=4;
  v3(42,2)=1136;
  v3(42,3)=  v3(38,3);
  v3(43,1)=4;
  v3(43,2)=2228;
  v3(43,3)=  v3(38,3);
  v3(44,1)=4;
  v3(44,2)=2172;
  v3(44,3)=(-(T44*params(5)*T135));
  v3(45,1)=4;
  v3(45,2)=222;
  v3(45,3)=  v3(44,3);
  v3(46,1)=4;
  v3(46,2)=352;
  v3(46,3)=  v3(44,3);
  v3(47,1)=8;
  v3(47,2)=1478;
  v3(47,3)=(-exp(y(8)));
  v3(48,1)=9;
  v3(48,2)=843;
  v3(48,3)=(-T111);
  v3(49,1)=9;
  v3(49,2)=453;
  v3(49,3)=  v3(48,3);
  v3(50,1)=9;
  v3(50,2)=817;
  v3(50,3)=  v3(48,3);
  v3(51,1)=9;
  v3(51,2)=845;
  v3(51,3)=(-((1+y(3))*T158));
  v3(52,1)=10;
  v3(52,2)=1056;
  v3(52,3)=(-(y(12)*T30*T181));
  v3(53,1)=10;
  v3(53,2)=272;
  v3(53,3)=(-(y(12)*T91*T130));
  v3(54,1)=10;
  v3(54,2)=1000;
  v3(54,3)=  v3(53,3);
  v3(55,1)=10;
  v3(55,2)=1052;
  v3(55,3)=  v3(53,3);
  v3(56,1)=10;
  v3(56,2)=216;
  v3(56,3)=(-(T80*y(12)*T121));
  v3(57,1)=10;
  v3(57,2)=268;
  v3(57,3)=  v3(56,3);
  v3(58,1)=10;
  v3(58,2)=996;
  v3(58,3)=  v3(56,3);
  v3(59,1)=10;
  v3(59,2)=212;
  v3(59,3)=(-(T44*y(12)*T170));
  v3(60,1)=10;
  v3(60,2)=2232;
  v3(60,3)=(-(T30*T130));
  v3(61,1)=10;
  v3(61,2)=1062;
  v3(61,3)=  v3(60,3);
  v3(62,1)=10;
  v3(62,2)=1140;
  v3(62,3)=  v3(60,3);
  v3(63,1)=10;
  v3(63,2)=2176;
  v3(63,3)=(-(T80*T91));
  v3(64,1)=10;
  v3(64,2)=278;
  v3(64,3)=  v3(63,3);
  v3(65,1)=10;
  v3(65,2)=356;
  v3(65,3)=  v3(63,3);
  v3(66,1)=10;
  v3(66,2)=1006;
  v3(66,3)=  v3(63,3);
  v3(67,1)=10;
  v3(67,2)=1136;
  v3(67,3)=  v3(63,3);
  v3(68,1)=10;
  v3(68,2)=2228;
  v3(68,3)=  v3(63,3);
  v3(69,1)=10;
  v3(69,2)=2172;
  v3(69,3)=(-(T44*T121));
  v3(70,1)=10;
  v3(70,2)=222;
  v3(70,3)=  v3(69,3);
  v3(71,1)=10;
  v3(71,2)=352;
  v3(71,3)=  v3(69,3);
  g3 = sparse(v3(:,1),v3(:,2),v3(:,3),10,2744);
end
end
end
end

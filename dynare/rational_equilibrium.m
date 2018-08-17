%
% Status : main Dynare file
%
% Warning : this file is generated automatically by Dynare
%           from model file (.mod)

if isoctave || matlab_ver_less_than('8.6')
    clear all
else
    clearvars -global
    clear_persistent_variables(fileparts(which('dynare')), false)
end
tic0 = tic;
% Save empty dates and dseries objects in memory.
dates('initialize');
dseries('initialize');
% Define global variables.
global M_ options_ oo_ estim_params_ bayestopt_ dataset_ dataset_info estimation_info ys0_ ex0_
options_ = [];
M_.fname = 'rational_equilibrium';
M_.dynare_version = '4.5.1';
oo_.dynare_version = '4.5.1';
options_.dynare_version = '4.5.1';
%
% Some global variables initialization
%
global_initialization;
diary off;
diary('rational_equilibrium.log');
M_.exo_names = 'eps';
M_.exo_names_tex = 'eps';
M_.exo_names_long = 'eps';
M_.endo_names = 'r';
M_.endo_names_tex = 'r';
M_.endo_names_long = 'r';
M_.endo_names = char(M_.endo_names, 'w');
M_.endo_names_tex = char(M_.endo_names_tex, 'w');
M_.endo_names_long = char(M_.endo_names_long, 'w');
M_.endo_names = char(M_.endo_names, 'c');
M_.endo_names_tex = char(M_.endo_names_tex, 'c');
M_.endo_names_long = char(M_.endo_names_long, 'c');
M_.endo_names = char(M_.endo_names, 'n');
M_.endo_names_tex = char(M_.endo_names_tex, 'n');
M_.endo_names_long = char(M_.endo_names_long, 'n');
M_.endo_names = char(M_.endo_names, 'l');
M_.endo_names_tex = char(M_.endo_names_tex, 'l');
M_.endo_names_long = char(M_.endo_names_long, 'l');
M_.endo_names = char(M_.endo_names, 'ltheta');
M_.endo_names_tex = char(M_.endo_names_tex, 'ltheta');
M_.endo_names_long = char(M_.endo_names_long, 'ltheta');
M_.endo_names = char(M_.endo_names, 'y');
M_.endo_names_tex = char(M_.endo_names_tex, 'y');
M_.endo_names_long = char(M_.endo_names_long, 'y');
M_.endo_names = char(M_.endo_names, 'nu');
M_.endo_names_tex = char(M_.endo_names_tex, 'nu');
M_.endo_names_long = char(M_.endo_names_long, 'nu');
M_.endo_names = char(M_.endo_names, 'k');
M_.endo_names_tex = char(M_.endo_names_tex, 'k');
M_.endo_names_long = char(M_.endo_names_long, 'k');
M_.endo_names = char(M_.endo_names, 'theta');
M_.endo_names_tex = char(M_.endo_names_tex, 'theta');
M_.endo_names_long = char(M_.endo_names_long, 'theta');
M_.endo_partitions = struct();
M_.param_names = 'sigma';
M_.param_names_tex = 'sigma';
M_.param_names_long = 'sigma';
M_.param_names = char(M_.param_names, 'gamma');
M_.param_names_tex = char(M_.param_names_tex, 'gamma');
M_.param_names_long = char(M_.param_names_long, 'gamma');
M_.param_names = char(M_.param_names, 'beta');
M_.param_names_tex = char(M_.param_names_tex, 'beta');
M_.param_names_long = char(M_.param_names_long, 'beta');
M_.param_names = char(M_.param_names, 'rho');
M_.param_names_tex = char(M_.param_names_tex, 'rho');
M_.param_names_long = char(M_.param_names_long, 'rho');
M_.param_names = char(M_.param_names, 'alpha');
M_.param_names_tex = char(M_.param_names_tex, 'alpha');
M_.param_names_long = char(M_.param_names_long, 'alpha');
M_.param_names = char(M_.param_names, 'delta');
M_.param_names_tex = char(M_.param_names_tex, 'delta');
M_.param_names_long = char(M_.param_names_long, 'delta');
M_.param_names = char(M_.param_names, 'sigma_eps');
M_.param_names_tex = char(M_.param_names_tex, 'sigma\_eps');
M_.param_names_long = char(M_.param_names_long, 'sigma_eps');
M_.param_names = char(M_.param_names, 'X');
M_.param_names_tex = char(M_.param_names_tex, 'X');
M_.param_names_long = char(M_.param_names_long, 'X');
M_.param_partitions = struct();
M_.exo_det_nbr = 0;
M_.exo_nbr = 1;
M_.endo_nbr = 10;
M_.param_nbr = 8;
M_.orig_endo_nbr = 10;
M_.aux_vars = [];
M_.Sigma_e = zeros(1, 1);
M_.Correlation_matrix = eye(1, 1);
M_.H = 0;
M_.Correlation_matrix_ME = 1;
M_.sigma_e_is_diagonal = 1;
M_.det_shocks = [];
options_.block=0;
options_.bytecode=0;
options_.use_dll=0;
M_.hessian_eq_zero = 0;
erase_compiled_function('rational_equilibrium_static');
erase_compiled_function('rational_equilibrium_dynamic');
M_.orig_eq_nbr = 10;
M_.eq_nbr = 10;
M_.ramsey_eq_nbr = 0;
M_.lead_lag_incidence = [
 0 3 0;
 0 4 0;
 0 5 0;
 0 6 0;
 0 7 0;
 1 8 0;
 0 9 0;
 0 10 13;
 2 11 0;
 0 12 0;]';
M_.nstatic = 7;
M_.nfwrd   = 1;
M_.npred   = 2;
M_.nboth   = 0;
M_.nsfwrd   = 1;
M_.nspred   = 2;
M_.ndynamic   = 3;
M_.equations_tags = {
};
M_.static_and_dynamic_models_differ = 0;
M_.exo_names_orig_ord = [1:1];
M_.maximum_lag = 1;
M_.maximum_lead = 1;
M_.maximum_endo_lag = 1;
M_.maximum_endo_lead = 1;
oo_.steady_state = zeros(10, 1);
M_.maximum_exo_lag = 0;
M_.maximum_exo_lead = 0;
oo_.exo_steady_state = zeros(1, 1);
M_.params = NaN(8, 1);
M_.NNZDerivatives = [31; 33; 71];
M_.params( 1 ) = 2.;
sigma = M_.params( 1 );
M_.params( 2 ) = 2.;
gamma = M_.params( 2 );
M_.params( 3 ) = 0.98;
beta = M_.params( 3 );
M_.params( 4 ) = 0.85;
rho = M_.params( 4 );
M_.params( 5 ) = 0.33;
alpha = M_.params( 5 );
M_.params( 6 ) = 0.023;
delta = M_.params( 6 );
M_.params( 7 ) = 0.014;
sigma_eps = M_.params( 7 );
M_.params( 8 ) = 1.97;
X = M_.params( 8 );
%
% INITVAL instructions
%
options_.initval_file = 0;
oo_.steady_state( 9 ) = 6.19;
oo_.steady_state( 3 ) = 0.0672;
oo_.steady_state( 5 ) = 0.7;
oo_.steady_state( 6 ) = 0;
oo_.exo_steady_state( 1 ) = 0;
oo_.steady_state( 1 ) = 1/M_.params(3)-1;
oo_.steady_state( 2 ) = 1.82;
oo_.steady_state( 10 ) = 1;
oo_.steady_state( 8 ) = 1.;
oo_.steady_state( 4 ) = 1-oo_.steady_state(5);
if M_.exo_nbr > 0
	oo_.exo_simul = ones(M_.maximum_lag,1)*oo_.exo_steady_state';
end
if M_.exo_det_nbr > 0
	oo_.exo_det_simul = ones(M_.maximum_lag,1)*oo_.exo_det_steady_state';
end
steady;
oo_.dr.eigval = check(M_,options_,oo_);
%
% SHOCKS instructions
%
M_.exo_det_length = 0;
M_.Sigma_e(1, 1) = M_.params(7)^2;
options_.k_order_solver = 1;
options_.drop = 1000;
options_.order = 3;
options_.periods = 10000;
var_list_ = char();
info = stoch_simul(var_list_);
save('rational_equilibrium_results.mat', 'oo_', 'M_', 'options_');
if exist('estim_params_', 'var') == 1
  save('rational_equilibrium_results.mat', 'estim_params_', '-append');
end
if exist('bayestopt_', 'var') == 1
  save('rational_equilibrium_results.mat', 'bayestopt_', '-append');
end
if exist('dataset_', 'var') == 1
  save('rational_equilibrium_results.mat', 'dataset_', '-append');
end
if exist('estimation_info', 'var') == 1
  save('rational_equilibrium_results.mat', 'estimation_info', '-append');
end
if exist('dataset_info', 'var') == 1
  save('rational_equilibrium_results.mat', 'dataset_info', '-append');
end
if exist('oo_recursive_', 'var') == 1
  save('rational_equilibrium_results.mat', 'oo_recursive_', '-append');
end


disp(['Total computing time : ' dynsec2hms(toc(tic0)) ]);
if ~isempty(lastwarn)
  disp('Note: warning(s) encountered in MATLAB/Octave code')
end
diary off

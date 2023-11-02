% Demo of how to change distribution into two permanent types
% Idea here is that there will be a reform that affects people differently
% based on their 'state/status/location' when the reform is implemented.
% We will compute the transition path for this reform.
%
% Specifically, here we will introduce a tax, that taxes people based on
% whether they were categorized as high or low assets when the reform was
% implemented. [This is a bit silly, obviously in practice it would be more
% standard to tax them based on their current assets, not what there assets
% were prior to the reform. But this is just a simple example about how to
% code such a thing so I want something easy to follow.]
%
% I just use the Aiyagari (1994) model, and then modify it to have two
% permanent types, one of which will be taxed while the other will not.


%% First, just copy-paste of code to solve the Aiyagari model
n_d=0;
n_k=2^10;
n_z=11;

% Parameters
Params.beta=0.96; %Model period is one-sixth of a year
Params.alpha=0.36;
Params.delta=0.08;
Params.mu=3;
Params.sigma=0.2;
Params.rho=0.6;

% Set up the exogenous shock process
% Create markov process for the exogenous labour productivity, l.
[z_grid,pi_z]=discretizeAR1_FarmerToda(0,Params.rho,sqrt((1-Params.rho^2)*Params.sigma^2),n_z);
% Note: sigma is standard deviations of s, input needs to be standard deviation of the innovations
% Because s is AR(1), the variance of the innovations is (1-rho^2)*sigma^2

[z_mean,z_variance,z_corr,~]=MarkovChainMoments(z_grid,pi_z);
z_grid=exp(z_grid);
% Get some info on the markov process
[Expectation_l,~,~,~]=MarkovChainMoments(z_grid,pi_z); %Since l is exogenous, this will be it's eqm value 
% Note: Aiyagari (1994) actually then normalizes l by dividing it by Expectation_l (so that the resulting process has expectation equal to 1)
z_grid=z_grid./Expectation_l;
[Expectation_l,~,~,~]=MarkovChainMoments(z_grid,pi_z);
% If you look at Expectation_l you will see it is now equal to 1
Params.Expectation_l=Expectation_l;

% In the absence of idiosyncratic risk, the steady state equilibrium is given by
r_ss=1/Params.beta-1;
K_ss=((r_ss+Params.delta)/Params.alpha)^(1/(Params.alpha-1)); %The steady state capital in the absence of aggregate uncertainty.

% Set grid for asset holdings
k_grid=10*K_ss*(linspace(0,1,n_k).^3)'; % linspace ^3 puts more points near zero, where the curvature of value and policy functions is higher and where model spends more time

a_grid=k_grid;
n_a=n_k;

% Create functions to be evaluated
FnsToEvaluate.K = @(aprime,a,s) a; %We just want the aggregate assets (which is this periods state)
% Now define the functions for the General Equilibrium conditions
GeneralEqmEqns.CapitalMarket = @(r,K,alpha,delta,Expectation_l) r-(alpha*(K^(alpha-1))*(Expectation_l^(1-alpha))-delta); %The requirement that the interest rate corresponds to the agg capital level
% Inputs can be any parameter, price, or aggregate of the FnsToEvaluate

DiscountFactorParamNames={'beta'};
ReturnFn=@(aprime, a, s, alpha,delta,mu,r) Aiyagari1994_ReturnFn(aprime, a, s,alpha,delta,mu,r);

% Initial guess for GE interest rate r
GEPriceParamNames={'r'};
Params.r=0.038;

% Solve for the stationary general equilbirium
vfoptions=struct(); % Use default options for solving the value function (and policy fn)
simoptions=struct(); % Use default options for solving for stationary distribution
heteroagentoptions.verbose=1; % verbose means that you want it to give you feedback on what is going on

fprintf('Calculating price vector corresponding to the stationary general eqm \n')
[p_eqm,~,GeneralEqmCondn]=HeteroAgentStationaryEqm_Case1(n_d, n_a, n_z, 0, pi_z, [], a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, [], [], [], GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);
% Now that we have the GE, let's calculate a bunch of related objects
Params.r=p_eqm.r; % Put the equilibrium interest rate into Params so we can use it to calculate things based on equilibrium parameters
Params.w=(1-Params.alpha)*((Params.r+Params.delta)/Params.alpha)^(Params.alpha/(Params.alpha-1));% Equilibrium wage

fprintf('Calculating various equilibrium objects \n')
[V,Policy]=ValueFnIter_Case1(n_d,n_a,n_z,[],a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, [], vfoptions);
StationaryDist=StationaryDist_Case1(Policy,n_d,n_a,n_z,pi_z, simoptions);

%% Done. 

% Let's just plot cdf of agent distribution over assets.
% This is relevant to how we are going to divide the two permanent types.
figure(1)
plot(a_grid,cumsum(sum(StationaryDist,2)))
% Based on looking at this I have decided to set the two permanent types as
% being more/less than assets of 5 (so there will be decent mass of each)

%% Next we want to create two permanent types based on their 'positions' in this initial equilibrium.
% Specifically, I will create 'lowa' for those with assets of less than 5, and 'higha' for those with 
% more than this mean assets. I will then setup an new initial distribution
% based on these two permanent types (with appropriate mass of each).

Names_i={'lowa','higha'};

% First, set up the new version of agent distribution based on these two types.
InitialDist.lowa=StationaryDist.*(a_grid<5);
InitialDist.higha=StationaryDist.*(a_grid>=5);
% Plot so we can see what is going on
figure(2)
plot(a_grid,cumsum(sum(StationaryDist,2)))
hold on
plot(a_grid,cumsum(sum(InitialDist.lowa,2)))
plot(a_grid,cumsum(sum(InitialDist.higha,2)))
hold off
legend('wholedist','lowa','higha')
% For PType agent distribution, the toolkit has them being of mass one
% conditional on ptype, and then stores the actual masses as a seperate
% field of the structure, so need to do this.
InitialDist.ptweights=[sum(InitialDist.lowa(:)),sum(InitialDist.higha(:))];
InitialDist.lowa=InitialDist.lowa./sum(InitialDist.lowa(:));
InitialDist.higha=InitialDist.higha./sum(InitialDist.higha(:));

% We also need to store the mass of each permanent type in Params
Params.ptypemass=InitialDist.ptweights;
PTypeDistParamNames={'ptypemass'};

% I am going to set up a new version of the return function, that uses the
% permanent types to make the tax rise over the transition different for
% each type. Specifically, the tax will only be paid by those who are
% 'higha' type.
% [Note that the type is based on initial assets, not current
% assets, hence why permanent types are being used. If it was just based on
% current assets you could just use an if-statement inside the return
% function to deal with this.]

Params.agentptype=[0,1];

ReturnFn=@(aprime, a, s, alpha,delta,mu,r,tau,agentptype) Aiyagari1994TwoTypes_ReturnFn(aprime, a, s,alpha,delta,mu,r, tau,agentptype);

% Next, compute the final equilibrium with these taxes
Params.tau=0.3;

% Note, switch to PType commands
[p_eqm_final,~,GeneralEqmCondn]=HeteroAgentStationaryEqm_Case1_PType(n_d, n_a, n_z, Names_i, 0, pi_z, [], a_grid, z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Params, DiscountFactorParamNames, PTypeDistParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions);
% Now that we have the GE, let's calculate a bunch of related objects
Params.r=p_eqm.r; % Put the equilibrium interest rate into Params so we can use it to calculate things based on equilibrium parameters
Params.w=(1-Params.alpha)*((Params.r+Params.delta)/Params.alpha)^(Params.alpha/(Params.alpha-1));% Equilibrium wage

fprintf('Calculating various equilibrium objects \n')
[V_final,Policy_final]=ValueFnIter_Case1_PType(n_d,n_a,n_z,Names_i,[],a_grid,z_grid, pi_z, ReturnFn, Params, DiscountFactorParamNames, vfoptions);
StationaryDist_final=StationaryDist_Case1_PType(PTypeDistParamNames,Policy_final,n_d,n_a,n_z,Names_i,pi_z,Params, simoptions);

%% Now we are ready to do the transition path
T=150;

ParamPath.tau=Params.tau*ones(1,T);

PricePath0.r=[linspace(p_eqm.r,p_eqm_final.r,50), p_eqm_final.r*ones(1,T-50)];

% General equilibrium conditions (for the transition path)
TransPathGeneralEqmEqns.CapitalMarket = @(r,K,alpha,delta,Expectation_l) r-(alpha*(K^(alpha-1))*(Expectation_l^(1-alpha))-delta);
% Note: For this model the transition path has the same general equilibrium conditions as the stationary equilibrium, but this will not always be true for more complex models.

transpathoptions.GEnewprice=3;
% Need to explain to transpathoptions how to use the GeneralEqmEqns to
% update the general eqm transition prices (in PricePath).
transpathoptions.GEnewprice3.howtoupdate=... % a row is: GEcondn, price, add, factor
    {'CaptialMarket','r',0,0.1}; % CaptialMarket GE condition will be positive if r is too big, so subtract
% Note: the update is essentially new_price=price+factor*add*GEcondn_value-factor*(1-add)*GEcondn_value
% Notice that this adds factor*GEcondn_value when add=1 and subtracts it what add=0
% A small 'factor' will make the convergence to solution take longer, but too large a value will make it 
% unstable (fail to converge). Technically this is the damping factor in a shooting algorithm.

% Now just run the TransitionPath_Case1 command (all of the other inputs
% are things we had already had to define to be able to solve for the
% initial and final equilibria)
transpathoptions.weightscheme=1;
transpathoptions.verbose=1;


vfoptions=struct();
simoptions=struct();
PricePath=TransitionPath_Case1_PType(PricePath0, ParamPath, T, V_final, InitialDist, n_d, n_a, n_z, Names_i, [],a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, TransPathGeneralEqmEqns, Params, DiscountFactorParamNames, transpathoptions, simoptions, vfoptions);


% You can calculate the value and policy functions for the transition path
[VPath,PolicyPath]=ValueFnOnTransPath_Case1_PType(PricePath, ParamPath, T, V_final, Policy_final, Params, n_d, n_a, n_z, Names_i, pi_z, [], a_grid,z_grid, DiscountFactorParamNames, ReturnFn, vfoptions);

% You can then use these to calculate the agent distribution for the transition path
AgentDistPath=AgentDistOnTransPath_Case1_PType(InitialDist,PricePath, ParamPath, PolicyPath,n_d,n_a,n_z,Names_i,pi_z,T, Params, simoptions);

% And then we can calculate AggVars for the path
AggVarsPath=EvalFnOnTransPath_AggVars_Case1_PType(FnsToEvaluate, PricePath, ParamPath, Params, T, PolicyPath, AgentDistPath, n_d, n_a, n_z, Names_i, pi_z, [], a_grid,z_grid, simoptions);




















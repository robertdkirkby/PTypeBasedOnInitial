function F=Aiyagari1994TwoTypes_ReturnFn(aprime, a, s,alpha,delta,mu,r,tau,agentptype)
% The return function is essentially the combination of the utility
% function and the constraints.

F=-Inf;
w=(1-alpha)*((r+delta)/alpha)^(alpha/(alpha-1));
c=(1-tau*agentptype)*w*s+(1+r)*a-aprime; % Budget Constraint
% c=w l_t+(1+r)a_t-a_{t+1}
if c>0
    if mu==1
        F=log(c);
    else
        F=(c^(1-mu) -1)/(1-mu);
    end
end

end
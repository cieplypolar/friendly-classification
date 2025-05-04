We want to observe:
- lower asymptotic error in linear regression
- faster convergence of Bayes log(n)

We will use Bayes with Laplace smoothing (maybe we will look at raw Bayes??)

Proposition 1:  
error_dis <= error_gen  
proof.  
from the fact that we approch infimum, and Bayes is just some classifier

Proposition 2:  
error_dis <= error_dis_population + O(sqrt(n\m log(m\n)))  
proof.  
from VC dimensions (never heard about VC dimensions)

Lemma 3:  
we proof log(n) convergence  



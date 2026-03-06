import scipy.stats as si
import numpy as np


S = 100 #Spot price
K = 98 #Strike price
T = 1 #Duration in year
rf = 0.04 #Risk-free rate 
vol = 0.4 #Volatility calculated

def black_scholes(S, K, T, rf, vol) :
    d1 = (np.log(S / K) + (rf + 0.5 * vol ** 2) * T) / (vol * np.sqrt(T))
    #d1 score ajuusté 
    d2 = d1 - vol * np.sqrt(T)
    #d2 représente la proba que l'option soit in the money à l'expiration
    
    #Calcul prix d'un call/put avec loi normale cumulative
    call_price = (S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-rf * T) * si.norm.cdf(d2, 0.0, 1.0))
    put_price = (K * np.exp(-rf * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0))
    return call_price, put_price


# Call the function
call_price, put_price = black_scholes(S, K, T, rf, vol)
print('Call price =', call_price.round(4), 'Put price =', put_price.round(4))

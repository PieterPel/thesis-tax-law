import numpy as np
from scipy.stats import norm

N_prime = norm.pdf
N = norm.cdf


def black_scholes_price(S, K, T, r, sigma, option_type="call") -> float:
    """
    :param S: Asset price
    :param K: Strike price
    :param tau: Time to maturity
    :param r: risk-free rate (treasury bills)
    :param sigma: volatility
    :return: call price
    """

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return price


# Function to calculate payoff
def calculate_payoff(
    S, K, option_type="call", position="long", option_price=0
):
    if option_type == "call":
        payoff = np.maximum(S - K, 0)
    elif option_type == "put":
        payoff = np.maximum(K - S, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    if position == "long":
        return payoff - option_price  # Negative option price for long position
    elif position == "short":
        return option_price - payoff
    else:
        raise ValueError("position must be 'long' or 'short'")


# Black-Scholes Delta function
def black_scholes_delta(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if option_type == "call":
        delta = norm.cdf(d1)
    elif option_type == "put":
        delta = norm.cdf(d1) - 1
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return delta


# Function to calculate the exercise probability
def exercise_probability(S, K, T, r, sigma, option_type="call"):
    d2 = (np.log(S / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return norm.cdf(d2)
    elif option_type == "put":
        return norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

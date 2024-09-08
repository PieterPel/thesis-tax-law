import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.integrate import simpson

from src.black_scholes import (
    black_scholes_price,
    black_scholes_delta,
    calculate_payoff,
)


def get_option_payoff_plot(
    S: np.linspace, K: float, option_price: float
) -> plt:

    # Payoff formulas for the titles
    formulas = {
        "Long Call": r"Payoff = $max(S_T - K, 0) - C$",
        "Short Call": r"Payoff = $C - max(S_T - K, 0)$",
        "Long Put": r"Payoff = $max(K - S_T, 0) - P$",
        "Short Put": r"Payoff = $P - max(K - S_T, 0)$",
    }

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot settings
    plot_settings = [
        ("Long Call", "call", "long", "blue"),
        ("Short Call", "call", "short", "red"),
        ("Long Put", "put", "long", "green"),
        ("Short Put", "put", "short", "orange"),
    ]

    # Create plots
    for ax, (title, option_type, position, color) in zip(
        axes.flatten(), plot_settings
    ):
        payoff = calculate_payoff(S, K, option_type, position, option_price)
        ax.plot(S, payoff, label=title, color=color)

        # Add horizontal line at y=0
        ax.axhline(0, color="black", lw=0.5)

        # Add vertical line at the strike price
        ax.axvline(
            K, color="black", linestyle="--", lw=0.5, label="Strike Price"
        )

        # Set titles and labels
        ax.set_title(f"{title}\n{formulas[title]}", fontsize=14)
        ax.set_xlabel("Aandelenprijs bij einde looptijd optie")
        ax.set_ylabel("Payoff")

        if "Long" in title:
            sign = "-"
            multiplier = -1
        else:
            sign = ""
            multiplier = 1

        # Custom ticks and labels
        if "Call" in title:
            ax.set_yticks([multiplier * option_price, 0])
            ax.set_yticklabels([f"{sign}C", "0"])
        elif "Put" in title:
            ax.set_yticks([multiplier * option_price, 0])
            ax.set_yticklabels([f"{sign}P", "0"])

        # Custom x-axis tick only at strike price
        ax.set_xticks([K])
        ax.set_xticklabels(["K"])

        # Hide default tick labels
        ax.tick_params(
            axis="x",
            which="both",
            bottom=True,
            top=False,
            labelbottom=True,
            labelsize=12,
        )
        ax.tick_params(
            axis="y",
            which="both",
            left=True,
            right=False,
            labelleft=True,
            labelsize=12,
        )

        # Hide plot borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Customize x-axis to look like an arrow
        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["left"].set_linewidth(1.5)
        ax.xaxis.set_tick_params(width=0)  # Hide default x-ticks

        # Add grid for clarity
        ax.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout
    plt.tight_layout()

    return plt


def get_black_scholes_price_plot(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type="call",
    S_integration=None,
) -> plt:

    # Integration range
    if S_integration is None:
        S_integration = np.linspace(0.001, 2 * S0, 10000)

    option_price = black_scholes_price(S0, K, T, r, sigma, option_type)

    # Plotting range (narrower)
    S_plot = np.linspace(S0 * 0.5, S0 * 1.5, 10000)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    # Risk-neutral probabilities (log-normal distribution)
    mean = np.log(S0) + (r - 0.5 * sigma**2) * T
    std_dev = sigma * np.sqrt(T)

    distribution_integration = (
        1 / (S_integration * std_dev * np.sqrt(2 * np.pi))
    ) * np.exp(-((np.log(S_integration) - mean) ** 2) / (2 * std_dev**2))
    distribution_plot = (1 / (S_plot * std_dev * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(S_plot) - mean) ** 2) / (2 * std_dev**2)
    )

    # Payoff functions
    if option_type == "call":
        discounted_payoff_integration = (
            np.maximum(S_integration - K, 0) * np.exp(-r * T) - option_price
        )
        discounted_payoff_plot = (
            np.maximum(S_plot - K, 0) * np.exp(-r * T) - option_price
        )
    elif option_type == "put":
        discounted_payoff_integration = (
            np.maximum(K - S_integration, 0) * np.exp(-r * T) - option_price
        )
        discounted_payoff_plot = (
            np.maximum(K - S_plot, 0) * np.exp(-r * T) - option_price
        )
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    # Discounted product of distribution and payoff for integration
    product_integration = (
        distribution_integration * discounted_payoff_integration
    )

    # Calculate the area under the product curve to verify it equals 0
    integral = simpson(y=product_integration, x=S_integration)
    print(
        f"Calculated area under the product curve by integration: {integral}"
    )

    # Normalize for better visualization (optional)
    distribution_plot_norm = distribution_plot / np.max(distribution_plot)
    discounted_payoff_plot_norm = discounted_payoff_plot / np.max(
        np.abs(discounted_payoff_plot)
    )
    product_plot_norm = (distribution_plot * discounted_payoff_plot) / (
        np.max(np.abs(distribution_plot * discounted_payoff_plot)) * 4
    )

    # Split product into parts above and below the y-axis
    product_above = np.maximum(product_plot_norm, 0)
    product_below = np.minimum(product_plot_norm, 0)

    # Plot
    plt.figure(figsize=(9, 4.5))

    # Plot Risk-neutral PDF
    plt.plot(
        S_plot,
        distribution_plot_norm,
        label="Verdeling aandelenprijs",
        color="black",
        linestyle="-",
    )

    # Plot Option Payoff
    plt.plot(
        S_plot,
        discounted_payoff_plot_norm,
        label=f"Verdisconteerde calloptie payoff",
        color="purple",
        linestyle="dashdot",
    )

    # Plot Product of PDF and Payoff (above y-axis in green)
    plt.fill_between(S_plot, product_above, alpha=0.3, color="green")

    # Plot Product of PDF and Payoff (below y-axis in red)
    plt.fill_between(S_plot, product_below, alpha=0.3, color="red")

    # Labels and Title
    plt.xlabel("Aandelenprijs bij einde looptijd optie")
    # plt.ylabel('Normalized Values')
    plt.title(
        "Log-normale verdeling, verdisconteerde optiepayoff en hun product"
    )
    # plt.grid(True)

    plt.yticks([0], ["0"])

    # Get current x-ticks
    current_ticks = plt.xticks()[0]

    # Combine current x-ticks with the custom tick for K
    new_ticks = np.append(current_ticks, K)

    # Set x-ticks with custom labels for K
    plt.xticks(
        new_ticks, [f"{tick:.0f}" if tick != K else "K" for tick in new_ticks]
    )

    plt.axvline(
        K, color="black", linestyle="--", label="Uitoefenprijs", alpha=0.5
    )

    plt.legend()

    plt.tight_layout()

    return plt


def get_method_compare_plot(
    asset_prices: np.linspace,
    strike_price: float,
    tau: float,
    rf_rate: float,
    sigma: float,
):

    # Calculate for each strike price
    bs_prices = [
        black_scholes_price(
            asset_price, strike_price, tau, rf_rate, sigma, "call"
        )
        for asset_price in asset_prices
    ]
    deltas = [
        black_scholes_delta(
            asset_price, strike_price, tau, rf_rate, sigma, "call"
        )
        for asset_price in asset_prices
    ]
    ratios = [
        (bs_price) / (asset_price)
        for bs_price, asset_price in zip(bs_prices, asset_prices)
    ]

    # Plot
    plt.figure(figsize=(10, 6))

    # Plot Delta
    plt.plot(asset_prices, deltas, label="Delta", color="blue")

    # Plot Black-Scholes price / asset price
    plt.plot(
        asset_prices,
        ratios,
        label="Black-Scholes prijs / aandelenprijs",
        color="green",
        linestyle="--",
    )

    # Customize axes labels and legend
    plt.xlabel("Aandelenprijs", fontsize=12)
    plt.ylabel("Waarde", fontsize=12)
    plt.legend(fontsize=12)
    plt.tick_params(axis="both", which="major", labelsize=12)

    # Add grid
    # plt.grid(True, linestyle='--', alpha=0.5)

    # Add title
    plt.title(
        "Delta en Black-Scholes-prijs / Aandelenprijs voor verschillende aandelenprijzen",
        fontsize=14,
    )

    # Show plot
    plt.tight_layout()

    return plt


def get_realised_economic_percentage_plot(
    asset_price: float,
    strike_price: float,
    tau: float,
    rf_rate: float,
    sigma: float,
) -> plt:

    # Future stock prices
    prices_at_expiry = np.linspace(0, asset_price * 2, 100)

    # Calculate initial Delta and option price for the current asset price
    initial_bs_price = black_scholes_price(
        asset_price, strike_price, tau, rf_rate, sigma, "call"
    )
    initial_delta = black_scholes_delta(
        asset_price, strike_price, tau, rf_rate, sigma, "call"
    )

    # Calculate percentage change for different future stock prices
    percent_changes = [
        (bs_price - initial_bs_price) / (price - asset_price)
        for bs_price, price in zip(
            [
                black_scholes_price(
                    price, strike_price, tau, rf_rate, sigma, "call"
                )
                for price in prices_at_expiry
            ],
            prices_at_expiry,
        )
    ]

    # Predicted percentages using Delta and BS/Price Quotient
    predicted_by_delta = [initial_delta for _ in prices_at_expiry]
    current_quotient_line = initial_bs_price / asset_price
    predicted_by_quotient = [current_quotient_line for _ in prices_at_expiry]

    # Identify regions where BS-price method performs better
    better_bs_price = np.abs(
        np.array(percent_changes) - np.array(predicted_by_quotient)
    ) < np.abs(np.array(percent_changes) - np.array(predicted_by_delta))

    # Calculate the PDF of the stock price at maturity
    mu = np.log(asset_price) + (rf_rate - 0.5 * sigma**2) * tau
    std = sigma * np.sqrt(tau)
    pdf = (1 / (prices_at_expiry * std * np.sqrt(2 * np.pi))) * np.exp(
        -((np.log(prices_at_expiry) - mu) ** 2) / (2 * std**2)
    )

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot percentage of stock gain/loss going to option holder
    ax1.plot(
        prices_at_expiry,
        percent_changes,
        label="Gerealiseerd economisch belang",
        color="green",
    )

    # Plot predicted percentage changes using Delta and Quotient
    ax1.plot(
        prices_at_expiry,
        predicted_by_delta,
        label="Economisch belang deltamethode",
        color="blue",
        linestyle="--",
    )
    ax1.plot(
        prices_at_expiry,
        predicted_by_quotient,
        label="Economisch belang prijsmethode",
        color="red",
        linestyle="dashdot",
    )

    # Shade area where BS-price method is better
    ax1.fill_between(
        prices_at_expiry,
        0,
        1,
        where=better_bs_price,
        color="gray",
        alpha=0.2,
        label="Prijsmethode beter",
    )

    # Customize axes labels and legend for the first y-axis
    ax1.set_xlabel("Aandelenprijs bij expiratie optie", fontsize=12)
    ax1.set_ylabel("Economisch belang (%)", fontsize=12)
    ax1.legend(loc="upper left", fontsize=12)
    ax1.tick_params(axis="both", which="major", labelsize=12)

    # Format the left y-axis to show percentage values
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1))

    # Ensure that the grid on ax1 is turned off
    ax1.grid(False)

    # Change limits for the first y-axis
    ax1.set_ylim(0, 1)  # Increase the upper limit for better space

    # Add title
    ax1.set_title(
        "Gerealiseerd economisch belang tegenover aandelenprijzen bij expiratie",
        fontsize=14,
    )

    # Create a second y-axis to show the PDF
    ax2 = ax1.twinx()
    ax2.plot(
        prices_at_expiry,
        pdf,
        color="purple",
        linestyle=":",
        linewidth=3,
        label="Verdeling Aandelenprijs",
    )
    ax2.set_ylabel("Waarschijnlijkheid", fontsize=12)
    ax2.tick_params(axis="y", labelsize=12)
    ax2.set_ylim(-0.08, 0.08)  # Adjust to fit the PDF curve well

    # Add legend for the second y-axis
    ax2.legend(loc="upper right", fontsize=12)
    ax2.set_yticks([0])
    ax2.set_yticklabels(["0"])

    # Ensure that the grid on ax2 is turned off
    ax2.grid(False)

    # Show plot
    plt.tight_layout()

    return plt

# CS134 - HW3
# Andria Chen

#
# Exercise B: Modeling with an Exponential Distribution
#

# Generate plots of:
# I. prior and posterior of the parameter of the model (mean λ)
# II. prior predictive, posterior predictive, data (histogram)
# Discuss qualitative fit of model with data on the basis of the plot of the 
# posterior predictive and the data

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import matplotlib as mpl
import matplotlib.pyplot as plt
import arviz as az

# load data 
data = pd.read_csv('/Users/Andria/Desktop/cs134/hw3/messages.csv')
data.columns = data.columns.str.strip()
mrt_data = data['MRT'].values

# Global vars for PyMC
mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)
GLOBAL_SEED = 42
CORES = 8
CHAINS = 4
NSAMPLES = 1000

# Prior for lambda
# Using Gamma(alpha = 2, beta = 20) as prior for lambda 
# This gives mean lambda = 2/20 = 0.1 corresponding to 10 minutes 
# Allows for a wide range of values 
prior_alpha = 2.0
prior_beta = 20.0

# Build Exponential Model 
with pm.Model() as ExponentialModel:
    # Prior: Gamma distribution for lambda
    lam = pm.Gamma("lambda", alpha=prior_alpha, beta=prior_beta)

    # Likelihood: Exponential distribution for MRT
    mrt_obs = pm.Exponential("mrt_obs", lam=lam, observed=mrt_data)

    # Sample from prior predictive
    prior_Exponential = pm.sample_prior_predictive(
        samples=4 * NSAMPLES, 
        return_inferencedata=True,
        random_seed=42
        )

    # Sample from posterior
    trace_Exponential = pm.sample(
        random_seed=42, 
        draws=NSAMPLES, 
        cores=CORES, 
        chains=CHAINS, 
        return_inferencedata=True
        )
    
    # Sample from posterior predictive
    posterior_Exponential = pm.sample_posterior_predictive(
        trace_Exponential,
        return_inferencedata=True
        )
    
# Plot I: Prior and Posterior of lambda
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_dist(
    prior_Exponential.prior["lambda"], 
    color="orange", 
    ax=ax, 
    label="Prior")
az.plot_dist(
    trace_Exponential.posterior["lambda"], 
    color="green", 
    ax=ax, 
    label="Posterior")
ax.set_xlabel("Lambda")
ax.set_ylabel("Density")
ax.set_title("Prior and Posterior of Lambda")
ax.legend()
plt.tight_layout()
plt.savefig("B_plot_1_lambda.pdf")
plt.close()

# Plot II: Prior Predictive, Posterior Predictive and Data 
fig, ax = plt.subplots(figsize=(10,6))
az.plot_dist(
    prior_Exponential.prior_predictive.mrt_obs, 
    color="red", 
    ax=ax, 
    label="Prior Predictive")
az.plot_dist(
    posterior_Exponential.posterior_predictive.mrt_obs, 
    color="blue", 
    ax=ax, 
    label="Posterior Predictive")
ax.hist(
    mrt_data,
    bins=30,
    density=True,
    alpha=0.5,
    color="green",
    label="Data",
    edgecolor="black")
max_plot_time = max(500, mrt_data.max() * 1.2)
ax.set_xlim(0, max_plot_time)
ax.set_xlabel("Message Response Time (minutes)")
ax.set_ylabel("Density")
ax.set_title("Prior Predictive, Posterior Predictive, and Data")
ax.legend()
plt.tight_layout()
plt.savefig("B_plot_2_predictive.pdf")
plt.close()

#
# Exercise C: Modeling with a Gamma Distribution
#

# Priors for Gamma Distribution parameters
# 
# Prior for alpha (shape)
# Gamma(alpha = 2, beta = 1) which gives us a mean of 2 
prior_alpha_shape = 2.0
prior_alpha_rate = 1.0

# Prior for beta (rate)
# Mathematica = Gamma(alpha = 3, beta = 0.02) which gives us a mean of 0.06
# Gamma(alpha = 3, beta = 50) since PyMC beta = 1/Mathematica beta 
# so 1/0.02 = 50 
# 2/0.06 = 33 minutes mean response time
prior_beta_shape = 3.0
prior_beta_rate = 50.0

# Build Gamma Distribution model
with pm.Model() as GammaModel:
    # Priors
    alpha = pm.Gamma("alpha", alpha=prior_alpha_shape, beta=prior_alpha_rate)
    beta = pm.Gamma("beta", alpha=prior_beta_shape, beta=prior_beta_rate)
    # Gamma likelihood
    mrt_obs = pm.Gamma("mrt_obs", alpha=alpha, beta=beta, observed=mrt_data)
    
    # Sample from prior predictive
    prior_Gamma = pm.sample_prior_predictive(
        samples=4 * NSAMPLES, 
        return_inferencedata=True,
        random_seed=42
    )
    
    # Sample from posterior
    trace_Gamma = pm.sample(
        random_seed=GLOBAL_SEED,
        draws=NSAMPLES,
        cores=CORES,
        chains=CHAINS,
        return_inferencedata=True
    )
    
    # Sample from posterior preditive
    posterior_Gamma = pm.sample_posterior_predictive(
        trace_Gamma,
        return_inferencedata=True
    )

# Plot I: Prior and Posterior of alpha
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_dist(
    prior_Gamma.prior["alpha"],
    color="orange", 
    ax=ax, 
    label="Prior")
az.plot_dist(
    trace_Gamma.posterior["alpha"], 
    color="green", 
    ax=ax, 
    label="Posterior")
ax.set_xlabel("Alpha")
ax.set_ylabel("Density")
ax.set_title("Prior and Posterior of Alpha")
ax.legend()
plt.tight_layout()
plt.savefig("C_plot_1_alpha.pdf")
plt.close()

# Plot II: Prior and Posterior of beta
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_dist(
    prior_Gamma.prior["beta"],
    color="orange", 
    ax=ax, 
    label="Prior")
az.plot_dist(
    trace_Gamma.posterior["beta"], 
    color="green", 
    ax=ax, 
    label="Posterior")
ax.set_xlabel("Beta")
ax.set_ylabel("Density")
ax.set_title("Prior and Posterior of Beta")
ax.legend()
plt.tight_layout()
plt.savefig('C_plot_2_beta.pdf')
plt.close()

# Plot III: Prior Predictive, Posterior Predictive, and Data
fig, ax = plt.subplots(figsize=(10, 6))

max_plot_time = max(500, mrt_data.max() * 1.2)

az.plot_dist(
    prior_Gamma.prior_predictive.mrt_obs, 
    color="red", 
    ax=ax, 
    label="Prior Predictive")
az.plot_dist(
    posterior_Gamma.posterior_predictive.mrt_obs, 
    color="blue", 
    ax=ax, 
    label="Posterior Predictive")
ax.hist(
    mrt_data,
    bins=30,
    density=True,
    alpha=0.5,
    color="green",
    label="Data"
)
ax.set_xlim(0, max_plot_time)
ax.set_xlabel("Message Response Time in Minutes")
ax.set_ylabel("Density")
ax.set_title("Predictive Distributions")
ax.legend()
plt.tight_layout()
plt.savefig('C_plot_3_predictive.pdf')
plt.close()

#
# Exercise D - Modeling with a Weibull Distribution
#

# Priors for Weibull Distribution parameters
# 
# Prior for alpha (shape)
# Gamma(alpha = 2, beta = 0.5) which gives us a mean of 1
# Gamma(alpha = 2, beta = 2) since PyMC beta = 1/0.5 = 2
prior_alpha_shape = 2.0
prior_alpha_rate = 2.0

# Prior for beta (scale)
# Gamma(alpha = 3, beta = 10) which gives us a mean of 30
# Gamma(alpha = 3, beta = 0.1) since PyMC beta = 1/10 = 0.1
prior_beta_shape = 3.0
prior_beta_rate = 0.1

# Build Weibull Distribution model
with pm.Model() as WeibullModel:
    # Priors
    alpha = pm.Gamma("alpha", alpha=prior_alpha_shape, beta=prior_alpha_rate)
    beta = pm.Gamma("beta", alpha=prior_beta_shape, beta=prior_beta_rate)
    # Gamma likelihood
    mrt_obs = pm.Weibull("mrt_obs", alpha=alpha, beta=beta, observed=mrt_data)
    
    # Sample from prior predictive
    prior_Weibull = pm.sample_prior_predictive(
        samples=4 * NSAMPLES, 
        return_inferencedata=True,
        random_seed=42
    )
    
    # Sample from posterior
    trace_Weibull = pm.sample(
        random_seed=GLOBAL_SEED,
        draws=NSAMPLES,
        cores=CORES,
        chains=CHAINS,
        return_inferencedata=True
    )
    
    # Sample from posterior preditive
    posterior_Weibull = pm.sample_posterior_predictive(
        trace_Weibull,
        return_inferencedata=True
    )

# Plot I: Prior and Posterior of alpha
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_dist(
    prior_Weibull.prior["alpha"],
    color="orange", 
    ax=ax, 
    label="Prior")
az.plot_dist(
    trace_Gamma.posterior["alpha"], 
    color="green", 
    ax=ax, 
    label="Posterior")
ax.set_xlabel("Alpha")
ax.set_ylabel("Density")
ax.set_title("Prior and Posterior of Alpha")
ax.legend()
plt.tight_layout()
plt.savefig("D_plot_1_alpha.pdf")
plt.close()

# Plot II: Prior and Posterior of beta
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_dist(
    prior_Weibull.prior["beta"],
    color="orange", 
    ax=ax, 
    label="Prior")
az.plot_dist(
    trace_Weibull.posterior["beta"], 
    color="green", 
    ax=ax, 
    label="Posterior")
ax.set_xlabel("Beta")
ax.set_ylabel("Density")
ax.set_title("Prior and Posterior of Beta")
ax.legend()
plt.tight_layout()
plt.savefig('D_plot_2_beta.pdf')
plt.close()

# Plot III: Prior Predictive, Posterior Predictive, and Data
fig, ax = plt.subplots(figsize=(10, 6))

max_plot_time = max(500, mrt_data.max() * 1.2)

az.plot_dist(
    prior_Weibull.prior_predictive.mrt_obs, 
    color="red", 
    ax=ax, 
    label="Prior Predictive")
az.plot_dist(
    posterior_Weibull.posterior_predictive.mrt_obs, 
    color="blue", 
    ax=ax, 
    label="Posterior Predictive")
ax.hist(
    mrt_data,
    bins=30,
    density=True,
    alpha=0.5,
    color="green",
    label="Data"
)
ax.set_xlim(0, max_plot_time)
ax.set_xlabel("Message Response Time in Minutes")
ax.set_ylabel("Density")
ax.set_title("Predictive Distributions")
ax.legend()
plt.tight_layout()
plt.savefig('D_plot_3_predictive.pdf')
plt.close()

#
# Exercise E - Comparison and Discussion
#

# Calculate Bayes Factor 
# BF = P(Data | Model A) / P(Data | Model B)
def marginal_llk_smc(
    model: pm.Model, n_samples: int, cores: int, chains: int, seed: int
) -> float:
    """
    Compute the marginal log likelihood using the sequential monte carlo (SMC)
    sampler.

    Parameters
    ----------
    model : pm.Model
    n_samples : int
    cores : int
    chains : int

    Returns
    -------
    float
        Marginal log likelihood estimate.
    """
    trace = pm.sample_smc(
        n_samples,
        model=model,
        cores=cores,
        chains=chains,
        return_inferencedata=True,
        random_seed=seed,
    )

    try:
        return trace.sample_stats["log_marginal_likelihood"].mean().item()
    except:
        raise ValueError("Unable to compute BF due to convergence error")


def bayes_factor(
    model1: pm.Model,
    model2: pm.Model,
    n_samples: int = 15000,
    cores: int = 1,
    chains: int = 4,
    seed: int = None,
) -> float:
    """
    Compute an estimate of the Bayes factor p(y|model1) / p(y|model2)..

    Parameters
    ----------
    model1 : pm.Model
        The model in the numerator of the bayes factor.
    model2 : pm.Model
        The model in the denominator of the bayes factor
    n_samples : int, optional
        Number of samples to draw during estimate, by default 5000
    cores : int, optional
        Number of cores to use when generating trace, by default 1
    Returns
    -------
    float
        The bayes factor estimate
    """

    # Compute the log marginal likelihoods for the models
    log_marginal_ll1 = marginal_llk_smc(
        model=model1, n_samples=n_samples, cores=cores, chains=chains, seed=seed
    )
    log_marginal_ll2 = marginal_llk_smc(
        model=model2, n_samples=n_samples, cores=cores, chains=chains, seed=seed
    )
    return np.exp(log_marginal_ll1 - log_marginal_ll2)

def report_bf(bf, model1_name: str, model2_name: str) -> None:
    """
    Parameters
    -----------
    bf : float
      Bayes factor value
    model1_name : str
      Name of the model in the numerator of the BF
    model2_name : str
      Name of the model in the denominator of the BF.
    """

    model_names = [model1_name, model2_name]
    bf_curr = bf
    if bf < 1:
        bf_curr = 1 / bf
        model_names.reverse()
    print(
        f"The data is more likely under the '{model_names[0]}' than the '{model_names[1]}', BF_{model_names[0]}_{model_names[1]}={bf_curr}\n"
    )

if __name__ == '__main__':
    # Comparison 1 between Exponential and Gamma
    BF_Exp_Gamma = bayes_factor(
        model1=ExponentialModel,
        model2=GammaModel,
        n_samples=10000,
        cores=CORES,
        chains=4,
        seed=GLOBAL_SEED,
    )
    report_bf(BF_Exp_Gamma, "Exponential", "Gamma")

    # Comparison 2 between Exponential and Weibull
    BF_Exp_Weibull = bayes_factor(
        model1=ExponentialModel,
        model2=WeibullModel,
        n_samples=10000,
        cores=CORES,
        chains=4,
        seed=GLOBAL_SEED,
    )
    report_bf(BF_Exp_Weibull, "Exponential", "Weibull")

    # Comparison 3 between Gamma and Weibull
    BF_Gamma_Weibull = bayes_factor(
        model1=GammaModel,
        model2=WeibullModel,
        n_samples=10000,
        cores=CORES,
        chains=4,
        seed=GLOBAL_SEED,
    )
    report_bf(BF_Gamma_Weibull, "Gamma", "Weibull")



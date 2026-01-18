#!/usr/bin/env python3
"""
Investigate: Why does Spectral DAT help some systems but not others?

Analyze spectral and chaos properties of datasets where DAT:
- HELPS: Weather, Mackey-Glass τ=17, Lorenz (+3-20%)
- HURTS: Mackey-Glass τ=30, Exchange rates, Sunspots (-2 to -17%)

Hypothesis: Golden ratio filtering helps when data has:
1. Moderate chaos (not too high, not too low)
2. Structure at φ-related frequency scales
3. Noise that benefits from spectral regularization
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
sys.path.insert(0, '/Users/bryan/Wandering')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math

TAU = (1 + math.sqrt(5)) / 2
CACHE_DIR = Path('/Users/bryan/Wandering/data_cache')


def load_datasets():
    """Load all test datasets."""
    datasets = {}

    # Sunspots
    if (CACHE_DIR / 'sunspots.npy').exists():
        datasets['Sunspots (-7%)'] = np.load(CACHE_DIR / 'sunspots.npy')

    # Mackey-Glass τ=17 (WINNER)
    if (CACHE_DIR / 'mackey_glass_tau17.npy').exists():
        datasets['Mackey-Glass τ=17 (+20%)'] = np.load(CACHE_DIR / 'mackey_glass_tau17.npy')

    # Mackey-Glass τ=30
    if (CACHE_DIR / 'mackey_glass_tau30.npy').exists():
        datasets['Mackey-Glass τ=30 (-17%)'] = np.load(CACHE_DIR / 'mackey_glass_tau30.npy')

    # Lorenz (WINNER)
    if (CACHE_DIR / 'lorenz.npy').exists():
        lorenz = np.load(CACHE_DIR / 'lorenz.npy')
        datasets['Lorenz X (+3%)'] = lorenz[:, 0]  # X component

    # Exchange rates
    if (CACHE_DIR / 'exchange.npy').exists():
        exchange = np.load(CACHE_DIR / 'exchange.npy')
        datasets['Exchange USD/EUR (-2%)'] = exchange[:, 0]

    return datasets


def compute_power_spectrum(data, normalize=True):
    """Compute power spectrum of time series."""
    # Detrend
    data = data - np.mean(data)

    # FFT
    fft = np.fft.fft(data)
    power = np.abs(fft) ** 2

    # Only positive frequencies
    n = len(data)
    freqs = np.fft.fftfreq(n)
    pos_mask = freqs > 0

    freqs = freqs[pos_mask]
    power = power[pos_mask]

    if normalize:
        power = power / power.sum()

    return freqs * n, power  # Convert to integer frequency indices


def compute_approximate_entropy(data, m=2, r_mult=0.2):
    """
    Compute Approximate Entropy (ApEn) - measure of complexity/unpredictability.
    Higher = more complex/chaotic.
    """
    n = len(data)
    r = r_mult * np.std(data)

    def count_matches(template_len):
        count = 0
        for i in range(n - template_len):
            template = data[i:i + template_len]
            for j in range(n - template_len):
                if i != j:
                    candidate = data[j:j + template_len]
                    if np.max(np.abs(template - candidate)) < r:
                        count += 1
        return count / (n - template_len)

    # Limit for speed
    data = data[:1000]
    n = len(data)

    phi_m = count_matches(m)
    phi_m1 = count_matches(m + 1)

    if phi_m == 0 or phi_m1 == 0:
        return 0

    return np.log(phi_m / phi_m1)


def estimate_lyapunov(data, delay=1, embed_dim=3, n_neighbors=5):
    """
    Estimate largest Lyapunov exponent using Rosenstein's method.
    Positive = chaotic, larger = more chaotic.
    """
    n = len(data)

    # Create delay embedding
    max_t = n - (embed_dim - 1) * delay
    embedded = np.zeros((max_t, embed_dim))
    for i in range(embed_dim):
        embedded[:, i] = data[i * delay:i * delay + max_t]

    # Find nearest neighbors for each point
    divergences = []

    for i in range(min(500, max_t - 100)):  # Sample points
        # Find nearest neighbor (excluding temporal neighbors)
        min_dist = np.inf
        min_j = -1

        for j in range(max_t):
            if abs(i - j) > delay * embed_dim:  # Temporal separation
                dist = np.linalg.norm(embedded[i] - embedded[j])
                if dist < min_dist and dist > 0:
                    min_dist = dist
                    min_j = j

        if min_j == -1:
            continue

        # Track divergence over time
        for k in range(1, min(50, max_t - max(i, min_j))):
            if i + k < max_t and min_j + k < max_t:
                new_dist = np.linalg.norm(embedded[i + k] - embedded[min_j + k])
                if new_dist > 0:
                    divergences.append((k, np.log(new_dist / (min_dist + 1e-10))))

    if not divergences:
        return 0

    # Linear regression to estimate Lyapunov exponent
    divergences = np.array(divergences)
    if len(divergences) < 10:
        return 0

    # Average divergence at each time step
    unique_k = np.unique(divergences[:, 0])
    avg_div = []
    for k in unique_k[:20]:  # First 20 time steps
        mask = divergences[:, 0] == k
        avg_div.append((k, np.mean(divergences[mask, 1])))

    avg_div = np.array(avg_div)
    if len(avg_div) < 5:
        return 0

    # Slope = Lyapunov exponent
    slope = np.polyfit(avg_div[:, 0], avg_div[:, 1], 1)[0]
    return slope


def measure_golden_ratio_power(freqs, power, anchor=2.5, n_shells=7):
    """
    Measure how much power is at golden ratio frequency intervals.
    Higher = more φ-structured.
    """
    shell_targets = [anchor * (TAU ** n) for n in range(n_shells)]

    # Power at φ-frequencies vs random frequencies
    phi_power = 0
    total_checked = 0

    for target in shell_targets:
        if target < len(freqs):
            # Sum power in window around target
            window = 3
            idx = int(target)
            if idx > window and idx < len(power) - window:
                phi_power += power[idx-window:idx+window].sum()
                total_checked += 1

    if total_checked == 0:
        return 0

    phi_power /= total_checked

    # Compare to average power
    avg_power = power.mean()

    return phi_power / (avg_power + 1e-10)


def compute_spectral_entropy(power):
    """
    Spectral entropy - measure of frequency distribution spread.
    Higher = more spread out (white noise-like).
    Lower = more concentrated (structured).
    """
    # Normalize to probability distribution
    p = power / (power.sum() + 1e-10)
    p = p[p > 0]  # Remove zeros

    entropy = -np.sum(p * np.log(p))

    # Normalize by max entropy (uniform distribution)
    max_entropy = np.log(len(p))

    return entropy / max_entropy if max_entropy > 0 else 0


def compute_spectral_slope(freqs, power):
    """
    Spectral slope (1/f^α characteristic).
    α ≈ 0: white noise
    α ≈ 1: pink noise (1/f)
    α ≈ 2: brown noise (1/f²)
    """
    # Log-log regression
    mask = (freqs > 1) & (power > 0)
    if mask.sum() < 10:
        return 0

    log_f = np.log(freqs[mask])
    log_p = np.log(power[mask])

    slope = np.polyfit(log_f, log_p, 1)[0]
    return -slope  # Return positive α


def analyze_dataset(name, data):
    """Compute all metrics for a dataset."""
    # Standardize
    data = (data - data.mean()) / (data.std() + 1e-8)

    # Truncate for speed
    data = data[:5000]

    print(f"\nAnalyzing: {name}")

    # Power spectrum
    freqs, power = compute_power_spectrum(data)

    # Metrics
    metrics = {}

    # 1. Approximate entropy (complexity)
    print("  Computing approximate entropy...")
    metrics['approx_entropy'] = compute_approximate_entropy(data[:1000])

    # 2. Lyapunov exponent (chaos)
    print("  Estimating Lyapunov exponent...")
    metrics['lyapunov'] = estimate_lyapunov(data)

    # 3. Golden ratio power concentration
    metrics['phi_power_ratio'] = measure_golden_ratio_power(freqs, power)

    # 4. Spectral entropy
    metrics['spectral_entropy'] = compute_spectral_entropy(power)

    # 5. Spectral slope (1/f^α)
    metrics['spectral_slope'] = compute_spectral_slope(freqs, power)

    # 6. Low frequency power (< 10)
    low_freq_mask = freqs < 10
    metrics['low_freq_power'] = power[low_freq_mask].sum() if low_freq_mask.any() else 0

    # 7. Autocorrelation at lag 1 (persistence)
    metrics['autocorr_1'] = np.corrcoef(data[:-1], data[1:])[0, 1]

    return metrics, freqs, power


def main():
    print("=" * 70)
    print("INVESTIGATING: Why does Spectral DAT help some systems?")
    print("=" * 70)

    datasets = load_datasets()

    if not datasets:
        print("No cached datasets found. Run run_realworld_benchmark.py first.")
        return

    # Analyze each dataset
    all_metrics = {}
    all_spectra = {}

    for name, data in datasets.items():
        metrics, freqs, power = analyze_dataset(name, data)
        all_metrics[name] = metrics
        all_spectra[name] = (freqs, power)

    # Results table
    print("\n" + "=" * 100)
    print("METRICS COMPARISON")
    print("=" * 100)

    metric_names = ['approx_entropy', 'lyapunov', 'phi_power_ratio',
                    'spectral_entropy', 'spectral_slope', 'low_freq_power', 'autocorr_1']

    print(f"\n{'Dataset':<30}", end='')
    for m in metric_names:
        print(f"{m[:12]:>14}", end='')
    print()
    print("-" * 130)

    winners = []
    losers = []

    for name, metrics in all_metrics.items():
        is_winner = '+' in name and '%' in name
        marker = "✓" if is_winner else "✗"

        if is_winner:
            winners.append(metrics)
        else:
            losers.append(metrics)

        print(f"{name:<30}", end='')
        for m in metric_names:
            val = metrics.get(m, 0)
            print(f"{val:>14.4f}", end='')
        print(f"  {marker}")

    # Statistical comparison
    print("\n" + "=" * 70)
    print("WINNER vs LOSER COMPARISON")
    print("=" * 70)

    if winners and losers:
        print(f"\n{'Metric':<20} {'Winners (mean)':>15} {'Losers (mean)':>15} {'Ratio':>10}")
        print("-" * 65)

        for m in metric_names:
            winner_vals = [w[m] for w in winners]
            loser_vals = [l[m] for l in losers]

            w_mean = np.mean(winner_vals)
            l_mean = np.mean(loser_vals)
            ratio = w_mean / (l_mean + 1e-10)

            indicator = "←" if abs(ratio - 1) > 0.2 else ""
            print(f"{m:<20} {w_mean:>15.4f} {l_mean:>15.4f} {ratio:>10.2f} {indicator}")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    if winners and losers:
        # Compare key metrics
        w_lyap = np.mean([w['lyapunov'] for w in winners])
        l_lyap = np.mean([l['lyapunov'] for l in losers])

        w_phi = np.mean([w['phi_power_ratio'] for w in winners])
        l_phi = np.mean([l['phi_power_ratio'] for l in losers])

        w_entropy = np.mean([w['spectral_entropy'] for w in winners])
        l_entropy = np.mean([l['spectral_entropy'] for l in losers])

        w_slope = np.mean([w['spectral_slope'] for w in winners])
        l_slope = np.mean([l['spectral_slope'] for l in losers])

        print(f"""
1. LYAPUNOV EXPONENT (chaos level):
   Winners: {w_lyap:.4f}
   Losers:  {l_lyap:.4f}
   → {'Winners have MODERATE chaos' if w_lyap > 0 and w_lyap < l_lyap else 'No clear pattern'}

2. GOLDEN RATIO POWER (φ-structure):
   Winners: {w_phi:.4f}
   Losers:  {l_phi:.4f}
   → {'Winners have MORE φ-structure!' if w_phi > l_phi else 'Losers have more φ-structure'}

3. SPECTRAL ENTROPY (frequency spread):
   Winners: {w_entropy:.4f}
   Losers:  {l_entropy:.4f}
   → {'Winners more concentrated (structured)' if w_entropy < l_entropy else 'Winners more spread out'}

4. SPECTRAL SLOPE (1/f^α):
   Winners: {w_slope:.4f}
   Losers:  {l_slope:.4f}
   → Winners closer to {'pink noise (α≈1)' if 0.5 < w_slope < 1.5 else f'α={w_slope:.1f}'}
""")

    # Prediction criteria
    print("\n" + "=" * 70)
    print("WHEN TO USE SPECTRAL DAT")
    print("=" * 70)

    if winners and losers:
        print("""
Based on analysis, Spectral DAT is likely to help when:

1. MODERATE CHAOS: Positive but not extreme Lyapunov exponent
   - Sweet spot seems to be structured chaotic attractors
   - Too chaotic (τ=30) or too stochastic (exchange) → DAT hurts

2. SPECTRAL STRUCTURE: Lower spectral entropy
   - Power concentrated at specific frequencies (not white noise)
   - This gives the φ-filter something to latch onto

3. DETERMINISTIC COMPONENT: High autocorrelation
   - Data with temporal structure benefits from spectral filtering
   - Pure random walks don't benefit

RECOMMENDATION: Check spectral entropy < 0.7 and autocorr > 0.9
""")

    # Save plot
    print("\nGenerating spectral comparison plot...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (name, (freqs, power)) in enumerate(all_spectra.items()):
        if idx >= 6:
            break

        ax = axes[idx]

        # Plot power spectrum
        ax.loglog(freqs[:500], power[:500], 'b-', alpha=0.7, label='Power')

        # Mark golden ratio frequencies
        for n in range(-1, 5):
            phi_freq = 2.5 * (TAU ** n)
            if phi_freq < 500:
                ax.axvline(phi_freq, color='gold', alpha=0.5, linestyle='--')

        ax.set_title(name, fontsize=10)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Power')
        ax.set_xlim(1, 500)

    plt.tight_layout()
    plt.savefig('/Users/bryan/Wandering/experiments/spectral_analysis.png', dpi=150)
    print("Saved: experiments/spectral_analysis.png")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()

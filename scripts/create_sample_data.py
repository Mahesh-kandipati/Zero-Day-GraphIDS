"""
Synthetic Network Flow Dataset Generator for Zero-Day GraphIDS.
Generates realistic CICFlowMeter-compliant flow records with distinct benign,
known attack, and designated ZERO_DAY_UNKNOWN traffic distributions.
"""

import os
import numpy as np
import pandas as pd

def generate_flow_dataset(
    n_benign: int = 700,
    n_known_attack: int = 150,
    n_zero_day: int = 150,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic tabular network flow dataset.
    Features follow standard CICFlowMeter schemas.
    """
    np.random.seed(random_seed)

    # 1. Benign Normal Network Flows: Typical browsing, API traffic, DNS
    # Long durations, balanced packets, moderate rates
    benign_dur = np.random.exponential(scale=150000, size=n_benign) + 1000
    benign_fwd_pkts = np.random.poisson(lam=12, size=n_benign) + 1
    benign_bwd_pkts = np.random.poisson(lam=15, size=n_benign) + 1
    benign_fwd_len = benign_fwd_pkts * np.random.normal(loc=120, scale=30, size=n_benign).clip(min=40)
    benign_bwd_len = benign_bwd_pkts * np.random.normal(loc=600, scale=100, size=n_benign).clip(min=40)
    benign_bytes_s = (benign_fwd_len + benign_bwd_len) / (benign_dur / 1e6 + 1e-4)
    benign_pkts_s = (benign_fwd_pkts + benign_bwd_pkts) / (benign_dur / 1e6 + 1e-4)
    benign_fwd_mean = benign_fwd_len / benign_fwd_pkts
    benign_bwd_mean = benign_bwd_len / benign_bwd_pkts
    benign_pkt_mean = (benign_fwd_len + benign_bwd_len) / (benign_fwd_pkts + benign_bwd_pkts)
    benign_labels = ['BENIGN'] * n_benign

    # 2. Known Attacks (e.g. DDoS / PortScan): Very high packet rate, small packets, short duration
    known_dur = np.random.exponential(scale=2000, size=n_known_attack) + 50
    known_fwd_pkts = np.random.poisson(lam=150, size=n_known_attack) + 10
    known_bwd_pkts = np.random.poisson(lam=2, size=n_known_attack)
    known_fwd_len = known_fwd_pkts * np.random.normal(loc=40, scale=5, size=n_known_attack).clip(min=20)
    known_bwd_len = known_bwd_pkts * np.random.normal(loc=40, scale=5, size=n_known_attack).clip(min=20)
    known_bytes_s = (known_fwd_len + known_bwd_len) / (known_dur / 1e6 + 1e-4)
    known_pkts_s = (known_fwd_pkts + known_bwd_pkts) / (known_dur / 1e6 + 1e-4)
    known_fwd_mean = known_fwd_len / known_fwd_pkts
    known_bwd_mean = np.where(known_bwd_pkts > 0, known_bwd_len / np.maximum(known_bwd_pkts, 1), 0.0)
    known_pkt_mean = (known_fwd_len + known_bwd_len) / (known_fwd_pkts + known_bwd_pkts)
    known_labels = ['DDoS'] * (n_known_attack // 2) + ['PortScan'] * (n_known_attack - n_known_attack // 2)

    # 3. Novel Zero-Day Intrusion (e.g. Stealth Data Exfiltration / C2 Beaconing)
    # Extremely persistent, periodic intervals, asymmetric packet ratios
    zero_dur = np.random.exponential(scale=600000, size=n_zero_day) + 50000
    zero_fwd_pkts = np.random.poisson(lam=80, size=n_zero_day) + 20
    zero_bwd_pkts = np.random.poisson(lam=3, size=n_zero_day) + 1
    zero_fwd_len = zero_fwd_pkts * np.random.normal(loc=950, scale=120, size=n_zero_day).clip(min=200)
    zero_bwd_len = zero_bwd_pkts * np.random.normal(loc=64, scale=10, size=n_zero_day).clip(min=30)
    zero_bytes_s = (zero_fwd_len + zero_bwd_len) / (zero_dur / 1e6 + 1e-4)
    zero_pkts_s = (zero_fwd_pkts + zero_bwd_pkts) / (zero_dur / 1e6 + 1e-4)
    zero_fwd_mean = zero_fwd_len / zero_fwd_pkts
    zero_bwd_mean = zero_bwd_len / zero_bwd_pkts
    zero_pkt_mean = (zero_fwd_len + zero_bwd_len) / (zero_fwd_pkts + zero_bwd_pkts)
    zero_labels = ['ZERO_DAY_UNKNOWN'] * n_zero_day

    # Combine All Subpopulations
    duration = np.concatenate([benign_dur, known_dur, zero_dur])
    fwd_pkts = np.concatenate([benign_fwd_pkts, known_fwd_pkts, zero_fwd_pkts])
    bwd_pkts = np.concatenate([benign_bwd_pkts, known_bwd_pkts, zero_bwd_pkts])
    fwd_len = np.concatenate([benign_fwd_len, known_fwd_len, zero_fwd_len])
    bwd_len = np.concatenate([benign_bwd_len, known_bwd_len, zero_bwd_len])
    bytes_s = np.concatenate([benign_bytes_s, known_bytes_s, zero_bytes_s])
    pkts_s = np.concatenate([benign_pkts_s, known_pkts_s, zero_pkts_s])
    fwd_mean = np.concatenate([benign_fwd_mean, known_fwd_mean, zero_fwd_mean])
    bwd_mean = np.concatenate([benign_bwd_mean, known_bwd_mean, zero_bwd_mean])
    pkt_mean = np.concatenate([benign_pkt_mean, known_pkt_mean, zero_pkt_mean])
    labels = benign_labels + known_labels + zero_labels

    df = pd.DataFrame({
        'Flow Duration': np.round(duration, 2),
        'Total Fwd Packets': fwd_pkts.astype(int),
        'Total Backward Packets': bwd_pkts.astype(int),
        'Total Length of Fwd Packets': np.round(fwd_len, 2),
        'Total Length of Bwd Packets': np.round(bwd_len, 2),
        'Flow Bytes/s': np.round(bytes_s, 4),
        'Flow Packets/s': np.round(pkts_s, 4),
        'Fwd Packet Length Mean': np.round(fwd_mean, 2),
        'Bwd Packet Length Mean': np.round(bwd_mean, 2),
        'Packet Length Mean': np.round(pkt_mean, 2),
        'Label': labels
    })

    # Shuffle rows to simulate real continuous packet capture stream
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    return df

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sample_dir = os.path.join(base_dir, 'sample_data')
    os.makedirs(sample_dir, exist_ok=True)

    print("[*] Generating demonstration network flow datasets...")

    # 1. CICIDS2017 Benchmark Sample (1000 flows)
    df_cicids = generate_flow_dataset(n_benign=700, n_known_attack=150, n_zero_day=150, random_seed=42)
    path_cicids = os.path.join(sample_dir, 'CICIDS2017_sample.csv')
    df_cicids.to_csv(path_cicids, index=False)
    print(f"[+] Created: {path_cicids} ({len(df_cicids)} flows)")

    # 2. CTU-13 Sample (800 flows)
    df_ctu = generate_flow_dataset(n_benign=600, n_known_attack=100, n_zero_day=100, random_seed=101)
    df_ctu['Label'] = df_ctu['Label'].replace({'DDoS': 'Botnet_C2', 'PortScan': 'NTP_Amplification'})
    path_ctu = os.path.join(sample_dir, 'CTU13_sample.csv')
    df_ctu.to_csv(path_ctu, index=False)
    print(f"[+] Created: {path_ctu} ({len(df_ctu)} flows)")

    # 3. USTC-TFC2016 Encrypted Malware Sample (800 flows)
    df_ustc = generate_flow_dataset(n_benign=600, n_known_attack=100, n_zero_day=100, random_seed=202)
    df_ustc['Label'] = df_ustc['Label'].replace({'DDoS': 'Trojan_Zeus', 'PortScan': 'Adware_Elex'})
    path_ustc = os.path.join(sample_dir, 'USTC_TFC2016_sample.csv')
    df_ustc.to_csv(path_ustc, index=False)
    print(f"[+] Created: {path_ustc} ({len(df_ustc)} flows)")

    # 4. Uninspected Live Traffic Capture Stream (500 flows)
    # Suitable for direct upload in the "Scan Traffic" module
    df_uninspected = generate_flow_dataset(n_benign=350, n_known_attack=50, n_zero_day=100, random_seed=303)
    path_uninspected = os.path.join(sample_dir, 'uninspected_traffic.csv')
    df_uninspected.to_csv(path_uninspected, index=False)
    print(f"[+] Created: {path_uninspected} ({len(df_uninspected)} flows)")

    print("[+] All sample datasets successfully created in 'sample_data/'.")

if __name__ == '__main__':
    main()

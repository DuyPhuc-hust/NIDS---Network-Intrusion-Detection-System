import os
import pandas as pd
import joblib
import numpy as np
from collections import Counter
from src.data.preprocess import clean_data
from scapy.all import ICMP, IP, TCP, UDP, PcapReader, rdpcap
import cicflowmeter.flow as cic_flow
import cicflowmeter.utils as cic_utils
from cicflowmeter.features.packet_time import PacketTime
from cicflowmeter.sniffer import create_sniffer
from datetime import datetime

DEFAULT_MODEL_DIR = os.path.join("models", "final")
TEMP_CSV = "temp_extracted_flow.csv"

CICFLOWMETER_COLUMN_MAP = {
    "flow_duration": "Flow Duration",
    "tot_fwd_pkts": "Total Fwd Packets",
    "tot_bwd_pkts": "Total Backward Packets",
    "totlen_fwd_pkts": "Total Length of Fwd Packets",
    "totlen_bwd_pkts": "Total Length of Bwd Packets",
    "fwd_pkt_len_max": "Fwd Packet Length Max",
    "fwd_pkt_len_min": "Fwd Packet Length Min",
    "fwd_pkt_len_mean": "Fwd Packet Length Mean",
    "fwd_pkt_len_std": "Fwd Packet Length Std",
    "bwd_pkt_len_max": "Bwd Packet Length Max",
    "bwd_pkt_len_min": "Bwd Packet Length Min",
    "bwd_pkt_len_mean": "Bwd Packet Length Mean",
    "bwd_pkt_len_std": "Bwd Packet Length Std",
    "flow_byts_s": "Flow Bytes/s",
    "flow_pkts_s": "Flow Packets/s",
    "flow_iat_mean": "Flow IAT Mean",
    "flow_iat_std": "Flow IAT Std",
    "flow_iat_max": "Flow IAT Max",
    "flow_iat_min": "Flow IAT Min",
    "fwd_iat_tot": "Fwd IAT Total",
    "fwd_iat_mean": "Fwd IAT Mean",
    "fwd_iat_std": "Fwd IAT Std",
    "fwd_iat_max": "Fwd IAT Max",
    "fwd_iat_min": "Fwd IAT Min",
    "bwd_iat_tot": "Bwd IAT Total",
    "bwd_iat_mean": "Bwd IAT Mean",
    "bwd_iat_std": "Bwd IAT Std",
    "bwd_iat_max": "Bwd IAT Max",
    "bwd_iat_min": "Bwd IAT Min",
    "fwd_psh_flags": "Fwd PSH Flags",
    "bwd_psh_flags": "Bwd PSH Flags",
    "fwd_urg_flags": "Fwd URG Flags",
    "bwd_urg_flags": "Bwd URG Flags",
    "fwd_header_len": "Fwd Header Length",
    "bwd_header_len": "Bwd Header Length",
    "fwd_pkts_s": "Fwd Packets/s",
    "bwd_pkts_s": "Bwd Packets/s",
    "pkt_len_min": "Min Packet Length",
    "pkt_len_max": "Max Packet Length",
    "pkt_len_mean": "Packet Length Mean",
    "pkt_len_std": "Packet Length Std",
    "pkt_len_var": "Packet Length Variance",
    "fin_flag_cnt": "FIN Flag Count",
    "syn_flag_cnt": "SYN Flag Count",
    "rst_flag_cnt": "RST Flag Count",
    "psh_flag_cnt": "PSH Flag Count",
    "ack_flag_cnt": "ACK Flag Count",
    "urg_flag_cnt": "URG Flag Count",
    "cwe_flag_count": "CWE Flag Count",
    "ece_flag_cnt": "ECE Flag Count",
    "down_up_ratio": "Down/Up Ratio",
    "pkt_size_avg": "Average Packet Size",
    "fwd_seg_size_avg": "Avg Fwd Segment Size",
    "bwd_seg_size_avg": "Avg Bwd Segment Size",
    "fwd_byts_b_avg": "Fwd Avg Bytes/Bulk",
    "fwd_pkts_b_avg": "Fwd Avg Packets/Bulk",
    "fwd_blk_rate_avg": "Fwd Avg Bulk Rate",
    "bwd_byts_b_avg": "Bwd Avg Bytes/Bulk",
    "bwd_pkts_b_avg": "Bwd Avg Packets/Bulk",
    "bwd_blk_rate_avg": "Bwd Avg Bulk Rate",
    "subflow_fwd_pkts": "Subflow Fwd Packets",
    "subflow_fwd_byts": "Subflow Fwd Bytes",
    "subflow_bwd_pkts": "Subflow Bwd Packets",
    "subflow_bwd_byts": "Subflow Bwd Bytes",
    "init_fwd_win_byts": "Init_Win_bytes_forward",
    "init_bwd_win_byts": "Init_Win_bytes_backward",
    "fwd_act_data_pkts": "act_data_pkt_fwd",
    "fwd_seg_size_min": "min_seg_size_forward",
    "active_mean": "Active Mean",
    "active_std": "Active Std",
    "active_max": "Active Max",
    "active_min": "Active Min",
    "idle_mean": "Idle Mean",
    "idle_std": "Idle Std",
    "idle_max": "Idle Max",
    "idle_min": "Idle Min",
}


def normalize_flow_columns(df):
    df = df.rename(columns=CICFLOWMETER_COLUMN_MAP)
    if "Fwd Header Length" in df.columns and "Fwd Header Length.1" not in df.columns:
        df["Fwd Header Length.1"] = df["Fwd Header Length"]
    return df


def _patched_cicflowmeter_statistics(values):
    values = [float(value) for value in values]
    if len(values) > 1:
        values_array = np.asarray(values, dtype=float)
        return {
            "total": float(values_array.sum()),
            "max": float(values_array.max()),
            "min": float(values_array.min()),
            "mean": float(values_array.mean()),
            "std": float(values_array.std()),
        }
    return {"total": 0, "max": 0, "min": 0, "mean": 0, "std": 0}


def extract_flows_with_cicflowmeter(pcap_path, temp_csv):
    cic_utils.get_statistics = _patched_cicflowmeter_statistics
    cic_flow.get_statistics = _patched_cicflowmeter_statistics
    PacketTime.get_time_stamp = lambda self: datetime.fromtimestamp(
        float(self.flow.packets[0][0].time)
    ).strftime("%Y-%m-%d %H:%M:%S")

    sniffer = create_sniffer(
        input_file=pcap_path,
        input_interface=None,
        output_mode="flow",
        output_file=temp_csv,
        url_model=None,
    )
    sniffer.start()
    sniffer.join()

    if not os.path.exists(temp_csv) or os.path.getsize(temp_csv) == 0:
        return pd.DataFrame()
    return normalize_flow_columns(pd.read_csv(temp_csv))


def _stats(values):
    if len(values) == 0:
        return 0, 0, 0, 0
    values = np.asarray(values, dtype=float)
    return values.mean(), values.std(), values.max(), values.min()


def _tcp_header_len(packet):
    return int(packet[TCP].dataofs * 4) if TCP in packet and packet[TCP].dataofs else 0


def _packet_flags(packet):
    if TCP not in packet:
        return set()
    return set(str(packet[TCP].flags))


def _build_flow_row(packets, forward_key):
    first_time = float(packets[0].time)
    times = [float(packet.time) for packet in packets]
    duration_us = max((times[-1] - first_time) * 1_000_000, 1.0)
    duration_s = duration_us / 1_000_000

    fwd_lengths = []
    bwd_lengths = []
    fwd_times = []
    bwd_times = []
    all_lengths = []
    fwd_header_lengths = []
    bwd_header_lengths = []
    flag_counts = {flag: 0 for flag in ["F", "S", "R", "P", "A", "U", "C", "E"]}
    init_win_fwd = -1
    init_win_bwd = -1
    act_data_pkt_fwd = 0
    min_seg_size_forward = 0

    for packet in packets:
        if IP not in packet:
            continue

        if TCP in packet:
            sport, dport = packet[TCP].sport, packet[TCP].dport
            header_len = _tcp_header_len(packet)
            payload_len = len(packet[TCP].payload)
        elif UDP in packet:
            sport, dport = packet[UDP].sport, packet[UDP].dport
            header_len = 8
            payload_len = len(packet[UDP].payload)
        elif ICMP in packet:
            sport, dport = 0, 0
            header_len = int(packet[IP].ihl * 4) if packet[IP].ihl else 20
            payload_len = len(packet[ICMP].payload)
        else:
            continue

        current_key = (packet[IP].src, sport, packet[IP].dst, dport, packet[IP].proto)
        is_forward = current_key == forward_key
        packet_len = len(packet)
        all_lengths.append(packet_len)

        for flag in _packet_flags(packet):
            if flag in flag_counts:
                flag_counts[flag] += 1

        if is_forward:
            fwd_lengths.append(packet_len)
            fwd_times.append(float(packet.time))
            fwd_header_lengths.append(header_len)
            if payload_len > 0:
                act_data_pkt_fwd += 1
            if header_len and (min_seg_size_forward == 0 or header_len < min_seg_size_forward):
                min_seg_size_forward = header_len
            if TCP in packet and init_win_fwd == -1:
                init_win_fwd = int(packet[TCP].window)
        else:
            bwd_lengths.append(packet_len)
            bwd_times.append(float(packet.time))
            bwd_header_lengths.append(header_len)
            if TCP in packet and init_win_bwd == -1:
                init_win_bwd = int(packet[TCP].window)

    flow_iat = np.diff(times) * 1_000_000 if len(times) > 1 else []
    fwd_iat = np.diff(fwd_times) * 1_000_000 if len(fwd_times) > 1 else []
    bwd_iat = np.diff(bwd_times) * 1_000_000 if len(bwd_times) > 1 else []

    fwd_mean, fwd_std, fwd_max, fwd_min = _stats(fwd_lengths)
    bwd_mean, bwd_std, bwd_max, bwd_min = _stats(bwd_lengths)
    pkt_mean, pkt_std, pkt_max, pkt_min = _stats(all_lengths)
    flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = _stats(flow_iat)
    fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = _stats(fwd_iat)
    bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = _stats(bwd_iat)

    total_fwd_bytes = sum(fwd_lengths)
    total_bwd_bytes = sum(bwd_lengths)
    total_packets = len(fwd_lengths) + len(bwd_lengths)
    total_bytes = total_fwd_bytes + total_bwd_bytes

    return {
        "Flow Duration": duration_us,
        "Total Fwd Packets": len(fwd_lengths),
        "Total Backward Packets": len(bwd_lengths),
        "Total Length of Fwd Packets": total_fwd_bytes,
        "Total Length of Bwd Packets": total_bwd_bytes,
        "Fwd Packet Length Max": fwd_max,
        "Fwd Packet Length Min": fwd_min,
        "Fwd Packet Length Mean": fwd_mean,
        "Fwd Packet Length Std": fwd_std,
        "Bwd Packet Length Max": bwd_max,
        "Bwd Packet Length Min": bwd_min,
        "Bwd Packet Length Mean": bwd_mean,
        "Bwd Packet Length Std": bwd_std,
        "Flow Bytes/s": total_bytes / duration_s,
        "Flow Packets/s": total_packets / duration_s,
        "Flow IAT Mean": flow_iat_mean,
        "Flow IAT Std": flow_iat_std,
        "Flow IAT Max": flow_iat_max,
        "Flow IAT Min": flow_iat_min,
        "Fwd IAT Total": sum(fwd_iat) if len(fwd_iat) else 0,
        "Fwd IAT Mean": fwd_iat_mean,
        "Fwd IAT Std": fwd_iat_std,
        "Fwd IAT Max": fwd_iat_max,
        "Fwd IAT Min": fwd_iat_min,
        "Bwd IAT Total": sum(bwd_iat) if len(bwd_iat) else 0,
        "Bwd IAT Mean": bwd_iat_mean,
        "Bwd IAT Std": bwd_iat_std,
        "Bwd IAT Max": bwd_iat_max,
        "Bwd IAT Min": bwd_iat_min,
        "Fwd PSH Flags": sum(1 for packet in packets if IP in packet and TCP in packet and "P" in _packet_flags(packet)),
        "Bwd PSH Flags": 0,
        "Fwd URG Flags": sum(1 for packet in packets if IP in packet and TCP in packet and "U" in _packet_flags(packet)),
        "Bwd URG Flags": 0,
        "Fwd Header Length": sum(fwd_header_lengths),
        "Bwd Header Length": sum(bwd_header_lengths),
        "Fwd Packets/s": len(fwd_lengths) / duration_s,
        "Bwd Packets/s": len(bwd_lengths) / duration_s,
        "Min Packet Length": pkt_min,
        "Max Packet Length": pkt_max,
        "Packet Length Mean": pkt_mean,
        "Packet Length Std": pkt_std,
        "Packet Length Variance": pkt_std ** 2,
        "FIN Flag Count": flag_counts["F"],
        "SYN Flag Count": flag_counts["S"],
        "RST Flag Count": flag_counts["R"],
        "PSH Flag Count": flag_counts["P"],
        "ACK Flag Count": flag_counts["A"],
        "URG Flag Count": flag_counts["U"],
        "CWE Flag Count": flag_counts["C"],
        "ECE Flag Count": flag_counts["E"],
        "Down/Up Ratio": len(bwd_lengths) / len(fwd_lengths) if fwd_lengths else 0,
        "Average Packet Size": total_bytes / total_packets if total_packets else 0,
        "Avg Fwd Segment Size": fwd_mean,
        "Avg Bwd Segment Size": bwd_mean,
        "Fwd Header Length.1": sum(fwd_header_lengths),
        "Subflow Fwd Packets": len(fwd_lengths),
        "Subflow Fwd Bytes": total_fwd_bytes,
        "Subflow Bwd Packets": len(bwd_lengths),
        "Subflow Bwd Bytes": total_bwd_bytes,
        "Init_Win_bytes_forward": init_win_fwd,
        "Init_Win_bytes_backward": init_win_bwd,
        "act_data_pkt_fwd": act_data_pkt_fwd,
        "min_seg_size_forward": min_seg_size_forward,
    }


def extract_flows_with_scapy(pcap_path):
    packets = rdpcap(pcap_path)
    flows = {}
    forward_keys = {}

    for packet in packets:
        if IP not in packet or (TCP not in packet and UDP not in packet and ICMP not in packet):
            continue

        if TCP in packet:
            sport = packet[TCP].sport
            dport = packet[TCP].dport
        elif UDP in packet:
            sport = packet[UDP].sport
            dport = packet[UDP].dport
        else:
            sport = 0
            dport = 0
        proto = packet[IP].proto
        forward_key = (packet[IP].src, sport, packet[IP].dst, dport, proto)
        reverse_key = (packet[IP].dst, dport, packet[IP].src, sport, proto)
        canonical_key = tuple(sorted([forward_key, reverse_key]))

        if canonical_key not in flows:
            flows[canonical_key] = []
            forward_keys[canonical_key] = forward_key
        flows[canonical_key].append(packet)

    rows = [
        _build_flow_row(flow_packets, forward_keys[key])
        for key, flow_packets in flows.items()
        if flow_packets
    ]
    return pd.DataFrame(rows)


def extract_flows(pcap_path, temp_csv=TEMP_CSV):
    try:
        df = extract_flows_with_cicflowmeter(pcap_path, temp_csv)
        if not df.empty:
            return df, "cicflowmeter-patched"
    except Exception as exc:
        print(f"[!] CICFlowMeter extraction failed, using Scapy fallback: {exc}")

    return extract_flows_with_scapy(pcap_path), "scapy-fallback"


def detect_pcap_signatures(pcap_path):
    icmp_pairs = Counter()
    icmp_destinations = Counter()
    icmp_sources = Counter()
    dns_query_destinations = Counter()
    dns_response_destinations = Counter()
    dns_sources = Counter()
    total_dns_udp = 0
    first_time = None
    last_time = None
    total_icmp = 0

    for packet in PcapReader(pcap_path):
        if IP not in packet:
            continue

        packet_time = float(packet.time)
        first_time = packet_time if first_time is None else min(first_time, packet_time)
        last_time = packet_time if last_time is None else max(last_time, packet_time)

        if ICMP in packet:
            total_icmp += 1
            icmp_pairs[(packet[IP].src, packet[IP].dst)] += 1
            icmp_sources[packet[IP].src] += 1
            icmp_destinations[packet[IP].dst] += 1

        if UDP in packet and (packet[UDP].sport == 53 or packet[UDP].dport == 53):
            total_dns_udp += 1
            dns_sources[packet[IP].src] += 1
            if packet[UDP].dport == 53:
                dns_query_destinations[packet[IP].dst] += 1
            if packet[UDP].sport == 53:
                dns_response_destinations[packet[IP].dst] += 1

    alerts = []
    duration = max((last_time - first_time), 1.0) if first_time is not None and last_time is not None else 1.0
    top_pair, top_count = icmp_pairs.most_common(1)[0] if icmp_pairs else ((None, None), 0)
    top_destination, top_destination_count = icmp_destinations.most_common(1)[0] if icmp_destinations else (None, 0)
    icmp_rate = total_icmp / duration

    if total_icmp >= 1000 or top_count >= 1000 or top_destination_count >= 1000 or icmp_rate >= 50:
        alerts.append({
            "event_type": "ICMPFlood",
            "severity": "HIGH",
            "source": top_pair[0],
            "destination": top_destination or top_pair[1],
            "evidence": (
                f"{total_icmp} ICMP packets, {len(icmp_sources)} unique sources, "
                f"top destination={top_destination} ({top_destination_count}), rate={icmp_rate:.2f} pkt/s"
            ),
        })

    dns_rate = total_dns_udp / duration
    top_query_destination, top_query_count = (
        dns_query_destinations.most_common(1)[0] if dns_query_destinations else (None, 0)
    )
    top_response_destination, top_response_count = (
        dns_response_destinations.most_common(1)[0] if dns_response_destinations else (None, 0)
    )

    if total_dns_udp >= 5000 and (top_query_count >= 5000 or top_response_count >= 5000 or dns_rate >= 500):
        if top_response_count >= top_query_count:
            destination = top_response_destination
            evidence_mode = "responses"
            evidence_count = top_response_count
        else:
            destination = top_query_destination
            evidence_mode = "queries"
            evidence_count = top_query_count

        alerts.append({
            "event_type": "DNSAmplification",
            "severity": "HIGH",
            "source": dns_sources.most_common(1)[0][0] if dns_sources else None,
            "destination": destination,
            "evidence": (
                f"{total_dns_udp} DNS/UDP packets, {evidence_count} {evidence_mode} for top target, "
                f"{len(dns_sources)} unique DNS sources, rate={dns_rate:.2f} pkt/s"
            ),
        })

    return alerts


def live_analyze(pcap_path, model_dir=DEFAULT_MODEL_DIR, use_signatures=False):
    if not os.path.exists(pcap_path):
        print(f"Error: File {pcap_path} not found.")
        return

    model_dir = model_dir or DEFAULT_MODEL_DIR
    model_file = os.path.join(model_dir, "model.pkl")
    scaler_file = os.path.join(model_dir, "scaler.pkl")
    label_encoder_file = os.path.join(model_dir, "label_encoder.pkl")
    features_file = os.path.join(model_dir, "features.pkl")

    required_files = [model_file, scaler_file, label_encoder_file, features_file]
    if any(not os.path.exists(path) for path in required_files):
        print(f"Error: Missing saved artifacts in {model_dir}. Please train or copy a model first.")
        return

    temp_csv = os.path.join(
        os.path.dirname(TEMP_CSV) or ".",
        f"temp_extracted_flow_{os.getpid()}_{os.path.basename(pcap_path)}.csv"
    )
    if os.path.exists(temp_csv):
        os.remove(temp_csv)

    try:
        df, extractor_name = extract_flows(pcap_path, temp_csv=temp_csv)
        if df.empty:
            print("Analysis Error: No flows extracted.")
            return

        print(f"[+] Flow extractor: {extractor_name}")
        X_processed = clean_data(df)

        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        label_encoder = joblib.load(label_encoder_file)

        expected_features = None
        if os.path.exists(features_file):
            expected_features = joblib.load(features_file)
        elif hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
        else:
            try:
                expected_features = list(model.get_booster().feature_names)
            except Exception:
                expected_features = None

        if expected_features is not None:
            for col in expected_features:
                if col not in X_processed.columns:
                    X_processed[col] = 0
            X_processed = X_processed[expected_features]
        else:
            print("[!] Warning: No feature list found, using raw extracted features.")

        X_processed = scaler.transform(X_processed)
        predictions = model.predict(X_processed)
        predictions = np.asarray(predictions)
        if predictions.ndim > 1:
            predictions = predictions.argmax(axis=1)

        try:
            class_labels = label_encoder.inverse_transform(predictions.astype(int))
        except Exception:
            class_labels = predictions

        results = pd.Series(class_labels).value_counts()
        total_flows = int(results.sum())
        signature_alerts = detect_pcap_signatures(pcap_path) if use_signatures else []

        print("\n" + "="*50)
        print(f" NIDS ANALYSIS REPORT | Source: {os.path.basename(pcap_path)}")
        print("="*50)
        print(f"{'TYPE':<20} | {'STATUS':<10} | {'FLOWS':<10}")
        print("-" * 50)

        for label, count in results.items():
            status = "CLEAN" if str(label) == "Normal" else "THREAT"
            print(f"{label:<20} | {status:<10} | {count:<10}")

        print("-" * 50)
        print(f"{'TOTAL':<20} | {'':<10} | {total_flows:<10}")

        if signature_alerts:
            print("-" * 50)
            print(" PCAP SIGNATURE ALERTS")
            for alert in signature_alerts:
                print(
                    f"{alert['event_type']:<20} | {alert['severity']:<10} | "
                    f"{alert['source']} -> {alert['destination']} | {alert['evidence']}"
                )
        print("="*50 + "\n")

    except Exception as e:
        print(f"Prediction Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 live_analyze.py <path_to_pcap>")
    else:
        live_analyze(sys.argv[1])

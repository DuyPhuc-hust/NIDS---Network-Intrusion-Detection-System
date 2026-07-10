from pathlib import Path
from random import Random

from scapy.all import DNS, DNSQR, ICMP, IP, TCP, UDP, Raw, wrpcap


OUT_DIR = Path("data/pcap/generated")
RNG = Random(42)


def stamp_packets(packets, start=1_725_000_000.0, step=0.001):
    for index, packet in enumerate(packets):
        packet.time = start + index * step
    return packets


def write_pcap(name, packets):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    wrpcap(str(path), packets)
    print(f"{path}: {len(packets)} packets")


def syn_scan():
    attacker = "198.51.100.10"
    victim = "203.0.113.20"
    packets = []
    for port in range(1, 1201):
        packets.append(
            IP(src=attacker, dst=victim)
            / TCP(sport=RNG.randint(20_000, 60_000), dport=port, flags="S", seq=port)
        )
    return stamp_packets(packets, step=0.0008)


def udp_flood():
    attacker = "198.51.100.30"
    victim = "203.0.113.40"
    payload = Raw(b"A" * 900)
    packets = []
    for index in range(2500):
        packets.append(
            IP(src=attacker, dst=victim)
            / UDP(sport=20_000 + (index % 20_000), dport=7777)
            / payload
        )
    return stamp_packets(packets, step=0.00025)


def icmp_flood():
    attacker = "198.51.100.50"
    victim = "203.0.113.60"
    payload = Raw(b"ping-flood" * 32)
    packets = []
    for index in range(1800):
        packets.append(
            IP(src=attacker, dst=victim)
            / ICMP(type=8, id=1234, seq=index)
            / payload
        )
    return stamp_packets(packets, step=0.0003)


def dns_burst():
    victim = "203.0.113.80"
    resolver_ips = [f"198.51.100.{index}" for index in range(80, 130)]
    packets = []
    for index in range(2200):
        resolver = resolver_ips[index % len(resolver_ips)]
        query = f"burst{index}.example.test"
        packets.append(
            IP(src=resolver, dst=victim)
            / UDP(sport=53, dport=RNG.randint(20_000, 60_000))
            / DNS(id=index % 65535, qr=1, aa=1, qd=DNSQR(qname=query), ancount=0)
        )
    return stamp_packets(packets, step=0.00035)


def mixed_attack():
    packets = []
    packets.extend(syn_scan()[:500])
    packets.extend(udp_flood()[:800])
    packets.extend(icmp_flood()[:500])
    packets.extend(dns_burst()[:700])
    return stamp_packets(packets, step=0.0004)


def main():
    write_pcap("synthetic_syn_scan.pcap", syn_scan())
    write_pcap("synthetic_udp_flood.pcap", udp_flood())
    write_pcap("synthetic_icmp_flood.pcap", icmp_flood())
    write_pcap("synthetic_dns_burst.pcap", dns_burst())
    write_pcap("synthetic_mixed_attack.pcap", mixed_attack())


if __name__ == "__main__":
    main()

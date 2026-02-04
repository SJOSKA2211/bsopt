#include <linux/bpf.h>
#include <bpf/bpf_helpers.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/udp.h>

SEC("xdp_market_filter")
int xdp_prog(struct xdp_md *ctx) {
    void *data_end = (void *)(long)ctx->data_end;
    void *data = (void *)(long)ctx->data;
    
    struct ethhdr *eth = data;
    if (data + sizeof(*eth) > data_end) return XDP_PASS;
    
    if (eth->h_proto != bpf_htons(ETH_P_IP)) return XDP_PASS;
    
    struct iphdr *ip = data + sizeof(*eth);
    if ((void*)ip + sizeof(*ip) > data_end) return XDP_PASS;
    
    if (ip->protocol == IPPROTO_UDP) {
        struct udphdr *udp = (void*)ip + sizeof(*ip);
        if ((void*)udp + sizeof(*udp) > data_end) return XDP_PASS;
        
        // Port 5555 is used for internal high-speed Market Data Mesh
        if (udp->dest == bpf_htons(5555)) {
            // REDIRECT to the AF_XDP socket in userspace (Pricing/Ingestion)
            // This bypasses the entire Linux TCP/IP stack.
            return XDP_REDIRECT;
        }
    }
    
    return XDP_PASS;
}

char _license[] SEC("license") = "GPL";

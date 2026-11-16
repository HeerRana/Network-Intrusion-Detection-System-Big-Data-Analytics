import os
import shutil
import sys

# 1. Exact 49 column headers
header = (
    "srcip,sport,dstip,dsport,proto,state,dur,sbytes,dbytes,sttl,dttl,"
    "sloss,dloss,service,Sload,Dload,Spkts,Dpkts,swin,dwin,stcpb,dtcpb,"
    "smeansz,dmeansz,trans_depth,res_bdy_len,Sjit,Djit,Stime,Ltime,"
    "Sintpkt,Dintpkt,tcprtt,synack,ackdat,is_sm_ips_ports,ct_state_ttl,"
    "ct_flw_http_mthd,is_ftp_login,ct_ftp_cmd,ct_srv_src,ct_srv_dst,"
    "ct_dst_ltm,ct_src_ltm,ct_src_dport_ltm,ct_dst_sport_ltm,"
    "ct_dst_src_ltm,attack_cat,Label\n"
)

filenames = [
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv",
]

os.makedirs("data", exist_ok=True)
output_path = os.path.join("data", "UNSW-NB15.csv")

# Resolve paths whether placed in data/ or root directory
files = []
for fname in filenames:
    data_path = os.path.join("data", fname)
    if os.path.exists(data_path):
        files.append(data_path)
    elif os.path.exists(fname):
        files.append(fname)
    else:
        files.append(data_path)

# Verify source files exist
missing = [f for f in files if not os.path.exists(f)]
if missing:
    print(f"Error: Could not find partition files: {missing}")
    print("Please place UNSW-NB15_1.csv through UNSW-NB15_4.csv in the data/ directory.")
    sys.exit(1)

# 2. Stream-merge files without loading everything into RAM
with open(output_path, "wb") as outfile:
    outfile.write(header.encode("utf-8"))
    for fpath in files:
        print(f"Merging {fpath}...")
        with open(fpath, "rb") as infile:
            shutil.copyfileobj(infile, outfile)

print("Done! Saved to:", output_path)

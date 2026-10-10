"""One GPU machine on EC2, for the model comparison and for training.

    python -m training.ec2 launch --hours 4 --setup ollama   # prints the instance id and public IP
    python -m training.ec2 status
    python -m training.ec2 terminate

Safety first: the security group admits SSH only from this computer's public
IP; the instance is told to TERMINATE itself after --hours whatever happens
(`shutdown -h` + instance-initiated-shutdown-behavior=terminate), so a forgotten
machine can't keep billing. The SSH key lives in ~/.ssh, never in the repo.
"""
import argparse
import json
import os
import stat
import subprocess
import time
import urllib.request
from pathlib import Path

import boto3

REGION = "us-east-1"
NAME = "f1-chat-gpu"
KEY_NAME = "f1-chat-train"
KEY_PATH = Path.home() / ".ssh" / f"{KEY_NAME}.pem"
SG_NAME = "f1-chat-ssh"
EBS_GB_MONTH_USD = 0.08   # gp3
# tried in order until AWS has capacity: L40S 48 GB (training), then A10G / L4 24 GB (enough to serve a 14B Q4 model)
INSTANCE_TYPES = ["g6e.xlarge", "g5.xlarge", "g6.xlarge", "g5.2xlarge", "g6.2xlarge"]
AMI_NAME = "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04) *"
STATE = Path(__file__).resolve().parent / "data" / "ec2_instance.json"

SETUP = {
    "ollama": """
curl -fsSL https://ollama.com/install.sh | sh
mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\\nEnvironment="OLLAMA_NUM_PARALLEL=4"\\nEnvironment="OLLAMA_MAX_LOADED_MODELS=1"\\nEnvironment="OLLAMA_KEEP_ALIVE=30m"\\n' > /etc/systemd/system/ollama.service.d/override.conf
systemctl daemon-reload && systemctl restart ollama
sleep 5
for m in {models}; do ollama pull "$m" >> /var/log/ollama-pull.log 2>&1; done
touch /var/log/setup-done
""",
    "none": "touch /var/log/setup-done\n",
}


def ec2():
    return boto3.client("ec2", region_name=REGION)


def my_ip() -> str:
    return urllib.request.urlopen("https://checkip.amazonaws.com", timeout=10).read().decode().strip()


def ensure_key() -> None:
    if KEY_PATH.exists():
        return
    KEY_PATH.parent.mkdir(exist_ok=True)
    try:
        ec2().delete_key_pair(KeyName=KEY_NAME)  # a key whose private half isn't here is useless
    except Exception:
        pass
    material = ec2().create_key_pair(KeyName=KEY_NAME, KeyType="ed25519")["KeyMaterial"]
    KEY_PATH.write_bytes(material.encode())  # bytes: Windows text mode would write CRLF, which OpenSSH cannot load
    os.chmod(KEY_PATH, stat.S_IRUSR)
    if os.name == "nt":  # OpenSSH on Windows refuses keys other accounts can read
        user = os.environ.get("USERNAME", "")
        subprocess.run(["icacls", str(KEY_PATH), "/inheritance:r", "/grant:r", f"{user}:R"], check=True, capture_output=True)


def ensure_sg() -> str:
    c = ec2()
    vpc = c.describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
    found = c.describe_security_groups(Filters=[{"Name": "group-name", "Values": [SG_NAME]}, {"Name": "vpc-id", "Values": [vpc]}])["SecurityGroups"]
    sg = found[0]["GroupId"] if found else c.create_security_group(GroupName=SG_NAME, Description="SSH from the owner IP only", VpcId=vpc)["GroupId"]
    # the owner's laptop switches between its own IP and Cloudflare WARP, whose exit IP rotates inside
    # 104.28.0.0/16 mid-session (dropping the SSH tunnel): allow both (SSH is key-only; machines self-terminate)
    ranges = sorted({f"{my_ip()}/32", "104.28.0.0/16"})
    current = found[0]["IpPermissions"] if found else []
    if current:
        c.revoke_security_group_ingress(GroupId=sg, IpPermissions=current)  # only ever today's IP
    c.authorize_security_group_ingress(GroupId=sg, IpPermissions=[{"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
                                                                  "IpRanges": [{"CidrIp": r, "Description": "owner"} for r in ranges]}])
    return sg


def latest_ami() -> str:
    images = ec2().describe_images(Owners=["amazon"], Filters=[{"Name": "name", "Values": [AMI_NAME]}, {"Name": "state", "Values": ["available"]}])["Images"]
    return max(images, key=lambda i: i["CreationDate"])["ImageId"]


def hourly_price(instance_type: str) -> float:
    """On-demand Linux price in us-east-1, from the AWS price list API."""
    pricing = boto3.client("pricing", region_name="us-east-1")
    filters = {"instanceType": instance_type, "location": "US East (N. Virginia)", "operatingSystem": "Linux", "tenancy": "Shared",
               "preInstalledSw": "NA", "capacitystatus": "Used"}
    item = json.loads(pricing.get_products(ServiceCode="AmazonEC2", Filters=[{"Type": "TERM_MATCH", "Field": k, "Value": v} for k, v in filters.items()])["PriceList"][0])
    return float(next(d["pricePerUnit"]["USD"] for t in item["terms"]["OnDemand"].values() for d in t["priceDimensions"].values()))


def _cost(hours: float, disk_gb: int, rate: float) -> float:
    return hours * rate + disk_gb * EBS_GB_MONTH_USD * hours / 730


def launch(hours: float, setup: str, models: list[str], disk_gb: int, types: list[str] | None = None) -> dict:
    """Start the first instance type (in `types` order) that any us-east-1 zone has capacity for."""
    from botocore.exceptions import ClientError

    from training import budget_guard

    types = types or INSTANCE_TYPES
    rates = {t: hourly_price(t) for t in types}
    budget_guard.check(_cost(hours, disk_gb, max(rates.values())), f"GPU machine for up to {hours} h")  # self-terminates by then: the most it can cost
    ensure_key()
    sg = ensure_sg()
    # cloud-init runs without $HOME, and the ollama CLI panics without it
    user_data = "#!/bin/bash\nset -x\nexport HOME=/root\n" + f"shutdown -h +{int(hours * 60)}\n" + SETUP[setup].replace("{models}", " ".join(models))
    vpc = ec2().describe_vpcs(Filters=[{"Name": "isDefault", "Values": ["true"]}])["Vpcs"][0]["VpcId"]
    subnets = ec2().describe_subnets(Filters=[{"Name": "vpc-id", "Values": [vpc]}, {"Name": "default-for-az", "Values": ["true"]}])["Subnets"]
    r, chosen = None, None
    for itype in types:
        for subnet in subnets:
            try:
                r = ec2().run_instances(
                    ImageId=latest_ami(), InstanceType=itype, KeyName=KEY_NAME, SecurityGroupIds=[sg], SubnetId=subnet["SubnetId"], MinCount=1, MaxCount=1,
                    InstanceInitiatedShutdownBehavior="terminate", UserData=user_data,
                    BlockDeviceMappings=[{"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": disk_gb, "VolumeType": "gp3", "DeleteOnTermination": True}}],
                    TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": NAME}, {"Key": "project", "Value": "f1-chat"}]}],
                )
                chosen = (itype, subnet["AvailabilityZone"])
                break
            except ClientError as exc:
                code = exc.response["Error"]["Code"]
                if code not in ("InsufficientInstanceCapacity", "Unsupported", "InvalidParameterValue"):
                    raise
                print(f"  no {itype} in {subnet['AvailabilityZone']} ({code})", flush=True)
        if r:
            break
    if not r:
        raise SystemExit("no capacity for any GPU type in any us-east-1 zone right now; try again later")
    print(f"launched {chosen[0]} in {chosen[1]} at ${rates[chosen[0]]:.3f}/h", flush=True)
    iid = r["Instances"][0]["InstanceId"]
    ec2().get_waiter("instance_running").wait(InstanceIds=[iid])
    ip = ec2().describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]["PublicIpAddress"]
    info = {"instance_id": iid, "type": chosen[0], "zone": chosen[1], "ip": ip, "launched": time.strftime("%Y-%m-%d %H:%M:%S"),
            "terminates_after_hours": hours, "setup": setup, "models": models}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({**info, "launched_epoch": time.time(), "disk_gb": disk_gb, "hourly_usd": rates[chosen[0]]}, indent=1))
    budget_guard.record(f"GPU machine {iid} {chosen[0]} (up to {hours} h, until terminated)", _cost(hours, disk_gb, rates[chosen[0]]), key=iid)
    return info


def ssh_cmd(*remote: str) -> list[str]:
    info = json.loads(STATE.read_text())
    return ["ssh", "-i", str(KEY_PATH), "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30", f"ubuntu@{info['ip']}", *remote]


def status() -> dict:
    info = json.loads(STATE.read_text())
    inst = ec2().describe_instances(InstanceIds=[info["instance_id"]])["Reservations"][0]["Instances"][0]
    return {**info, "state": inst["State"]["Name"]}


def terminate() -> None:
    from training import budget_guard

    info = json.loads(STATE.read_text())
    ec2().terminate_instances(InstanceIds=[info["instance_id"]])
    used_h = min(info["terminates_after_hours"], (time.time() - info["launched_epoch"]) / 3600 + 1 / 60)  # AWS bills per second
    cost = _cost(used_h, info.get("disk_gb", 150), info["hourly_usd"])
    budget_guard.record(f"GPU machine {info['instance_id']} {info['type']} ({used_h:.2f} h)", cost, key=info["instance_id"])
    print(f"terminating {info['instance_id']} after {used_h:.2f} h (~${cost:.2f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["launch", "status", "terminate"])
    ap.add_argument("--hours", type=float, default=4.0, help="the instance terminates itself after this long")
    ap.add_argument("--setup", choices=list(SETUP), default="none")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--disk", type=int, default=150)
    ap.add_argument("--types", nargs="*", help=f"instance types to try in order (default {INSTANCE_TYPES})")
    args = ap.parse_args()
    if args.action == "launch":
        print(json.dumps(launch(args.hours, args.setup, args.models, args.disk, args.types), indent=1))
    elif args.action == "status":
        print(json.dumps(status(), indent=1))
    else:
        terminate()

#!/usr/bin/env bash
# Launch one self-terminating Trn2 instance that runs scripts/aws/userdata.sh.
#
# Required env: S3 (s3://bucket/prefix holding code/ ; results land in results/),
#               SUBNET (needs outbound internet: HF Hub + pip), PROFILE (instance profile
#               with read on $S3/code/* and write on $S3/results/*).
# Optional env: REGION (sa-east-1), TYPE (trn2.3xlarge), AMI (latest Neuron multi-framework DLAMI),
#               SG, MAX_MIN (180), CAPACITY_RESERVATION (a Capacity Block id),
#               MATRIX (full | verify | check; see scripts/run_experiments.sh).
# Stage the code first:  aws s3 sync . "$S3/code/" --exclude '.venv/*' --exclude 'runs/*'
set -euo pipefail
: "${S3:?set S3}" "${SUBNET:?set SUBNET}" "${PROFILE:?set PROFILE}"
REGION=${REGION:-sa-east-1}
TYPE=${TYPE:-trn2.3xlarge}
MAX_MIN=${MAX_MIN:-180}
HERE=$(cd "$(dirname "$0")" && pwd)

if [ -z "${AMI:-}" ]; then
  # Latest Neuron multi-framework DLAMI (has /opt/aws_neuronx_venv_torch_neuronx_2_14_0).
  AMI=$(aws ssm get-parameter --region "$REGION" \
    --name /aws/service/neuron/dlami/multi-framework/ubuntu-24.04/latest/image_id \
    --query Parameter.Value --output text)
fi
echo "AMI $AMI  type $TYPE  region $REGION"

UD=$(mktemp)
sed -e "s#__S3__#$S3#" -e "s#__MAX_MIN__#$MAX_MIN#" -e "s#__MATRIX__#${MATRIX:-full}#" "$HERE/userdata.sh" > "$UD"

extra=()
[ -n "${SG:-}" ] && extra+=(--security-group-ids "$SG")
if [ -n "${CAPACITY_RESERVATION:-}" ]; then
  extra+=(--instance-market-options MarketType=capacity-block
          --capacity-reservation-specification "CapacityReservationTarget={CapacityReservationId=$CAPACITY_RESERVATION}")
fi

aws ec2 run-instances --region "$REGION" --image-id "$AMI" --instance-type "$TYPE" \
  --subnet-id "$SUBNET" --iam-instance-profile "Name=$PROFILE" "${extra[@]}" \
  --instance-initiated-shutdown-behavior terminate \
  --metadata-options HttpTokens=required \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=200,VolumeType=gp3,DeleteOnTermination=true}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=dinov3-torchneuron-finetune}]' \
  --user-data "file://$UD" \
  --query 'Instances[0].[InstanceId,Placement.AvailabilityZone,State.Name]' --output text
rm -f "$UD"

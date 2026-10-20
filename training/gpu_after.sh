#!/bin/bash
# Runs on the GPU machine beside training: when training ends, export the model,
# check it answers in Ollama, upload everything to the bucket, and shut down
# (the machine terminates itself on shutdown). Any failure stops the chain and
# leaves the machine up for a look; it still terminates at its time limit.
#   setsid nohup bash gpu_after.sh <bucket> > after.log 2>&1 &
set -x
BUCKET=$1
cd ~ && source venv/bin/activate
while pgrep -f '^python sft.py train' > /dev/null; do sleep 60; done
aws s3 cp ~/train.log "s3://$BUCKET/model/train.log" --only-show-errors
test -d run/adapter || { echo "no adapter: training did not finish"; exit 1; }
aws s3 sync run/adapter "s3://$BUCKET/model/adapter" --only-show-errors
python sft.py export --out run > ~/export.log 2>&1 || { aws s3 cp ~/export.log "s3://$BUCKET/model/export.log"; exit 1; }
aws s3 cp ~/export.log "s3://$BUCKET/model/export.log" --only-show-errors
cd run
printf 'FROM ./f1-analyst-q4_k_m.gguf\nTEMPLATE {{ .Prompt }}\nPARAMETER stop "<|im_end|>"\n' > Modelfile
ollama create f1-analyst -f Modelfile
sed 's/"f1-test"/"f1-analyst"/' ~/base/probe.json > probe.json
curl -s localhost:11434/api/generate -d @probe.json > probe_reply.json
aws s3 cp probe_reply.json "s3://$BUCKET/model/probe_reply.json" --only-show-errors
aws s3 cp f1-analyst-q4_k_m.gguf "s3://$BUCKET/model/f1-analyst-q4_k_m.gguf" --only-show-errors || exit 1
echo uploaded > ~/after-done
sudo shutdown -h +2

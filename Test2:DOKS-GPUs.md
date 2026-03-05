# Test2: DOKS with GPUs


## Goals

- Familiar with AMD GPUs and their monitoring tools, including rocm-smi and amd-smi, for troubleshooting and performance analysis.

- Provision a DOKS cluster with both CPU and GPU nodes using the doctl CLI.

- Deploy CPU and GPU workloads within the DOKS cluster.

- Review the Pod-to-Pod and Service networking in Kubernetes, and explore different ways to access workloads.

- Consider the key differences between AI inference services and traditional web services from the infrastructure solution perspective.

## Provision a DOKS cluster from the Management Droplet

**For this test, we only require on-demand 1-GPU droplets (AMD mi300 or mi325, more available) — not 8-GPU or contracted GPU droplets.**

[Provision a DOKS cluster](https://docs.digitalocean.com/reference/doctl/reference/kubernetes/cluster/create/#example) with at least one CPU droplet and one 1-GPU droplet. If GPUs are unavailable, try using different slugs or regions. 

```
# Example: create a DOKS with 2 node pools

doctl kubernetes cluster create rs-atl1-test1 \
  --region atl1 \
  --version 1.34.1-do.4 \
  --node-pool "name=rs-atl1-test1-cpu-pool;size=s-4vcpu-8gb-amd;count=1" \
  --node-pool "name=rs-atl1-test1-mi325x1-pool;size=gpu-mi325x1-256gb;count=1" \
  --tag "rs-atl1-test1"

# 1-GPU Droplet Slugs - AMD

gpu-mi300x1-192gb
gpu-mi325x1-256gb

# List all CPUs and GPUs

doctl compute size list | grep vcpu 
doctl compute size list | grep gpu
doctl compute size list | grep mi300x1
doctl compute size list | grep mi325x1
```

Check the newly-created DOKS cluster in the DO Console. 

You should see both the CPU and GPU droplets, that are managed by DOKS. While you can log into these nodes by adding your SSH keys and configuring them, this is not recommended, as it may create conflicts with DOKS management.

Under **Kubernetes → Connecting to Kubernetes** in the Console, copy the provided script and paste it on the management droplet to configure access, where you will maanage the DOKS cluster using `kubectl` commands.

```
doctl kubernetes cluster get rs-atl1-test1 -o json

kubectl get nodes -o wide
NAME                               STATUS   ROLES    AGE     VERSION   INTERNAL-IP    EXTERNAL-IP       OS-IMAGE                       KERNEL-VERSION        CONTAINER-RUNTIME
rs-atl1-test1-cpu-pool-hhjja       Ready    <none>   8m53s   v1.34.1   10.128.1.129   134.199.204.225   Debian GNU/Linux 13 (trixie)   6.12.73+deb13-amd64   containerd://1.7.28
rs-atl1-test1-mi325x1-pool-hhjj6   Ready    <none>   7m44s   v1.34.1   10.128.1.130   165.245.135.17    Debian GNU/Linux 13 (trixie)   6.12.73+deb13-amd64   containerd://1.7.28

# Check the node details, including labels and annotations
kubectl describe node rs-atl1-test1-cpu-pool-hhjja 
kubectl describe node rs-atl1-test1-mi325x1-pool-hhjj6

kubectl get namespaces
kubectl get pods -ALL -o wide 

# Research these types of pods
kubectl -n kube-system describe pod amd-gpu-device-plugin-x57v6 
kubectl -n kube-system describe pod do-node-agent-amd-device-metrics-exporter-mhn76
```


## Deploy a vLLM Llama (70B) GPU workload from the Management Droplet

[Llama 70B](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct/tree/main) (FP16) requires approximately 140 GB of VRAM for the model weights alone. In addition, it requires runtime KV cache, which can consume tens of gigabytes of VRAM, depending on the number of concurrent sessions and the context length of each session. A single MI300 or MI325 GPU can run this mid-size model without any issues.

The vLLM server provides [various parameters](https://docs.vllm.ai/en/stable/cli/serve/#arguments) (flags) that control model loading, inference behavior, GPU usage, and server performance. Take a moment to review the common ones, including host, port, gpu-memory-utilization, and max-model-len, etc.

Create a Kubernetes Secret to store your Hugging Face token, which will be used by the vLLM Llama workload to download the gated models from Hugging Face:

```
kubectl create secret generic hf-token-secret --from-literal=HF_TOKEN=<HUGGING_FACE TOKEN>

kubectl get secret
kubectl describe secret hf-token-secret

# To delete the secret
kubectl delete secret hf-token-secret 
```

Create the vLLM Llama deployment (GPU Workload) using the provided yaml:

```
kubectl apply -f Test2/amd-vllm-llama-70b.yaml

kubectl get pods -o wide 
watch kubectl get pods

# Measure the time of image pulling and start
kubectl describe pod vllm-llama-86449cd99d-f4jn8

# Measure the time of model downloading, loading to GPU(s), and warm-up
kubectl logs -f vllm-llama-86449cd99d-f4jn8

# To delete the workload
kubectl delete -f Test2/amd-vllm-llama-70b.yaml
```

Execute into the pod and verify:

```
kubectl exec -it vllm-llama-86449cd99d-f4jn8 -- /bin/bash

root@vllm-llama-86449cd99d-f4jn8:/app# ps -ef

root@vllm-llama-86449cd99d-f4jn8:/app# ls -ls /root/.cache/huggingface/hub/
root@vllm-llama-86449cd99d-f4jn8:/app# du -sh /root/.cache/huggingface/hub/

root@vllm-llama-86449cd99d-f4jn8:/app# rocm-smi 
root@vllm-llama-86449cd99d-f4jn8:/app# amd-smi
root@vllm-llama-86449cd99d-f4jn8:/app# amd-smi version

root@vllm-llama-86449cd99d-f4jn8:/app# env | grep HF_TOKEN

root@vllm-llama-86449cd99d-f4jn8:/app# curl http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "meta-llama/Llama-3.1-70B-Instruct",
    "messages": [{"role": "user", "content": "who are you?"}]
  }'

root@vllm-llama-86449cd99d-f4jn8:/app# vllm bench serve --model meta-llama/Llama-3.1-70B-Instruct --dataset-name random \
--random-input-len 1000 --random-output-len 100 --request-rate inf \
--num-prompts 16 --base-url http://localhost:8000 \
--endpoint /v1/completions --backend openai
root@vllm-llama-86449cd99d-f4jn8:/app# 
```

**Notes:** AMD rocm-smi is being deprecated in favor of amd-smi, a newer system management interface that provides functionality similar to nvidia-smi for NVIDIA GPUs.

Use the Web Console in the DO Console to log in to the DOKS-managed GPU droplet and verify:

```
root@rs-atl1-test1-mi325x1-pool-hhjj6:~# ps -ef | grep vllm

root@rs-atl1-test1-mi325x1-pool-hhjj6:~# ls -ls /root/.cache/huggingface/hub/
root@rs-atl1-test1-mi325x1-pool-hhjj6:~# du -sh /root/.cache/huggingface/hub/

root@rs-atl1-test1-mi325x1-pool-hhjj6:~# rocm-smi
root@rs-atl1-test1-mi325x1-pool-hhjj6:~# amd-smi
root@rs-atl1-test1-mi325x1-pool-hhjj6:~# amd-smi version
```

**Notes:** AMD ROCm is a user-space library stack. The pod runs its own ROCm user-space components provided by the container image, while relying on the GPU kernel driver from the host. Therefore, the ROCm version inside the pod does not need to exactly match the ROCm version installed on the host. 

Check the droplet metrics and GPU metrics using the DO Console, including CPU memory, Disk I/O, Bandwidth, GPU Memory Untilizaiton and PCIe Throughput, etc.

## Expose the vLLM Llama workload from the Management Droplet

Use the provided YAML to expose the workload both internally and externally. 

While a NodePort automatically creates a ClusterIP for internal access, it’s generally better to define separate services for different use cases.

```
kubectl apply -f Test2/service-exposure.yaml


# Get the ClusterIP

kubectl get svc
NAME            TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)        AGE
vllm-internal   ClusterIP   10.128.158.240   <none>        80/TCP         8m35s
vllm-nodeport   NodePort    10.128.142.45    <none>        80:30080/TCP   8m35s

# Get the pod's IP

kubectl get pod -o wide
kubectl get pod vllm-llama-765f79fffb-k7l7b -o jsonpath='{.status.podIP}'

10.129.0.227

# Get the node’s external IPs

kubectl get nodes -o wide
kubectl get nodes -o jsonpath='{range .items[*]}{.status.addresses[?(@.type=="ExternalIP")].address}{"\n"}{end}'

134.199.204.225
165.245.135.17
```

Test the inference using the NodePort from the management droplet or your laptop:

```
curl http://134.199.204.225:30080/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "meta-llama/Llama-3.1-70B-Instruct",
    "messages": [{"role": "user", "content": "How to learn AI/ML? 200 words"}]
  }'
```

Run a temporary pod within the DOKS to test the inference using the internal ClusterIP DNS:

```
kubectl run curl-test --rm -i -t --restart=Never --image=curlimages/curl -- \
  curl http://vllm-internal.default.svc.cluster.local:80/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"meta-llama/Llama-3.1-70B-Instruct","messages":[{"role":"user","content":"How are you?"}]}'
```

Run a temporary pod within the DOKS to test the inference using the internal ClusterIP:
```
kubectl run curl-test --rm -i -t --restart=Never --image=curlimages/curl -- \
  curl http://10.128.158.240:80/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"meta-llama/Llama-3.1-70B-Instruct","messages":[{"role":"user","content":"How are you?"}]}'
```

Run a temporary pod within the DOKS to test the inference using the internal Pod IP:
```
kubectl run curl-test --rm -i -t --restart=Never --image=curlimages/curl -- \
  curl http://10.129.0.227:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"meta-llama/Llama-3.1-70B-Instruct","messages":[{"role":"user","content":"How are you?"}]}'
```

## Deploy a vLLM benchmarker CPU workload from the Management Droplet and Benchamrk the vLLM Llama workload


Create the vLLM benchmarker deployment (CPU Workload) using the provided yaml:

```
kubectl apply -f Test2/vllm-benchmarker.yaml

kubectl get pods -o wide 
watch kubectl get pods
kubectl describe pod vllm-benchmarker-7994b7977c-kf9cc
kubectl logs -f vllm-benchmarker-7994b7977c-kf9cc

# To delete the workload
kubectl delete -f Test2/vllm-benchmarker.yaml
```

Execute into the pod and benchmark the vLLM Llama workload:

```
kubectl exec -it vllm-benchmarker-7994b7977c-kf9cc -- /bin/bash

# Benchmarking using the internal ClusterIP DNS

root@vllm-benchmarker-7994b7977c-kf9cc:/app# curl http://vllm-internal.default.svc.cluster.local:80/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"meta-llama/Llama-3.1-70B-Instruct","messages":[{"role":"user","content":"How are you?"}]}'

root@vllm-benchmarker-7994b7977c-kf9cc:/app# vllm bench serve \
  --base-url http://vllm-internal.default.svc.cluster.local:80 \
  --model "meta-llama/Llama-3.1-70B-Instruct" \
  --dataset-name random \
  --random-input-len 256 \
  --random-output-len 256 \
  --request-rate 10000 \
  --num-prompts 4 \
  --ignore-eos \
  --trust-remote-code  \
  --endpoint /v1/completions \
  --backend openai \
  --save-result \
  --result-filename result.json

root@vllm-benchmarker-7994b7977c-kf9cc:/app# cat result.json

# Benchmarking using the internal Pod IP

root@vllm-benchmarker-7994b7977c-kf9cc:/app# vllm bench serve \
  --base-url http://10.129.0.227:8000 \
  --model "meta-llama/Llama-3.1-70B-Instruct" \
  --dataset-name random \
  --random-input-len 256 \
  --random-output-len 256 \
  --request-rate 10000 \
  --num-prompts 4 \
  --ignore-eos \
  --trust-remote-code  \
  --endpoint /v1/completions \
  --backend openai \
  --save-result \
  --result-filename result.json
```

## Other Tests

You can run additional tests by adding a CPU or GPU node pool, increasing the replicas of a deployment, and upgrading the DOKS cluster to the latest version (1.34.1-do.5).
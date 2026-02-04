# Test1: 1 Node with 1 GPU (NVIDIA)

## Goals

- Understand the concepts of model and inference.

- Provision a GPU Droplet, run a vLLM container with Llama 3.2 1B, and test LLM inference using OpenAI-compatible APIs.

- Run vLLM benchmarks to evaluate performance.

## Concepts - AI Inference, Model and Inference Server

LLM inference can be viewed as y = f(x), where x is the input prompt, y is the generated tokens, and f is the model defined by both its architecture (the connections between parameters) and its learned parameters (often billions or more). 

An inference server takes an input x, applies the model f, and produces the output y.

There are many open-source LLM inference servers. [vLLM](https://docs.vllm.ai/en/latest/) is one of the most popular options and provides a ready-to-use container image. Alternatively, we can write low-level Python code to perform LLM inference directly.

## Provision a GPU Droplet using the Management Droplet

Check GPU availability in different regions using [this link](https://digitalocean.enterprise.slack.com/docs/T024FPVD5/F09KEUASL6B).

Provision a Droplet:

```
doctl compute droplet create rs-test1 \
 --image 203838782 \
 --region tor1 \
 --size gpu-6000adax1-48gb \
 --enable-monitoring \ 
 --ssh-keys <YOUR_SSH_KEY_ID>

# Slugs 

gpu-6000adax1-48gb
gpu-4000adax1-20gb 
gpu-l40sx1-48gb
gpu-h100x1-80gb

# Example

doctl compute droplet create rs-test1 \
 --image 203838782 \
 --region tor1 \
 --size gpu-6000adax1-48gb \
 --enable-monitoring \
 --ssh-keys 51482720

ID           Name        Public IPv4    Private IPv4    Public IPv6    Memory    VCPUs    Disk    Region    Image                        VPC UUID    Status    Tags    Features                    Volumes
549043682    rs-test1                                                  65536     8        500     tor1      Ubuntu NVIDIA AI/ML Ready                new               monitoring,droplet_agent    

doctl compute droplet list | grep rs-test1

549043682    rs-test1                                                     159.203.45.54      10.137.0.27                      65536      8        500     tor1      Ubuntu NVIDIA AI/ML Ready                                                          39c985c2-dc7f-11e8-b1a9-3cfdfea9ee58    active                                                                                                                 monitoring,droplet_agent,private_networking   
```

Access the newly created GPU droplet from your laptop using VS Code or a terminal, and perform a basic check:

```
cat /etc/os-release
uname -r

nvidia-smi

dcgmi --version
dcgm-exporter --version

systemctl status dcgm-exporter
curl http://127.0.0.1:5000/metrics

systemctl status do-agent 

docker version
docker ps
```

## Run vLLM Container on the GPU Droplet

We use [the official vllm image](https://hub.docker.com/r/vllm/vllm-openai) for NVIDIA GPUs and [the Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) for this test.

### Step 1: Run the vLLM server with the Llama 3.2 1B

```
TOKEN=<HUGGING_FACE TOKEN>
VOLUME=/root/.cache/huggingface
MODEL=meta-llama/Llama-3.2-1B-Instruct

docker run -it --rm --gpus all \
    -v $VOLUME:$VOLUME \
    -p 8000:8000 \
    -e HF_TOKEN=$TOKEN \
    vllm/vllm-openai:latest \
    --model $MODEL
```

### Step 2: Check the container and its image

```
root@rs-test1:~# docker ps
CONTAINER ID   IMAGE                     COMMAND                  CREATED          STATUS          PORTS                                         NAMES
8882b93c027c   vllm/vllm-openai:latest   "vllm serve --model …"   29 minutes ago   Up 29 minutes   0.0.0.0:8000->8000/tcp, [::]:8000->8000/tcp   strange_johnson
root@rs-test1:~# 
root@rs-test1:~# docker image ls
REPOSITORY         TAG       IMAGE ID       CREATED      SIZE
vllm/vllm-openai   latest    59c018a60d96   5 days ago   20.1GB
root@rs-test1:~# 
```

### Step 3: Check the location of the downloaded model on the host (or within the container) amd compare it with [the source](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/tree/main) on Hugging Face

The downloaded model includes the model weights, related metadata, and a tokenizer. The tokenizer converts text into numerical tokens that can be processed by the model.

Softlinks are used so that multiple model versions can share the same files without duplication.

```
root@rs-test1:~/.cache/huggingface# pwd
/root/.cache/huggingface
root@rs-test1:~/.cache/huggingface# 
root@rs-test1:~/.cache/huggingface# ls -ls
total 8
4 drwxr-xr-x 4 root root 4096 Feb  3 23:13 hub
root@rs-test1:~/.cache/huggingface# 
root@rs-test1:~/.cache/huggingface# du -sh hub
2.4G    hub
root@rs-test1:~/.cache/huggingface# 
root@rs-test1:~/.cache/huggingface# tree hub
hub
└── models--meta-llama--Llama-3.2-1B-Instruct
    ├── blobs
    │   ├── 02ee80b6196926a5ad790a004d9efd6ab1ba6542
    │   ├── 1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f
    │   ├── 3e3aaf51a035cb5092d9f6827a0dc074657ba88c
    │   ├── 4ff488a165e900e5129cda7c20ab32d568d2a475
    │   ├── 5cc5f00a5b203e90a27a3bd60d1ec393b07971e8
    │   └── 75ae08310d6d23df373ee2644b497192b3cce6d8
    ├── refs
    │   └── main
    └── snapshots
        └── 9213176726f574b556790deb65791e0c5aa438b6
            ├── config.json -> ../../blobs/3e3aaf51a035cb5092d9f6827a0dc074657ba88c
            ├── generation_config.json -> ../../blobs/75ae08310d6d23df373ee2644b497192b3cce6d8
            ├── model.safetensors -> ../../blobs/1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f
            ├── special_tokens_map.json -> ../../blobs/02ee80b6196926a5ad790a004d9efd6ab1ba6542
            ├── tokenizer.json -> ../../blobs/5cc5f00a5b203e90a27a3bd60d1ec393b07971e8
            └── tokenizer_config.json -> ../../blobs/4ff488a165e900e5129cda7c20ab32d568d2a475

5 directories, 13 files

root@rs-test1:~/.cache/huggingface# cat hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6/tokenizer.json 
```

### Step 4: Check the Droplet metrics and GPU metrics using the DO Console

### Step 5: Check the GPU Info on the host (or within the container)

VRAM usage for LLM inference includes both the model weights and the runtime cache, which depends on the number of sessions and the context length of each session.

```
root@rs-test1:~/.cache/huggingface# nvidia-smi
Tue Feb  3 23:30:28 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 575.57.08              Driver Version: 575.57.08      CUDA Version: 12.9     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX 6000 Ada Gene...    On  |   00000000:01:00.0 Off |                  Off |
| 30%   32C    P8             23W /  300W |   44578MiB /  49140MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           86711      C   VLLM::EngineCore                      44568MiB |
+-----------------------------------------------------------------------------------------+
root@rs-test1:~/.cache/huggingface# 
root@rs-test1:~/.cache/huggingface# watch -n 1 nvidia-smi
```

### Step 6: Test the inference on the host (or within the container) while monitoring the container's logs

Open a new terminal on you laptop and connect to the GPU Droplet:

```
root@rs-test1:~# docker ps
CONTAINER ID   IMAGE                     COMMAND                  CREATED          STATUS          PORTS                                         NAMES
8882b93c027c   vllm/vllm-openai:latest   "vllm serve --model …"   29 minutes ago   Up 29 minutes   0.0.0.0:8000->8000/tcp, [::]:8000->8000/tcp   strange_johnson
root@rs-test1:~# 
root@rs-test1:~# docker exec -it 888 /bin/bash

ps -ef

env | grep HF_TOKEN

# Non Streaming mode

curl http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "who are you?"}]
  }'

curl http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "How to learn AI/ML as a solution architect? Please answer my answer with 100+ words"}]
  }'

# Streaming mode

curl http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "How to learn AI/ML as a solution architect? Please answer my answer with 100+ words"}],
    "stream": true
  }'
```

### Step 7: Test the inference from the Management Droplet (or your laptop) while monitoring the container's logs

```
curl http://<GPU_DROPLET_PUBLIC_IP>:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "who are you?"}]
  }'

curl http://<GPU_DROPLET_PUBLIC_IP>:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "How to learn AI/ML as a solution architect? Please answer my answer with 100+ words"}]
  }'

curl http://<GPU_DROPLET_PUBLIC_IP>:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "How to learn AI/ML as a solution architect? Please answer my answer with 100+ words"}],
    "stream": true
  }'

# Examples

curl http://159.203.45.54:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "who are you?"}]
  }'

curl http://159.203.45.54:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "How to learn AI/ML as a solution architect? Please answer my answer with 100+ words"}]
  }'

curl http://159.203.45.54:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "How to learn AI/ML as a solution architect? Please answer my answer with 100+ words"}],
    "stream": true
  }'
```

## Run vLLM Benchmarker within the container on the GPU Droplet

The vLLM image comes with [the benchmark tool](https://docs.vllm.ai/en/latest/cli/bench/serve/), which we can run directly within the vLLM container:

While benchmaring, monitor the Droplet metrics and GPU metrics using the DO Console and running "watch -n 1 nvidia-smi" on the host (or within the container).

Pay attention to CPU usage, bandwidth, temperature, power consumption, and other relevant metrics.

```
# Run 1 inference

vllm bench serve --model meta-llama/Llama-3.2-1B-Instruct --dataset-name random \
--random-input-len 1000 --random-output-len 100 --request-rate inf \
--num-prompts 1 --base-url http://localhost:8000 \
--endpoint /v1/completions --backend openai

# Run 10 inference concurrently

vllm bench serve --model meta-llama/Llama-3.2-1B-Instruct --dataset-name random \
--random-input-len 1000 --random-output-len 100 --request-rate inf \
--num-prompts 10 --base-url http://localhost:8000 \
--endpoint /v1/completions --backend openai

# Run 1000 inference concurrently

vllm bench serve --model meta-llama/Llama-3.2-1B-Instruct --dataset-name random \
--random-input-len 1000 --random-output-len 100 --request-rate 10 \
--num-prompts 1000 --base-url http://localhost:8000 \
--endpoint /v1/completions --backend openai

# --request-rate inf, all requests are sent simultaneously; TTFT might be less meaningful in this scenario because many requests could be queued before being processed by the inference server.
```

## Run vLLM benchmarker on the Management Droplet

To simulate real-world scenarios, where client applications and inference servers may be deployed in different environments and network latency can affect performance, we can run the vLLM benchmarker on the Management Droplet.

On the Management Droplet, we run the same vLLM image without requiring GPUs. We still need the HF token to download the model’s tokenizer (which is very small) which will be used by the vLLM benchmarker, but not the model weights.

### Step 1: Run the vLLM container while overriding the entrypoint on the Management Droplet

```
TOKEN=<HUGGING_FACE TOKEN>
VOLUME=/root/.cache/huggingface

docker run -it --rm \
    --entrypoint /bin/bash \
    -v $VOLUME:$VOLUME \
    -e HF_TOKEN=$TOKEN \
    vllm/vllm-openai:latest
```

### Step 2: Run vLLM Benchmarker within the container

```
vllm bench serve --model meta-llama/Llama-3.2-1B-Instruct --dataset-name random \
--random-input-len 1000 --random-output-len 100 --request-rate inf \
--num-prompts 10 --base-url http://<GPU_DROPLET_PUBLIC_IP>:8000 \
--endpoint /v1/completions --backend openai

# Example

vllm bench serve --model meta-llama/Llama-3.2-1B-Instruct --dataset-name random \
--random-input-len 1000 --random-output-len 100 --request-rate inf \
--num-prompts 10 --base-url http://159.203.45.54:8000 \
--endpoint /v1/completions --backend openai
```

### Step 3: Check the downloaded tokenizer on the host (or within the container)

```
root@8e17ddff6101:/vllm-workspace# cd ~/.cache/huggingface/hub/
root@8e17ddff6101:~/.cache/huggingface/hub# 
root@8e17ddff6101:~/.cache/huggingface/hub# pwd
/root/.cache/huggingface/hub
root@8e17ddff6101:~/.cache/huggingface/hub# 
root@8e17ddff6101:~/.cache/huggingface/hub# ls
models--meta-llama--Llama-3.2-1B-Instruct
root@8e17ddff6101:~/.cache/huggingface/hub# 
root@8e17ddff6101:~/.cache/huggingface/hub# du -sh models--meta-llama--Llama-3.2-1B-Instruct/
8.8M    models--meta-llama--Llama-3.2-1B-Instruct/
root@8e17ddff6101:~/.cache/huggingface/hub# 
root@8e17ddff6101:~/.cache/huggingface/hub# 
```

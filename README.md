# awesome-huggingface
Repository containing awesome resources regarding Hugging Face tooling.

Note: I haven't made most these resources, I just want to centralize them so that people can easily find them.

# Fine-tuning open-source LLMs

Fine-tuning an LLM typically involves 2 steps:

* supervised fine-tuning (SFT), also called instruction tuning.
* human preference fine-tuning.

## General guides

### Alignment Handbook

Hugging Face maintains the Alignment Handbook, which contains scripts to fine-tune any decoder-only LLM for supervised fine-tuning (SFT) and direct preference optimization (DPO): https://github.com/huggingface/alignment-handbook. The scripts support both full fine-tuning and QLora.

## Model-specific guides

### LLaMa-2

LLaMa-2 is an open LLM by Meta. It improves upon LLaMa v1.

Fine-tune LLaMa-2 on your own data using QLoRa: https://github.com/brevdev/notebooks/blob/main/llama2-finetune-own-data.ipynb.

Fine-tune LLaMa-2 in Google Colab: https://github.com/mlabonne/llm-course/blob/main/Fine_tune_Llama_2_in_Google_Colab.ipynb.

Fine-tune LLaMa-2 with DPO: https://huggingface.co/blog/dpo-trl.

### Mistral-7B

Mistral-7B is the first open-source LLM by Mistral.ai. It improves upon LLaMa-2.

Fine-tune Mistral-7B: https://blog.neuralwork.ai/an-llm-fine-tuning-cookbook-with-mistral-7b/.

Fine-tune Mistral-7B on 3090s, a100s, h100s: https://github.com/abacaj/fine-tune-mistral.

Fine-tune Mistral-7B on your own data using QLoRa: https://github.com/brevdev/notebooks/blob/main/mistral-finetune-own-data.ipynb.

Fine-tune Mistral-7B with DPO (direct preference optimization): https://towardsdatascience.com/fine-tune-a-mistral-7b-model-with-direct-preference-optimization-708042745aac.

### Mixtral-8x7B

Mistral-8x7B is the second open-source LLM by Mistral.ai. It improves upon Mistral-7B by incorporating a mixture-of-experts (MoE) architecture.

Fine-tune Mixtral on your own data using QLoRa: https://github.com/brevdev/notebooks/blob/main/mixtral-finetune-own-data.ipynb.

# Fine-tuning multimodal LLMs

### BLIP-2

BLIP-2 is a multimodal LLM by Salesforce, leveraging frozen image encoders (EVA-CLIP) and frozen large language models (OPT, Flan-T5).

Fine-tune BLIP-2 using PEFT: https://colab.research.google.com/drive/16XbIysCzgpAld7Kd9-xz-23VPWmqdWmW?usp=sharing#scrollTo=6cCVhsmJxxjH.

### LLaVa-1.5

LLaVa is a multimodal LLM by Microsoft.

Deploying LLaVa-1.5 on AWS: https://dev.to/denisyay/create-a-visual-chatbot-on-aws-ec2-with-llava-15-transformers-and-runhouse-hm1

# Deploying open-source LLMs

For deploying open-source models, there are a few options.

## vLLM

vLLM is a framework for setting up an inference server of open-source large language models. It also supports an OpenAI-compatible server. This allows to use the [OpenAI Python library](https://github.com/openai/openai-python) to interact with the model.

One typically sets up a virtual machine (VM) on a cloud of choice which hosts the vLLM server.

- deploying vLLM on Google Cloud (Vertex AI endpoints): https://cloud.google.com/blog/products/ai-machine-learning/serve-open-source-llms-on-google-cloud
- deploying vLLM and FastAPI on AWS: https://medium.com/@chinmayd49/self-host-llm-with-ec2-vllm-langchain-fastapi-llm-cache-and-huggingface-model-7a2efa2dcdab

## TGI (text-generation-inference)

TGI is similar to vLLM in that in provides an inference server for open-source large language models.

## Inference Endpoints

Any Hugging Face model can be deployed using Inference Endpoints: https://huggingface.co/docs/inference-endpoints/index. This automatically spins up an endpoint for you, on a cloud provider of choice.
# nanochat

![nanochat logo](dev/nanochat.png)
![scaling laws](dev/scaling_laws_jan26.png)

nanochat is the simplest experimental harness for training LLMs. It is designed to run on a single GPU node, the code is minimal/hackable, and it covers all major LLM stages including tokenization, pretraining, finetuning, evaluation, inference, and a chat UI. For example, you can train your own GPT-2 capability LLM (which cost ~$43,000 to train in 2019) for only $72 (~3 hours of 8XH100 GPU node) and then talk to it in a familiar ChatGPT-like web UI. On a spot instance, the total cost can be closer to ~$20. More generally, nanochat is configured out of the box to train an entire miniseries of compute-optimal models by setting one single complexity dial: `--depth`, the number of layers in the GPT transformer model (GPT-2 capability happens to be approximately depth 26). All other hyperparameters (the width of the transformer, number of heads, learning rate adjustments, training horizons, weight decays, ...) are calculated automatically in an optimal way.

For questions about the repo, I recommend either using [DeepWiki](https://deepwiki.com/karpathy/nanochat) from Devin/Cognition to ask questions about the repo, or use the [Discussions tab](https://github.com/karpathy/nanochat/discussions), or come by the [#nanochat](https://discord.com/channels/1020383067459821711/1427295580895314031) channel on Discord.

## Time-to-GPT-2 Leaderboard

Presently, the main focus of development is on tuning the pretraining stage, which takes the most amount of compute. Inspired by the modded-nanogpt repo and to incentivise progress and community collaboration, nanochat maintains a leaderboard for a "GPT-2 speedrun", which is the wall-clock time required to train a nanochat model to GPT-2 grade capability, as measured by the DCLM CORE score. The [runs/speedrun.sh](runs/speedrun.sh) script always reflects the reference way to train GPT-2 grade model and talk to it. The current leaderboard looks as follows:

| # | time | val_bpb | CORE | Description | Date | Commit | Contributors |
|---|-------------|---------|------|-------------|------|--------|--------------|
| 0 | 168 hours | - | 0.2565 | Original OpenAI GPT-2 checkpoint | 2019 | - | OpenAI |
| 1 | 3.04 | 0.74833 | 0.2585 | d24 baseline, slightly overtrained | Jan 29 2026 | 348fbb3 | @karpathy |
| 2 | 2.91 | 0.74504 | 0.2578 | d26 slightly undertrained **+fp8** | Feb 2 2026 | a67eba3 | @karpathy |
| 3 | 2.76 | 0.74645 | 0.2602 | bump total batch size to 1M tokens | Feb 5 2026 | 2c062aa | @karpathy |

The primary metric we care about is "time to GPT-2" - the wall clock time needed to outperform the GPT-2 (1.6B) CORE metric on an 8XH100 GPU node. The GPT-2 CORE score is 0.256525. In 2019, the training of GPT-2 cost approximately $43,000 so it is incredible that due to many advances over 7 years across the stack, we can now do so much faster and for well below $100 (e.g. at the current ~$3/GPU/hr, an 8XH100 node is ~$24/hr, so 3 hours is ~$72).

See [dev/LEADERBOARD.md](dev/LEADERBOARD.md) for more docs on how to interpret and contribute to the leaderboard.

## Getting started

### Reproduce and talk to GPT-2

The most fun you can have is to train your own GPT-2 and talk to it. The entire pipeline to do so is contained in the single file [runs/speedrun.sh](runs/speedrun.sh), which is designed to be run on an 8XH100 GPU node. Boot up a new 8XH100 GPU box from your favorite provider (e.g. I use and like [Lambda](https://lambda.ai/service/gpu-cloud)), and kick off the training script:

```bash
bash runs/speedrun.sh
```

You may wish to do so in a screen session as this will take ~3 hours to run. Once it's done, you can talk to it via the ChatGPT-like web UI. Make sure again that your local uv virtual environment is active (run `source .venv/bin/activate`), and serve it:

```bash
python -m scripts.chat_web
```

And then visit the URL shown. Make sure to access it correctly, e.g. on Lambda use the public IP of the node you're on, followed by the port, so for example [http://209.20.xxx.xxx:8000/](http://209.20.xxx.xxx:8000/), etc. Then talk to your LLM as you'd normally talk to ChatGPT! Get it to write stories or poems. Ask it to tell you who you are to see a hallucination. Ask it why the sky is blue. Or why it's green. The speedrun is a 4e19 FLOPs capability model so it's a bit like talking to a kindergartener :).

---

<img width="2672" height="1520" alt="image" src="https://github.com/user-attachments/assets/ed39ddf8-2370-437a-bedc-0f39781e76b5" />

---

## Notes

- Works on Ampere 8XA100 GPU nodes, slightly slower than H100 nodes.

- Can run on a single GPU by omitting torchrun (training will take ~8x longer).

- For GPUs <80GB VRAM, tune --device_batch_size in scripts to avoid OOM errors.

- Code is vanilla PyTorch and should run on any compatible device (xpu, mps, etc.).

## Research

For experimentation, check runs/scaling_laws.sh and runs/miniseries.sh
Quick pretraining (~5 min) can be done with smaller models:

```
OMP_NUM_THREADS=1 torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=12 \
    --run="d12" \
    --model-tag="d12" \
    --core-metric-every=999999 \
    --sample-every=-1 \
    --save-every=-1 \
```

This uses wandb (run name "d12"), only evaluates CORE at the last step, and skips intermediate checkpoint saves. You can iterate by changing parameters and monitoring plots for:

1. val_bpb (validation loss)

2. core_metric (CORE score)

3. VRAM utilization, train/mfu (FLOPS), train/tok_per_sec (throughput)

## Running on CPU / MPS

runs/runcpu.sh shows an example on CPU or Apple Silicon. Models are shrunk for fast training (~10–20 minutes).

## Guides

Some helpful guides:

- Beating GPT-2 for <$100: the nanochat journey

- Miniseries v1

- Adding new abilities: Counting R in strawberry

- Customizing personality through synthetic data: Infusing identity



## File structure

```
.
├── README.md
├── dev
│   ├── gen_synthetic_data.py       
│   ├── generate_logo.html
│   ├── nanochat.png
│   └── repackage_data_reference.py 
├── nanochat
│   ├── __init__.py                 
│   ├── checkpoint_manager.py       
│   ├── common.py                  
│   ├── core_eval.py               
│   ├── dataloader.py               
│   ├── dataset.py                 
│   ├── engine.py                   
│   ├── execution.py   
│   ├── gpt.py              
│   ├── logo.svg
│   ├── loss_eval.py               
│   ├── optim.py                    
│   ├── report.py                   
│   ├── tokenizer.py               
│   └── ui.html                     
├── pyproject.toml
├── runs
│   ├── miniseries.sh               
│   ├── runcpu.sh                  
│   ├── scaling_laws.sh             
│   └── speedrun.sh                 
├── scripts
│   ├── base_eval.py               
│   ├── base_train.py               
│   ├── chat_cli.py                
│   ├── chat_eval.py                
│   ├── chat_rl.py                  
│   ├── chat_sft.py               
│   ├── chat_web.py                 
│   ├── tok_eval.py         
│   └── tok_train.py                
├── tasks
│   ├── arc.py                      
│   ├── common.py                   
│   ├── customjson.py              
│   ├── gsm8k.py                   
│   ├── humaneval.py                
│   ├── mmlu.py                     
│   ├── smoltalk.py                
│   └── spellingbee.py              
├── tests
│   └── test_engine.py
└── uv.lock
```

## Contributing

The goal of nanochat is to provide a minimal, end-to-end LLM framework that runs on budgets <$1000. It’s designed to be readable, hackable, and fully forkable, with a strong baseline. Improving pretraining and CORE score latency is the main focus.

## Acknowledgements

- Thank you to HuggingFace for datasets.
- Thank you to Lambda for compute resources.
- Thank you to contributors and community members for guidance and support.




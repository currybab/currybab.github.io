+++
title = 'running nanochat speedrun with only one rtx 5090'
date = 2025-10-15T11:41:14+09:00
tags = ['nanochat', 'speedrun']
categories = ['nanochat']
+++

Yesterday, Andrej Karpathy posted a [nanochat](https://github.com/karpathy/nanochat).
It's about building a chatbot with small amount of resources and newest architecture.

I want to study all of the code, but in weekday, I have to work, so I don't have enough time to study. Also I have another resources to study yet.
I found speedrun.sh in nanochat repository. So, I decided to run it.

I have only 1 RTX 5090, so I can't run it with 8 GPUs. I should change some arguments. 
```
nproc_per_node=1
device_batch_size=8 (for pre train & mid train)
device_batch_size=2 (for sft)
device_batch_size=1 (for rl)
```

It's running now, about 1 day passed. My experiment is logging to [wandb](https://wandb.ai/currybab/nanochat/workspace?nw=nwusercurrybab).
I will update this post when it's finished. That would be great.

- About 71 hours passed, pre-train & mid-train is finished. 
- About 46 minutes passed, sft is finished. 


### wandb log for training

- [pre-train](https://wandb.ai/currybab/nanochat?nw=nwusercurrybab)
- [mid-train](https://wandb.ai/currybab/nanochat-mid?nw=nwusercurrybab)
- [sft](https://wandb.ai/currybab/nanochat-sft?nw=nwusercurrybab)
- [rl](https://wandb.ai/currybab/nanochat-rl?nw=nwusercurrybab)

### result

| Metric          | BASE     | MID      | SFT      | RL       |
|-----------------|----------|----------|----------|----------|
| CORE            | 0.2108   | -        | -        | -        |
| ARC-Challenge   | -        | 0.2713   | 0.2713   | -        |
| ARC-Easy        | -        | 0.2803   | 0.3178   | -        |
| GSM8K           | -        | 0.0243   | 0.0379   | 0.0652   |
| HumanEval       | -        | 0.0488   | 0.0549   | -        |
| MMLU            | -        | 0.2898   | 0.2837   | -        |
| ChatCORE        | -        | 0.0390   | 0.0513   | -        |


- [modified seedrun.sh](https://github.com/currybab/nanochat/blob/working/speedrun.sh)
- [report.md](https://github.com/currybab/nanochat/blob/working/report.md)
- My result is little bad against [Karpathy's result](https://github.com/karpathy/nanochat/discussions/1#discussion-9022446).


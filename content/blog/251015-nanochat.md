+++
title = 'running nanochat speedrun with rtx 5090'
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
device_batch_size=8
```

It's running now, about 1 day passed. My experiment is logging to [wandb](https://wandb.ai/currybab/nanochat/workspace?nw=nwusercurrybab).
I will update this post when it's finished. That would be great.

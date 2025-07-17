+++
title = "2507W3 Recently Read"
date = "2025-07-17T11:39:16+09:00"

#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = []
+++

- [To be a better programmer, write little proofs in your head](https://the-nerve-blog.ghost.io/to-be-a-better-programmer-write-little-proofs-in-your-head/)
- [Recsys Keynote: Improving Recommendation Systems & Search in the Age of LLMs - Eugene Yan](https://www.youtube.com/watch?v=2vlCqD6igVA)
    - Good introductory review for LLM-based recommendation systems, I summarized the key points in the video.
    - LLMs for Recsys
        - Word2Vec(2013): Item embedding from co-occurrence in user interaction sequences
        - GRU4Rec(2015): Short-term, next-item indent from user interaction sequences
        - SASRec(2018), BERT4Rec(2019): Attention on long-range depedencies in user interaction sequences
        - LLMs(2024): Semantic IDs, Data Augementation, Unified Models
    - Challenge 1: Hash based item IDs don't encode item content; thus struggle with cold-start and sparsity
        - Problem: Help users discover new items, faster (@Tiktok)
        - Solution: Trainable, multimodal, semantic IDs
        - Benefits: Address cold-start, Recs that understand content
    - Challenge 2: High-quality metadata is essential for search (and RecSys) but costly and high effort to get
        - General Solution -> using LLMs for synthetic data & labels
        - Problem 1: Poor user experience and lost trust due to low-quality job recommendation(@Indeed)
        - Solution 1: Lightweight classifier to filter bad recs
        - Problem 2: Help users search for new items (podcasts, audiobooks) in catalog of known items (song, artist) (@Spotify)
        - Solution 2: Query recommendation system
        - Benefits: Richer, higher-quality data at larger scale, Far lower cost and effort than human annotation
    - Challenge 3: Task-specific models duplicate engineering, increase maintenance cost, and don't benefit from transfer learning
        - General Solution -> Unified models(it works for vision and language, so why not recsys? Even work for payments and fraud at Stripe, too!)
        - Problem 1: Teams deal with complexity from bespoke models for search, similar item recs, pre-query recs (@Netflix)
        - Solution 1: Unified ranker for all of the above
        - Problem 2: Help users get better results with highly specific or broad queries, on ever-changing inventory (@Etsy)
        - Solution 2: Unified Embedding + retrieval
        - Benefits: Simplifies systems, reduced maintenance overhead, transfer learning(but there may be the alignment tax), Gains to unified model benefit other use cases
    - Three takeawys
        - Semantic IDs: Better performance, address cold-start
        - Data Augmentation: Enrich data, address sparsity
        - Unified Models: Simplify systems, address fragmentation
- [360Brew: LLM-based Personalized Ranking and Recommendation - Hamed and Maziar, LinkedIn AI](https://www.youtube.com/watch?v=U0S6CfzAY5c)
    - 
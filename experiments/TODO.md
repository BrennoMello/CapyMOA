# TODO: 
    # - Implement experiments mini-imagenet dataset
    # - Implement FLOPs calculation during training and evaluation
    # - Implement additional baselines: ER-ACE, DER++, SER, CLS-ER, AGEM
    #- Implement RAR with random augmentation
    # - Implement additional strategies suggested by reviewers: OCM [B], GSA [C], MOSE [D], and CCLDC [E], IDER [F], SCR [G], Myopia [H]
    # [A] Csaba, Botos, et al. "Label delay in online continual learning." Advances in Neural Information Processing Systems 37 (2024): 119976-120012.
    # [B] Guo, Yiduo, Bing Liu, and Dongyan Zhao. "Online continual learning through mutual information maximization." International Conference on Machine Learning. PMLR, 2022.
    # [C] Guo, Yiduo, Bing Liu, and Dongyan Zhao. "Dealing with cross-task class discrimination in online continual learning." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
    # [D] Yan, Hongwei, et al. "Orchestrate latent expertise: Advancing online continual learning with multi-level supervision and reverse self-distillation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.
    # [E] Wang, Maorong, et al. "Improving plasticity in online continual learning via collaborative learning." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.
    # [F] LEARNING, ABLE CONTINUAL. "IDER: IDEMPOTENT EXPERIENCE REPLAY FOR RELI-ABLE CONTINUAL LEARNING."
    # [G] Zheda Mai, Ruiwen Li, Hyunwoo Kim, and Scott Sanner. Supervised contrastive replay: Revisiting the nearest class mean classifier in online class-incremental continual learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3589–3599, 2021.
    # [H] Wang, Xinrui, et al. "Forgetting, ignorance or myopia: Revisiting key challenges in online continual learning." Advances in Neural Information Processing Systems 37 (2024): 58341-58375.
    # - Implement additional metrics: forgetting, backward transfer, forward transfer
    # - Implement ViT backbone model
    # - Implement batches available variation after delay 
    # - Implement command line interface for running experiments with different configurations
    
    #TODO: ER with additional loss based of ER-ACE

# TODO: Experiments

    #- Change range of no delay prob
    #- Change range of delay batches number
    #- Change range of buffer size

    #- Add anytime accuracy seen
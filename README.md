
##Preprocessing
Similar to the previous competition solution, but simpler. Every landmark and xyz was used and standardized. Flip left-handed signer(rather than augmentation). MAX_LEN=768 was used. Any hand-crafted feature was not significant, likely due to the absence of complex relations between movements in frames.

##Model
Encoder is the same as the previous competition (Stacked Conv1DBlock + TransformerBlock) but increased in size (expand ratio 2->4 in Conv1DBlock) and depth (8 layers -> 17 layers). Single model has ~6.5M parameters. I applied padding='same'(rather than 'causal') and output stride=2 (requires slightly more logic for handling masking). In my case, mixing in Transformer blocks wasn't as effective as in the previous competition. Perhaps global features were less crucial in this comp. Additionally, I added one BN to input of the Conv1DBlock for more training stability(especially with awp + more epochs).

CTC decoder used a single GRU layer followed by one FC layer. Attention Decoder used a single-layer Transformer decoder. Introducing augmentation to the Decoder input and adding up to 4 Decoder layers improves the performance of the attention decoder (up to +0.004). However, considering the number of parameters and inference speed, I considered it inefficient and thus used a single-layer decoder.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F5003978%2F7d1f7aee69242af5e6aa5ef19c9460f6%2Fmodeldesign.drawio-2.svg?generation=1692993585923781&alt=media)
##Augmentations
* Random resample (0.5x ~ 1.5x to original length)
* Random Affine
* Random Cutout
* Random token replacement on decoder input(prob=0.2)

What I overlooked this time is that augmentations which showed no performance improvement or even degraded performance in shorter epochs might actually help improve performance in longer epochs. I experienced a similar phenomenon in the previous competition, but it seemed more pronounced in this one. By selecting augmentations solely based on the results from a 60epoch experiment, I think I missed out on many potentially beneficial augmentations when testing the 400epoch training in the final week.


##Training
Epoch = 400
bs = 16 * num_replicas = 128
Lr = 5e-4 * num_replicas = 4e-3
AWP = 0.2 starts at 0.1 * Epoch
Schedule = CosineDecay with warmup ratio 0.1
Optimizer = AdamW (slightly better than RAdam with Lookahead)
Loss = CTC(weight=0.25) + CCE with label smoothing=0.1~0.25(weight=0.75)

Training takes around 14 hours with colab TPUv2-8(as colab TPU runtime recently reduced to 3~4 hours, needed 4 consecutive sessions to complete training).
Longer Epoch always gave better CV(5fold split by id) and LB but got no time to try over 400 epochs.

##LB history
|  | public LB | private LB | 
| --- | --- | --- |
| prevcompsinglemodel + CTC (or Attention) | 0.76 | 0.74 |
| + deeper and wider model, add pose | 0.79 | 0.78 |
| + 3 seed ensemble with Attention | 0.80 | 0.79 |
| + ctc attention joint decoding | 0.81 | 0.80 | 
| + use all landmarks, longer epoch | 0.82 | 0.81 |



| | troughput (iterations/s) | latency (ms) | model size (Mb)| Public LB | Private LB |
| --- | --- | --- | --- | --- | --- |
| CTCGreedy | 20.14 | 49.66 | 11.41 | 0.815 | 0.807 |
| ATTGreedy | 10.16 | 98.42 | 11.77 | 0.816 | 0.808 |
| CTCATTJointGreedy | 5.26 | 190.22 | 12.49 | 0.820 | 0.812 |
| CTCATTJointGreedy -2xseed | 2.71 | 368.95 | 24.90 | 0.825 | 0.817 |
| CTCATTJointGreedy -3xseed | 1.75 | 570.77 | 37.30 | 0.825 | 0.819 |

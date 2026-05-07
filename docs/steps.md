What To Do Next

  Immediate (low-hanging fruit)

  1. Fix LoRA hyperparameters for RoBERTa/MPNet                                                                                                                      
  The LoRA experiments didn't fail because the idea is bad — they failed because lr=2e-4 + 20 epochs is a poor match. Try:
  - lr=5e-5 or 1e-5 with longer warmup (500 steps)                                                                                                                   
  - More epochs (30-50)                                                                                                                                              
  - Smaller LoRA rank (r=8) to reduce overfitting risk                                                                                                               
                                                                                                                                                                     
  2. Run evaluation on held-out test split                                                                                                                           
  You have a test split set up. Run scripts/evaluate.py on the best LUAR LoRA checkpoint to get final PAN metrics (AUC, c@1, F0.5u, EER) — these are the numbers that
   matter for reporting.                                                                                                                                             
    
  3. Increase number of training senders                                                                                                                             
  You used 100 senders. The Enron dataset has ~150+ usable senders. More senders = more diverse contrastive negatives = likely better generalization.
                                                                                                                                                                     
  Medium-term (next experiments)
                                                                                                                                                                     
  4. Cross-domain transfer to PAN benchmarks                                                                                                                         
  Take your best LUAR LoRA checkpoint and evaluate zero-shot on PAN 2020/2021 authorship verification fanfiction/news datasets. If LUAR's stylometric representations
   transfer cross-domain, that's a strong paper result.                                                                                                              
    
  5. Hard negative mining with phishing / LLM-impersonation data                                                                                                     
  Your current negatives are random other Enron senders. The actual adversarial case is an LLM mimicking a sender's style. Add synthetically impersonated emails as
  hard negatives to the contrastive batches — this directly trains the model to distinguish real vs. LLM-mimicked style.                                             
    
  6. Mahalanobis scoring                                                                                                                                             
  The prototypical head uses cosine distance from a centroid. Switch to Mahalanobis distance (with Ledoit-Wolf covariance shrinkage as the README notes) — this
  accounts for the shape of the embedding cluster, not just distance from center, and is known to improve anomaly detection significantly.                           
    
  Longer-term                                                                                                                                                        

  7. Per-sender threshold calibration
  Some senders write more consistently than others. A single global threshold for anomaly detection will have poor precision. Calibrate a per-sender threshold during
   the profile-building phase using held-out emails.                                                                                                                 
    
  8. Ensemble LUAR embeddings with lexical stylometric features                                                                                                      
  Character n-gram frequencies, function word distributions, punctuation patterns — these classical stylometric features are orthogonal to neural representations. A
  late-fusion ensemble could push AUROC further and be more robust to adversarial paraphrasing.      
# Perona-SURF-Task

Results:

For the CLIP model, I had a final train accuracy of 97.16779% and final test accuracy of 95.165%. 

For the LLaVA model, I had an accuracy of 31.667%, with 95 total correct responses out of 300 samples. This was a limited sample (instead of the 1303 images), because the Google Colab script was taking a very long time to run (~1.5 hours). While this is a potential source of error, I was still able to receive a good measure of the sample accuracy. 

In conclusion, the CLIP model had a much higher accuracy, which was to be expected in this task. While I trained a small neural network on the Caltech101 dataset along with the CLIP model, LLaVA had never been trained on the dataset. The LLaVA model was able to describe images well, it did not perform as well for specific classification tasks. The CLIP model was finetuned on the dataset and did significantly better. The results demonstrate that for specialized classification tasks, a smaller, trained model is more effective than a large, general-purpose one.

BONUS:

The PCA plots are in the images folder. The results make sense because as the layers progress, we see that the two clusters become more and more differentiated and separable. 

The original dimensionality of the embeddings is 4096. Each layer outputs a vector of this size. 

The LLaVA embeddings were acquired at the last sequence index because the very last token could see all of the sequence before it; the LLaVA model processes the tokens sequentially, so to find all the embeddings, we can index by -1. 

Finally, I interpreted the difference between the distinct layers as the stages of processing. The early layers start feature mixing and assessing which tokens are important and begins to comprehend the image features with the text prompt. However, the deep layers represent the refining stage with separate regions of the latent vector space to have greater ease in predicting the next word.
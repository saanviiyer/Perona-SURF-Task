# Perona-SURF-Task

Results:

For the CLIP model, I had a final train accuracy of 97.16779% and final test accuracy of 95.165%. 

For the LLaVA model, I had an accuracy of 31.667%, with 95 total correct responses out of 300 samples. This was a limited sample (instead of the 1303 images), because the Google Colab script was taking a very long time to run (~1.5 hours). While this is a potential source of error, I was still able to receive a good measure of the sample accuracy. 

In conclusion, the CLIP model had a much higher accuracy, which was to be expected in this task. While I trained a small neural network on the Caltech101 dataset along with the CLIP model, LLaVA had never been trained on the dataset. The LLaVA model was able to describe images well, it did not perform as well for specific classification tasks. The CLIP model was finetuned on the dataset and did significantly better. The results demonstrate that for specialized classification tasks, a smaller, trained model is more effective than a large, general-purpose one.


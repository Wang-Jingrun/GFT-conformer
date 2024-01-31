# **Learnable Graph Ratio Mask Based Speech Enhancement in the Graph Frequency Domain**

## Abstract:
Deep neural network (DNN) based speech enhancement methods in the time-frequency domain generally work on the complex-valued time-frequency representations obtained through short-time Fourier transform (STFT). With the fine-detailed structures of noisy speech in terms of complex-valued spectrogram as the input, both the magnitude and phase of clean speech can be estimated by DNNs. Such methods mainly employ complex neural networks to deal with complex-valued inputs directly or two-path networks to deal with real and imaginary parts separately, resulting in high computational complexity and a large number of model parameters. To address this problem, this paper proposes a DNN-based speech enhancement method in the graph frequency domain by utilizing the theory of graph signal processing (GSP) to obtain real-valued inputs instead of complex-valued inputs. Specifically, a novel real symmetric adjacency matrix is defined based on the positional relationships among the samples of speech signals so that the speech signals are represented as undirected graph signals. Through eigenvalue decomposition of the real symmetric adjacency matrix, the graph Fourier transform (GFT) basis is obtained and then utilized to extract the real-valued features of the speech graph signals in the graph frequency domain. Since the GFT basis is closely related to the adjacency matrix, these real-valued features in the graph frequency domain implicitly exploit the relationships among speech samples. Furthermore, by combining the convolution-augmented transformer (conformer) and the convolutional recurrent network (CRN), this paper constructs the GFT-conformer model, which is an essentially convolutional encoder-decoder (CED) with four two-stage conformer blocks (TS-conformers) to capture both local and global dependencies of the features in both the time and graph-frequency dimensions, for estimating the targets based on masking to achieve better speech enhancement. Moreover, considering the differences in characteristics between speech and noise across various graph frequency components, this paper introduces the learnable graph ratio mask (LGRM), which allows separate control over the mask ranges for different graph frequency components, enabling fine-grained denoising of various graph frequency components to further improve the speech enhancement performance of the GFT-conformer. We evaluate the performance of the proposed GFT-conformer with LGRM on the Voice Bank+DEMAND dataset and Deep Xi dataset in terms of five commonly used metrics. Experimental results show that the proposed GFT-conformer with LGRM achieves a better performance with the smallest model size of 1.4M parameters, as compared to several other state-of-the-art DNN-based time-domain and time-frequency domain methods.  

## Datasets:

### Voice Bank + DEMAND:
Download VCTK-DEMAND dataset with 16 kHz, change the dataset dir:
```
-VCTK-DEMAND/
  -train/
    -noisy/
    -clean/
  -test/
    -noisy/
    -clean/
```
 https://datashare.ed.ac.uk/handle/10283/2791

### Deep Xi:

https://ieee-dataport.org/open-access/deep-xi-dataset

**Note that the sample rate for all wavs needs to be converted to 16000.**


## How to use:

### training:
The config parameters can be modified in config.yaml.

### evaluation:

## Model and Comparison:

## citation:
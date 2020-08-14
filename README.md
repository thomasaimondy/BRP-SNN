# BRP-SNN
The code for tuning Spiking Neural Network based on Biologically-plausible Reward Propagation

Spiking Neural Networks (SNNs) contain more biology-realistic structures and biology-inspired learning principles compared with that in standard Artificial Neural Networks (ANNs). SNNs were considered as the third generation of ANNs, powerful on robust computation with low computation cost. The dynamic neurons in SNNs are non-differential, containing decayed historical states and generating event-based spikes after their states reaching the threshold. This dynamic characteristic of SNN made it hard to be directly trained with standard back propagation (BP) which is also considered not biologically plausible. In this paper, a Biologically-plausible Reward Propagation (BRP) algorithm is proposed and applied on a deep SNN architecture with both spiking-convolution (with both 1D and 2D kernels) and full-connection layers. Different with standard BP that propagated the error signals from post to pre synaptic neurons layer by layer, the BRP propagated the target labels instead of target errors directly from the output layer to all of the pre hidden layers. This effort was  more consistent with the top-down reward-guiding learning in cortical column of neocortex. Then the synaptic modifications with only local gradient difference were induced, and during this procedure, a pseudo-BP was used for the gradient propagation. The performance of the proposed BRP-SNN was further verified on  spatial (including MNIST and Cifar-10) and temporal (including TIDigits and DvsGesture) tasks, with shallow and deep architectures. The experimental result showned that the BRP played roles on convergent learning of SNN, and also reached comparable accuracy and lower computational cost compared with other state-of-the-art SNNs tuned with biologically-plausible algorithms. We think the introduce of biologically-plausible learning rules to the training procedure of biologically-realistic SNNs might give us more hints and inspirations towards better understanding of the intelligent nature of the biological system.

The author of this work is : Tielin Zhang, Shuncheng Jia, Bo Xu.


# Important announcement

This code is refined from the source code of "Direct Random Target Projection (DRTP) - PyTorch-based implementation";

The source code of DRTP is with Apache LICENSE-2.0.

The github link of DRTP is :https://github.com/ChFrenkel/DirectRandomTargetProjection


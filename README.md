# YOAO
You only attack once (Single-step DeepFool adversarial attack algorithm)
The following code provides the implementation of both the DeepFool algorithm and the innovative YOAO (You Only Attack Once) algorithm. These algorithms are designed to be tested with the CIFAR-10 dataset, a widely recognized collection of images commonly used in machine learning for object recognition tasks. By default, the code is set to work with a ResNet classifier, which is a deep neural network known for its accuracy and efficiency in image classification.

To utilize this code for your own experiments or to tailor it to your specific needs, you simply need to replace the "RESNET_cifar10_3.pth" with the path to the weights file of your trained model. This allows you to integrate the code with your own dataset and model, leveraging the power of DeepFool and YOAO algorithms for generating adversarial examples that can challenge and enhance the robustness of your deep learning models.

Alternatively, if you do not have a custom-trained model, you can use publicly available weight files. Many open-source repositories and datasets offer pre-trained models that can be directly applied to this code, enabling you to explore the capabilities of these algorithms without the need for extensive training procedures.

Whether you're a researcher looking to test the vulnerability of a new model or a developer seeking to improve the security of an existing system, this code serves as a powerful tool. It not only demonstrates the effectiveness of the DeepFool and YOAO algorithms but also provides a foundation for further exploration and development in the field of adversarial machine learning.

So, embark on this journey of enhancing your models' resilience against adversarial attacks by replacing the weight file and witnessing the algorithms in action. It's a step towards building more robust and secure AI systems that can stand up to the sophisticated challenges of today's digital landscape.

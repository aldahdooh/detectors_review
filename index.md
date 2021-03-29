### Authors

<p align="center">
  <table>
      <tr>
        <td>Ahmed Aldahdooh</td>
        <td>Wassim Hamidouche</td>
        <td>Olivier Deforges</td>
      </tr>
      <tr>
        <td colspan="3">Univ Rennes, INSA Rennes, CNRS, IETR - UMR 6164, F-35000 Rennes, France</td>
      </tr>
      <tr>
        <td colspan="3"><a href = "mailto:ahmed.aldahdooh@insa-rennes.fr">ahmed.aldahdooh@insa-rennes.fr</a></td>
      </tr>
  </table>
</p>

### Paper
[Preprint](X)

# Abstract
DL has shown great success in many human-related tasks, which has led to its adoption in many computer vision  based applications, such as security surveillance system, autonomous vehicles and healthcare. Such safety-critical applications have to draw its path to success deployment once it has the capability to overcome safety-critical challenges. One of these challenges is the defense against or/and the detection of adversarial examples (AEs). Adversary can carefully craft small, often imperceptible, noise called perturbations, to be added to the clean image to generate the AE. The aim of AE is to fool the DL model which makes it a potential risk for DL applications. Many test-time evasion attacks and countermeasures, i.e., defense or detection methods, are proposed in the literature. Moreover, few reviews and surveys were published and theoretically showed the taxonomy of the threats and the countermeasure methods with little focus in AE detection methods. In this paper, we attempt to provide a theoretical and experimental review for AE detection methods. A detailed discussion for such methods is provided and experimental results for eight state-of-the-art detectors are provided under different scenarios on four datasets. We also provide potential challenges and future perspectives for this  research direction. 


### Datasets
<table border="0">
  <tbody>
    <tr>
      <td><strong>Dataset</strong></td>
      <td><strong>CNN Model</strong></td>
    </tr>
    <tr>
      <td><strong>MNIST(98.73)</strong></td>
      <td>
        <p>2 (CONV(32, 3x3)+ReLU)+MaxPool, 2 (CONV(64, 3x3)+ReLU)+MaxPool,</p>
        <p>Dense (256) + ReLU + Dropout (0.3), Dense (256) + ReLU,&nbsp;</p>
        <p>Dense(10) + Softmax</p>
      </td>
    </tr>
    <tr>
      <td><strong>CIFAR-10 (89.11)</strong></td>
       <td>
         <p>2(Conv(64, 3x3) + BatchNorm + ReLU) + MaxPool + Dropout(0.1),</p>
         <p>2(Conv(128, 3x3) + BatchNorm + ReLU) + MaxPool + Dropout(0.2),</p>
         <p>2(Conv(256, 3x3) + BatchNorm + ReLU) + MaxPool + Dropout(0.3),</p>
         <p>Conv(512, 3x3) + BatchNorm + ReLU + MaxPool + Dropout(0.4),</p>
         <p>Dense (512) ,</p>
         <p>Dense(10) + Softmax</p>
      </td>
    </tr>
    <tr>
      <td><strong>SVHN (94.98)</strong></td>
      <td>
        <p>2 (CONV(32, 3x3)+ReLU)+MaxPool, 2 (CONV(64, 3x3)+ReLU)+MaxPool,</p>
        <p>Dense (512) + ReLU + Dropout (0.3), Dense (128) + ReLU,&nbsp;</p>
        <p>Dense(10) + Softmax</p>
      </td>
    </tr>
    <tr>
      <td><strong>Tiny-ImageNet (64.48)</strong></td>
      <td>
        <div>
          <div>DenseNet201</div>
        </div>
      </td>
    </tr>
  </tbody>
</table>

### Attacks

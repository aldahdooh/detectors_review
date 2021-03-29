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


### Citation
```

```

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
      <td>2 (CONV(32, 3x3)+ReLU) + MaxPool,<br>
          2 (CONV(64, 3x3)+ReLU) + MaxPool,<br>
          Dense (256) + ReLU + Dropout (0.3), Dense (256) + ReLU,<br>
          Dense(10) + Softmax
      </td>
    </tr>
    <tr>
      <td><strong>CIFAR-10 (89.11)</strong></td>
       <td>2(Conv(64, 3x3) + BatchNorm + ReLU) + MaxPool + Dropout(0.1),<br>
         2(Conv(128, 3x3) + BatchNorm + ReLU) + MaxPool + Dropout(0.2),<br>
         2(Conv(256, 3x3) + BatchNorm + ReLU) + MaxPool + Dropout(0.3),<br>
         Conv(512, 3x3) + BatchNorm + ReLU + MaxPool + Dropout(0.4),<br>
         Dense (512) ,<br>
         Dense(10) + Softmax
      </td>
    </tr>
    <tr>
      <td><strong>SVHN (94.98)</strong></td>
      <td>2 (CONV(32, 3x3)+ReLU)+MaxPool, 2 (CONV(64, 3x3)+ReLU)+MaxPool,<br>
        Dense (512) + ReLU + Dropout (0.3), Dense (128) + ReLU,<br>
        Dense(10) + Softmax
      </td>
    </tr>
    <tr>
      <td><strong>Tiny-ImageNet (64.48)</strong></td>
      <td>DenseNet201</td>
    </tr>
  </tbody>
</table>


### Attacks
<table border="0">
  <tbody>
    <tr>
      <td><strong>Scenario (Zero Knowledge of the detector)</strong></td>
      <td><strong>Attack</strong></td>
    </tr>
    <tr>
      <td><strong>White-box</strong></td>
      <td>FGSM, BIM, PGD-L1, PGD-L2, PGD-Linf, CWinf, CW-HCA, DeepFool</td>
    </tr>
    <tr>
      <td><strong>Black-box</strong></td>
      <td>Square attack, SkipHopJump, Spatial Transformation attack, Zoo</td>
    </tr>
  </tbody>
</table>

# Results, MNIST
### Results, White-box
### Results, Black-box
### Results, Gray-box

# Results, CIFAR-10
### Results, White-box
### Results, Black-box
### Results, Gray-box

# Results, SVHN
### Results, White-box
### Results, Black-box
### Results, Gray-box

# Results, Tiny-ImageNet
### Results, White-box
### Results, Black-box
### Results, Gray-box


<label for="dataset-select">Choose a dataset:</label>

<select name="datasets" id="dataset-select">
    <option value="">--Please choose an option--</option>
    <option value="mnist">MNIST</option>
    <option value="cifar10">CIFAR-10</option>
    <option value="svhn">SVHN</option>
    <option value="tiny">Tiny-ImageNet</option>
</select>


<select id="type">
    <option value="item1">item1</option>
    <option value="item2">item2</option>
    <option value="item3">item3</option>
</select>

<select id="size">
    <option value="">-- select one -- </option>
</select>



<script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.3/jquery.min.js"></script>
<script>
  $(document).ready(function() {

    $("#type").change(function() {
        var val = $(this).val();
        if (val == "item1") {
            $("#size").html("<option value='test'>item1: test 1</option><option value='test2'>item1: test 2</option>");
        } else if (val == "item2") {
            $("#size").html("<option value='test'>item2: test 1</option><option value='test2'>item2: test 2</option>");

        } else if (val == "item3") {
            $("#size").html("<option value='test'>item3: test 1</option><option value='test2'>item3: test 2</option>");
            }
        });
    });
</script>

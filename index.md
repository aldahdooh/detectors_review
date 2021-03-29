# The benchmark
The aim of this benchmark is to have a framework that is able to test the performance of the adversarial examples detection methods under the same attack scenarios. This will help researchers to follow-up the up-to-date progress on the domain. Here, we start with the results published in the review paper; "Adversarial Example Detection for DNN Models: A Review" ([Link](X)). 


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


### Your contribution
We are welcoming your contribution to enrich this benchmark either by adding new detectors' performance evaluation or by including current detectors' performance with more attacks and with different baseline classifiers. Please contact us by opening an isuue to include your updates to the code and to the results.

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

### Related Paper(s)
[Preprint](X)


# Results

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

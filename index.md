<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
<script src="https://www.kryogenix.org/code/browser/sorttable/sorttable.js"></script>
<script src="https://aldahdooh.github.io/detectors_review/js/tableManager.js"></script>

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

<label for="attacks-select">Select an attack:</label>
<select name="attacks" id="attacks-select">
    <option value="fgsm1">FGSM(8)</option>
    <option value="fgsm2">FGSM(16)</option>
    <option value="fgsm3">FGSM(32)</option>
    <option value="fgsm4">FGSM(64)</option>
    <option value="fgsm5">FGSM(80)</option>
    <option value="bim1">BIM(8)</option>
    <option value="bim2">BIM(16)</option>
    <option value="bim3">BIM(32)</option>
    <option value="bim4">BIM(64)</option>
    <option value="bim5">BIM(80)</option>
    <option value="pgd11">PGD-L1(5)</option>
    <option value="pgd12">PGD-L1(10)</option>
    <option value="pgd13">PGD-L1(15)</option>
    <option value="pgd14">PGD-L1(20)</option>
    <option value="pgd15">PGD-L1(25)</option>
    <option value="pgd21">PGD-L2(0.25)</option>
    <option value="pgd22">PGD-L2(0.3125)</option>
    <option value="pgd23">PGD-L2(0.5)</option>
    <option value="pgd24">PGD-L2(1.0)</option>
    <option value="pgd25">PGD-L2(1.5)</option>
    <option value="pgd26">PGD-L2(2.0)</option>
    <option value="pgdi1">PGD-Linf(8)</option>
    <option value="pgdi2">PGD-Linf(16)</option>
    <option value="pgdi3">PGD-Linf(32)</option>
    <option value="pgdi4">PGD-Linf(64)</option>
    <option value="cwi">CW-Linf</option>
    <option value="hca1">CW-HCA(8)</option>
    <option value="hca2">CW-HCA(16)</option>
    <option value="hca3">CW-HCA(80)</option>
    <option value="hca4">CW-HCA(128)</option>
    <option value="df">DeepFool</option>
    <option value="sa">SquareAttack</option>
    <option value="hop">HopSkipJumpAttack</option>
    <option value="sta">SpatialTransformationAttack</option>
</select>

<div id="tables">
</div>

<p><strong>MNIST</strong></p>
<table id='minst' class="tablemanager" border='1'>
<thead>
<tr>
<th>Detector</th>
<th>Model 1 &ndash; DR</th>
<th>Model 1 &ndash; FPR</th>
</tr>
</thead>
<tbody>
<tr>
<td>KD+BU</td>
<td>85.54</td>
<td>3.46</td>
</tr>
<tr>
<td>LID</td>
<td>81.66</td>
<td>1.41</td>
</tr>
<tr>
<td>NSS</td>
<td>100</td>
<td>0</td>
</tr>
<tr>
<td>FS</td>
<td>97.8</td>
<td>5.27</td>
</tr>
<tr>
<td>MagNet</td>
<td>100</td>
<td>0.2</td>
</tr>
<tr>
<td>DNR</td>
<td>59.28</td>
<td>10.01</td>
</tr>
<tr>
<td>SFAD</td>
<td>97.76</td>
<td>10.79</td>
</tr>
<tr>
<td>NIC</td>
<td>100</td>
<td>10.12</td>
</tr>
</tbody>
</table>

<script>
$('.tablemanager').tablemanager({
			firstSort: [[3,0],[2,0],[1,'asc']],
			appendFilterby: true,
			dateFormat: [[4,"mm-dd-yyyy"]],
			debug: true,
			pagination: true,
			showrows: [5,10,20,50,100],
			disableFilterBy: [1]
		});
</script>
  
<script>
  $(document).ready(function() {
    $("#attacks-select").change(function() {
      var val = $(this).val();
      if (val == "fgsm1") {
        $("#tables").html("<p><strong>MNIST</strong></p><table id='mnist' class='sortable' border='0'> <tbody> <tr> <th style="text-align: center;">Detector</th> <th style="text-align: center;">Model 1 &ndash; DR</th> <th style="text-align: center;">Model 1 &ndash; FPR</th> </tr> <tr> <td style="text-align: center;">KD+BU</td> <td style="text-align: center;">85.54</td> <td style="text-align: center;">3.46</td> </tr> <tr> <td style="text-align: center;">LID</td> <td style="text-align: center;">81.66</td> <td style="text-align: center;">1.41</td> </tr> <tr> <td style="text-align: center;">NSS</td> <td style="text-align: center;">100</td> <td style="text-align: center;">0</td> </tr> <tr> <td style="text-align: center;">FS</td> <td style="text-align: center;">97.8</td> <td style="text-align: center;">5.27</td> </tr> <tr> <td style="text-align: center;">MagNet</td> <td style="text-align: center;">100</td> <td style="text-align: center;">0.2</td> </tr> <tr> <td style="text-align: center;">DNR</td> <td style="text-align: center;">59.28</td> <td style="text-align: center;">10.01</td> </tr> <tr> <td style="text-align: center;">SFAD</td> <td style="text-align: center;">97.76</td> <td style="text-align: center;">10.79</td> </tr> <tr> <td style="text-align: center;">NIC</td> <td style="text-align: center;">100</td> <td style="text-align: center;">10.12</td> </tr> </tbody> </table>");
      } 
      else if (val == "fgsm2") {
        $("#tables").html("");
      } 
      else if (val == "fgsm3") {
        $("#tables").html("");
      }
      else if (val == "fgsm4") {
        $("#tables").html("");
      }
      else if (val == "fgsm5") {
        $("#tables").html("");
      }
      else if (val == "bim1") {
        $("#tables").html("");
      }
      else if (val == "bim2") {
        $("#tables").html("");
      }
      else if (val == "bim3") {
        $("#tables").html("");
      }
      else if (val == "bim4") {
        $("#tables").html("");
      }
      else if (val == "bim5") {
        $("#tables").html("");
      }
      else if (val == "pgd11") {
        $("#tables").html("");
      }
      else if (val == "pgd12") {
        $("#tables").html("");
      }
      else if (val == "pgd13") {
        $("#tables").html("");
      }
      else if (val == "pgd14") {
        $("#tables").html("");
      }
      else if (val == "pgd15") {
        $("#tables").html("");
      }
      else if (val == "pgd21") {
        $("#tables").html("");
      }
      else if (val == "pgd22") {
        $("#tables").html("");
      }
      else if (val == "pgd23") {
        $("#tables").html("");
      }
      else if (val == "pgd24") {
        $("#tables").html("");
      }
      else if (val == "pgd25") {
        $("#tables").html("");
      }
      else if (val == "pgd26") {
        $("#tables").html("");
      }
      else if (val == "pgdi1") {
        $("#tables").html("");
      }
      else if (val == "pgdi2") {
        $("#tables").html("");
      }
      else if (val == "pgdi3") {
        $("#tables").html("");
      }
      else if (val == "pgdi4") {
        $("#tables").html("");
      }
      else if (val == "cwi") {
        $("#tables").html("");
      }
      else if (val == "hca1") {
        $("#tables").html("");
      }
      else if (val == "hca2") {
        $("#tables").html("");
      }
      else if (val == "hca3") {
        $("#tables").html("");
      }
      else if (val == "hca4") {
        $("#tables").html("");
      }
      else if (val == "df") {
        $("#tables").html("");
      }
      else if (val == "sq") {
        $("#tables").html("");
      }
      else if (val == "hop") {
        $("#tables").html("");
      }
      else if (val == "sta") {
        $("#tables").html("");
      }
    });
  });
</script>

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
  <option value="">--Please choose an option--</option>
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
    $("#attacks").change(function() {
      var val = $(this).val();
      if (val == "fgsm1") {
        $("#tables").html(" <table border="0">
                                <tbody>
                                  <tr>
                                    <td><strong>1 Scenario (Zero Knowledge of the detector)</strong></td>
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
                              </table>");
      } 
      else if (val == "fgsm2") {
        $("#tables").html(" <table border="0">
                                <tbody>
                                  <tr>
                                    <td><strong>2 Scenario (Zero Knowledge of the detector)</strong></td>
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
                              </table>");
      } 
      else if (val == "fgsm3") {
        $("#tables").html(" <table border="0">
                              <tbody>
                                <tr>
                                  <td><strong>3 Scenario (Zero Knowledge of the detector)</strong></td>
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
                           </table>");
      }
    });
  });
</script>

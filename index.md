<script src="https://code.jquery.com/jquery-3.5.1.js"></script>
<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.24/css/jquery.dataTables.min.css"/>
<script type="text/javascript" src="https://cdn.datatables.net/1.10.24/js/jquery.dataTables.min.js"></script>


# The benchmark
The aim of this benchmark is to have a framework that is able to test the performance of the adversarial examples detection methods under the same attack scenarios. This will help researchers to follow-up the up-to-date progress on the domain. Here, we start with the results published in the review paper; "Adversarial Example Detection for DNN Models: A Review" ([Link](X)). 

<center>
    <a href="https://aldahdooh.github.io/detectors_review/imgs/all_sumary.png" target="_blank">
        <figure>
            <img src="https://aldahdooh.github.io/detectors_review/imgs/all_sumary.png" width="500" height="300">
            <figcaption><p>Fig.1 - Average detection rates (%) for detectors against adversarial examples for each dataset</p></figcaption>
        </figure>
    </a>
</center>

# Results

<label for="attacks-select">Select an attack:</label>
<select name="attacks" id="attacks-select">
    <option value="item0">--- Please select an attack ---</option>
    <option value="item1" disabled="disabled">******White-box Attacks******</option>
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
    <option value="item2" disabled="disabled">******Black-box Attacks******</option>
    <option value="sa">SquareAttack</option>
    <option value="hop">HopSkipJumpAttack</option>
    <option value="sta">SpatialTransformationAttack</option>
</select>

<blockquote>
<p><span style="color: #0000ff;"><em><strong>Note:</strong></em> In this website, we only report the detection rate (DR) and the false positive rate (FPR). Other performance results, like TP, TN, FP, and FN, can be accquired from the genenerated CSV file for each detector (visit the gitub repository).</span></p>
</blockquote>

<div id="tables">
</div>  

<script>
  $(document).ready(function() {
    $("#attacks-select").change(function() {
      var val = $(this).val();
      if (val == "item0") {
        $("#tables").html("");
      } 
      else if (val == "fgsm1") {
        $("#tables").html("<hr><p><strong>CIFAR</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>35.03</td><td>7.3</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>53</td><td>3.84</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>87.59</td><td>6.56</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>29.33</td><td>5.07</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.72</td><td>0.77</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>32.09</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>67.94</td><td>10.9</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>43.64</td><td>10.08</td></tr> </tbody> </table> <hr><p><strong>SVHN</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>46.75</td><td>10.02</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>72.18</td><td>7.37</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>98.95</td><td>0.54</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>47.5</td><td>5.1</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>8.57</td><td>0.49</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>45.45</td><td>10</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>81.26</td><td>11.02</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>67.35</td><td>9.99</td></tr> </tbody> </table> <hr><p><strong>Tiny-ImageNet</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>4.76</td><td>1.64</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>0</td><td>0</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>83.71</td><td>21.81</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>23.04</td><td>5.33</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.56</td><td>0.9</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>-</td><td>-</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>50.08</td><td>16.38</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.09</td></tr> <tbody> </tbody> </table>");
      } 
      else if (val == "fgsm2") {
        $("#tables").html("<hr><p><strong>CIFAR</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>33.23</td><td>4.5</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>81.23</td><td>1.44</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>99.94</td><td>6.56</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>35.34</td><td>5.07</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>3.11</td><td>0.77</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>31.35</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>79.9</td><td>10.9</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>58.48</td><td>10.08</td></tr> </tbody> </table> <hr><p><strong>SVHN</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>44.3</td><td>9.28</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>89.79</td><td>3.47</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>99.85</td><td>0.54</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>51.88</td><td>5.1</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>18.75</td><td>0.49</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>50.63</td><td>10</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>88.57</td><td>11.02</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>59.86</td><td>9.99</td></tr> </tbody> </table> <hr><p><strong>Tiny-ImageNet</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>0</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>27.89</td><td>6.52</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>97.01</td><td>21.81</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>23.88</td><td>5.33</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>1.16</td><td>0.9</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>-</td><td>-</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>57.74</td><td>16.38</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.09</td></tr> </tbody> </table>");
      } 
      else if (val == "fgsm3") {
        $("#tables").html("<hr><p><strong>MNIST</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>85.54</td><td>3.46</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>81.66</td><td>1.41</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>100</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>97.8</td><td>5.27</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>100</td><td>0.2</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>59.28</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>97.76</td><td>10.79</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.12</td></tr> </tbody> </table> <hr><p><strong>CIFAR</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>0.08</td><td>0.04</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>94.23</td><td>0.11</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>99.67</td><td>6.56</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>32.83</td><td>5.07</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>100</td><td>0.77</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>27.24</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>92.58</td><td>10.9</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>87.32</td><td>10.08</td></tr> </tbody> </table>");
      }
      else if (val == "fgsm4") {
        $("#tables").html("<hr><p><strong>MNIST</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>53.8</td><td>0.64</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>77.09</td><td>0.07</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>100</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>98.06</td><td>5.27</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>100</td><td>0.2</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>87.81</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>98.74</td><td>10.79</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.12</td></tr> </tbody> </table>");
      }
      else if (val == "fgsm5") {
        $("#tables").html("<hr><p><strong>MNIST</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead>  <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>49.03</td><td>0.17</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>73.64</td><td>0.07</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>100</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>98.02</td><td>5.27</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>100</td><td>0.2</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>91.91</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>99.47</td><td>10.79</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.12</td></tr> </tbody> </table>");
      }
      else if (val == "bim1") {
        $("#tables").html("<hr><p><strong>CIFAR</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>84.47</td><td>2.18</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>88.05</td><td>3.65</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>52.16</td><td>6.56</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>8.74</td><td>5.07</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.56</td><td>0.77</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>4.27</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>18.12</td><td>10.9</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>99.95</td><td>10.08</td></tr> </tbody> </table> <hr><p><strong>SVHN</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>49.49</td><td>11.01</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>52.38</td><td>11.01</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>92.08</td><td>0.54</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>11.71</td><td>5.1</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>54.29</td><td>0.49</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>24.8</td><td>10</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>26.07</td><td>11.02</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>92.91</td><td>9.99</td></tr> </tbody> </table> <hr><p><strong>Tiny-ImageNet</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>56.43</td><td>9.75</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>65.74</td><td>9.29</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>33.13</td><td>21.81</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>9.14</td><td>5.33</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.65</td><td>0.9</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>-</td><td>-</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>9.23</td><td>16.38</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.09</td></tr> </tbody> </table>");
      }
      else if (val == "bim2") {
        $("#tables").html("<hr><p><strong>CIFAR</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>99.55</td><td>0.07</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>98.55</td><td>0.44</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>87.74</td><td>6.56</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>0.34</td><td>5.07</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.69</td><td>0.77</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>17.07</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>45.35</td><td>10.9</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.08</td></tr> </tbody> </table> <hr><p><strong>SVHN</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>93.64</td><td>2.79</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>86.64</td><td>5.4</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>99.85</td><td>0.54</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>0.73</td><td>5.1</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>88.08</td><td>0.49</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>14.74</td><td>10</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>14.22</td><td>11.02</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>92.91</td><td>9.99</td></tr> </tbody> </table> <hr><p><strong>Tiny-ImageNet</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>85.93</td><td>2.57</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>90.09</td><td>3.75</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>59.46</td><td>21.81</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>1.92</td><td>5.33</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.65</td><td>0.9</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>-</td><td>-</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>6.81</td><td>16.38</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.09</td></tr> </tbody> </table>");
      }
      else if (val == "bim3") {
        $("#tables").html("<hr><p><strong>MNIST</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>58.66</td><td>3.77</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>80.06</td><td>0.94</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>100</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>99.18</td><td>5.27</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>100</td><td>0.2</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>67.19</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>93.81</td><td>10.79</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>99.46</td><td>10.12</td></tr> </tbody> </table> <hr><p><strong>Tiny-ImageNet</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>96.66</td><td>0.21</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>96.77</td><td>1.13</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>92.88</td><td>21.81</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>0.5</td><td>5.33</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.81</td><td>0.9</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>-</td><td>-</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>5.63</td><td>16.38</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.09</td></tr> </tbody> </table>");
      }
      else if (val == "bim4") {
        $("#tables").html("<hr><p><strong>MNIST</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>48.2</td><td>5.52</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>74.71</td><td>0.4</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>100</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>95.08</td><td>5.27</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>100</td><td>0.2</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>51.24</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>70.34</td><td>10.79</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>99.99</td><td>10.12</td></tr> </tbody> </table>");
      }
      else if (val == "bim5") {
        $("#tables").html("<hr><p><strong>MNIST</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>80.76</td><td>2.09</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>77.63</td><td>0.57</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>100</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>89.73</td><td>5.27</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>100</td><td>0.2</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>62.99</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>66.53</td><td>10.79</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.12</td></tr> </tbody> </table>");
      }
      else if (val == "pgd11") {
        $("#tables").html("<hr><p><strong>CIFAR</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>51.96</td><td>7.12</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>0</td><td>0</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>5.32</td><td>6.56</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>75.61</td><td>5.07</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.4</td><td>0.77</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>38.66</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>66.06</td><td>10.9</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>56.12</td><td>10.08</td></tr> </tbody> </table>");
      }
      else if (val == "pgd12") {
        $("#tables").html("<hr><p><strong>MNIST</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>71.34</td><td>3.03</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>77.71</td><td>2.49</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>65.32</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>97.8</td><td>5.27</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>5</td><td>0.2</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>57.56</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>95.66</td><td>10.79</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.12</td></tr> </tbody> </table> <hr><p><strong>CIFAR</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>9.67</td><td>1.81</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>48.18</td><td>24.71</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>8.02</td><td>6.56</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>70.7</td><td>5.07</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.61</td><td>0.77</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>28.92</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>30.34</td><td>10.9</td></tr><tr> </tbody> </table>");
      }
      else if (val == "pgd13") {
        $("#tables").html("<hr><p><strong>MNIST</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>52.41</td><td>3.4</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>73.87</td><td>2.19</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>88.61</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>94.56</td><td>5.27</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>51.51</td><td>0.2</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>56.57</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>88.3</td><td>10.79</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>99.99</td><td>10.12</td></tr> </tbody> </table> <hr><p><strong>CIFAR</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>34.48</td><td>10.14</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>69.43</td><td>21.1</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>11.38</td><td>6.56</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>56.61</td><td>5.07</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.68</td><td>0.77</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>18.07</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>13.7</td><td>10.9</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>92.32</td><td>10.08</td></tr> </tbody> </table> <hr><p><strong>SVHN</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>9.84</td><td>6.97</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>43.03</td><td>19.99</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>0.48</td><td>0.54</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>43.32</td><td>5.1</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>20.43</td><td>0.49</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>36.9</td><td>10</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>46.9</td><td>11.02</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>91.99</td><td>9.99</td></tr> </tbody> </table> <hr><p><strong>Tiny-ImageNet</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>0</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>0</td><td>0</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>19.69</td><td>21.81</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>50.68</td><td>5.33</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.7</td><td>0.9</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>-</td><td>-</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>44.2</td><td>16.38</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.09</td></tr> </tbody> </table>");
      }
      else if (val == "pgd14") {
        $("#tables").html("<hr><p><strong>MNIST</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>28.16</td><td>3.46</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>65.96</td><td>1.61</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>98.05</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>88.1</td><td>5.27</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>94.55</td><td>0.2</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>49.47</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>78.18</td><td>10.79</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>98.77</td><td>10.12</td></tr></tbody> </table><hr><p><strong>SVHN</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>22.24</td><td>14.22</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>48.8</td><td>19.75</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>0.59</td><td>0.54</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>30.79</td><td>5.1</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>32.03</td><td>0.49</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>34.64</td><td>10</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>37.62</td><td>11.02</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>88.7</td><td>9.99</td></tr> </tbody> </table> <hr><p><strong>Tiny-ImageNet</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>0</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>0</td><td>0</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>19.83</td><td>21.81</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>54.48</td><td>5.33</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.63</td><td>0.9</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>-</td><td>-</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>37.29</td><td>16.38</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.09</td></tr></tbody> </table>");
      }
      else if (val == "pgd15") {
        $("#tables").html("<hr><p><strong>SVHN</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>42.67</td><td>17.09</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>53.67</td><td>18.86</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>0.78</td><td>0.54</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>21.62</td><td>5.1</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>41.71</td><td>0.49</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>30.16</td><td>10</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>31.06</td><td>11.02</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>93.45</td><td>9.99</td></tr> </tbody> </table> <hr><p><strong>Tiny-ImageNet</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>32.52</td><td>1.78</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>69.17</td><td>0.17</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>58.62</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>100</td><td>5.27</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>100</td><td>0.2</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>79.79</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>98.71</td><td>10.79</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.12</td></tr><tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead><tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>0</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>54.93</td><td>34.09</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>20.23</td><td>21.81</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>55.64</td><td>5.33</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.6</td><td>0.9</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>-</td><td>-</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>30.99</td><td>16.38</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.09</td></tr> </tbody> </table>");
      }
      else if (val == "pgd21") {
        $("#tables").html("<hr><p><strong>CIFAR</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>36.99</td><td>6.79</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>30.53</td><td>16.78</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>7.38</td><td>6.56</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>73.59</td><td>5.07</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.55</td><td>0.77</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>30.49</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>34.95</td><td>10.9</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>72.18</td><td>10.08</td></tr> </tbody> </table>");
      }
      else if (val == "pgd22") {
        $("#tables").html("<hr><p><strong>CIFAR</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>0.12</td><td>0.11</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>51.83</td><td>23.39</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>8.75</td><td>6.56</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>67.14</td><td>5.07</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.62</td><td>0.77</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>26.12</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>24.08</td><td>10.9</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>89.1</td><td>10.08</td></tr> </tbody> </table> <hr><p><strong>SVHN</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>20.4</td><td>5.26</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>22.88</td><td>9.13</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>0.38</td><td>0.54</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>59.33</td><td>5.1</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>7.73</td><td>0.49</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>37.34</td><td>10</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>60.69</td><td>11.02</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>71.26</td><td>9.99</td></tr> </tbody> </table>");
      }
      else if (val == "pgd23") {
        $("#tables").html("<hr><p><strong>CIFAR</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>55.01</td><td>9.26</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>77.97</td><td>17.71</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>13.72</td><td>6.56</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>45.36</td><td>5.07</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.7</td><td>0.77</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>10.65</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>10.95</td><td>10.9</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>97.21</td><td>10.08</td></tr> </tbody> </table> <hr><p><strong>SVHN</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>10.99</td><td>9.26</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>46.48</td><td>20.56</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>0.53</td><td>0.54</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>37.59</td><td>5.1</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>24.86</td><td>0.49</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>35.41</td><td>10</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>42.13</td><td>11.02</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>83.26</td><td>9.99</td></tr> </tbody> </table> <hr><p><strong>Tiny-ImageNet</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>0</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>63.45</td><td>29.11</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>21.38</td><td>21.81</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>51.83</td><td>5.33</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.63</td><td>0.9</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>-</td><td>-</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>20.09</td><td>16.38</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.09</td></tr></tbody> </table>");
      }
      else if (val == "pgd24") {
        $("#tables").html("<hr><p><strong>MNIST</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>73.87</td><td>2.69</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>81.27</td><td>2.93</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>62.36</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>98.07</td><td>5.27</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>11</td><td>0.2</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>54.52</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>96.42</td><td>10.79</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.12</td></tr></tbody> </table> <hr><p><strong>SVHN</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>71.9</td><td>13.07</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>64.86</td><td>16.04</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>2.34</td><td>0.54</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>9.89</td><td>5.1</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>59.34</td><td>0.49</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>19.98</td><td>10</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>22.69</td><td>11.02</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>99.45</td><td>9.99</td></tr> </tbody> </table> <hr><p><strong>Tiny-ImageNet</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>40.73</td><td>3.49</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>75.43</td><td>14.99</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>23.55</td><td>21.81</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>28.31</td><td>5.33</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.77</td><td>0.9</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>-</td><td>-</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>10.96</td><td>16.38</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.09</td></tr> </tbody> </table>");
      }
      else if (val == "pgd25") {
        $("#tables").html("<hr><p><strong>MNIST</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>0.04</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>59.34</td><td>3.46</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>88.51</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>96.07</td><td>5.27</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>60.79</td><td>0.2</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>57.83</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>89.61</td><td>10.79</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>98.74</td><td>10.12</td></tr></tbody> </table>");
      }
      else if (val == "pgd26") {
        $("#tables").html("<hr><p><strong>MNIST</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>0.71</td><td>0.07</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>57.41</td><td>1.45</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>98.63</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>85.58</td><td>5.27</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>93.31</td><td>0.2</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>48.21</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>75.44</td><td>10.79</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.12</td></tr></tbody> </table>");
      }
      else if (val == "pgdi1") {
        $("#tables").html("<hr><p><strong>CIFAR</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>92.27</td><td>0.96</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>94.39</td><td>1.81</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>57.06</td><td>6.56</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>8.2</td><td>5.07</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.57</td><td>0.77</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>11.34</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>29.49</td><td>10.9</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.08</td></tr> </tbody> </table> <hr><p><strong>SVHN</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>61.21</td><td>10.25</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>55.83</td><td>10.64</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>95.54</td><td>0.54</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>10.35</td><td>5.1</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>65.74</td><td>0.49</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>21.83</td><td>10</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>24.08</td><td>11.02</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>92.98</td><td>9.99</td></tr> </tbody> </table> <hr><p><strong>Tiny-ImageNet</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>80.75</td><td>4.93</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>91.48</td><td>2.82</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>58.23</td><td>21.81</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>9.74</td><td>5.33</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.64</td><td>0.9</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>-</td><td>-</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>7.62</td><td>16.38</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.09</td></tr> </tbody> </table>");
      }
      else if (val == "pgdi2") {
        $("#tables").html("<hr><p><strong>CIFAR</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>99.89</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>99.22</td><td>0.26</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>93.24</td><td>6.56</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>0.2</td><td>5.07</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.66</td><td>0.77</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>25.11</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>52.9</td><td>10.9</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.08</td></tr> </tbody> </table> <hr><p><strong>SVHN</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>95.75</td><td>1.82</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>89.77</td><td>4.13</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>99.95</td><td>0.54</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>0.53</td><td>5.1</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>92.95</td><td>0.49</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>16.19</td><td>10</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>13.96</td><td>11.02</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>99.99</td><td>9.99</td></tr> </tbody> </table> <hr><p><strong>Tiny-ImageNet</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>96.71</td><td>0.41</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>97.54</td><td>0.87</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>83.65</td><td>21.81</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>1.99</td><td>5.33</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.64</td><td>0.9</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>-</td><td>-</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>5.91</td><td>16.38</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.09</td></tr> </tbody> </table>");
      }
      else if (val == "pgdi3") {
        $("#tables").html("<hr><p><strong>MNIST</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>55.32</td><td>3.97</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>79.04</td><td>0.98</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>100</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>99.2</td><td>5.27</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>100</td><td>0.2</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>67.13</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>93.47</td><td>10.79</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.12</td></tr> </tbody> </table>");
      }
      else if (val == "pgdi4") {
        $("#tables").html("<hr><p><strong>MNIST</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>49.71</td><td>5.55</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>75.01</td><td>0.4</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>100</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>95.18</td><td>5.27</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>100</td><td>0.2</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>51.28</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>70.18</td><td>10.79</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.12</td></tr> </tbody> </table>");
      }
      else if (val == "cwi") {
        $("#tables").html("<hr><p><strong>MNIST</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>42.77</td><td>0.71</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>64.43</td><td>3.94</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>2.47</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>98.41</td><td>5.27</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>40.56</td><td>0.2</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>57.98</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>98.24</td><td>10.79</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.12</td></tr> </tbody> </table> <hr><p><strong>CIFAR</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>21.12</td><td>4.54</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>64.52</td><td>20.58</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>27.48</td><td>6.56</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>56.18</td><td>5.07</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>13.23</td><td>0.77</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>44.15</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>87.68</td><td>10.9</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>61.68</td><td>10.08</td></tr> </tbody> </table> <hr><p><strong>SVHN</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>55.14</td><td>8.67</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>43.37</td><td>11.96</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>11.02</td><td>0.54</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>67.01</td><td>5.1</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>8.3</td><td>0.49</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>46.19</td><td>10</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>87.09</td><td>11.02</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>85.83</td><td>9.99</td></tr> </tbody> </table> <hr><p><strong>Tiny-ImageNet</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>0</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>0</td><td>0</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>32.35</td><td>21.81</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>24.78</td><td>5.12</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>7.93</td><td>0.9</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>-</td><td>-</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>68.13</td><td>16.38</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.09</td></tr> </tbody> </table>");
      }
      else if (val == "hca1") {
        $("#tables").html("<hr><p><strong>CIFAR</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>37.29</td><td>6.34</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>44.59</td><td>15.01</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>40.94</td><td>6.56</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>68.33</td><td>5.07</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.61</td><td>0.77</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>34.91</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>57.76</td><td>10.9</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>75.18</td><td>10.08</td></tr> </tbody> </table> <hr><p><strong>SVHN</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>15.3</td><td>4.73</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>24.9</td><td>7.07</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>79.08</td><td>0.54</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>32.45</td><td>5.1</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>35.43</td><td>0.49</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>33.04</td><td>10</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>49.68</td><td>11.02</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>81.92</td><td>9.99</td></tr> </tbody> </table> <hr><p><strong>Tiny-ImageNet</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>21.46</td><td>3.49</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>31.89</td><td>6.62</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>38.02</td><td>21.81</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>44.76</td><td>5.33</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.44</td><td>0.9</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>-</td><td>-</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>42.89</td><td>16.38</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.09</td></tr> </tbody> </table>");
      }
      else if (val == "hca2") {
        $("#tables").html("<hr><p><strong>CIFAR</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>29.69</td><td>3.95</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>65.46</td><td>19.25</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>65.12</td><td>6.56</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>44.28</td><td>5.07</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.44</td><td>0.77</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>24.79</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>33.94</td><td>10.9</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>71.39</td><td>10.08</td></tr> </tbody> </table> <hr><p><strong>SVHN</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>47.89</td><td>6.87</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>53.27</td><td>9.47</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>91.47</td><td>0.54</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>16.07</td><td>5.1</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>68.6</td><td>0.49</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>23.87</td><td>10</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>33.46</td><td>11.02</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>93.84</td><td>9.99</td></tr> </tbody> </table> <hr><p><strong>Tiny-ImageNet</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>35.34</td><td>5.44</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>53.87</td><td>10.73</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>34.35</td><td>21.81</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>39.01</td><td>5.33</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.19</td><td>0.9</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>-</td><td>-</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>33.45</td><td>16.38</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.09</td></tr> </tbody> </table>");
      }
      else if (val == "hca3") {
        $("#tables").html("<hr><p><strong>MNIST</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>32.52</td><td>1.78</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>69.17</td><td>0.17</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>58.62</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>100</td><td>5.27</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>100</td><td>0.2</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>79.79</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>98.71</td><td>10.79</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.12</td></tr> </tbody> </table>");
      }
      else if (val == "hca4") {
        $("#tables").html("<hr><p><strong>MNIST</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>86</td><td>3.13</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>99.85</td><td>0</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>5.9</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>99.98</td><td>5.27</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>100</td><td>0.2</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>100</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>100</td><td>10.79</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.12</td></tr> </tbody> </table>");
      }
      else if (val == "df") {
        $("#tables").html("<hr><p><strong>MNIST</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>48.97</td><td>0.37</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>93.3</td><td>0.1</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>98.5</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>66.96</td><td>5.27</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>96.99</td><td>0.2</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>95.6</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>99.58</td><td>10.79</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.12</td></tr> </tbody> </table> <hr><p><strong>CIFAR</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>54.02</td><td>1.44</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>63.57</td><td>6.12</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>50.15</td><td>6.56</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>39.18</td><td>5.07</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>57.33</td><td>0.77</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>30.2</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>89.57</td><td>10.9</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>84.91</td><td>10.08</td></tr> </tbody> </table> <hr><p><strong>SVHN</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>58.47</td><td>7.58</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>64.74</td><td>2.3</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>58.8</td><td>0.54</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>62.33</td><td>5.1</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>44.98</td><td>0.49</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>66.7</td><td>10</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>89.55</td><td>11.02</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>83.25</td><td>9.99</td></tr> </tbody> </table> <hr><p><strong>Tiny-ImageNet</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>0</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>30.88</td><td>17.73</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>27.2</td><td>22.68</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>36.96</td><td>5.86</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>1.28</td><td>1.02</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>-</td><td>-</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>72.32</td><td>16.38</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>9.3</td></tr> </tbody> </table>");
      }
      else if (val == "sa") {
        $("#tables").html("<hr><p><strong>MNIST</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead>  <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>53.53</td><td>0.24</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>42.78</td><td>0.03</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>87.64</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>99.96</td><td>5.27</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>99.93</td><td>0.2</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>81.27</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>98.85</td><td>10.79</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>99.68</td><td>10.12</td></tr> </tbody> </table> <hr><p><strong>CIFAR</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>0</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>85.76</td><td>4.72</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>1.49</td><td>6.56</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>17.82</td><td>5.07</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>94.04</td><td>0.77</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>52.86</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>93.91</td><td>10.9</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>61.88</td><td>10.08</td></tr> </tbody> </table> <hr><p><strong>SVHN</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>70.02</td><td>7.45</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>52.01</td><td>5.95</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>33.36</td><td>0.54</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>74.76</td><td>5.1</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>28.32</td><td>0.49</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>45.56</td><td>10</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>93.23</td><td>11.02</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>78.59</td><td>9.99</td></tr> </table> <hr><p><strong>Tiny-ImageNet</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>33.11</td><td>5.13</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>89.25</td><td>4.47</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>95.03</td><td>21.81</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>25.72</td><td>5.33</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>81.71</td><td>0.9</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>-</td><td>-</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>75.69</td><td>16.38</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.09</td></tr> </tbody> </table>");
      }
      else if (val == "hop") {
        $("#tables").html("<hr><p><strong>MNIST</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead>  <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>61.82</td><td>0.57</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>61.52</td><td>2.22</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>99.88</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>99.98</td><td>5.27</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>98.32</td><td>0.2</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>59.98</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>99.91</td><td>10.79</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.12</td></tr> </tbody> </table> <hr><p><strong>CIFAR</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>28.03</td><td>7.19</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>88.34</td><td>11.18</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>21.42</td><td>6.56</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>84.16</td><td>5.07</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.58</td><td>0.77</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>38.81</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>95.57</td><td>10.9</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>67.53</td><td>10.08</td></tr> </tbody> </table> <hr><p><strong>SVHN</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>84.85</td><td>5.72</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>58.26</td><td>10.6</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>57.59</td><td>0.54</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>94.42</td><td>5.1</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>6.13</td><td>0.49</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>34.57</td><td>10</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>96.47</td><td>11.02</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>86.33</td><td>9.99</td></tr> </tbody> </table> <hr><p><strong>Tiny-ImageNet</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>0</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>0</td><td>0</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>60.19</td><td>21.81</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>48.92</td><td>4.81</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>8.42</td><td>0.9</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>-</td><td>-</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>69.17</td><td>16.38</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.09</td></tr> </tbody> </table>");
      }
      else if (val == "sta") {
        $("#tables").html("<hr><p><strong>MNIST</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>47.94</td><td>0.5</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>93.81</td><td>0.64</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>12.26</td><td>0</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>77.49</td><td>5.27</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>1.61</td><td>0.2</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>88</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>97.61</td><td>10.79</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>99.83</td><td>10.12</td></tr> </tbody> </table> <hr><p><strong>CIFAR</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>44.15</td><td>3.1</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>94.23</td><td>5.27</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>31.73</td><td>6.56</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>22.46</td><td>5.07</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>2.32</td><td>0.77</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>56.2</td><td>10.01</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>92.9</td><td>10.9</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>48.77</td><td>10.08</td></tr> </tbody> </table> <hr><p><strong>SVHN</strong></p><hr> <table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>73.21</td><td>7.72</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>90.12</td><td>1.82</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>99.89</td><td>0.54</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>32.63</td><td>5.1</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>14.56</td><td>0.49</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>73.41</td><td>10</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>96.81</td><td>11.02</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>94.4</td><td>9.99</td></tr> </tbody> </table> <hr><p><strong>Tiny-ImageNet</strong></p><hr><table id='' class='display compact' style='width:100%'> <thead> <tr> <th rowspan='2'>Detector</th> <th rowspan='2'>Category</th> <th colspan='2'>Model 1</th> </tr> <tr> <th>DR</th> <th>FPR</th> </tr> </thead> <tbody> <tr><td><a href='https://arxiv.org/pdf/1703.00410.pdf'>KD+BU</a></td><td>Supervised</td><td>13.94</td><td>1.03</td></tr><tr><td><a href='https://arxiv.org/pdf/1801.02613.pdf'>LID</a></td><td>Supervised</td><td>74.11</td><td>24.08</td></tr><tr><td><a href='https://ieeexplore.ieee.org/document/9287056'>NSS</a></td><td>Supervised</td><td>0.06</td><td>21.81</td></tr><tr><td><a href='https://arxiv.org/pdf/1704.01155.pdf'>FS</a></td><td>Unsupervised</td><td>25.5</td><td>5.33</td></tr><tr><td><a href='https://arxiv.org/pdf/1705.09064.pdf'>MagNet</a></td><td>Unsupervised</td><td>0.26</td><td>0.9</td></tr><tr><td><a href='https://arxiv.org/pdf/1910.00470.pdf'>DNR</a></td><td>Unsupervised</td><td>-</td><td>-</td></tr><tr><td><a href='https://arxiv.org/pdf/2103.05354.pdf'>SFAD</a></td><td>Unsupervised</td><td>73.42</td><td>16.38</td></tr><tr><td><a href='https://www.ndss-symposium.org/wp-content/uploads/2019/02/ndss2019_03A-4_Ma_paper.pdf'>NIC</a></td><td>Unsupervised</td><td>100</td><td>10.09</td></tr> </tbody> </table>");
      }
      $(document).ready(function() {
        $('table.display').DataTable({
          "lengthMenu": [[10, 15, 25, 50, -1], [10, 115, 25, 50, "All"]]
          });
        });
    });
  });
</script>

<br>
<hr>

# About
### Citation
```
```

### Authors
- [Ahmed Aldahdooh](https://scholar.google.com/citations?user=7BLBJC0AAAAJ&hl=en) [<img src="https://aldahdooh.github.io/detectors_review/imgs/email.png" width="12" height="12" />](mailto:ahmed.aldahdooh@insa-rennes.fr)
- [Wassim Hamidouche](https://scholar.google.fr/citations?user=ywBnUIAAAAAJ&hl=en) [<img src="https://aldahdooh.github.io/detectors_review/imgs/email.png" width="12" height="12" />](mailto:Wassim.Hamidouche@insa-rennes.fr)
- [Olivier Deforges](https://scholar.google.fr/citations?user=c5DiiBUAAAAJ&hl=en) [<img src="https://aldahdooh.github.io/detectors_review/imgs/email.png" width="12" height="12" />](mailto:Olivier.Deforges@insa-rennes.fr)

### Your contribution
We are welcoming your contribution to enrich this benchmark either by adding new detectors' performance evaluation or by including current detectors' performance with more attacks and with different baseline classifiers. Please 1)Follow the instruction [here](https://github.com/aldahdooh/detectors_review#steps-to-add-new-detector-method) 2)Contact us by opening an isuue to include your updates to the code and to the results. 

# Datasets and Attacks
### Datasets
<table border="0">
  <thead>
      <tr>
      <th><strong>Dataset</strong></th>
      <th><strong>Neural Network Model(s)</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>MNIST</strong></td>
      <td><ul>
            <li>Model 1<sup>**</sup></li>
         </ul> 
      </td>
    </tr>
    <tr>
      <td><strong>CIFAR-10</strong></td>
       <td><ul>
            <li>Model 1<sup>**</sup></li>
         </ul> 
      </td>
    </tr>
    <tr>
      <td><strong>SVHN</strong></td>
      <td><ul>
            <li>Model 1<sup>**</sup></li>
         </ul> 
      </td>
    </tr>
    <tr>
      <td><strong>Tiny-ImageNet </strong></td>
      <td><ul>
            <li>Model 1<sup>**</sup></li>
         </ul>
      </td>
    </tr>
  </tbody>
</table>


_**Models Description_
<table>
<thead>
<tr>
<th>Model Name</th>
<th>Description</th>
</tr>
</thead>
<tbody>
<tr>
<td>MNIST - Model 1 (98.73)</td>
<td>2 (CONV(32, 3x3)+ReLU) + MaxPool,<br/>2 (CONV(64, 3x3)+ReLU) + MaxPool,<br/>Dense (256) + ReLU + Dropout (0.3), Dense (256) + ReLU,<br/>Dense(10) + Softmax</td>
</tr>
<tr>
<td>CIFAR-10 - Model 1 (89.11)</td>
<td>2(Conv(64, 3x3) + BatchNorm + ReLU) + MaxPool + Dropout(0.1),<br/>2(Conv(128, 3x3) + BatchNorm + ReLU) + MaxPool + Dropout(0.2),<br/>2(Conv(256, 3x3) + BatchNorm + ReLU) + MaxPool + Dropout(0.3),<br/>Conv(512, 3x3) + BatchNorm + ReLU + MaxPool + Dropout(0.4),<br/>Dense (512) ,<br/>Dense(10) + Softmax</td>
</tr>
<tr>
<td>SVHN - Model 1 (94.98)</td>
<td>2 (CONV(32, 3x3)+ReLU)+MaxPool, 2 (CONV(64, 3x3)+ReLU)+MaxPool,<br/>Dense (512) + ReLU + Dropout (0.3), Dense (128) + ReLU,<br/>Dense(10) + Softmax</td>
</tr>
<tr>
<td>Tiny-ImageNet - Model 1 (64.48)</td>
<td>DenseNet201</td>
</tr>
</tbody>
</table>


### Attacks
<table>
<thead>
<tr>
<th>Scenario</th>
<th>Attack</th>
<th>Norm</th>
<th>(Un)Targeted</th>
<th>Parameters</th>
</tr>
</thead>
<tbody>
<tr>
<td rowspan="8">White-box</td>
<td >FGSM</td>
<td>L-inf</td>
<td>U</td>
<td >eps = (8, 16, 32, 64, 80, 128)/255<br/>
  eps_step = 0.01
</td>
</tr>
<tr>
<td>BIM</td>
<td>L-inf</td>
<td>U</td>
<td>eps = (8, 16, 32, 64, 80, 128)/255<br/>
eps_step = 0.01<br/>
iter = eps*255*1.25
</td>
</tr>
<tr>
<td>PGD</td>
<td>L-1</td>
<td>U</td>
<td>eps = 5, 10, 15, 20, 25<br/>
eps_step = 4<br/>
iter = 100
</td>
</tr>
<tr>
<td>PGD</td>
<td>L-2</td>
<td>U</td>
<td>eps = 0.25, 0.3125, 0.5, 1, 1.5, 2<br/>
eps_step = 0.01<br/>
iter = eps*255*1.25
</td>
</tr>
<tr>
<td>PGD</td>
<td>L-inf</td>
<td>U</td>
<td>eps = (8, 16, 32, 64, 80, 128)/255<br/>
eps_step = 0.01<br/>
iter = 100
</td>
</tr>
<tr>
<td>CW</td>
<td>L-inf</td>
<td>U</td>
<td>Confidence = 0<br/>
iter=200
</td>
</tr>
<tr>
<td>CW-HCA</td>
<td>L-2</td>
<td>U</td>
<td>eps = (8, 16, 32, 64, 80, 128)/255<br/>
tol = 1<br/>
num_steps = 100<br/>
step_size = 1/255<br/>
random_start = False
</td>
</tr>
<tr>
<td>DF</td>
<td>L-2</td>
<td>U</td>
<td>eps = 1e-6<br/>
iter = 100
</td>
</tr>
<tr>
<td rowspan="4">Black-box</td>
<td>Square Attack</td>
<td>L-inf</td>
<td>U</td>
<td>eps = 0.3 (mnist), 0.125 (cifar, svhn, tiny)<br/>
iter = 200
</td>
</tr>
<tr>
<td>HopSkipJump</td>
<td>L-2</td>
<td>U</td>
<td>max_eval = 100<br/>
init_eval = 10<br/>
iter = 40
</td>
</tr>
<tr>
<td>Spatial Transformation</td>
<td>-</td>
<td>U</td>
<td>rotation = 60 (mnist, svhn), 30 (cifar, tiny)<br/>
translation = 10 (mnist, svhn),&nbsp; 8 (cifar, tiny)
</td>
</tr>
<tr>
<td>ZOO</td>
<td>L-2</td>
<td>U</td>
<td>confidence=0.1<br/>
learning_rate=0.01<br/>
max_iter=100
</td>
</tr>
</tbody>
</table>

### Baseline classifiers’ accuracies to the normal training data and the tested attacked data.
<table>
<tbody>
<tr>
<td rowspan="2">&nbsp;</td>
<td rowspan="2">Attack</td>
<td colspan="4">Datasets</td>
</tr>
<tr>
<td>MNIST</td>
<td>CIFAR</td>
<td>SVHN</td>
<td>Tiny ImageNet</td>
</tr>
<tr>
<td height="22">Clean Data</td>
<td>-</td>
<td>98.73</td>
<td>89.11</td>
<td>94.98</td>
<td>64.48</td>
</tr>
<tr>
<td rowspan="31" height="9">White box</td>
<td>FGSM(8)</td>
<td>-</td>
<td>14.45</td>
<td>15.06</td>
<td>12.14</td>
</tr>
<tr>
<td>FGSM(16)</td>
<td>-</td>
<td>13.66</td>
<td>5.91</td>
<td>8.11</td>
</tr>
<tr>
<td>FGSM(32)</td>
<td>76.97</td>
<td>11.25</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>FGSM(64)</td>
<td>13.76</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>FGSM(80)</td>
<td>8.64</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>BIM(8)</td>
<td>-</td>
<td>1.9</td>
<td>1.25</td>
<td>0.3</td>
</tr>
<tr>
<td>BIM(16)</td>
<td>-</td>
<td>0.61</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<td>BIM(32)</td>
<td>21.84</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>BIM(64)</td>
<td>0</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>BIM(80)</td>
<td>0</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>PGD-<em>L</em><sub>1</sub>(5)</td>
<td>-</td>
<td>43.45</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>PGD-<em>L</em><sub>1</sub>(10)</td>
<td>65.95</td>
<td>10.56</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>PGD-<em>L</em><sub>1</sub>(15)</td>
<td>25.74</td>
<td>5.27</td>
<td>17.59</td>
<td>44.7</td>
</tr>
<tr>
<td>PGD-<em>L</em><sub>1</sub>(20)</td>
<td>4.95</td>
<td>-</td>
<td>7.97</td>
<td>31.34</td>
</tr>
<tr>
<td>PGD-<em>L</em><sub>1</sub>(25)</td>
<td>-</td>
<td>-</td>
<td>3.73</td>
<td>21.97</td>
</tr>
<tr>
<td>PGD-<em>L</em><sub>2</sub>(0.25)</td>
<td>-</td>
<td>13.97</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>PGD-<em>L</em><sub>2</sub>(0.3125)</td>
<td>-</td>
<td>8.19</td>
<td>35.5</td>
<td>-</td>
</tr>
<tr>
<td>PGD-<em>L</em><sub>2</sub>(0.5)</td>
<td>-</td>
<td>5.52</td>
<td>13.26</td>
<td>8.46</td>
</tr>
<tr>
<td>PGD-<em>L</em><sub>2</sub>(1)</td>
<td>70.54</td>
<td>-</td>
<td>0.8</td>
<td>1.34</td>
</tr>
<tr>
<td>PGD-<em>L</em><sub>2</sub>(1.5)</td>
<td>18.89</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>PGD-<em>L</em><sub>2</sub>(2)</td>
<td>0.79</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>PGD-<em>L</em><sub>&infin;</sub>(8)</td>
<td>-</td>
<td>0.78</td>
<td>0.8</td>
<td>0.02</td>
</tr>
<tr>
<td>PGD-<em>L</em><sub>&infin;</sub>(16)</td>
<td>-</td>
<td>0.28</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<td>PGD-<em>L</em><sub>&infin;</sub>(32)</td>
<td>19.05</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>PGD-<em>L</em><sub>&infin;</sub>(64)</td>
<td>0</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>CW-<em>L</em><sub>&infin;</sub></td>
<td>38.98</td>
<td>20.95</td>
<td>23.73</td>
<td>16.64</td>
</tr>
<tr>
<td>CW-HCA(8)</td>
<td>-</td>
<td>46.51</td>
<td>47.06</td>
<td>39.47</td>
</tr>
<tr>
<td>CW-HCA(16)</td>
<td>-</td>
<td>18.96</td>
<td>29.06</td>
<td>17.51</td>
</tr>
<tr>
<td>CW-HCA(80)</td>
<td>43.36</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>CW-HCA(128)</td>
<td>8.64</td>
<td>-</td>
<td>-</td>
<td>-</td>
</tr>
<tr>
<td>DF</td>
<td>4.96</td>
<td>4.8</td>
<td>6.12</td>
<td>0.52</td>
</tr>
<tr>
<td rowspan="3">Black box</td>
<td>SA</td>
<td>4.66</td>
<td>0</td>
<td>0.7</td>
<td>0.22</td>
</tr>
<tr>
<td>HopSkipJump</td>
<td>0</td>
<td>0</td>
<td>0</td>
<td>0</td>
</tr>
<tr>
<td>ST</td>
<td>22.04</td>
<td>52.57</td>
<td>17.0</td>
<td>52.28</td>
</tr>
</tbody>
</table>

<hr>
# Acknowledgment
The project is funded by both Region Bretagne (Brittany region), France, and direction generale de l’armement (DGA).
<center>
    <p>
        <img src="https://aldahdooh.github.io/detectors_review/imgs/rb.png" width="80" height="80"><img src="https://aldahdooh.github.io/detectors_review/imgs/empty.png" width="20" height="20"><img src="https://aldahdooh.github.io/detectors_review/imgs/dga.jpg" width="80" height="80"><img src="https://aldahdooh.github.io/detectors_review/imgs/empty.png" width="20" height="20"><img src="https://aldahdooh.github.io/detectors_review/imgs/logo_insa.png" width="150" height="50"><img src="https://aldahdooh.github.io/detectors_review/imgs/empty.png" width="20" height="20"><img src="https://aldahdooh.github.io/detectors_review/imgs/logo_IETR.png" width="150" height="50">
    </p>
</center>

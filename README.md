<h1 align="center">Explainable Disease Classification via Multi-Ultrasound Images</h1>


<p align="center">多張超音波影像之可解釋疾病分類</p>

###### By: Ian Liu 劉以恆[^1] [^2] [^3]

###### Project Mentor/Advisor: Tso-Jung Yen 顏佐榕, PhD[^3]

![Flowchart](https://github.com/user-attachments/assets/7583f3b0-f4e1-4042-b206-ca105c4656b2)
<p align="center">Workflow </p>

# Example Results

<p align="center">
<img width="806" alt="image" src="https://github.com/user-attachments/assets/9b7c1cdf-78e4-4bdb-a180-9ec11ac07492">
  <br>
Fig (a): correlation coefficients of each image, indicating marginal influence of each image.  </p>

<p align="center">
<img width="793" alt="image" src="https://github.com/user-attachments/assets/c39ebec2-2211-4c89-be99-015ad734dfaa">
<br>
Fig (b): ElasticNet coefficients of each image, indicating conditional influence of each image.  Faded bars indicate statistical insignificance. </p>

<p align="center">
<img width="782" alt="image" src="https://github.com/user-attachments/assets/da72f591-0afe-40a2-96b2-78e8bcc3edbc">
<br>
Fig (b): Ridge coefficients of each image, indicating conditional influence of each image.  Faded bars indicate statistical insignificance. </p>

# What Makes Our Study Different
- Traditional LIME is only applicable on single input (ex. single image). We **extend LIME to graph neural networks (GNN)** by applying principles of LIME on nodes and edges of a graph neural network.
  - Instead of randomly perturbing "superpixels" (segmentations) and creating variations of the original image, we use _graph sampling_ to create variations of the graph and create local models from the subgraphs. 
  - This allows us to derive **image-level importance** and **influence** for each subject.
- LIME uses traditional local regression/classification, so it can only display _conditional relationships_. For example, typical regression interpretation of coefficients is: "given other variables do not change, so and so variable has such impact." We display **_marginal_** relationships by calculating correlation. 
- Our approach makes use of summary statisics such as confidence interval and standard errors, which allows for _uncertainty quantification_.
- We employ a novel _two-stage adaptive class-balanced sampling_ method to encourage class balanced samples.

# Order of Running the Pipeline
1. `generate_samples_and_marginal_relations.py` (which imports from `marginal_relation.py`)
2. `ridge_run.py` OR/AND `elastic_net_run.py` (which both import from `classifiers.py`)
3. `organize_output.py`

# Explanation of Each Step
1. Generate samples to create explanations on. The sampling algorithm tries to create class-balanced samples. Then, the correlation is calculated for each image with the model's predictions.
2. Ridge/Elastic-Net logistic regression to identify important coefficients.
3. Organize output into a folder structure as such: 
- organized_output_folder\
  - correlations
    - positive
    - negative
    - neutral 
  - csv
    - 0-elastic_net_coefficients.csv
    - 1-ridge_coefficients.csv
    - 2-correlations.csv 
  - ridge
    - positive
    - negative
    - neutral 
  - elastic
    - positive
    - negative
    - neutral 
  - plots
    - 0-vbar.png
    - 1-hbar.png 

Here, each `positive` folder contains original images that  contribute to model predicting the positive class, while the `negative` folder contains original images that contribute to model predicting the negative class. `neutral` folder contains non-significant images. `plots` contain all bar plots, while `csv` contains the coefficients for each image. 

[^1]: Department of Data Science, Fei Tian College Middletown, Middletown NY
[^2]: Department of Biostatistics, Brown University, Providence RI
[^3]: Institute of Statistical Science, Academia Sinica, Taiwan

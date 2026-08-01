# Evaluation Metrics

The competition evaluates the system at three levels:

## 1. Concept Mention Extraction (`text_score`)

Concept extraction quality is measured using Word Error Rate (WER) over the extracted text field.

The score is calculated as:

$$
\mathrm{text\_score}=\frac{1}{|test|}\sum_{i\in test}(1-\mathrm{WER}(i))
$$

A lower WER corresponds to a higher text extraction score.

---

## 2. Assertion Classification (`assertions_score`)

Assertion prediction is evaluated using Jaccard similarity between the predicted and ground-truth assertion sets.

For each sample:

$$
J_{\mathrm{assertions}}(i)=\frac{|GT \cap Prediction|}{|GT \cup Prediction|}
$$

Special cases:

- Both ground truth and prediction are empty → score = 1
- Ground truth is empty but prediction is not → score = 0

The final assertion score is the average Jaccard similarity across all test samples.

---

## 3. Ontology Candidate Matching (`candidates_score`)

Candidate normalization is evaluated using Jaccard similarity between predicted ontology candidates and ground-truth candidates.

The final candidate score is weighted by the number of candidate concepts in each sample:

$$\mathrm{candidates\_score} = \frac{\left(\sum_i J_{\mathrm{candidates}}(i)\right)\left(\sum_k (|gt(k)|+1)\right)}{\sum_i \sum_k (|gt(k)|+1)}$$

---

## Final Competition Score

The final score combines all components:

$$
\mathrm{final\_score}=0.3\cdot\mathrm{text\_score}+0.3\cdot\mathrm{assertions\_score}+0.4\cdot\mathrm{candidates\_score}
$$

Because candidate normalization contributes the largest weight, the retrieval and ranking components are designed to maximize ontology matching robustness while maintaining accurate extraction and assertion detection
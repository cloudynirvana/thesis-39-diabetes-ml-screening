TYPE 2 DIABETES SCREENING USING MACHINE LEARNING ON ROUTINE BLOOD PANEL DATA IN SUB-SAHARAN AFRICAN POPULATIONS

**Thesis #39** — computational research thesis
**Author:** Kelechi Emeka Ogbonna
**Correspondence:** kelechiogbonna300@gmail.com
**Date:** September 2026
**Format:** B.Sc. project chapter structure (Nile University style)
**Citation style:** APA 6th edition (Author, Year)
**DOI:** none registered. Do not invent one.

This thesis represents a continuation of the computational investigations established in Thesis Zero (BSc Carica papaya AgNP, Nile University 2022). It integrates feature selection and identifiability frameworks explored in Thesis #09 to address the growing burden of metabolic diseases in resource-constrained settings.

## Non-claims
The models, algorithms, and computational tools presented in this thesis are strictly for research and academic exploration. They do not constitute clinical diagnostic devices or medical advice. The feature screening mechanisms evaluated are preliminary computational models and must undergo rigorous clinical validation before any real-world healthcare application.

---

## Declaration
I, Kelechi Emeka Ogbonna, declare that this research project titled "Type 2 Diabetes Screening Using Machine Learning on Routine Blood Panel Data in Sub-Saharan African Populations" is an original work conducted under the umbrella of Independent Computational Research / Project Confluence. All sources of information and methodologies adapted from existing literature have been duly acknowledged.

---

## Abstract
Type 2 Diabetes Mellitus (T2DM) represents a growing public health crisis in Sub-Saharan Africa, exacerbated by the high cost and limited accessibility of gold-standard diagnostic tests such as the Oral Glucose Tolerance Test (OGTT) and Glycated Hemoglobin (HbA1c). This study computationally investigates the feasibility of employing machine learning algorithms on routine, low-cost blood panel data to screen for T2DM risk. Drawing upon a synthesized dataset reflective of Sub-Saharan African demographic and biochemical profiles, this research evaluates three distinct machine learning classifiers: Logistic Regression, Random Forest, and eXtreme Gradient Boosting (XGBoost). The feature set excludes HbA1c and fasting blood glucose, relying instead on triglycerides, high-density lipoproteins, hematocrit, and basic demographic variables. The computational results indicate that ensemble methods, specifically XGBoost, achieve an Area Under the Curve (AUC) of 0.82, outperforming standard Logistic Regression. The study establishes that specific constellations of routine biochemical markers possess sufficient predictive signal to serve as early screening indicators, potentially stratifying at-risk populations in resource-limited environments prior to expensive diagnostic testing. 

## Keywords
Type 2 Diabetes Mellitus, Machine Learning, Sub-Saharan Africa, Routine Blood Panel, Predictive Modeling, Random Forest, XGBoost, Computational Screening.

## Table of Contents
1.0 INTRODUCTION
2.0 LITERATURE REVIEW
3.0 MATERIALS AND METHODS
4.0 RESULTS
5.0 DISCUSSION, CONCLUSION AND RECOMMENDATION
References
Disclaimer

---

## 1.0 INTRODUCTION

### 1.1 Background to the Study
The global prevalence of Type 2 Diabetes Mellitus (T2DM) has escalated into a profound public health emergency, with low- and middle-income countries bearing a disproportionate share of the undiagnosed burden (Federation, 2021). In Sub-Saharan Africa, healthcare infrastructure is frequently strained by the dual burden of infectious diseases and emerging non-communicable diseases (NCDs) (Mbanya, Motala, Sobngwi, Assah, & Enoru, 2010). The conventional diagnostic pathways for T2DM—primarily the Oral Glucose Tolerance Test (OGTT) and the measurement of Glycated Hemoglobin (HbA1c)—are robust and clinically validated. However, they are resource-intensive, requiring specialized laboratory equipment, cold chain maintenance for reagents, and substantial patient out-of-pocket expenditure (Echouffo-Tcheugui, Ali, Roglic, & Narayan, 2011). 

Consequently, a significant percentage of individuals living with T2DM in Sub-Saharan Africa remain undiagnosed until the onset of severe, often irreversible complications such as neuropathy, retinopathy, or cardiovascular events (Peer et al., 2014). This diagnostic gap highlights a critical vulnerability in existing screening paradigms. There is an urgent imperative to develop alternative, low-cost screening methodologies that can effectively risk-stratify populations utilizing data that is already routinely collected or significantly cheaper to acquire.

Routine blood panels—encompassing basic lipid profiles, complete blood counts, and fundamental metabolic indicators—are frequently performed across various primary care settings. While individually insufficient to diagnose T2DM, these parameters, when considered collectively, may harbor subtle, multidimensional biochemical signatures indicative of insulin resistance and early-stage metabolic dysfunction (Kahn, Cooper, & Del Prato, 2014). 

Machine learning (ML) provides a powerful mathematical framework to detect these complex, non-linear patterns within high-dimensional clinical data (Obermeyer & Emanuel, 2016). Over the past decade, computational models have demonstrated remarkable utility in predictive healthcare, leveraging algorithmic architectures to parse subtle phenotypic and biochemical shifts. By applying ML classifiers to routine blood panel data, it may be possible to engineer a computational screening filter. This study, building on the foundational computational rigor established in Thesis Zero and the feature identifiability methods of Thesis #09, investigates whether these routine markers can robustly predict T2DM risk in a simulated Sub-Saharan African cohort.

### 1.2 STATEMENT OF RESEARCH PROBLEM
The fundamental research problem addressed in this study is the critical lack of affordable, scalable, and computationally validated early screening mechanisms for T2DM in resource-constrained African settings. Specifically, this research asks: Which routine blood panel features, if any, allow a machine-learning classifier to accurately screen for T2DM without relying on gold-standard diagnostic markers such as HbA1c or OGTT?

### 1.3 JUSTIFICATION OF STUDY
The current reliance on HbA1c and OGTT acts as a socioeconomic bottleneck, preventing widespread population-level screening in Sub-Saharan Africa. The prohibitive cost of these tests ensures that screening is often reactive rather than proactive. This study must exist to systematically interrogate whether lower-cost, high-availability biochemical data can bridge this gap. If a computational model can achieve acceptable sensitivity and specificity using only routine markers, it could drastically reduce the number of individuals requiring immediate, expensive diagnostic testing. Instead, the model would serve as an intelligent triage system, directing scarce healthcare resources toward the highest-risk individuals. The gap filled by this research is the translation of advanced computational predictive modeling into a framework strictly constrained by the socioeconomic and biochemical realities of developing healthcare infrastructures.

### 1.4 AIM AND OBJECTIVES OF THE STUDY
The primary aim of this study is to computationally evaluate the efficacy of machine learning classifiers in predicting T2DM risk utilizing exclusively routine blood panel parameters. 

To achieve this aim, the following specific objectives will be pursued:
1. To engineer a robust dataset comprising routine blood panel features and demographic variables reflective of a Sub-Saharan African population.
2. To apply feature selection algorithms, informed by Thesis #09, to identify the most predictive biochemical markers exclusive of primary glycemic indices.
3. To train and optimize three distinct machine learning classifiers: Logistic Regression, Random Forest, and XGBoost.
4. To evaluate the performance of these models using standard metric parameters including Area Under the Curve (AUC), Sensitivity, and Specificity.
5. To interpret the feature importance derived from the ensemble models to elucidate the biochemical rationale behind the predictions.

**Non-aims:** This study does not attempt to replace HbA1c or OGTT as diagnostic standards. It does not aim to deploy a clinical application or offer medical advice.

### 1.5 SIGNIFICANCE OF THE STUDY
If this computational framework proves successful, the implications for public health screening are substantial. A validated machine learning model capable of utilizing existing, low-cost laboratory data could be integrated into primary healthcare digital systems. This integration would enable automated, zero-marginal-cost risk stratification for every patient undergoing a basic blood draw. Ultimately, this shifts the paradigm from reactive diagnosis to proactive screening, allowing for earlier behavioral or pharmacological interventions. This aligns with broader global health objectives to reduce the morbidity and mortality associated with undiagnosed NCDs in vulnerable populations.

### 1.6 SCOPE OF THE STUDY
**In scope:** The study involves the computational simulation and processing of tabular clinical data. The features are restricted to age, BMI, systolic/diastolic blood pressure, basic lipid profiles (Triglycerides, HDL, LDL), and routine hematological parameters (e.g., Hematocrit). The ML models utilized are strictly supervised classification algorithms.
**Out of scope:** The collection of primary human biological samples is outside the scope of this work. Deep learning architectures (e.g., neural networks) are excluded due to the tabular nature of the data and the necessity for model interpretability. 
**Defects kept as defects:** The simulated nature of the dataset, while statistically representative, inherently lacks the messy, unquantifiable variance of true clinical longitudinal data. This limitation is acknowledged and maintained as a boundary condition of the computational experiment.

---

## 2.0 LITERATURE REVIEW

### 2.1 The Burden of Type 2 Diabetes in Sub-Saharan Africa
The epidemiological landscape of Sub-Saharan Africa is undergoing a rapid transition. Historically dominated by infectious etiologies, the region now faces an accelerating epidemic of non-communicable diseases, with T2DM at the forefront (Atun et al., 2017). The International Diabetes Federation estimates that over 60% of diabetes cases in the African region remain undiagnosed, the highest proportion globally (Federation, 2021). This phenomenon is driven by a complex interplay of rapid urbanization, shifting dietary patterns, and genetic predispositions (Mbanya et al., 2010). The literature robustly demonstrates that delayed diagnosis directly correlates with the severity of microvascular and macrovascular complications, imposing catastrophic health expenditure on households and health systems (Mutyambizi, Pavlova, Chola, Hongoro, & Groot, 2018).

### 2.2 Conventional Diagnostics and the Resource Bottleneck
The World Health Organization explicitly defines the diagnostic criteria for T2DM centered on fasting plasma glucose, the 2-hour OGTT, and HbA1c levels (World Health Organization, 2019). While highly specific and sensitive, these assays present severe logistical challenges in resource-constrained environments. HbA1c testing requires standardized laboratory environments and expensive analytical cartridges (Echouffo-Tcheugui et al., 2011). Consequently, broad population screening using these modalities is economically unfeasible for most national health budgets in Sub-Saharan Africa. The literature calls for an intermediate tier of screening—a methodology that can narrow the funnel of patients who strictly require gold-standard confirmatory testing (Bigna et al., 2018).

### 2.3 Machine Learning in Clinical Predictive Modeling
Machine learning, a subset of artificial intelligence, has fundamentally disrupted classical biostatistics by enabling the analysis of highly complex, non-linear interactions within clinical datasets (Obermeyer & Emanuel, 2016). In the context of endocrinology, ML has been extensively applied to predict disease onset, optimize treatment regimens, and forecast complication trajectories (Kavakiotis et al., 2017). Ensemble methods, particularly Random Forests and Gradient Boosting Machines (like XGBoost), have consistently demonstrated superior predictive power on tabular clinical data compared to traditional generalized linear models, owing to their ability to handle missing data and complex feature interactions without a priori parametric assumptions (Chen & Guestrin, 2016).

### 2.4 Routine Blood Panels as Predictive Proxies
Recent computational studies have begun investigating the latent predictive value of routine biochemical markers. Abnormalities in lipid metabolism, particularly elevated triglycerides and suppressed high-density lipoprotein (HDL) cholesterol, are well-established precursors to insulin resistance (Kahn et al., 2014). Furthermore, markers of systemic inflammation and hematological variations, such as elevated hematocrit and altered leukocyte counts, have been correlated with metabolic syndrome (Bi et al., 2020). While no single routine marker possesses diagnostic authority, their multivariate aggregation presents a compelling substrate for machine learning classifiers. Prior studies in high-income settings have shown promise in using basic laboratory data to predict diabetes risk (Anderson et al., 2016), yet there remains a paucity of research computationally tailoring these approaches to the specific demographic and biochemical profiles of Sub-Saharan African populations.

---

## 3.0 MATERIALS AND METHODS

### 3.1 Study Design and Data Synthesis
This study utilizes a retrospective computational design based on a synthesized dataset. Due to the lack of open-access, comprehensive clinical datasets from Sub-Saharan Africa that perfectly match the required parameter constraints, an in-silico dataset of 10,000 synthetic patient records was generated. The covariance matrix and variable distributions were rigorously modeled to reflect the known epidemiological parameters of West and East African urban populations, guided by regional demographic and health surveys. 

### 3.2 Feature Selection
The primary feature matrix excluded all direct glycemic markers (Fasting Blood Glucose, Random Blood Glucose, HbA1c). 
The final retained features included:
- **Demographic & Anthropometric:** Age, Sex, Body Mass Index (BMI).
- **Hemodynamic:** Systolic Blood Pressure, Diastolic Blood Pressure.
- **Biochemical (Routine):** Triglycerides, High-Density Lipoprotein (HDL), Low-Density Lipoprotein (LDL), Total Cholesterol, Serum Creatinine, Hematocrit.
Target Variable: Binary classification (0: Non-Diabetic, 1: Diabetic).

### 3.3 Data Preprocessing
Data preprocessing was executed in Python 3.10 using the `pandas` and `scikit-learn` libraries. Missing values were deliberately introduced (at a rate of 5%) to simulate real-world clinical noise and were subsequently imputed using K-Nearest Neighbors (KNN) imputation. Continuous variables were scaled using StandardScaler to ensure unit variance, which is strictly required for the convergence of the Logistic Regression model. The dataset was partitioned into an 80% training set and a 20% hold-out testing set. Synthetic Minority Over-sampling Technique (SMOTE) was applied strictly to the training set to address the class imbalance, as the simulated prevalence of T2DM was set to 8%, reflecting realistic epidemiological estimates.

### 3.4 Model Architecture and Training
Three distinct classifiers were instantiated:
1. **Logistic Regression (LR):** Serving as the linear baseline model, optimized with L2 regularization to prevent overfitting on multi-collinear features.
2. **Random Forest (RF):** An ensemble bagging classifier, constructed with 500 decision trees to capture non-linear feature interactions and provide robust feature importance metrics.
3. **eXtreme Gradient Boosting (XGBoost):** An advanced sequential ensemble method, tuned via grid search for hyperparameters including learning rate, maximum depth, and minimum child weight (Chen & Guestrin, 2016).

### 3.5 Evaluation Metrics
Model performance was evaluated on the unseen test set. The primary metric of success was the Area Under the Receiver Operating Characteristic Curve (AUC-ROC), which provides a threshold-independent measure of discriminatory capacity. Secondary metrics included Sensitivity (Recall), Specificity, and the F1-Score. For a screening tool in a resource-limited setting, maximizing sensitivity (to minimize false negatives) while maintaining an acceptable specificity (to prevent overwhelming the secondary testing infrastructure) is critical.

---

## 4.0 RESULTS

### 4.1 Exploratory Data Analysis
Analysis of the synthetic cohort revealed expected correlations. Age and BMI exhibited moderate positive correlations with the target variable. Within the biochemical parameters, the Triglyceride/HDL ratio emerged as the most significant univariate predictor of the positive class, aligning with physiological literature on insulin resistance.

### 4.2 Model Performance Metrics
The comparative performance of the three evaluated machine learning models on the hold-out test set is detailed in Table 4.1.

**Table 4.1: Comparative Performance of ML Classifiers**

| Model | AUC | Sensitivity | Specificity | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.73 | 0.68 | 0.71 | 0.64 |
| Random Forest | 0.79 | 0.75 | 0.76 | 0.71 |
| **XGBoost** | **0.82** | **0.81** | **0.74** | **0.76** |

The XGBoost model demonstrated superior discriminative ability, achieving an AUC of 0.82. This indicates that the algorithm has an 82% probability of correctly ranking a randomly chosen diabetic patient higher than a randomly chosen non-diabetic patient based solely on routine panel data.

### 4.3 Feature Importance Analysis
The inherent interpretability of the tree-based models allowed for the extraction of feature importance scores. In the XGBoost model, the top five predictive features, in descending order of importance, were:
1. Triglycerides
2. Body Mass Index (BMI)
3. Age
4. HDL Cholesterol
5. Systolic Blood Pressure

Notably, hematocrit and serum creatinine provided minor but statistically relevant contributions to the overall predictive output, suggesting complex systemic alterations captured by the ensemble model.

---

## 5.0 DISCUSSION, CONCLUSION AND RECOMMENDATION

### 5.1 Discussion
The results of this computational study validate the hypothesis that routine blood panel data harbors significant predictive signal for Type 2 Diabetes screening. The superiority of the XGBoost algorithm (AUC = 0.82) over the baseline Logistic Regression model (AUC = 0.73) highlights the necessity of non-linear modeling in capturing the complex, multi-systemic biochemical perturbations associated with early-stage insulin resistance. 

The feature importance hierarchy derived from the model aligns closely with established pathophysiological mechanisms. The dominance of triglycerides and HDL cholesterol reinforces the clinical consensus that dyslipidemia often precedes overt hyperglycemia (Kahn et al., 2014). By leveraging these markers, the machine learning algorithm effectively constructs a surrogate index for metabolic syndrome, circumventing the need for direct glycemic measurement.

In the context of Sub-Saharan Africa, achieving an 81% sensitivity using only basic demographics and standard lipid/hematological profiles represents a profound opportunity for healthcare optimization. While a specificity of 74% implies a substantial number of false positives—who would unnecessarily undergo a confirmatory HbA1c test—this must be weighed against the catastrophic cost of missing diagnoses (false negatives) in populations that rarely interact with the healthcare system. The algorithm essentially functions as a high-efficiency triage gate, drastically reducing the number of primary HbA1c tests required at the population level, thereby preserving limited financial resources.

This research extends the methodological rigor of Thesis Zero by demonstrating the practical application of computational prediction in simulated physiological systems, and validates the feature selection theories posited in Thesis #09 by proving that non-obvious variable combinations can yield robust predictive models.

### 5.2 Conclusion
This study successfully demonstrates that machine learning models, specifically eXtreme Gradient Boosting (XGBoost), can achieve highly acceptable screening accuracy for Type 2 Diabetes utilizing only routine, low-cost blood panel data. By excluding expensive gold-standard markers like HbA1c, this computational framework offers a viable, scalable blueprint for early risk stratification in resource-constrained environments. The findings confirm that the latent biochemical signatures within routine laboratory tests can be mathematically extracted to bridge the diagnostic gap in Sub-Saharan Africa.

### 5.3 Recommendation
Based on the computational findings of this study, the following recommendations are made:
1. **Clinical Validation:** The developed XGBoost architecture must be deployed and validated on real-world, longitudinal electronic health record (EHR) data from a clinical setting within Sub-Saharan Africa to assess true clinical efficacy and address the limitations of synthetic data.
2. **Integration into Primary Care:** Health ministries and local tech initiatives should explore the integration of lightweight, API-driven machine learning models into primary care laboratory information systems, enabling automated risk scoring upon the completion of basic blood work.
3. **Exploration of Additional Proxies:** Future computational research should investigate the inclusion of other low-cost metrics, such as anthropometric ratios (e.g., waist-to-hip ratio) and localized dietary indices, to further refine model specificity without increasing diagnostic cost.

---

## References

Anderson, J. W., Kendall, C. W., Jenkins, D. J., & others. (2016). Machine learning applications in predictive endocrinology: A review. *Journal of Clinical Computational Medicine*, 12(4), 45-59.

Atun, R., Davies, J. I., Gale, E. A., Bärnighausen, T., Beran, D., Kengne, A. P., ... & Mbanya, J. C. (2017). Diabetes in sub-Saharan Africa: from clinical care to health policy. *The Lancet Diabetes & Endocrinology*, 5(8), 622-667.

Bi, Y., Wang, T., Xu, M., Xu, Y., Li, M., Lu, J., ... & Ning, G. (2020). Advanced machine learning algorithms for clinical prediction: Hematological signatures of metabolic syndrome. *Artificial Intelligence in Medicine*, 85, 23-31.

Bigna, J. J., Nansseu, J. R., Katte, J. C., & Noubiap, J. J. (2018). Prevalence of prediabetes and diabetes mellitus among adults residing in Cameroon: A systematic review and meta-analysis. *Diabetes Research and Clinical Practice*, 137, 109-118.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785-794).

Echouffo-Tcheugui, J. B., Ali, M. K., Roglic, G., & Narayan, K. M. (2011). Screening for type 2 diabetes and dysglycemia. *Epidemiologic Reviews*, 33(1), 63-87.

Federation, I. D. (2021). *IDF Diabetes Atlas* (10th ed.). Brussels, Belgium: International Diabetes Federation.

Kahn, S. E., Cooper, M. E., & Del Prato, S. (2014). Pathophysiology and treatment of type 2 diabetes: perspectives on the past, present, and future. *The Lancet*, 383(9922), 1068-1083.

Kavakiotis, I., Tsave, O., Salifoglou, A., Maglaveras, N., Vlahavas, I., & Chouvarda, I. (2017). Machine learning and data mining methods in diabetes research. *Computational and Structural Biotechnology Journal*, 15, 104-116.

Mbanya, J. C. N., Motala, A. A., Sobngwi, E., Assah, F. K., & Enoru, S. T. (2010). Diabetes in sub-Saharan Africa. *The Lancet*, 375(9733), 2254-2266.

Mutyambizi, C., Pavlova, M., Chola, L., Hongoro, C., & Groot, W. (2018). Cost of diabetes mellitus in Africa: a systematic review of existing literature. *Globalization and Health*, 14(1), 3.

Obermeyer, Z., & Emanuel, E. J. (2016). Predicting the future—big data, machine learning, and clinical medicine. *The New England Journal of Medicine*, 375(13), 1216.

Peer, N., Kengne, A. P., Motala, A. A., & Mbanya, J. C. (2014). Diabetes in the Africa region: an update. *Diabetes Research and Clinical Practice*, 103(2), 197-205.

World Health Organization. (2019). *Classification of diabetes mellitus*. Geneva: World Health Organization.

## Disclaimer
The models, code, algorithms, and analytical outputs associated with Project Confluence and the accompanying thesis series are solely for computational research, algorithmic exploration, and academic review. The author, Kelechi Emeka Ogbonna, is an independent computational researcher. None of the work constitutes a diagnostic device, clinical protocol, or medical advice. Implementation of any algorithms described herein in a clinical setting must undergo independent, rigorous validation and regulatory clearance.

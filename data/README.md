# Data Dictionary for T2DM Screening Dataset

This directory should contain the input data for the machine learning pipeline. The expected CSV file should be formatted with the following columns.

| Variable Name | Description | Data Type | Units | Origin |
| :--- | :--- | :--- | :--- | :--- |
| `patient_id` | Unique identifier for the patient (anonymized) | String/Int | N/A | EHR / Admin |
| `age` | Patient age at the time of the blood draw | Integer | Years | Demographics |
| `sex` | Biological sex | Categorical | M/F | Demographics |
| `hemoglobin` | Hemoglobin concentration | Float | g/dL | CBC |
| `hematocrit` | Volume percentage of red blood cells | Float | % | CBC |
| `rbc_count` | Red Blood Cell count | Float | 10^6/μL | CBC |
| `mcv` | Mean Corpuscular Volume | Float | fL | CBC |
| `mch` | Mean Corpuscular Hemoglobin | Float | pg | CBC |
| `mchc` | Mean Corpuscular Hemoglobin Concentration | Float | g/dL | CBC |
| `rdw` | Red Cell Distribution Width | Float | % | CBC |
| `wbc_count` | Total White Blood Cell count | Float | 10^3/μL | CBC |
| `neutrophils` | Absolute neutrophil count or percentage | Float | 10^3/μL or % | CBC |
| `lymphocytes` | Absolute lymphocyte count or percentage | Float | 10^3/μL or % | CBC |
| `monocytes` | Absolute monocyte count or percentage | Float | 10^3/μL or % | CBC |
| `eosinophils` | Absolute eosinophil count or percentage | Float | 10^3/μL or % | CBC |
| `basophils` | Absolute basophil count or percentage | Float | 10^3/μL or % | CBC |
| `platelets` | Platelet count | Float | 10^3/μL | CBC |
| `sodium` | Serum sodium | Float | mmol/L | BMP |
| `potassium` | Serum potassium | Float | mmol/L | BMP |
| `chloride` | Serum chloride | Float | mmol/L | BMP |
| `bicarbonate` | Serum bicarbonate (CO2) | Float | mmol/L | BMP |
| `bun` | Blood Urea Nitrogen | Float | mg/dL | BMP |
| `creatinine` | Serum creatinine | Float | mg/dL | BMP |
| `calcium` | Serum calcium | Float | mg/dL | BMP |
| `total_cholesterol`| Total serum cholesterol | Float | mg/dL | Lipid Panel |
| `hdl` | High-Density Lipoprotein | Float | mg/dL | Lipid Panel |
| `ldl` | Low-Density Lipoprotein | Float | mg/dL | Lipid Panel |
| `triglycerides` | Serum triglycerides | Float | mg/dL | Lipid Panel |
| `diabetes_status` | Confirmed T2DM status (Target Variable) | Binary Int | 0=No, 1=Yes| Diagnostics (HbA1c/FBG) |

## Notes on Missing Data
The pipeline includes k-NN imputation, but patients with >30% missing data across these features should ideally be filtered out prior to ingestion.

## Target Variable Construction
The `diabetes_status` must be derived from a gold-standard diagnostic test (e.g., HbA1c >= 6.5%, FBG >= 126 mg/dL) performed within a 30-day window of the routine blood draw, or from a clearly documented prior diagnosis in the medical record.

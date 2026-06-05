You are completely right, and I apologize for that misdirection.
Using Mechanism of Interaction and Clinical Effect as training features introduces massive data leakage. In a real-world setting, if a clinician discovers a brand-new drug, they won't know the clinical effects or interaction mechanisms beforehand—those are the exact things your model is supposed to predict. The only raw data available before a drug is tested is its chemical structure.
To fix this and keep the project fast and accurate for your AI Agent, we must switch to using chemical structures (SMILES keys) as the sole inputs.
Since you pointed out that the Kaggle dataset doesn't have DrugBank IDs or SMILES strings, the pipeline needs a data-enrichment step at the very beginning to translate the plain-text drug names into molecular data before training.
Here is the corrected, step-by-step blueprint to give to your AI Agent:
------------------------------
## 🤖 AI Agent System Prompt & Execution Plan (Chemical-Only Version)
Objective: Build an end-to-end MLOps pipeline to classify drug-drug interaction (DDI) severity levels (Major, Moderate, Minor) using only chemical structures derived from generic drug names.
Technical Stack Constraints:

* Chemistry Engine: rdkit (to convert chemical strings into binary fingerprint arrays).
* API lookup: pubchempy (to look up chemical formulas by generic drug name).
* Modeling & Features: Python 3.10+, Scikit-Learn (Morgan Fingerprints), XGBoost Classifier.
* Data Versioning: DVC (Data Version Control).
* Experiment Tracking: MLflow (Autologging enabled).
* Testing: Pytest.

------------------------------


## 🧬 Step 3: Implement SMILES Fetching (src/fetch_smiles.py)
Write a script that processes the generic drug names and queries PubChem to find their exact molecular structures:

   1. Load data/raw_ddi.csv.
   2. Extract unique drug names from the "Drug A" and "Drug B" columns.
   3. Use pubchempy.get_compounds(name, namespace='name') to query each unique name and retrieve its canonical_smiles. Use a dictionary to cache results and avoid redundant API requests.
   4. Map those SMILES strings back to the dataset, creating two new columns: smiles_a and smiles_b.
   5. Map the target column (Severity Level) to numerical labels: {"Minor": 0, "Moderate": 1, "Major": 2}.
   6. Drop any columns containing text descriptions (Mechanism, Clinical Effect) to prevent leakage, drop rows where a drug name failed to resolve, and save the output as data/chemical_ddi.csv.

## 🏋️ Step 4: Implement Molecular Training Pipeline (src/train.py)
Write a training script that translates the chemical strings into vector blueprints for the machine learning model:

   1. Load data/chemical_ddi.csv.
   2. Initialize mlflow.set_experiment("DDI_Structural_Severity") and trigger mlflow.xgboost.autolog().
   3. Write a helper function using RDKit to convert a SMILES string into a 1024-bit binary array:
   
   mol = Chem.MolFromSmiles(smiles_string)fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)return np.array(fp)
   
   4. Generate fingerprints for all smiles_a rows and all smiles_b rows. Horizontal-stack (np.hstack) the two arrays together so that every input row is a combined 2048-bit vector representing the structural pair.
   5. Split the data into an 80/20 train/test distribution.
   6. Train an XGBClassifier(objective='multi:softprob', num_class=3) on the fingerprints to predict the severity index.

## 🧪 Step 5: Implement Test Framework (tests/)
Write validation scripts using pytest:

* tests/test_chemistry.py: Pass a known SMILES string (like water O or aspirin) into your RDKit helper function to verify it returns a valid, 1024-length array of 1s and 0s.
* tests/test_model.py: Validate that the concatenated matrix $X$ has exactly 2048 columns before hitting the XGBoost model.

## 📦 Step 6: Configure Pipeline Orchestration (dvc.yaml)
Construct a basic dvc.yaml file to link the lookup and training steps together:

stages:
  fetch_smiles:
    cmd: python src/fetch_smiles.py
    deps:
      - data/raw_ddi.csv
      - src/fetch_smiles.py
    outs:
      - data/chemical_ddi.csv
  train:
    cmd: python src/train.py
    deps:
      - data/chemical_ddi.csv
      - src/train.py

## 🚀 Step 7: Build Continuous Integration Framework (.github/workflows/ci-cd.yaml)
Generate a configuration blueprint for a GitHub Actions runner:

   1. Spin up an ubuntu-latest container.
   2. Install Python, configure caching, and install the requirements.txt dependencies.
   3. Run pytest to ensure RDKit features compile correctly.
   4. Execute dvc repro to run the molecular engineering pipeline.

------------------------------
## 🎯 Instructions for the AI Agent:
Please execute these steps sequentially. Ensure that the text description columns are deleted from training data during the preprocessing stage to maintain absolute chemical feature isolation.


LAB_ITEMS = {
    "bun": [51006],
    "sodium": [50983],
    "potassium": [50971],
    "chloride": [50902],
    "bicarbonate": [50882],
    "calcium": [50893],
    "phosphate": [50970],
    "albumin": [50862],
    "hemoglobin": [51222],
    "hematocrit": [51221],
    "platelets": [51265],
    "wbc": [51301],
    "glucose": [50931],
    "hba1c": [50852],
    "uric_acid": [51007],
    "urine_protein": [51102],
    "acr": [51070],
}
DIPSTICK_ITEM = 51492
DIPSTICK_MAP = {"NEG": 0, "NEGATIVE": 0, "TR": 1, "TRACE": 1, "30": 2, "100": 3, "300": 4,
                ">300": 5, ">=300": 5, ">600": 5, "1000": 5}

LAB_VARS = ["creatinine", "egfr", "bun", "sodium", "potassium", "chloride", "bicarbonate", "calcium",
            "phosphate", "albumin", "hemoglobin", "hematocrit", "platelets", "wbc", "glucose", "hba1c",
            "uric_acid", "urine_protein", "acr", "urine_dipstick"]
VITAL_VARS = ["sbp", "dbp", "bmi"]
EVENT_VARS = ["admission", "ed_visit", "icu", "aki", "rasi_start", "sglt2_start", "diuretic_start",
              "nsaid_start", "insulin_start"]
COMORB_VARS = ["diabetes", "hypertension", "heart_failure", "coronary", "pvd", "malignancy"]
CARE_VARS = ["outpatient", "emergency", "inpatient"]

X_VARS = LAB_VARS + VITAL_VARS + EVENT_VARS + COMORB_VARS
D_X = len(X_VARS)
D_C = len(CARE_VARS)
N_MEAS = len(LAB_VARS) + len(VITAL_VARS)
MEAS_IDX = list(range(0, N_MEAS))
EVENT_IDX = list(range(N_MEAS, N_MEAS + len(EVENT_VARS)))
COMORB_IDX = list(range(N_MEAS + len(EVENT_VARS), D_X))
VAR_INDEX = {v: i for i, v in enumerate(X_VARS)}
EGFR_COL = VAR_INDEX["egfr"]
CREAT_COL = VAR_INDEX["creatinine"]
ACR_COL = VAR_INDEX["acr"]
DIABETES_COL = VAR_INDEX["diabetes"]

OMR_BP_NAMES = ["Blood Pressure", "Blood Pressure Sitting"]
OMR_BMI_NAME = "BMI (kg/m2)"
CHART_SBP = [220179, 220050]
CHART_DBP = [220180, 220051]

CKD_REGEX = r"^(585[1-9]|N18[1-9])"
AKI_ICD9, AKI_ICD10 = ["584"], ["N17"]
KRT_ICD9_DX = ["V4511", "V56", "V420"]
KRT_ICD10_DX = ["Z992", "Z49", "Z940"]
KRT_ICD9_PROC = ["3995", "5498", "556"]
KRT_ICD10_PROC = ["5A1D", "3E1M39Z", "0TY"]


def _c(letter, lo, hi):
    return [f"{letter}{i:02d}" for i in range(lo, hi + 1)]


COMORBIDITY_CODES = {
    "diabetes": (["250", "3572", "3620", "36641"], ["E08", "E09", "E10", "E11", "E13"]),
    "hypertension": (["401", "402", "403", "404", "405"], ["I10", "I11", "I12", "I13", "I15", "I16"]),
    "heart_failure": (["428", "39891", "40201", "40211", "40291", "40401", "40403", "40411", "40413",
                       "40491", "40493"],
                      ["I099", "I110", "I130", "I132", "I255", "I420", "I425", "I426", "I427", "I428",
                       "I429", "I43", "I50", "P290"]),
    "coronary": ([str(i) for i in range(410, 415)], _c("I", 20, 25)),
    "pvd": (["440", "441", "4431", "4432", "4433", "4434", "4435", "4436", "4437", "4438", "4439", "4471",
             "5571", "5579", "V434"],
            ["I70", "I71", "I731", "I738", "I739", "I771", "I790", "I792", "K551", "K558", "K559", "Z958",
             "Z959"]),
    "malignancy": ([str(i) for i in list(range(140, 173)) + list(range(174, 196)) + list(range(200, 209))]
                   + ["2386"],
                   _c("C", 0, 26) + _c("C", 30, 34) + _c("C", 37, 41) + ["C43"] + _c("C", 45, 58)
                   + _c("C", 60, 76) + _c("C", 81, 85) + ["C88"] + _c("C", 90, 97)),
}

DRUG_CLASSES = {
    "rasi_start": r"lisinopril|enalapril|ramipril|captopril|benazepril|quinapril|fosinopril|perindopril|"
                  r"losartan|valsartan|irbesartan|candesartan|olmesartan|telmisartan|sacubitril",
    "sglt2_start": r"empagliflozin|dapagliflozin|canagliflozin|ertugliflozin",
    "diuretic_start": r"furosemide|torsemide|bumetanide|hydrochlorothiazide|chlorthalidone|metolazone|indapamide",
    "nsaid_start": r"ibuprofen|naproxen|ketorolac|diclofenac|meloxicam|celecoxib|indomethacin|etodolac|nabumetone",
    "insulin_start": r"insulin",
}

VALUE_RANGES = {
    "bun": (1, 300), "sodium": (100, 180), "potassium": (1.5, 9), "chloride": (60, 140),
    "bicarbonate": (5, 50), "calcium": (4, 16), "phosphate": (0.5, 15), "albumin": (0.5, 6),
    "hemoglobin": (2, 22), "hematocrit": (5, 70), "platelets": (5, 2000), "wbc": (0.1, 500),
    "glucose": (10, 2000), "hba1c": (3, 20), "uric_acid": (0.5, 25), "urine_protein": (0, 5000),
    "acr": (0, 20000), "sbp": (50, 260), "dbp": (20, 160), "bmi": (10, 90),
}

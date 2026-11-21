import io
from typing import Tuple
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from audit_engine import run_deterministic_checks, enrich_with_gemini_citations

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB per uploaded file

ALLOWED_ORIGINS = [
    "https://veritax-audit-workspace.onrender.com",
    "https://veritax-audit-workspacenew.onrender.com",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]

app = FastAPI(
    title="VeriTax AI Auditor Engine",
    description="Deterministic Indian GST & Statutory Audit Backend powered by Google Gemini",
    version="1.3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

DEMO_PROFILES = {
    "deccan": {
        "entity_name": "Deccan Precision Components Pvt Ltd",
        "gstin": "36AAACD1234F1Z5",
        "fy": "FY 2025-26",
        "pr": [
            {
                "invoice_no": "INV/25-26/0891",
                "invoice_date": "2026-01-12",
                "vendor_name": "Apex Logistics & Fleet Pvt Ltd",
                "vendor_gstin": "36AABCA4412M1Z5",
                "description": "Freight & Warehousing Services",
                "taxable_value": 450000.0,
                "itc_claimed": 81000.0,
                "is_msme_udyam": False,
                "payment_date": "2026-02-05"
            },
            {
                "invoice_no": "INV/25-26/1104",
                "invoice_date": "2026-02-03",
                "vendor_name": "Deccan Motors Corp",
                "vendor_gstin": "36AAACD9921L1Z9",
                "description": "Executive SUV Motor Vehicle Purchase",
                "taxable_value": 1200000.0,
                "itc_claimed": 336000.0,
                "is_msme_udyam": False,
                "payment_date": "2026-02-10"
            },
            {
                "invoice_no": "MSME/26/0042",
                "invoice_date": "2026-01-05",
                "vendor_name": "Sri Balaji Packaging Works",
                "vendor_gstin": "36AABCS7731P1Z1",
                "description": "Corrugated Boxes Supply",
                "taxable_value": 280000.0,
                "itc_claimed": 50400.0,
                "is_msme_udyam": True,
                "payment_date": ""
            }
        ],
        "gstr2b": [
            {
                "invoice_no": "INV/25-26/0891",
                "vendor_gstin": "36AABCA4412M1Z5",
                "itc_available_2b": 0.0,
                "vendor_filing_status": "GSTR-1_NOT_FILED"
            },
            {
                "invoice_no": "INV/25-26/1104",
                "vendor_gstin": "36AAACD9921L1Z9",
                "itc_available_2b": 336000.0,
                "vendor_filing_status": "FILED"
            },
            {
                "invoice_no": "MSME/26/0042",
                "vendor_gstin": "36AABCS7731P1Z1",
                "itc_available_2b": 50400.0,
                "vendor_filing_status": "FILED"
            }
        ]
    },
    "charminar": {
        "entity_name": "Charminar Mobility & Spares",
        "gstin": "36AABCC6620K1Z8",
        "fy": "FY 2025-26",
        "pr": [
            {
                "invoice_no": "DL/25-26/0310",
                "invoice_date": "2026-01-18",
                "vendor_name": "Hyderabad Express Cargo",
                "vendor_gstin": "36AABCH5501R1Z2",
                "description": "Linehaul Transport Charges",
                "taxable_value": 320000.0,
                "itc_claimed": 57600.0,
                "is_msme_udyam": False,
                "payment_date": "2026-02-01"
            },
            {
                "invoice_no": "MSME/26/0119",
                "invoice_date": "2026-01-10",
                "vendor_name": "Telangana Precision Die Works",
                "vendor_gstin": "36AABCT8840N1Z4",
                "description": "Machined Spares & Dies",
                "taxable_value": 195000.0,
                "itc_claimed": 35100.0,
                "is_msme_udyam": True,
                "payment_date": ""
            }
        ],
        "gstr2b": [
            {
                "invoice_no": "DL/25-26/0310",
                "vendor_gstin": "36AABCH5501R1Z2",
                "itc_available_2b": 12600.0,
                "vendor_filing_status": "PARTIAL_FILED"
            },
            {
                "invoice_no": "MSME/26/0119",
                "vendor_gstin": "36AABCT8840N1Z4",
                "itc_available_2b": 35100.0,
                "vendor_filing_status": "FILED"
            }
        ]
    }
}

REQUIRED_PR_COLS = ["invoice_no", "invoice_date", "vendor_name", "vendor_gstin", "itc_claimed"]
REQUIRED_2B_COLS = ["invoice_no", "vendor_gstin", "itc_available_2b"]


def read_bounded_upload(upload: UploadFile, label: str) -> bytes:
    content = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{label} exceeds the 5 MB prototype upload limit."
        )
    return content


def parse_tabular_upload(filename: str, raw_bytes: bytes, label: str) -> pd.DataFrame:
    if not raw_bytes:
        raise ValueError(f"{label} file is empty.")

    lower_name = (filename or "").lower()
    if lower_name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(raw_bytes))
    elif lower_name.endswith(".xlsx") or lower_name.endswith(".xls"):
        xls = pd.ExcelFile(io.BytesIO(raw_bytes))
        df = pd.read_excel(xls, sheet_name=0)
    else:
        raise ValueError(f"{label} file must be a .csv, .xlsx, or .xls file.")

    if df.empty:
        raise ValueError(f"{label} file contains zero data rows.")

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def validate_and_normalize_frames(pr_df: pd.DataFrame, g2b_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    missing_pr = [c for c in REQUIRED_PR_COLS if c not in pr_df.columns]
    if missing_pr:
        raise ValueError(f"Purchase Register missing required columns: {', '.join(missing_pr)}")

    missing_2b = [c for c in REQUIRED_2B_COLS if c not in g2b_df.columns]
    if missing_2b:
        raise ValueError(f"GSTR-2B missing required columns: {', '.join(missing_2b)}")

    pr_df = pr_df.copy()
    g2b_df = g2b_df.copy()

    pr_df["itc_claimed"] = pd.to_numeric(pr_df["itc_claimed"], errors="coerce").fillna(0.0)
    if "taxable_value" in pr_df.columns:
        pr_df["taxable_value"] = pd.to_numeric(pr_df["taxable_value"], errors="coerce").fillna(0.0)
    else:
        pr_df["taxable_value"] = 0.0

    if "description" not in pr_df.columns:
        pr_df["description"] = ""
    if "is_msme_udyam" not in pr_df.columns:
        pr_df["is_msme_udyam"] = False
    if "payment_date" not in pr_df.columns:
        pr_df["payment_date"] = ""

    parsed_dates = pd.to_datetime(pr_df["invoice_date"], format="mixed", dayfirst=True, errors="coerce")
    if parsed_dates.isna().any():
        bad_rows = parsed_dates[parsed_dates.isna()].index.tolist()
        raise ValueError(f"Purchase Register has invalid invoice_date values at row(s): {bad_rows}")
    pr_df["invoice_date"] = parsed_dates.dt.strftime("%Y-%m-%d")

    g2b_df["itc_available_2b"] = pd.to_numeric(g2b_df["itc_available_2b"], errors="coerce").fillna(0.0)
    if "vendor_filing_status" not in g2b_df.columns:
        g2b_df["vendor_filing_status"] = "REFLECTED_IN_2B"

    return pr_df, g2b_df


@app.get("/")
def health_check():
    return {
        "status": "online",
        "engine": "VeriTax Deterministic + Gemini 3.8 Flash",
        "version": "1.3.0",
        "security_hygiene": "strict_cors_and_5mb_upload_cap"
    }


@app.post("/api/v1/audit/run")
async def execute_audit(
    purchase_register: UploadFile = File(...),
    gstr_2b: UploadFile = File(...),
    cutoff_date: str = Form("2026-03-31"),
    entity_name: str = Form("Uploaded Entity Ledger"),
    gstin: str = Form("N/A")
):
    try:
        pr_bytes = read_bounded_upload(purchase_register, "Purchase Register")
        g2b_bytes = read_bounded_upload(gstr_2b, "GSTR-2B")

        raw_pr = parse_tabular_upload(purchase_register.filename or "purchase_register.csv", pr_bytes, "Purchase Register")
        raw_2b = parse_tabular_upload(gstr_2b.filename or "gstr_2b.csv", g2b_bytes, "GSTR-2B")

        pr_df, g2b_df = validate_and_normalize_frames(raw_pr, raw_2b)

        raw_anomalies = run_deterministic_checks(pr_df, g2b_df, audit_cutoff_date=cutoff_date)
        final_findings = enrich_with_gemini_citations(raw_anomalies)
        total_exposure = round(sum(f.exposure_inr for f in final_findings), 2)

        return {
            "entity_profile": {
                "profile_key": "custom_upload",
                "entity_name": (entity_name or "").strip()[:120] or "Uploaded Entity Ledger",
                "gstin": (gstin or "").strip().upper()[:15] or "N/A",
                "fy": "Custom Upload"
            },
            "audit_summary": {
                "total_rows_scanned": len(pr_df),
                "anomalies_flagged": len(final_findings),
                "total_exposure_inr": total_exposure,
                "cutoff_date": cutoff_date
            },
            "findings": [f.model_dump() for f in final_findings]
        }

    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception:
        raise HTTPException(status_code=400, detail="Unable to process uploaded audit files.")


@app.get("/api/v1/audit/demo")
def run_demo_audit(profile: str = "deccan"):
    """Instant test endpoint supporting selectable client/dealer profiles."""
    try:
        selected = DEMO_PROFILES.get(profile.lower(), DEMO_PROFILES["deccan"])
        sample_pr = pd.DataFrame(selected["pr"])
        sample_2b = pd.DataFrame(selected["gstr2b"])

        raw_anomalies = run_deterministic_checks(sample_pr, sample_2b)
        final_findings = enrich_with_gemini_citations(raw_anomalies)

        return {
            "entity_profile": {
                "profile_key": profile.lower(),
                "entity_name": selected["entity_name"],
                "gstin": selected["gstin"],
                "fy": selected["fy"]
            },
            "audit_summary": {
                "total_rows_scanned": len(sample_pr),
                "anomalies_flagged": len(final_findings),
                "total_exposure_inr": round(sum(f.exposure_inr for f in final_findings), 2),
                "cutoff_date": "2026-03-31"
            },
            "findings": [f.model_dump() for f in final_findings]
        }
    except Exception:
        raise HTTPException(status_code=500, detail="Demo audit engine temporarily unavailable.")

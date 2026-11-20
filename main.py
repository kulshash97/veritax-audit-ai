import io
import traceback
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from audit_engine import run_deterministic_checks, enrich_with_gemini_citations

app = FastAPI(
    title="VeriTax AI Auditor Engine",
    description="Deterministic Indian GST & Statutory Audit Backend powered by Google Gemini",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
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


@app.get("/")
def health_check():
    return {"status": "online", "engine": "VeriTax Deterministic + Gemini 3.8 Flash"}


@app.post("/api/v1/audit/run")
async def execute_audit(
    purchase_register: UploadFile = File(...),
    gstr_2b: UploadFile = File(...),
    cutoff_date: str = "2026-03-31"
):
    try:
        pr_bytes = await purchase_register.read()
        g2b_bytes = await gstr_2b.read()

        pr_df = pd.read_csv(io.BytesIO(pr_bytes))
        g2b_df = pd.read_csv(io.BytesIO(g2b_bytes))

        raw_anomalies = run_deterministic_checks(pr_df, g2b_df, audit_cutoff_date=cutoff_date)
        final_findings = enrich_with_gemini_citations(raw_anomalies)
        total_exposure = sum(f.exposure_inr for f in final_findings)

        return {
            "audit_summary": {
                "total_rows_scanned": len(pr_df),
                "anomalies_flagged": len(final_findings),
                "total_exposure_inr": round(total_exposure, 2),
                "cutoff_date": cutoff_date
            },
            "findings": [f.model_dump() for f in final_findings]
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Audit pipeline failed: {str(e)}")


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
                "total_exposure_inr": sum(f.exposure_inr for f in final_findings),
            },
            "findings": [f.model_dump() for f in final_findings]
        }
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Demo audit error: {type(exc).__name__}: {exc}")

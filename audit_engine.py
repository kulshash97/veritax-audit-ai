import os
import time
from typing import List, Literal
import pandas as pd
from pydantic import BaseModel, Field
from google import genai
from google.genai import types


class StatutoryExplanation(BaseModel):
    rule_code: str = Field(description="Internal rule code, e.g., SEC_17_5_BLOCKED_ITC")
    statutory_citation: str = Field(description="Exact Act, Section, Rule, or CBIC Circular")
    audit_memo_note: str = Field(description="Concise 2-sentence audit finding memo written for a Chartered Accountant")
    recommended_action: str = Field(description="Concrete compliance step, e.g., Reverse ITC in GSTR-3B Table 4(B)(1)")


class AuditFinding(BaseModel):
    finding_id: str
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "PASS"]
    category: str
    vendor_name: str
    vendor_gstin: str
    invoice_no: str
    invoice_date: str
    exposure_inr: float
    deterministic_proof: str
    statutory_citation: str
    audit_memo_note: str
    recommended_action: str
    status: str = "PENDING_CA_REVIEW"


BLOCKED_KEYWORDS_17_5 = [
    "motor vehicle", "car", "suv", "catering", "food", "restaurant",
    "club membership", "health insurance", "life insurance", "personal"
]

FALLBACK_STATUTES = {
    "SEC_16_2_AA_ITC_MISMATCH": (
        "Section 16(2)(aa) of CGST Act, 2017 & Rule 36(4) of CGST Rules",
        "Input Tax Credit claimed in books exceeds ITC auto-populated in GSTR-2B because the supplier has not yet reflected the invoice in GSTR-1. Reconcile with the vendor before claiming credit in returns.",
        "Withhold vendor payment for the GST component and reverse excess ITC in GSTR-3B Table 4(B)(2) until reflected in GSTR-2B."
    ),
    "SEC_17_5_BLOCKED_CREDIT": (
        "Section 17(5)(a)/(b) of CGST Act, 2017",
        "Input Tax Credit is statutorily restricted on specified motor vehicles, food/catering, and personal or club expenses unless used for eligible outward taxable supplies. Flagged invoice requires reviewer verification against statutory exceptions.",
        "Reverse ineligible ITC in GSTR-3B Table 4(B)(1) along with applicable interest under Section 50(3) unless an eligible exception applies."
    ),
    "IT_SEC_43B_H_MSME_DELAY": (
        "Section 43B(h) of Income Tax Act, 1961 r/w Section 15 of MSMED Act, 2006",
        "Payments to registered Micro or Small enterprises remaining unpaid or delayed beyond the statutory 45-day window (or 15 days without written agreement) face disallowance in the year of accrual. Verify Udyam registration status and payment dates.",
        "Disallow the unpaid principal expense in Tax Audit Form 3CD Clause 26 and claim deduction in the financial year of actual payment."
    ),
}


def run_deterministic_checks(
    purchase_df: pd.DataFrame,
    gstr2b_df: pd.DataFrame,
    audit_cutoff_date: str = "2026-03-31"
) -> List[dict]:
    raw_anomalies = []
    cutoff_dt = pd.to_datetime(audit_cutoff_date)

    purchase_df = purchase_df.copy()
    gstr2b_df = gstr2b_df.copy()

    purchase_df["join_key"] = (
        purchase_df["vendor_gstin"].astype(str).str.strip().str.upper()
        + "_"
        + purchase_df["invoice_no"].astype(str).str.strip().str.upper()
    )
    gstr2b_df["join_key"] = (
        gstr2b_df["vendor_gstin"].astype(str).str.strip().str.upper()
        + "_"
        + gstr2b_df["invoice_no"].astype(str).str.strip().str.upper()
    )

    merged = pd.merge(
        purchase_df,
        gstr2b_df[["join_key", "itc_available_2b", "vendor_filing_status"]],
        on="join_key",
        how="left"
    )
    merged["itc_available_2b"] = merged["itc_available_2b"].fillna(0.0)
    merged["vendor_filing_status"] = merged["vendor_filing_status"].fillna("NOT_FILED_IN_2B")

    for _, row in merged.iterrows():
        inv_no = str(row["invoice_no"])
        v_name = str(row["vendor_name"])
        v_gstin = str(row["vendor_gstin"])
        inv_date = str(row["invoice_date"])
        desc = str(row.get("description", "")).lower()
        itc_claimed = float(row.get("itc_claimed", 0.0))
        itc_2b = float(row.get("itc_available_2b", 0.0))
        taxable_val = float(row.get("taxable_value", 0.0))

        variance = round(itc_claimed - itc_2b, 2)
        if variance > 1.0:
            raw_anomalies.append({
                "rule_code": "SEC_16_2_AA_ITC_MISMATCH",
                "severity": "CRITICAL",
                "category": "GSTR-2B vs Books ITC Variance",
                "vendor_name": v_name,
                "vendor_gstin": v_gstin,
                "invoice_no": inv_no,
                "invoice_date": inv_date,
                "exposure_inr": variance,
                "deterministic_proof": (
                    f"Books ITC Claimed: INR {itc_claimed:,.2f} | "
                    f"GSTR-2B Reflected: INR {itc_2b:,.2f} | "
                    f"Excess Claimed: INR {variance:,.2f} (Status: {row['vendor_filing_status']})"
                )
            })

        matched_kw = [kw for kw in BLOCKED_KEYWORDS_17_5 if kw in desc]
        if matched_kw and itc_claimed > 0:
            raw_anomalies.append({
                "rule_code": "SEC_17_5_BLOCKED_CREDIT",
                "severity": "HIGH",
                "category": "Ineligible / Blocked ITC (Sec 17(5))",
                "vendor_name": v_name,
                "vendor_gstin": v_gstin,
                "invoice_no": inv_no,
                "invoice_date": inv_date,
                "exposure_inr": itc_claimed,
                "deterministic_proof": (
                    f"Line item '{row.get('description')}' matched restricted category "
                    f"({', '.join(matched_kw)}) with ITC claimed of INR {itc_claimed:,.2f}."
                )
            })

        is_msme = str(row.get("is_msme_udyam", "false")).lower() in ["true", "1", "yes"]
        if is_msme:
            inv_dt = pd.to_datetime(row["invoice_date"])
            pay_val = row.get("payment_date")
            pay_dt = pd.to_datetime(pay_val) if pd.notna(pay_val) and str(pay_val).strip() != "" else cutoff_dt
            days_elapsed = (pay_dt - inv_dt).days

            if days_elapsed > 45:
                raw_anomalies.append({
                    "rule_code": "IT_SEC_43B_H_MSME_DELAY",
                    "severity": "HIGH",
                    "category": "Sec 43B(h) MSME Disallowance Risk",
                    "vendor_name": v_name,
                    "vendor_gstin": v_gstin,
                    "invoice_no": inv_no,
                    "invoice_date": inv_date,
                    "exposure_inr": taxable_val,
                    "deterministic_proof": (
                        f"Registered Micro/Small vendor paid/unpaid after {days_elapsed} days "
                        f"(exceeds 45-day statutory limit by {days_elapsed - 45} days). "
                        f"Expense of INR {taxable_val:,.2f} subject to disallowance."
                    )
                })

    return raw_anomalies


def enrich_with_gemini_citations(raw_anomalies: List[dict]) -> List[AuditFinding]:
    enriched_findings: List[AuditFinding] = []
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    client = genai.Client(api_key=api_key) if api_key else None

    system_instruction = (
        "You are an Indian tax and statutory compliance analyst. "
        "Do NOT alter any numbers or dates provided. Return a crisp 2-sentence "
        "audit review memo note and a clear compliance action."
    )

    for i, item in enumerate(raw_anomalies, start=1):
        fallback_cit, fallback_memo, fallback_act = FALLBACK_STATUTES.get(
            item["rule_code"],
            ("CGST Act 2017 / Income Tax Act 1961", item["deterministic_proof"], "Review supporting voucher.")
        )

        citation = fallback_cit
        memo = fallback_memo
        action = fallback_act

        if client and i == 1:
            prompt = (
                f"Rule Code: {item['rule_code']}\n"
                f"Category: {item['category']}\n"
                f"Verified Proof: {item['deterministic_proof']}\n"
                f"Exposure: INR {item['exposure_inr']:,.2f}"
            )
            try:
                time.sleep(0.5)
                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=StatutoryExplanation,
                        temperature=0.1,
                    ),
                )
                ai_data = StatutoryExplanation.model_validate_json(response.text)
                citation = ai_data.statutory_citation
                memo = ai_data.audit_memo_note
                action = ai_data.recommended_action
            except Exception:
                pass

        enriched_findings.append(
            AuditFinding(
                finding_id=f"VTX-{202600 + i}",
                severity=item["severity"],
                category=item["category"],
                vendor_name=item["vendor_name"],
                vendor_gstin=item["vendor_gstin"],
                invoice_no=item["invoice_no"],
                invoice_date=item["invoice_date"],
                exposure_inr=item["exposure_inr"],
                deterministic_proof=item["deterministic_proof"],
                statutory_citation=citation,
                audit_memo_note=memo,
                recommended_action=action,
            )
        )

    return enriched_findings

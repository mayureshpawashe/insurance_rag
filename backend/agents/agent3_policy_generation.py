#!/usr/bin/env python3
"""
Agent 3: Policy Generation and Premium Calculation
Generates policy documents and calculates premiums based on risk assessment
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Policy generation models
class PolicyTerms(BaseModel):
    coverage_amount: float = Field(description="Total coverage amount in INR")
    premium_amount: float = Field(description="Annual premium amount in INR")
    deductible: float = Field(description="Deductible amount in INR")
    policy_term: int = Field(description="Policy term in years")
    waiting_periods: Dict[str, int] = Field(description="Waiting periods for different conditions (days)")
    special_conditions: List[str] = Field(description="Special conditions and requirements")
    exclusions: List[str] = Field(description="Policy exclusions")
    benefits: List[str] = Field(description="Policy benefits and features")
    claim_process: str = Field(description="Claims process description")

class PremiumCalculation(BaseModel):
    base_premium: float = Field(description="Base premium before adjustments")
    risk_adjustments: Dict[str, float] = Field(description="Risk-based adjustments")
    discounts: Dict[str, float] = Field(description="Applicable discounts")
    taxes: Dict[str, float] = Field(description="Taxes and charges")
    final_premium: float = Field(description="Final premium amount")
    payment_frequency: str = Field(description="Annual, Semi-annual, Quarterly, Monthly")
    installment_amount: float = Field(description="Per installment amount")

class PolicyDocument(BaseModel):
    policy_number: str = Field(description="Generated policy number")
    document_sections: Dict[str, str] = Field(description="Policy document sections")
    regulatory_disclosures: List[str] = Field(description="Required regulatory disclosures")
    contact_information: Dict[str, str] = Field(description="Contact details for claims/service")
    effective_date: datetime = Field(description="Policy effective date")
    expiry_date: datetime = Field(description="Policy expiry date")

class PolicyDraft(BaseModel):
    policy_id: str = Field(description="Unique policy identifier")
    policy_name: str = Field(description="Name of the policy")
    policy_type: str = Field(description="Type of insurance policy")
    customer_id: str = Field(description="Customer identifier")
    terms: PolicyTerms
    premium_calculation: PremiumCalculation
    policy_document: PolicyDocument
    risk_based_adjustments: List[str] = Field(description="Risk adjustments made")
    compliance_status: str = Field(description="Regulatory compliance status")
    generated_at: datetime = Field(description="Generation timestamp")
    agent_recommendations: List[str] = Field(description="Agent recommendations")

class PolicyTemplate(BaseModel):
    template_id: str
    template_name: str
    policy_type: str
    base_coverage: float
    base_premium: float
    standard_terms: Dict[str, Any]
    customizable_fields: List[str]
    regulatory_requirements: List[str]

class PolicyGenerationAgent:
    """
    Agent 3: Generates policies and calculates premiums based on risk assessment
    Creates complete policy documents with terms, conditions, and pricing
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1", temperature=0.2)
        self.policy_templates = self._load_policy_templates()
        self.base_premiums = self._load_base_premiums()
        self.regulatory_requirements = self._load_regulatory_requirements()
        self.tax_rates = self._load_tax_rates()
        self.discount_rules = self._load_discount_rules()
        
    async def generate_policy(
        self,
        customer_profile,
        risk_assessment,
        customer_needs: str,
        policy_type: str = None
    ) -> PolicyDraft:
        """
        Main function: Generate complete policy with terms and premium
        """
        try:
            generation_start = datetime.now()
            
            # Step 1: Determine appropriate policy type
            if not policy_type:
                policy_type = await self._determine_policy_type(customer_needs, customer_profile)
            
            # Step 2: Select appropriate policy template
            template = self._select_policy_template(policy_type, customer_profile)
            
            # Step 3: Calculate premium based on risk assessment
            premium_calc = self._calculate_premium(template, risk_assessment, customer_profile)
            
            # Step 4: Generate policy terms based on risk and needs
            terms = await self._generate_policy_terms(
                template, risk_assessment, customer_profile, premium_calc
            )
            
            # Step 5: Create policy document
            policy_doc = await self._generate_policy_document(
                policy_type, terms, customer_profile, risk_assessment
            )
            
            # Step 6: Validate compliance
            compliance_status = self._validate_compliance(terms, policy_type)
            
            # Step 7: Generate recommendations
            recommendations = await self._generate_agent_recommendations(
                risk_assessment, terms, customer_profile
            )
            
            # Step 8: Create final policy draft
            policy_id = self._generate_policy_id(customer_profile, policy_type)
            
            return PolicyDraft(
                policy_id=policy_id,
                policy_name=f"Custom {policy_type} Policy",
                policy_type=policy_type,
                customer_id=getattr(customer_profile, 'user_id', 'unknown'),
                terms=terms,
                premium_calculation=premium_calc,
                policy_document=policy_doc,
                risk_based_adjustments=self._get_risk_adjustments(risk_assessment),
                compliance_status=compliance_status,
                generated_at=generation_start,
                agent_recommendations=recommendations
            )
            
        except Exception as e:
            print(f"Error in policy generation: {str(e)}")
            return self._create_default_policy(customer_profile, policy_type or "Health Insurance")
    
    async def _determine_policy_type(self, customer_needs: str, customer_profile) -> str:
        """
        Analyze customer needs to determine most appropriate policy type
        """
        needs_analysis_prompt = f"""
        Based on the customer's expressed needs and profile, determine the most appropriate insurance policy type.
        
        Customer Needs: {customer_needs}
        
        Customer Profile Summary:
        {self._summarize_customer_profile(customer_profile)}
        
        Available Policy Types:
        - Health Insurance: Medical coverage, hospitalization, treatments
        - Life Insurance: Life cover, savings, investment
        - Term Insurance: Pure life cover, high coverage, low premium
        - Motor Insurance: Vehicle insurance, third-party, comprehensive
        - Home Insurance: Property protection, contents, liability
        - Travel Insurance: Trip coverage, medical emergencies abroad
        
        Consider:
        1. Primary needs expressed by customer
        2. Age and life stage
        3. Family situation
        4. Financial capacity
        5. Risk tolerance
        
        Return only the policy type name that best matches the customer's needs.
        """
        
        prompt = ChatPromptTemplate.from_template(needs_analysis_prompt)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            result = chain.invoke({}).strip()
            
            # Validate against available types
            valid_types = ["Health Insurance", "Life Insurance", "Term Insurance", 
                          "Motor Insurance", "Home Insurance", "Travel Insurance"]
            
            for policy_type in valid_types:
                if policy_type.lower() in result.lower():
                    return policy_type
            
            # Default fallback
            return "Health Insurance"
            
        except Exception as e:
            print(f"Error determining policy type: {str(e)}")
            return "Health Insurance"
    
    def _select_policy_template(self, policy_type: str, customer_profile) -> PolicyTemplate:
        """
        Select appropriate policy template based on policy type and customer profile
        """
        templates = self.policy_templates.get(policy_type, [])
        
        if not templates:
            return self._create_default_template(policy_type)
        
        # Simple selection logic - could be enhanced with ML
        age = self._extract_age(customer_profile)
        
        if age < 30:
            # Younger customers - basic coverage, affordable premiums
            return next((t for t in templates if "basic" in t.template_name.lower()), templates[0])
        elif age < 50:
            # Middle-aged - comprehensive coverage
            return next((t for t in templates if "comprehensive" in t.template_name.lower()), templates[0])
        else:
            # Older customers - senior-focused coverage
            return next((t for t in templates if "senior" in t.template_name.lower()), templates[0])
    
    def _calculate_premium(
        self, 
        template: PolicyTemplate, 
        risk_assessment, 
        customer_profile
    ) -> PremiumCalculation:
        """
        Calculate comprehensive premium based on risk assessment and template
        """
        # Step 1: Start with base premium
        base_premium = template.base_premium
        
        # Step 2: Apply risk adjustments
        risk_adjustments = {}
        
        # Overall risk score adjustment
        risk_multiplier = risk_assessment.recommended_premium_multiplier
        risk_adjustments['overall_risk'] = (risk_multiplier - 1.0) * base_premium
        
        # Individual risk factor adjustments
        for factor in risk_assessment.risk_factors:
            if factor.impact_score > 0.7:  # High impact factors
                adjustment = factor.impact_score * factor.weight * base_premium * 0.2
                risk_adjustments[f'{factor.factor_name}_surcharge'] = adjustment
        
        # Step 3: Apply discounts
        discounts = self._calculate_discounts(customer_profile, template)
        
        # Step 4: Calculate taxes and charges
        taxes = self._calculate_taxes_and_charges(base_premium + sum(risk_adjustments.values()))
        
        # Step 5: Calculate final premium
        adjusted_premium = base_premium + sum(risk_adjustments.values()) - sum(discounts.values())
        final_premium = adjusted_premium + sum(taxes.values())
        
        # Round to nearest rupee
        final_premium = float(Decimal(str(final_premium)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
        
        # Step 6: Calculate installment amounts
        installment_amount = final_premium / 12  # Monthly
        
        return PremiumCalculation(
            base_premium=base_premium,
            risk_adjustments=risk_adjustments,
            discounts=discounts,
            taxes=taxes,
            final_premium=final_premium,
            payment_frequency="Annual",
            installment_amount=round(installment_amount, 2)
        )
    
    async def _generate_policy_terms(
        self,
        template: PolicyTemplate,
        risk_assessment,
        customer_profile,
        premium_calc: PremiumCalculation
    ) -> PolicyTerms:
        """
        Generate comprehensive policy terms based on risk and customer needs
        """
        # Calculate coverage amount based on premium and risk
        coverage_multiplier = self._calculate_coverage_multiplier(risk_assessment)
        coverage_amount = premium_calc.final_premium * coverage_multiplier
        
        # Determine deductible based on risk level
        deductible = self._calculate_deductible(risk_assessment, coverage_amount)
        
        # Generate waiting periods
        waiting_periods = self._generate_waiting_periods(risk_assessment, template.policy_type)
        
        # Generate special conditions based on risk factors
        special_conditions = await self._generate_special_conditions(risk_assessment)
        
        # Generate exclusions
        exclusions = self._generate_exclusions(risk_assessment, template.policy_type)
        
        # Generate benefits
        benefits = self._generate_benefits(template, coverage_amount)
        
        # Generate claims process
        claims_process = await self._generate_claims_process(template.policy_type)
        
        return PolicyTerms(
            coverage_amount=coverage_amount,
            premium_amount=premium_calc.final_premium,
            deductible=deductible,
            policy_term=1,  # Standard 1-year term
            waiting_periods=waiting_periods,
            special_conditions=special_conditions,
            exclusions=exclusions,
            benefits=benefits,
            claim_process=claims_process
        )
    
    async def _generate_policy_document(
        self,
        policy_type: str,
        terms: PolicyTerms,
        customer_profile,
        risk_assessment
    ) -> PolicyDocument:
        """
        Generate complete policy document with all sections
        FIXED: Returns strings instead of dictionaries for document sections
        """
        policy_number = self._generate_policy_number()
        
        # Generate document sections using simplified approach
        document_sections = self._generate_document_sections_simple(
            policy_type, terms, customer_profile
        )
        
        # Add regulatory disclosures
        regulatory_disclosures = self._get_regulatory_disclosures(policy_type)
        
        # Contact information
        contact_info = self._get_contact_information()
        
        # Set policy dates
        effective_date = datetime.now() + timedelta(days=1)  # Next day
        expiry_date = effective_date + timedelta(days=365)   # 1 year
        
        return PolicyDocument(
            policy_number=policy_number,
            document_sections=document_sections,
            regulatory_disclosures=regulatory_disclosures,
            contact_information=contact_info,
            effective_date=effective_date,
            expiry_date=expiry_date
        )
    
    def _generate_document_sections_simple(
        self,
        policy_type: str,
        terms: PolicyTerms,
        customer_profile
    ) -> Dict[str, str]:
        """
        Generate document sections with proper string formatting
        FIXED: Returns strings instead of dictionaries
        """
        try:
            sections = {}
            
            # Policy Declaration - as string
            sections["POLICY_DECLARATION"] = f"""
POLICY DECLARATION
Policy Type: {policy_type}
Coverage Amount: ₹{terms.coverage_amount:,.2f}
Annual Premium: ₹{terms.premium_amount:,.2f}
Deductible: ₹{terms.deductible:,.2f}
Policy Term: {terms.policy_term} year(s)
"""
            
            # Coverage Details - as string
            benefits_text = ", ".join(terms.benefits[:5])  # First 5 benefits
            sections["COVERAGE_DETAILS"] = f"""
COVERAGE DETAILS
This {policy_type} policy provides comprehensive coverage including: {benefits_text}.
The policy covers up to ₹{terms.coverage_amount:,.2f} with an annual premium of ₹{terms.premium_amount:,.2f}.
"""
            
            # Terms and Conditions - as string
            conditions_text = "; ".join(terms.special_conditions[:3])  # First 3 conditions
            sections["TERMS_CONDITIONS"] = f"""
TERMS AND CONDITIONS
Policy Term: {terms.policy_term} year(s)
Special Conditions: {conditions_text if conditions_text else 'Standard terms apply'}
Deductible: ₹{terms.deductible:,.2f} applies to all claims.
"""
            
            # Claims Procedure - as string
            sections["CLAIMS_PROCEDURE"] = f"""
CLAIMS PROCEDURE
{terms.claim_process}
"""
            
            # Exclusions - as string
            exclusions_text = "; ".join(terms.exclusions[:5])  # First 5 exclusions
            sections["EXCLUSIONS"] = f"""
POLICY EXCLUSIONS
The following are excluded from coverage: {exclusions_text}.
Please refer to the complete policy document for full list of exclusions.
"""
            
            # Definitions - as string
            sections["DEFINITIONS"] = f"""
KEY DEFINITIONS
Policy Holder: The person in whose name this policy is issued
Sum Insured: The maximum amount payable under this policy (₹{terms.coverage_amount:,.2f})
Deductible: The amount you pay before insurance coverage begins (₹{terms.deductible:,.2f})
Premium: The amount paid for insurance coverage (₹{terms.premium_amount:,.2f} annually)
"""
            
            return sections
            
        except Exception as e:
            print(f"Error generating document sections: {str(e)}")
            # Return basic fallback sections
            return {
                "POLICY_DECLARATION": f"This {policy_type} policy provides coverage of ₹{terms.coverage_amount:,.2f} for an annual premium of ₹{terms.premium_amount:,.2f}.",
                "COVERAGE_DETAILS": f"This policy covers the benefits and features as specified in the terms.",
                "TERMS_CONDITIONS": "Standard terms and conditions apply as per IRDAI guidelines.",
                "CLAIMS_PROCEDURE": terms.claim_process,
                "EXCLUSIONS": f"Standard exclusions apply including: {', '.join(terms.exclusions[:3])}.",
                "DEFINITIONS": "Standard insurance definitions apply as per policy terms."
            }
    
    def _calculate_coverage_multiplier(self, risk_assessment) -> float:
        """
        Calculate coverage amount multiplier based on risk
        """
        # Lower risk = higher coverage for same premium
        # Higher risk = lower coverage for same premium
        
        if risk_assessment.overall_risk_level.value in ['very_low', 'low']:
            return 25.0  # 25x premium coverage
        elif risk_assessment.overall_risk_level.value == 'medium':
            return 20.0  # 20x premium coverage
        elif risk_assessment.overall_risk_level.value == 'high':
            return 15.0  # 15x premium coverage
        else:  # very_high, critical
            return 10.0  # 10x premium coverage
    
    def _calculate_deductible(self, risk_assessment, coverage_amount: float) -> float:
        """
        Calculate appropriate deductible based on risk and coverage
        """
        base_deductible_percentage = 0.02  # 2% of coverage
        
        # Adjust based on risk level
        if risk_assessment.overall_risk_level.value in ['high', 'very_high', 'critical']:
            base_deductible_percentage = 0.05  # 5% for high risk
        elif risk_assessment.overall_risk_level.value in ['very_low', 'low']:
            base_deductible_percentage = 0.01  # 1% for low risk
        
        deductible = coverage_amount * base_deductible_percentage
        
        # Ensure minimum and maximum limits
        min_deductible = 2500.0
        max_deductible = 50000.0
        
        return max(min_deductible, min(max_deductible, deductible))
    
    def _calculate_discounts(self, customer_profile, template: PolicyTemplate) -> Dict[str, float]:
        """
        Calculate applicable discounts
        """
        discounts = {}
        
        # Age-based discounts
        age = self._extract_age(customer_profile)
        if age < 30:
            discounts['young_customer_discount'] = template.base_premium * 0.05  # 5%
        
        # Family discount (if multiple family members)
        family_pref = next((p for p in customer_profile.preferences if 'family' in p.preference_type.lower()), None)
        if family_pref and any(term in family_pref.value.lower() for term in ['married', 'children', 'spouse']):
            discounts['family_discount'] = template.base_premium * 0.10  # 10%
        
        # Loyalty discount (if existing customer)
        if hasattr(customer_profile, 'policies_discussed') and len(customer_profile.policies_discussed) > 2:
            discounts['loyalty_discount'] = template.base_premium * 0.05  # 5%
        
        return discounts
    
    def _calculate_taxes_and_charges(self, premium_amount: float) -> Dict[str, float]:
        """
        Calculate taxes and charges
        """
        taxes = {}
        
        # GST on insurance premiums (18% in India)
        taxes['gst'] = premium_amount * 0.18
        
        # Service charges
        taxes['service_charges'] = 100.0  # Flat ₹100
        
        # Stamp duty (varies by state, using average)
        taxes['stamp_duty'] = premium_amount * 0.002  # 0.2%
        
        return taxes
    
    # Helper methods for policy generation
    def _generate_waiting_periods(self, risk_assessment, policy_type: str) -> Dict[str, int]:
        """Generate waiting periods based on risk and policy type"""
        waiting_periods = {}
        
        if policy_type == "Health Insurance":
            waiting_periods['pre_existing_conditions'] = 1095  # 3 years
            waiting_periods['specific_diseases'] = 730       # 2 years
            waiting_periods['maternity'] = 270              # 9 months
            
            # Adjust based on risk
            if risk_assessment.overall_risk_level.value in ['high', 'very_high']:
                waiting_periods['general_treatment'] = 30   # 30 days for high risk
        
        return waiting_periods
    
    async def _generate_special_conditions(self, risk_assessment) -> List[str]:
        """Generate special conditions based on risk factors"""
        conditions = []
        
        for factor in risk_assessment.risk_factors:
            if factor.impact_score > 0.7:
                if factor.factor_type == 'health':
                    conditions.append(f"Regular health monitoring required for {factor.factor_name}")
                elif factor.factor_type == 'behavioral':
                    conditions.append(f"Lifestyle modification recommendations for {factor.factor_name}")
                elif factor.factor_type == 'demographic' and factor.factor_name == 'occupation':
                    conditions.append(f"Occupational safety requirements apply")
        
        if risk_assessment.requires_manual_review:
            conditions.append("Subject to medical examination and document verification")
        
        return conditions
    
    def _generate_exclusions(self, risk_assessment, policy_type: str) -> List[str]:
        """Generate policy exclusions"""
        exclusions = [
            "Pre-existing conditions (subject to waiting period)",
            "Self-inflicted injuries",
            "War and terrorism",
            "Nuclear risks",
            "Substance abuse related claims"
        ]
        
        if policy_type == "Health Insurance":
            exclusions.extend([
                "Cosmetic surgery",
                "Dental treatment (unless accidental)",
                "Eye glasses and contact lenses",
                "Pregnancy complications (subject to waiting period)"
            ])
        
        # Add risk-specific exclusions
        high_risk_factors = [f for f in risk_assessment.risk_factors if f.impact_score > 0.8]
        if high_risk_factors:
            exclusions.append("Additional exclusions may apply based on risk assessment")
        
        return exclusions
    
    def _generate_benefits(self, template: PolicyTemplate, coverage_amount: float) -> List[str]:
        """Generate policy benefits"""
        benefits = [
            f"Coverage up to ₹{coverage_amount:,.0f}",
            "Cashless treatment at network hospitals",
            "Pre and post hospitalization coverage",
            "Day care procedures covered",
            "Emergency ambulance services"
        ]
        
        if template.policy_type == "Health Insurance":
            benefits.extend([
                "Annual health check-up",
                "AYUSH treatments covered",
                "Home healthcare benefits",
                "Organ donor coverage"
            ])
        
        return benefits
    
    async def _generate_claims_process(self, policy_type: str) -> str:
        """Generate claims process description"""
        if policy_type == "Health Insurance":
            return """
Step 1: Notify the insurance company within 24-48 hours of hospitalization
Step 2: Submit pre-authorization request for cashless treatment
Step 3: For reimbursement claims, submit original bills and documents
Step 4: Claims team will review and process within 15-30 days
Step 5: Approved amount will be paid directly to hospital or reimbursed
"""
        else:
            return """
Step 1: Notify the insurance company immediately after the incident
Step 2: Submit completed claim form with required documents
Step 3: Insurance company will investigate the claim
Step 4: Upon approval, claim amount will be processed
Step 5: Payment will be made as per policy terms
"""
    
    def _validate_compliance(self, terms: PolicyTerms, policy_type: str) -> str:
        """Validate policy compliance with regulations"""
        try:
            # Basic compliance checks
            if terms.coverage_amount <= 0:
                return "NON_COMPLIANT"
            
            if terms.premium_amount <= 0:
                return "NON_COMPLIANT"
            
            if not terms.exclusions:
                return "PENDING"
            
            if not terms.benefits:
                return "PENDING"
            
            return "COMPLIANT"
            
        except Exception as e:
            print(f"Error in compliance validation: {str(e)}")
            return "PENDING"
    
    async def _generate_agent_recommendations(self, risk_assessment, terms: PolicyTerms, customer_profile) -> List[str]:
        """Generate agent recommendations"""
        recommendations = []
        
        # Risk-based recommendations
        if risk_assessment.overall_risk_level.value in ['high', 'very_high']:
            recommendations.append("Consider additional health screening before policy issuance")
            recommendations.append("Recommend higher deductible to reduce premium")
        
        # Coverage recommendations
        if terms.coverage_amount < 500000:
            recommendations.append("Consider increasing coverage amount for better protection")
        
        # Premium recommendations
        if terms.premium_amount > 50000:
            recommendations.append("Explore payment plan options to manage premium affordability")
        
        return recommendations
    
    def _get_risk_adjustments(self, risk_assessment) -> List[str]:
        """Get list of risk adjustments made"""
        adjustments = []
        
        for factor in risk_assessment.risk_factors:
            if factor.impact_score > 0.6:
                adjustments.append(f"Adjustment for {factor.factor_name}: {factor.explanation}")
        
        return adjustments
    
    # Utility methods
    def _extract_age(self, customer_profile) -> int:
        """Extract age from customer profile"""
        age_pref = next((p for p in customer_profile.preferences if p.preference_type == 'age'), None)
        if age_pref and age_pref.value.isdigit():
            return int(age_pref.value)
        return 35  # Default age
    
    def _summarize_customer_profile(self, customer_profile) -> str:
        """Create customer profile summary"""
        summary_parts = []
        for pref in customer_profile.preferences:
            if pref.confidence > 0.7:
                summary_parts.append(f"{pref.preference_type}: {pref.value}")
        return "; ".join(summary_parts) if summary_parts else "Limited profile information"
    
    def _generate_policy_id(self, customer_profile, policy_type: str) -> str:
        """Generate unique policy ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        type_code = policy_type.replace(" ", "")[:4].upper()
        customer_id = getattr(customer_profile, 'user_id', 'CUST')[:8]
        return f"POL-{type_code}-{customer_id}-{timestamp}"
    
    def _generate_policy_number(self) -> str:
        """Generate policy number"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"HDFC-{timestamp}"
    
    # Data loading methods (would load from database in production)
    def _load_policy_templates(self) -> Dict[str, List[PolicyTemplate]]:
        """Load policy templates"""
        return {
            "Health Insurance": [
                PolicyTemplate(
                    template_id="HI_BASIC_001",
                    template_name="Basic Health Insurance",
                    policy_type="Health Insurance",
                    base_coverage=500000.0,
                    base_premium=12000.0,
                    standard_terms={},
                    customizable_fields=[],
                    regulatory_requirements=[]
                )
            ]
        }
    
    def _load_base_premiums(self) -> Dict[str, float]:
        """Load base premium rates"""
        return {
            "Health Insurance": 12000.0,
            "Life Insurance": 15000.0,
            "Term Insurance": 8000.0,
            "Motor Insurance": 10000.0,
            "Home Insurance": 6000.0,
            "Travel Insurance": 2000.0
        }
    
    def _load_regulatory_requirements(self) -> Dict[str, List[str]]:
        """Load regulatory requirements"""
        return {
            "Health Insurance": [
                "IRDAI health insurance guidelines",
                "Mandatory waiting periods",
                "Pre-existing conditions disclosure"
            ]
        }
    
    def _load_tax_rates(self) -> Dict[str, float]:
        """Load tax rates"""
        return {
            "gst": 0.18,
            "stamp_duty": 0.002,
            "service_charges": 100.0
        }
    
    def _load_discount_rules(self) -> Dict[str, Dict[str, float]]:
        """Load discount rules"""
        return {
            "age_discount": {"under_30": 0.05},
            "family_discount": {"family_policy": 0.10},
            "loyalty_discount": {"existing_customer": 0.05}
        }
    
    def _get_regulatory_disclosures(self, policy_type: str) -> List[str]:
        """Get required regulatory disclosures"""
        return [
            "This product is regulated by IRDAI",
            "Please read the policy terms carefully before purchase",
            "Insurance coverage is subject to terms and conditions",
            "Claims are subject to policy exclusions and waiting periods"
        ]
    
    def _get_contact_information(self) -> Dict[str, str]:
        """Get contact information"""
        return {
            "customer_service": "1800-XXX-XXXX",
            "claims_helpline": "1800-YYY-YYYY",
            "email": "support@company.com",
            "website": "www.company.com"
        }
    
    def _create_default_template(self, policy_type: str) -> PolicyTemplate:
        """Create default template if none found"""
        return PolicyTemplate(
            template_id=f"DEFAULT_{policy_type.replace(' ', '_')}",
            template_name=f"Standard {policy_type}",
            policy_type=policy_type,
            base_coverage=500000.0,
            base_premium=self.base_premiums.get(policy_type, 10000.0),
            standard_terms={},
            customizable_fields=[],
            regulatory_requirements=[]
        )
    
    def _create_default_policy(self, customer_profile, policy_type: str) -> PolicyDraft:
        """Create default policy in case of errors"""
        return PolicyDraft(
            policy_id=f"DEFAULT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            policy_name=f"Standard {policy_type}",
            policy_type=policy_type,
            customer_id=getattr(customer_profile, 'user_id', 'unknown'),
            terms=PolicyTerms(
                coverage_amount=500000.0,
                premium_amount=12000.0,
                deductible=5000.0,
                policy_term=1,
                waiting_periods={},
                special_conditions=[],
                exclusions=["Standard exclusions apply"],
                benefits=["Standard benefits included"],
                claim_process="Standard claims process"
            ),
            premium_calculation=PremiumCalculation(
                base_premium=12000.0,
                risk_adjustments={},
                discounts={},
                taxes={},
                final_premium=12000.0,
                payment_frequency="Annual",
                installment_amount=1000.0
            ),
            policy_document=PolicyDocument(
                policy_number="DEFAULT-POLICY",
                document_sections={
                    "POLICY_DECLARATION": "Standard policy declaration",
                    "COVERAGE_DETAILS": "Standard coverage details",
                    "TERMS_CONDITIONS": "Standard terms and conditions",
                    "CLAIMS_PROCEDURE": "Standard claims procedure",
                    "EXCLUSIONS": "Standard exclusions",
                    "DEFINITIONS": "Standard definitions"
                },
                regulatory_disclosures=[],
                contact_information={},
                effective_date=datetime.now(),
                expiry_date=datetime.now() + timedelta(days=365)
            ),
            risk_based_adjustments=[],
            compliance_status="PENDING",
            generated_at=datetime.now(),
            agent_recommendations=[]
        )

# Factory function for integration
def create_policy_generation_agent():
    """Factory function to create policy generation agent"""
    return PolicyGenerationAgent()

# Example usage
async def test_policy_generation(customer_profile, risk_assessment, customer_needs="Health insurance for family"):
    """Test the policy generation agent"""
    agent = create_policy_generation_agent()
    policy = await agent.generate_policy(customer_profile, risk_assessment, customer_needs)
    
    print(f"Generated Policy: {policy.policy_name}")
    print(f"Policy Type: {policy.policy_type}")
    print(f"Coverage: ₹{policy.terms.coverage_amount:,.2f}")
    print(f"Premium: ₹{policy.terms.premium_amount:,.2f}")
    print(f"Risk Adjustments: {len(policy.risk_based_adjustments)}")
    print(f"Compliance Status: {policy.compliance_status}")
    
    return policy


from typing import List, Dict, Any

async def recommend_top_policies(
    self,
    customer_profile,
    risk_assessment,
    customer_needs: str,
    policy_type: str = None,
    top_n: int = 3
) -> List[Dict[str, Any]]:
    """
    Recommends top N policy templates based on customer profile and affordability.
    """
    try:
        if not policy_type:
            policy_type = await self._determine_policy_type(customer_needs, customer_profile)

        templates = self.policy_templates.get(policy_type, [])
        if not templates:
            return []

        recommendations = []

        for template in templates:
            try:
                premium_calc = self._calculate_premium(template, risk_assessment, customer_profile)
                fit_score = 100 - abs(premium_calc.final_premium - 0.6 * 18000)  # heuristic: 60% of salary

                recommendations.append({
                    "template_id": template.template_id,
                    "policy_name": template.template_name,
                    "coverage": template.base_coverage,
                    "estimated_annual_premium": premium_calc.final_premium,
                    "monthly_cost": premium_calc.installment_amount,
                    "fit_score": round(fit_score, 2)
                })
            except Exception as e:
                print(f"Failed to calculate premium for template {template.template_name}: {e}")

        top_sorted = sorted(recommendations, key=lambda x: x["fit_score"], reverse=True)
        return top_sorted[:top_n]

    except Exception as e:
        print(f"Error in recommending policies: {e}")
        return []

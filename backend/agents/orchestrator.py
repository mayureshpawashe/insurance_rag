#!/usr/bin/env python3
"""
Smart Underwriting Agent Orchestrator
Coordinates all three agents to provide complete underwriting services
"""

import os
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime,timedelta
from pydantic import BaseModel, Field

# Import the three agents
from .agent1_historical_patterns import HistoricalRiskPatternAgent, create_historical_pattern_agent
from .agent2_risk_assessment import ActuarialRiskAssessmentAgent, create_risk_assessment_agent
from .agent3_policy_generation import PolicyGenerationAgent, create_policy_generation_agent

# Result models
class UnderwritingDecision(BaseModel):
    decision: str = Field(description="APPROVED, DECLINED, MANUAL_REVIEW")
    decision_reason: str = Field(description="Reason for the decision")
    confidence_level: float = Field(description="Confidence in decision (0-1)")
    manual_review_triggers: List[str] = Field(description="Factors requiring manual review")

class UnderwritingResult(BaseModel):
    customer_id: str
    request_id: str = Field(description="Unique request identifier")
    
    # Agent outputs
    historical_patterns: List[Any] = Field(description="Output from Agent 1")
    risk_assessment: Any = Field(description="Output from Agent 2")
    policy_draft: Any = Field(description="Output from Agent 3")
    
    # Final decision
    underwriting_decision: UnderwritingDecision
    
    # Process metadata
    processing_time: float = Field(description="Total processing time in seconds")
    agents_used: List[str] = Field(description="List of agents that were called")
    success: bool = Field(description="Whether underwriting completed successfully")
    errors: List[str] = Field(description="Any errors encountered")
    
    # Integration data
    external_data_sources: List[str] = Field(description="External APIs used")
    compliance_checks: Dict[str, bool] = Field(description="Regulatory compliance status")
    
    timestamp: datetime = Field(description="When underwriting was completed")

class SmartUnderwritingOrchestrator:
    """
    Main orchestrator that coordinates all three agents for complete underwriting
    Enhanced version of your existing process_message function
    """
    
    async def recommend_policies(
        self,
        customer_id: str,
        customer_profile,
        customer_needs: str,
        policy_type: str = None
    ) -> List[Dict[str, Any]]:
        """
        Lightweight endpoint to return top N recommended policies for this customer.
        """
        try:
            # Step 1: Run risk assessment (Agent 2)
            risk_assessment = await self.agent2.assess_customer_risk(
                customer_profile=customer_profile,
                historical_patterns=[],
                external_data_sources=[]
            )

            # Step 2: Recommend top policies (Agent 3)
            top_policies = await self.agent3.recommend_top_policies(
                customer_profile=customer_profile,
                risk_assessment=risk_assessment,
                customer_needs=customer_needs,
                policy_type=policy_type
            )

            return top_policies

        except Exception as e:
            print(f"Error in recommend_policies: {str(e)}")
            return []
    
    
    def __init__(self, vector_db, uploaded_vector_db, policy_list):
        # Initialize all three agents
        self.agent1 = create_historical_pattern_agent(vector_db, uploaded_vector_db, policy_list)
        self.agent2 = create_risk_assessment_agent()
        self.agent3 = create_policy_generation_agent()
        
        # Store database references for integration with your existing system
        self.vector_db = vector_db
        self.uploaded_vector_db = uploaded_vector_db
        self.policy_list = policy_list
        
        # Configuration
        self.max_processing_time = 300  # 5 minutes max
        self.enable_external_apis = False  # Can be enabled for production
        
    async def process_underwriting_request(
        self,
        customer_id: str,
        customer_profile,
        customer_needs: str,
        uploaded_documents: List[str] = None,
        policy_type: str = None,
        external_data_sources: List[str] = None
    ) -> UnderwritingResult:
        """
        Main underwriting process - coordinates all three agents
        Enhanced version of your existing process_message function
        """
        start_time = datetime.now()
        request_id = f"UW-{customer_id}-{start_time.strftime('%Y%m%d%H%M%S')}"
        agents_used = []
        errors = []
        
        try:
            print(f"🎯 Starting Smart Underwriting for Customer: {customer_id}")
            print(f"📋 Request ID: {request_id}")
            print(f"💼 Customer Needs: {customer_needs}")
            
            # Step 1: Process uploaded documents (using your existing functionality)
            if uploaded_documents:
                print(f"📄 Processing {len(uploaded_documents)} uploaded documents...")
                # This would integrate with your existing process_uploaded_pdf function
                # for doc_path in uploaded_documents:
                #     process_uploaded_pdf(doc_path, customer_id)
            
            # Step 2: Agent 1 - Historical Risk Pattern Analysis
            print(f"🔍 Agent 1: Analyzing historical risk patterns...")
            historical_patterns = []
            try:
                historical_patterns = await self.agent1.analyze_historical_patterns(customer_profile)
                agents_used.append("Agent1_HistoricalPatterns")
                print(f"   ✅ Found {len(historical_patterns)} historical patterns")
                
                # Log pattern summary
                if historical_patterns:
                    pattern_summary = self.agent1.get_pattern_summary(historical_patterns)
                    print(f"   📊 Risk signal: {pattern_summary.get('overall_risk_signal', 'unknown')}")
                
            except Exception as e:
                error_msg = f"Agent 1 error: {str(e)}"
                errors.append(error_msg)
                print(f"   ❌ {error_msg}")
            
            # Step 3: Agent 2 - Actuarial Risk Assessment
            print(f"📊 Agent 2: Conducting risk assessment...")
            risk_assessment = None
            try:
                risk_assessment = await self.agent2.assess_customer_risk(
                    customer_profile=customer_profile,
                    historical_patterns=historical_patterns,
                    external_data_sources=external_data_sources if self.enable_external_apis else None
                )
                agents_used.append("Agent2_RiskAssessment")
                print(f"   ✅ Risk Level: {risk_assessment.overall_risk_level.value}")
                print(f"   📈 Risk Score: {risk_assessment.risk_score:.1f}/100")
                print(f"   💰 Premium Multiplier: {risk_assessment.recommended_premium_multiplier:.2f}")
                
            except Exception as e:
                error_msg = f"Agent 2 error: {str(e)}"
                errors.append(error_msg)
                print(f"   ❌ {error_msg}")
                # Create default risk assessment
                risk_assessment = self._create_default_risk_assessment(customer_profile)
            
            # Step 4: Agent 3 - Policy Generation and Premium Calculation  
            print(f"📋 Agent 3: Generating policy draft...")
            policy_draft = None
            try:
                policy_draft = await self.agent3.generate_policy(
                    customer_profile=customer_profile,
                    risk_assessment=risk_assessment,
                    customer_needs=customer_needs,
                    policy_type=policy_type
                )
                agents_used.append("Agent3_PolicyGeneration")
                print(f"   ✅ Policy: {policy_draft.policy_name}")
                print(f"   💰 Premium: ₹{policy_draft.terms.premium_amount:,.2f}")
                print(f"   🛡️ Coverage: ₹{policy_draft.terms.coverage_amount:,.2f}")
                
            except Exception as e:
                error_msg = f"Agent 3 error: {str(e)}"
                errors.append(error_msg)
                print(f"   ❌ {error_msg}")
                # Create default policy
                policy_draft = self._create_default_policy_draft(customer_profile, risk_assessment)
            
            # Step 5: Make Final Underwriting Decision
            print(f"⚖️ Making underwriting decision...")
            underwriting_decision = self._make_underwriting_decision(
                risk_assessment, policy_draft, historical_patterns, errors
            )
            
            print(f"   🎯 Decision: {underwriting_decision.decision}")
            print(f"   📝 Reason: {underwriting_decision.decision_reason}")
            
            # Step 6: Perform Compliance Checks
            compliance_checks = self._perform_compliance_checks(policy_draft, risk_assessment)
            
            # Step 7: Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            print(f"⏱️ Processing completed in {processing_time:.2f} seconds")
            
            # Step 8: Create final result
            result = UnderwritingResult(
                customer_id=customer_id,
                request_id=request_id,
                historical_patterns=historical_patterns,
                risk_assessment=risk_assessment,
                policy_draft=policy_draft,
                underwriting_decision=underwriting_decision,
                processing_time=processing_time,
                agents_used=agents_used,
                success=len(errors) == 0,
                errors=errors,
                external_data_sources=external_data_sources or [],
                compliance_checks=compliance_checks,
                timestamp=end_time
            )
            
            # Step 9: Update customer profile with results (integration with your system)
            await self._update_customer_profile(customer_profile, result)
            
            return result
            
        except Exception as e:
            # Handle critical errors
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            error_msg = f"Critical orchestrator error: {str(e)}"
            
            print(f"❌ {error_msg}")
            
            return UnderwritingResult(
                customer_id=customer_id,
                request_id=request_id,
                historical_patterns=[],
                risk_assessment=self._create_default_risk_assessment(customer_profile),
                policy_draft=self._create_default_policy_draft(customer_profile, None),
                underwriting_decision=UnderwritingDecision(
                    decision="MANUAL_REVIEW",
                    decision_reason="System error during processing",
                    confidence_level=0.0,
                    manual_review_triggers=[error_msg]
                ),
                processing_time=processing_time,
                agents_used=agents_used,
                success=False,
                errors=[error_msg],
                external_data_sources=[],
                compliance_checks={},
                timestamp=end_time
            )
    
    def _make_underwriting_decision(
        self, 
        risk_assessment, 
        policy_draft, 
        historical_patterns, 
        errors: List[str]
    ) -> UnderwritingDecision:
        """
        Make final underwriting decision based on all agent outputs
        """
        manual_review_triggers = []
        
        # Check for errors
        if errors:
            manual_review_triggers.extend(errors)
        
        # Check risk assessment
        if risk_assessment:
            # High risk scores require manual review
            if risk_assessment.risk_score > 75:
                manual_review_triggers.append(f"High risk score: {risk_assessment.risk_score:.1f}")
            
            # Critical risk level
            if risk_assessment.overall_risk_level.value == "critical":
                manual_review_triggers.append("Critical risk level detected")
            
            # Agent specifically flagged for review
            if risk_assessment.requires_manual_review:
                manual_review_triggers.append("Risk assessment flagged for manual review")
            
            # Low confidence in assessment
            if risk_assessment.confidence_level < 0.6:
                manual_review_triggers.append(f"Low confidence in risk assessment: {risk_assessment.confidence_level:.2f}")
        
        # Check policy draft
        if policy_draft:
            # Very high premiums
            if policy_draft.terms.premium_amount > 100000:
                manual_review_triggers.append(f"High premium amount: ₹{policy_draft.terms.premium_amount:,.2f}")
            
            # Compliance issues
            if policy_draft.compliance_status != "COMPLIANT":
                manual_review_triggers.append(f"Compliance status: {policy_draft.compliance_status}")
        
        # Check historical patterns
        if historical_patterns:
            high_risk_patterns = [p for p in historical_patterns if p.similarity_score > 0.9]
            if high_risk_patterns:
                manual_review_triggers.append(f"High similarity to {len(high_risk_patterns)} high-risk historical cases")
        
        # Make decision
        if manual_review_triggers:
            return UnderwritingDecision(
                decision="MANUAL_REVIEW",
                decision_reason=f"Manual review required due to: {'; '.join(manual_review_triggers[:3])}",
                confidence_level=0.7,
                manual_review_triggers=manual_review_triggers
            )
        
        elif risk_assessment and risk_assessment.overall_risk_level.value in ["very_high", "critical"]:
            return UnderwritingDecision(
                decision="DECLINED",
                decision_reason=f"Risk level too high: {risk_assessment.overall_risk_level.value}",
                confidence_level=0.8,
                manual_review_triggers=[]
            )
        
        elif risk_assessment and risk_assessment.overall_risk_level.value in ["high"]:
            return UnderwritingDecision(
                decision="APPROVED",
                decision_reason="Approved with risk-based premium adjustments",
                confidence_level=0.7,
                manual_review_triggers=[]
            )
        
        else:
            return UnderwritingDecision(
                decision="APPROVED",
                decision_reason="Standard approval - acceptable risk profile",
                confidence_level=0.9,
                manual_review_triggers=[]
            )
    
    def _perform_compliance_checks(self, policy_draft, risk_assessment) -> Dict[str, bool]:
        """
        Perform regulatory compliance checks
        """
        checks = {}
        
        # Basic policy compliance
        if policy_draft:
            checks['policy_terms_valid'] = policy_draft.terms.coverage_amount > 0
            checks['premium_reasonable'] = 1000 <= policy_draft.terms.premium_amount <= 500000
            checks['deductible_valid'] = policy_draft.terms.deductible >= 0
            checks['document_complete'] = len(policy_draft.policy_document.document_sections) > 0
        else:
            checks['policy_terms_valid'] = False
            checks['premium_reasonable'] = False
            checks['deductible_valid'] = False
            checks['document_complete'] = False
        
        # Risk assessment compliance
        if risk_assessment:
            checks['risk_assessment_complete'] = len(risk_assessment.risk_factors) > 0
            checks['confidence_adequate'] = risk_assessment.confidence_level > 0.5
        else:
            checks['risk_assessment_complete'] = False
            checks['confidence_adequate'] = False
        
        # IRDAI compliance (placeholder - would need real regulatory checks)
        checks['irdai_compliant'] = all(checks.values())
        
        return checks
    
    async def _update_customer_profile(self, customer_profile, result: UnderwritingResult):
        """
        Update customer profile with underwriting results
        Integrates with your existing session management system
        """
        try:
            # Add policy to discussed/recommended lists (using your existing structure)
            if result.policy_draft:
                # Import from main system - adjust import path as needed
                try:
                    from main_new2 import PolicyEntity  # Import from your main file
                    
                    policy_entity = PolicyEntity(
                        policy_name=result.policy_draft.policy_name,
                        policy_type=result.policy_draft.policy_type,
                        mentioned_in_turn=0,
                        source_document="underwriting_agent",
                        is_uploaded=False
                    )
                    
                    # Add to recommended policies
                    if not hasattr(customer_profile, 'policies_recommended'):
                        customer_profile.policies_recommended = []
                    
                    customer_profile.policies_recommended.append(policy_entity)
                except ImportError:
                    print("Warning: Could not import PolicyEntity from main system")
            
            # Update last interaction timestamp
            customer_profile.last_interaction = datetime.now().isoformat()
            
            # Save profile using your existing save_session function
            try:
                from main_new2 import save_session  # Import from your main file
                save_session(customer_profile)
            except ImportError:
                print("Warning: Could not import save_session from main system")
            
        except Exception as e:
            print(f"Warning: Could not update customer profile: {str(e)}")
    
    # Default/fallback creation methods
    def _create_default_risk_assessment(self, customer_profile):
        """Create default risk assessment in case of errors"""
        from .agent2_risk_assessment import RiskAssessment, RiskLevel
        
        return RiskAssessment(
            customer_id=getattr(customer_profile, 'user_id', 'unknown'),
            overall_risk_level=RiskLevel.MEDIUM,
            risk_score=50.0,
            risk_factors=[],
            actuarial_models_used=[],
            recommended_premium_multiplier=1.0,
            confidence_level=0.5,
            explanation="Default risk assessment due to processing error",
            external_data_used=[],
            assessment_timestamp=datetime.now(),
            requires_manual_review=True
        )
    
    def _create_default_policy_draft(self, customer_profile, risk_assessment):
        """Create default policy draft in case of errors"""
        # This would use the default policy creation from Agent 3
        return self.agent3._create_default_policy(customer_profile, "Health Insurance")
    
    # Integration methods with your existing system
    def integrate_with_existing_faq(self, user_input: str, customer_profile, conversation_turn: int):
        """
        Integration point with your existing answer_faq function
        Determines when to use underwriting vs regular FAQ
        """
        # Check if user input suggests underwriting need
        underwriting_keywords = [
            'quote', 'premium', 'buy policy', 'apply', 'underwriting',
            'risk assessment', 'policy recommendation', 'new policy'
        ]
        
        needs_underwriting = any(keyword in user_input.lower() for keyword in underwriting_keywords)
        
        if needs_underwriting:
            # Use underwriting orchestrator
            return "underwriting_needed"
        else:
            # Use existing FAQ system
            return "use_existing_faq"
    
    async def enhanced_process_message(
        self, 
        user_id: str, 
        user_input: str, 
        conversation_turn: int = 0,
        customer_profile = None
    ):
        """
        Enhanced version of your process_message function that integrates underwriting
        """
        # Get customer profile (using your existing function)
        if not customer_profile:
            try:
                from main_new2 import get_or_create_session
                customer_profile = get_or_create_session(user_id)
            except ImportError:
                print("Warning: Could not import get_or_create_session from main system")
                # Create a minimal customer profile
                customer_profile = type('CustomerProfile', (), {
                    'user_id': user_id,
                    'preferences': [],
                    'policies_discussed': [],
                    'policies_recommended': []
                })()
        
        # Check if underwriting is needed
        decision = self.integrate_with_existing_faq(user_input, customer_profile, conversation_turn)
        
        if decision == "underwriting_needed":
            # Use smart underwriting
            result = await self.process_underwriting_request(
                customer_id=user_id,
                customer_profile=customer_profile,
                customer_needs=user_input
            )
            
            # Format response for user
            return self._format_underwriting_response(result)
        else:
            # Use your existing FAQ system
            try:
                from main_new2 import answer_faq
                return answer_faq(user_input, customer_profile, conversation_turn)
            except ImportError:
                return f"I understand you're asking about: {user_input}. However, I couldn't access the FAQ system. Please try using specific underwriting keywords like 'quote', 'premium', or 'recommend policy'."
    
    def _format_underwriting_response(self, result: UnderwritingResult) -> str:
        """
        Format underwriting result into user-friendly response
        """
        if not result.success:
            return f"I encountered some issues while processing your request. Please try again or contact our support team. Reference: {result.request_id}"
        
        response_parts = []
        
        # Decision summary
        if result.underwriting_decision.decision == "APPROVED":
            response_parts.append("✅ **Good news! Your application can be approved.**")
        elif result.underwriting_decision.decision == "DECLINED":
            response_parts.append("❌ **I'm sorry, but we cannot offer coverage at this time.**")
        else:
            response_parts.append("📋 **Your application requires manual review by our underwriting team.**")
        
        response_parts.append(f"**Reason:** {result.underwriting_decision.decision_reason}")
        
        # Policy details (if available)
        if result.policy_draft and result.underwriting_decision.decision == "APPROVED":
            response_parts.append(f"\n**📋 Policy Details:**")
            response_parts.append(f"• **Policy:** {result.policy_draft.policy_name}")
            response_parts.append(f"• **Coverage:** ₹{result.policy_draft.terms.coverage_amount:,.2f}")
            response_parts.append(f"• **Annual Premium:** ₹{result.policy_draft.terms.premium_amount:,.2f}")
            response_parts.append(f"• **Monthly Installment:** ₹{result.policy_draft.premium_calculation.installment_amount:,.2f}")
        
        # Risk summary
        if result.risk_assessment:
            response_parts.append(f"\n**📊 Risk Assessment:**")
            response_parts.append(f"• **Risk Level:** {result.risk_assessment.overall_risk_level.value.replace('_', ' ').title()}")
            response_parts.append(f"• **Risk Score:** {result.risk_assessment.risk_score:.1f}/100")
        
        # Historical insights
        if result.historical_patterns:
            response_parts.append(f"\n**🔍 Historical Analysis:**")
            response_parts.append(f"• Found {len(result.historical_patterns)} similar cases in our database")
            
        # Next steps
        response_parts.append(f"\n**🎯 Next Steps:**")
        if result.underwriting_decision.decision == "APPROVED":
            response_parts.append("• Review the policy details above")
            response_parts.append("• Contact us to proceed with policy issuance")
            response_parts.append("• Complete KYC and documentation")
        elif result.underwriting_decision.decision == "MANUAL_REVIEW":
            response_parts.append("• Our underwriting team will review your application")
            response_parts.append("• You may be contacted for additional information")
            response_parts.append("• Decision will be communicated within 3-5 business days")
        
        response_parts.append(f"\n**Reference ID:** {result.request_id}")
        
        return "\n".join(response_parts)

# Factory function for integration with your main system
def create_smart_underwriting_orchestrator(vector_db, uploaded_vector_db, policy_list):
    """
    Factory function to create orchestrator with your existing database connections
    """
    return SmartUnderwritingOrchestrator(vector_db, uploaded_vector_db, policy_list)

# Example integration with your existing main_new2.py
async def enhanced_api_handle_message(
    user_id: str,
    user_input: str,
    conversation_turn: int = 0,
    file_path: Optional[str] = None,
    extraction_method: str = 'auto',
    vector_db = None,
    uploaded_vector_db = None,
    policy_list = None
):
    """
    Enhanced version of your api_handle_message that includes smart underwriting
    """
    processing_info = None
    
    # Process uploaded file if provided (your existing functionality)
    if file_path:
        try:
            try:
                from main_new2 import process_uploaded_pdf, get_or_create_session
                
                policy = process_uploaded_pdf(file_path, user_id, extraction_method=extraction_method)
                processing_info = {
                    "policy_name": policy.policy_name,
                    "policy_type": policy.policy_type,
                    "extraction_method": extraction_method
                }
            except ImportError:
                processing_info = {"error": "Could not process uploaded file - main system not available"}
        except Exception as e:
            return {
                "response": f"Error processing your PDF: {str(e)}",
                "user_profile": {},
                "processing_info": {"error": str(e)}
            }
    
    # Create orchestrator
    orchestrator = create_smart_underwriting_orchestrator(vector_db, uploaded_vector_db, policy_list)
    
    # Process message with enhanced capabilities
    response = await orchestrator.enhanced_process_message(user_id, user_input, conversation_turn)
    
    # Get updated user profile
    try:
        from main_new2 import get_or_create_session
        user_profile = get_or_create_session(user_id)
    except ImportError:
        user_profile = {"user_id": user_id, "note": "Profile system not available"}
    
    return {
        "response": response,
        "user_profile": user_profile,
        "processing_info": processing_info
    }
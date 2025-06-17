#!/usr/bin/env python3
"""
Agent 2: Actuarial Risk Assessment
Performs quantitative risk assessment using actuarial models and external data
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Risk assessment models
class RiskLevel(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"

class RiskFactor(BaseModel):
    factor_name: str = Field(description="Name of the risk factor")
    factor_type: str = Field(description="Category: demographic, health, financial, behavioral")
    value: str = Field(description="Value or description of the factor")
    impact_score: float = Field(description="Impact on risk (0-1)")
    confidence: float = Field(description="Confidence in this assessment (0-1)")
    weight: float = Field(description="Weight in overall calculation (0-1)")
    explanation: str = Field(description="Why this factor affects risk")

class ActuarialModel(BaseModel):
    model_name: str = Field(description="Name of the actuarial model")
    model_type: str = Field(description="Type: demographic, health, financial")
    base_rate: float = Field(description="Base rate for this model")
    risk_multiplier: float = Field(description="Risk adjustment multiplier")
    confidence_interval: Tuple[float, float] = Field(description="95% confidence interval")
    last_updated: datetime = Field(description="When model was last calibrated")

class RiskAssessment(BaseModel):
    customer_id: str
    overall_risk_level: RiskLevel
    risk_score: float = Field(description="Numerical risk score (0-100)")
    risk_factors: List[RiskFactor]
    actuarial_models_used: List[str]
    recommended_premium_multiplier: float = Field(description="Premium adjustment factor")
    confidence_level: float = Field(description="Overall confidence in assessment")
    explanation: str = Field(description="Human-readable risk explanation")
    external_data_used: List[str] = Field(description="External data sources consulted")
    assessment_timestamp: datetime
    requires_manual_review: bool = Field(description="Whether manual review is needed")

class ExternalDataResult(BaseModel):
    source: str = Field(description="Data source name")
    data_type: str = Field(description="Type of data retrieved")
    status: str = Field(description="SUCCESS, FAILED, PARTIAL")
    data: Dict[str, Any] = Field(description="Retrieved data")
    risk_impact: float = Field(description="Impact on risk assessment")

class ActuarialRiskAssessmentAgent:
    """
    Agent 2: Performs comprehensive risk assessment using actuarial models
    Integrates with external data sources and builds on historical patterns
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1", temperature=0.1)  # Low temperature for consistency
        self.actuarial_models = self._load_actuarial_models()
        self.risk_weights = self._load_risk_weights()
        self.external_apis = self._initialize_external_apis()
        self.demographic_tables = self._load_demographic_tables()
        
    async def assess_customer_risk(
        self, 
        customer_profile, 
        historical_patterns: List = None,
        external_data_sources: List[str] = None
    ) -> RiskAssessment:
        """
        Main function: Comprehensive risk assessment for a customer
        """
        try:
            assessment_start = datetime.now()
            
            # Step 1: Extract risk factors from customer profile
            demographic_factors = self._assess_demographic_risk(customer_profile)
            health_factors = await self._assess_health_risk(customer_profile)
            financial_factors = self._assess_financial_risk(customer_profile)
            behavioral_factors = self._assess_behavioral_risk(customer_profile)
            
            # Step 2: Incorporate historical pattern insights
            pattern_factors = self._assess_historical_pattern_risk(historical_patterns or [])
            
            # Step 3: Gather external data if requested
            external_data = []
            if external_data_sources:
                external_data = await self._gather_external_data(customer_profile, external_data_sources)
                external_factors = self._assess_external_data_risk(external_data)
            else:
                external_factors = []
            
            # Step 4: Combine all risk factors
            all_risk_factors = (
                demographic_factors + health_factors + financial_factors + 
                behavioral_factors + pattern_factors + external_factors
            )
            
            # Step 5: Calculate overall risk score using actuarial models
            risk_score, premium_multiplier = self._calculate_overall_risk(all_risk_factors)
            
            # Step 6: Determine risk level and requirements
            risk_level = self._determine_risk_level(risk_score)
            requires_review = self._requires_manual_review(risk_score, all_risk_factors)
            
            # Step 7: Generate explanation
            explanation = await self._generate_risk_explanation(all_risk_factors, risk_score, risk_level)
            
            # Step 8: Calculate confidence level
            confidence = self._calculate_confidence_level(all_risk_factors, external_data)
            
            return RiskAssessment(
                customer_id=getattr(customer_profile, 'user_id', 'unknown'),
                overall_risk_level=risk_level,
                risk_score=risk_score,
                risk_factors=all_risk_factors,
                actuarial_models_used=list(self.actuarial_models.keys()),
                recommended_premium_multiplier=premium_multiplier,
                confidence_level=confidence,
                explanation=explanation,
                external_data_used=[data.source for data in external_data],
                assessment_timestamp=assessment_start,
                requires_manual_review=requires_review
            )
            
        except Exception as e:
            print(f"Error in risk assessment: {str(e)}")
            # Return default assessment in case of error
            return self._create_default_assessment(customer_profile)
    
    def _assess_demographic_risk(self, customer_profile) -> List[RiskFactor]:
        """
        Assess demographic risk factors using actuarial tables
        """
        factors = []
        
        # Age risk assessment
        age_pref = next((p for p in customer_profile.preferences if p.preference_type == 'age'), None)
        if age_pref and age_pref.value.isdigit():
            age = int(age_pref.value)
            age_risk = self._calculate_age_risk(age)
            factors.append(RiskFactor(
                factor_name="age",
                factor_type="demographic",
                value=str(age),
                impact_score=age_risk['impact'],
                confidence=0.95,  # High confidence in age data
                weight=0.3,  # Age is a major factor
                explanation=age_risk['explanation']
            ))
        
        # Gender risk assessment (if available)
        gender_pref = next((p for p in customer_profile.preferences if p.preference_type == 'gender'), None)
        if gender_pref:
            gender_risk = self._calculate_gender_risk(gender_pref.value)
            factors.append(RiskFactor(
                factor_name="gender",
                factor_type="demographic",
                value=gender_pref.value,
                impact_score=gender_risk['impact'],
                confidence=0.9,
                weight=0.1,
                explanation=gender_risk['explanation']
            ))
        
        # Location risk assessment
        location_pref = next((p for p in customer_profile.preferences if p.preference_type in ['location', 'city', 'state']), None)
        if location_pref:
            location_risk = self._calculate_location_risk(location_pref.value)
            factors.append(RiskFactor(
                factor_name="location",
                factor_type="demographic",
                value=location_pref.value,
                impact_score=location_risk['impact'],
                confidence=0.8,
                weight=0.15,
                explanation=location_risk['explanation']
            ))
        
        # Occupation risk assessment
        occupation_pref = next((p for p in customer_profile.preferences if p.preference_type == 'occupation'), None)
        if occupation_pref:
            occupation_risk = self._calculate_occupation_risk(occupation_pref.value)
            factors.append(RiskFactor(
                factor_name="occupation",
                factor_type="demographic",
                value=occupation_pref.value,
                impact_score=occupation_risk['impact'],
                confidence=0.85,
                weight=0.25,
                explanation=occupation_risk['explanation']
            ))
        
        return factors
    
    async def _assess_health_risk(self, customer_profile) -> List[RiskFactor]:
        """
        Assess health-related risk factors
        """
        factors = []
        
        # Health status assessment
        health_prefs = [p for p in customer_profile.preferences 
                       if p.preference_type in ['health', 'medical_history', 'conditions']]
        
        for health_pref in health_prefs:
            health_risk = await self._calculate_health_risk(health_pref.value)
            factors.append(RiskFactor(
                factor_name=f"health_{health_pref.preference_type}",
                factor_type="health",
                value=health_pref.value,
                impact_score=health_risk['impact'],
                confidence=health_pref.confidence,
                weight=0.4,  # Health is very important for health insurance
                explanation=health_risk['explanation']
            ))
        
        # Lifestyle health factors
        lifestyle_prefs = [p for p in customer_profile.preferences 
                          if p.preference_type in ['smoking', 'drinking', 'exercise', 'diet']]
        
        for lifestyle_pref in lifestyle_prefs:
            lifestyle_risk = self._calculate_lifestyle_risk(lifestyle_pref.preference_type, lifestyle_pref.value)
            factors.append(RiskFactor(
                factor_name=f"lifestyle_{lifestyle_pref.preference_type}",
                factor_type="health",
                value=lifestyle_pref.value,
                impact_score=lifestyle_risk['impact'],
                confidence=lifestyle_pref.confidence,
                weight=0.2,
                explanation=lifestyle_risk['explanation']
            ))
        
        return factors
    
    def _assess_financial_risk(self, customer_profile) -> List[RiskFactor]:
        """
        Assess financial risk factors
        """
        factors = []
        
        # Income assessment
        income_pref = next((p for p in customer_profile.preferences if p.preference_type in ['income', 'salary']), None)
        if income_pref:
            income_risk = self._calculate_income_risk(income_pref.value)
            factors.append(RiskFactor(
                factor_name="income",
                factor_type="financial",
                value=income_pref.value,
                impact_score=income_risk['impact'],
                confidence=income_pref.confidence,
                weight=0.25,
                explanation=income_risk['explanation']
            ))
        
        # Budget preferences
        budget_pref = next((p for p in customer_profile.preferences if p.preference_type == 'budget'), None)
        if budget_pref:
            budget_risk = self._calculate_budget_risk(budget_pref.value)
            factors.append(RiskFactor(
                factor_name="budget",
                factor_type="financial",
                value=budget_pref.value,
                impact_score=budget_risk['impact'],
                confidence=budget_pref.confidence,
                weight=0.15,
                explanation=budget_risk['explanation']
            ))
        
        return factors
    
    def _assess_behavioral_risk(self, customer_profile) -> List[RiskFactor]:
        """
        Assess behavioral risk factors based on user interactions
        """
        factors = []
        
        # Risk tolerance assessment
        risk_pref = next((p for p in customer_profile.preferences if p.preference_type == 'risk_tolerance'), None)
        if risk_pref:
            risk_tolerance_impact = self._calculate_risk_tolerance_impact(risk_pref.value)
            factors.append(RiskFactor(
                factor_name="risk_tolerance",
                factor_type="behavioral",
                value=risk_pref.value,
                impact_score=risk_tolerance_impact['impact'],
                confidence=risk_pref.confidence,
                weight=0.1,
                explanation=risk_tolerance_impact['explanation']
            ))
        
        # Policy interaction patterns
        policies_discussed = getattr(customer_profile, 'policies_discussed', [])
        if policies_discussed:
            interaction_risk = self._analyze_interaction_patterns(policies_discussed)
            factors.append(RiskFactor(
                factor_name="interaction_patterns",
                factor_type="behavioral",
                value=f"Discussed {len(policies_discussed)} policies",
                impact_score=interaction_risk['impact'],
                confidence=0.7,
                weight=0.05,
                explanation=interaction_risk['explanation']
            ))
        
        return factors
    
    def _analyze_interaction_patterns(self, policies_discussed) -> Dict[str, Any]:
        """
        Analyze interaction patterns to assess behavioral risk
        FIXED: Added the missing method
        """
        try:
            num_policies = len(policies_discussed)
            
            # Analyze pattern based on number of policies discussed
            if num_policies == 0:
                return {
                    'impact': 0.3,
                    'explanation': 'No policy discussions - limited engagement pattern'
                }
            elif num_policies <= 2:
                return {
                    'impact': 0.2,
                    'explanation': 'Focused policy discussions - indicates clear decision making'
                }
            elif num_policies <= 5:
                return {
                    'impact': 0.1,
                    'explanation': 'Moderate policy exploration - normal comparison behavior'
                }
            else:
                return {
                    'impact': 0.4,
                    'explanation': 'Extensive policy discussions - may indicate indecisiveness or complex needs'
                }
        except Exception as e:
            print(f"Error analyzing interaction patterns: {str(e)}")
            return {
                'impact': 0.2,
                'explanation': 'Standard interaction pattern assessment'
            }
    
    def _assess_historical_pattern_risk(self, historical_patterns) -> List[RiskFactor]:
        """
        Incorporate risk insights from historical patterns (Agent 1 output)
        """
        factors = []
        
        if not historical_patterns:
            return factors
        
        # Calculate average similarity and risk indicators
        avg_similarity = sum(p.similarity_score for p in historical_patterns) / len(historical_patterns)
        all_risk_indicators = []
        for pattern in historical_patterns:
            all_risk_indicators.extend(pattern.risk_indicators)
        
        unique_indicators = list(set(all_risk_indicators))
        
        # Create risk factor based on historical patterns
        pattern_impact = self._calculate_pattern_impact(avg_similarity, unique_indicators)
        
        factors.append(RiskFactor(
            factor_name="historical_patterns",
            factor_type="historical",
            value=f"{len(historical_patterns)} patterns, avg similarity {avg_similarity:.2f}",
            impact_score=pattern_impact['impact'],
            confidence=avg_similarity,  # Confidence based on similarity
            weight=0.2,
            explanation=pattern_impact['explanation']
        ))
        
        return factors
    
    def _calculate_pattern_impact(self, avg_similarity: float, unique_indicators: List[str]) -> Dict[str, Any]:
        """
        Calculate impact of historical patterns on risk assessment
        """
        # Higher similarity and more indicators = higher impact
        base_impact = avg_similarity * 0.5  # Base impact from similarity
        indicator_impact = min(len(unique_indicators) * 0.1, 0.3)  # Additional impact from indicators
        
        total_impact = min(base_impact + indicator_impact, 0.8)  # Cap at 0.8
        
        return {
            'impact': total_impact,
            'explanation': f'Historical patterns show {avg_similarity:.1%} similarity with {len(unique_indicators)} risk indicators'
        }
    
    async def _gather_external_data(self, customer_profile, data_sources: List[str]) -> List[ExternalDataResult]:
        """
        Gather data from external sources (APIs, databases)
        """
        external_data = []
        
        for source in data_sources:
            try:
                if source == 'credit_score':
                    data = await self._fetch_credit_data(customer_profile)
                elif source == 'medical_records':
                    data = await self._fetch_medical_data(customer_profile)
                elif source == 'government_verification':
                    data = await self._fetch_government_data(customer_profile)
                else:
                    continue
                
                external_data.append(data)
                
            except Exception as e:
                print(f"Error fetching {source}: {str(e)}")
                # Add failed attempt record
                external_data.append(ExternalDataResult(
                    source=source,
                    data_type="unknown",
                    status="FAILED",
                    data={},
                    risk_impact=0.0
                ))
        
        return external_data
    
    def _assess_external_data_risk(self, external_data: List[ExternalDataResult]) -> List[RiskFactor]:
        """
        Assess risk factors from external data
        """
        factors = []
        
        for data in external_data:
            if data.status == "SUCCESS":
                factors.append(RiskFactor(
                    factor_name=f"external_{data.source}",
                    factor_type="external",
                    value=str(data.data),
                    impact_score=data.risk_impact,
                    confidence=0.8,
                    weight=0.15,
                    explanation=f"External data from {data.source}"
                ))
        
        return factors
    
    def _calculate_overall_risk(self, risk_factors: List[RiskFactor]) -> Tuple[float, float]:
        """
        Calculate overall risk score using weighted combination of factors
        """
        if not risk_factors:
            return 50.0, 1.0  # Default medium risk
        
        # Calculate weighted risk score
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for factor in risk_factors:
            weighted_score = factor.impact_score * 100 * factor.weight * factor.confidence
            total_weighted_score += weighted_score
            total_weight += factor.weight * factor.confidence
        
        # Normalize to 0-100 scale
        if total_weight > 0:
            risk_score = min(100, max(0, total_weighted_score / total_weight))
        else:
            risk_score = 50.0
        
        # Calculate premium multiplier based on risk score
        # Lower risk = lower premiums, higher risk = higher premiums
        base_multiplier = 1.0
        risk_adjustment = (risk_score - 50) / 100  # -0.5 to +0.5
        premium_multiplier = max(0.5, min(2.0, base_multiplier + risk_adjustment))
        
        return risk_score, premium_multiplier
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """
        Convert numerical risk score to categorical risk level
        """
        if risk_score < 20:
            return RiskLevel.VERY_LOW
        elif risk_score < 35:
            return RiskLevel.LOW
        elif risk_score < 50:
            return RiskLevel.MEDIUM
        elif risk_score < 70:
            return RiskLevel.HIGH
        elif risk_score < 85:
            return RiskLevel.VERY_HIGH
        else:
            return RiskLevel.CRITICAL
    
    def _requires_manual_review(self, risk_score: float, risk_factors: List[RiskFactor]) -> bool:
        """
        Determine if manual review is required
        """
        # High risk scores require review
        if risk_score > 75:
            return True
        
        # Certain high-impact factors require review
        high_impact_factors = [f for f in risk_factors if f.impact_score > 0.8]
        if len(high_impact_factors) > 2:
            return True
        
        # Low confidence assessments require review
        low_confidence_factors = [f for f in risk_factors if f.confidence < 0.5]
        if len(low_confidence_factors) > 3:
            return True
        
        return False
    
    async def _generate_risk_explanation(
        self, 
        risk_factors: List[RiskFactor], 
        risk_score: float, 
        risk_level: RiskLevel
    ) -> str:
        """
        Generate human-readable explanation of risk assessment
        """
        # Sort factors by impact
        sorted_factors = sorted(risk_factors, key=lambda x: x.impact_score * x.weight, reverse=True)
        top_factors = sorted_factors[:5]  # Top 5 most impactful factors
        
        factors_text = "\n".join([
            f"- {f.factor_name} ({f.factor_type}): {f.value} - {f.explanation}"
            for f in top_factors
        ])
        
        explanation_prompt = f"""
        Generate a clear, professional explanation for this insurance risk assessment:
        
        Overall Risk Score: {risk_score:.1f}/100
        Risk Level: {risk_level.value.replace('_', ' ').title()}
        
        Key Risk Factors:
        {factors_text}
        
        Provide a concise explanation (2-3 paragraphs) that:
        1. Summarizes the overall risk assessment
        2. Explains the main factors influencing the assessment
        3. Is understandable to a customer without technical jargon
        4. Maintains a professional, helpful tone
        
        Focus on the most significant factors and their business impact.
        """
        
        prompt = ChatPromptTemplate.from_template(explanation_prompt)
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            explanation = chain.invoke({})
            return explanation
        except Exception as e:
            print(f"Error generating explanation: {str(e)}")
            return f"Risk assessment completed with score {risk_score:.1f}/100 ({risk_level.value.replace('_', ' ').title()}). Assessment based on {len(risk_factors)} risk factors including demographic, health, and behavioral considerations."
    
    def _calculate_confidence_level(self, risk_factors: List[RiskFactor], external_data: List) -> float:
        """
        Calculate overall confidence in the risk assessment
        """
        if not risk_factors:
            return 0.5
        
        # Average confidence of all factors, weighted by their weight
        total_weighted_confidence = 0.0
        total_weight = 0.0
        
        for factor in risk_factors:
            weighted_confidence = factor.confidence * factor.weight
            total_weighted_confidence += weighted_confidence
            total_weight += factor.weight
        
        base_confidence = total_weighted_confidence / total_weight if total_weight > 0 else 0.5
        
        # Boost confidence if we have external data
        if external_data:
            base_confidence = min(1.0, base_confidence + 0.1)
        
        return base_confidence
    
    # Risk calculation helper methods
    def _calculate_age_risk(self, age: int) -> Dict[str, Any]:
        """Calculate age-based risk using actuarial tables"""
        if age < 18:
            return {'impact': 0.3, 'explanation': 'Minor - limited risk assessment applicable'}
        elif age < 25:
            return {'impact': 0.6, 'explanation': 'Young adult - higher accident risk, limited medical history'}
        elif age < 35:
            return {'impact': 0.3, 'explanation': 'Young adult - generally lower risk period'}
        elif age < 45:
            return {'impact': 0.4, 'explanation': 'Middle age - moderate risk, some health concerns emerging'}
        elif age < 55:
            return {'impact': 0.6, 'explanation': 'Pre-senior - increased health risks and medical needs'}
        elif age < 65:
            return {'impact': 0.7, 'explanation': 'Senior - higher medical risk and healthcare needs'}
        else:
            return {'impact': 0.8, 'explanation': 'Elderly - significant health risks and care requirements'}
    
    def _calculate_gender_risk(self, gender: str) -> Dict[str, Any]:
        """Calculate gender-based risk"""
        gender_lower = gender.lower()
        if gender_lower in ['male', 'm']:
            return {'impact': 0.1, 'explanation': 'Male gender - slightly higher accident and lifestyle risks'}
        elif gender_lower in ['female', 'f']:
            return {'impact': 0.05, 'explanation': 'Female gender - generally lower risk profile'}
        else:
            return {'impact': 0.075, 'explanation': 'Standard gender risk assessment'}
    
    def _calculate_location_risk(self, location: str) -> Dict[str, Any]:
        """Calculate location-based risk"""
        location_lower = location.lower()
        
        high_risk_locations = ['mumbai', 'delhi', 'kolkata', 'chennai']
        medium_risk_locations = ['bangalore', 'pune', 'hyderabad', 'ahmedabad']
        
        if any(city in location_lower for city in high_risk_locations):
            return {'impact': 0.6, 'explanation': 'Metropolitan area - higher pollution, traffic risks, and healthcare costs'}
        elif any(city in location_lower for city in medium_risk_locations):
            return {'impact': 0.4, 'explanation': 'Urban area - moderate environmental and lifestyle risks'}
        else:
            return {'impact': 0.3, 'explanation': 'Standard geographic risk profile'}
    
    def _calculate_occupation_risk(self, occupation: str) -> Dict[str, Any]:
        """Calculate occupation-based risk"""
        occupation_lower = occupation.lower()
        
        high_risk_occupations = ['mining', 'construction', 'aviation', 'racing', 'military', 'police']
        medium_risk_occupations = ['driver', 'chef', 'mechanic', 'nurse', 'firefighter']
        low_risk_occupations = ['office', 'teacher', 'engineer', 'manager', 'accountant', 'analyst']
        
        if any(risk in occupation_lower for risk in high_risk_occupations):
            return {'impact': 0.8, 'explanation': 'High-risk occupation with elevated injury and health risks'}
        elif any(risk in occupation_lower for risk in medium_risk_occupations):
            return {'impact': 0.5, 'explanation': 'Moderate risk occupation with some occupational hazards'}
        elif any(risk in occupation_lower for risk in low_risk_occupations):
            return {'impact': 0.2, 'explanation': 'Low-risk occupation with minimal occupational hazards'}
        else:
            return {'impact': 0.4, 'explanation': 'Standard occupational risk assessment'}
    
    async def _calculate_health_risk(self, health_info: str) -> Dict[str, Any]:
        """Calculate health-based risk using LLM analysis"""
        health_analysis_prompt = f"""
        Analyze the following health information for insurance risk assessment:
        
        Health Information: {health_info}
        
        Determine the risk impact (0.0-1.0) and provide explanation.
        Consider: chronic conditions, hereditary factors, lifestyle diseases, current symptoms.
        
        Respond in JSON format:
        {{
            "impact": 0.5,
            "explanation": "brief explanation"
        }}
        """
        
        try:
            prompt = ChatPromptTemplate.from_template(health_analysis_prompt)
            chain = prompt | self.llm | StrOutputParser()
            result = chain.invoke({})
            
            # Parse result
            import re
            impact_match = re.search(r'"impact":\s*([0-9.]+)', result)
            explanation_match = re.search(r'"explanation":\s*"([^"]+)"', result)
            
            impact = float(impact_match.group(1)) if impact_match else 0.5
            explanation = explanation_match.group(1) if explanation_match else 'Health risk assessment completed'
            
            return {'impact': impact, 'explanation': explanation}
            
        except Exception as e:
            print(f"Error in health risk calculation: {str(e)}")
            return {'impact': 0.5, 'explanation': 'Standard health risk assessment'}
    
    def _calculate_lifestyle_risk(self, lifestyle_type: str, value: str) -> Dict[str, Any]:
        """Calculate lifestyle-based risk"""
        value_lower = value.lower()
        
        if lifestyle_type == 'smoking':
            if 'non' in value_lower or 'no' in value_lower:
                return {'impact': 0.1, 'explanation': 'Non-smoker - reduced health risks'}
            else:
                return {'impact': 0.7, 'explanation': 'Smoker - significantly increased health risks'}
        
        elif lifestyle_type == 'drinking':
            if 'no' in value_lower or 'never' in value_lower:
                return {'impact': 0.1, 'explanation': 'Non-drinker - reduced health risks'}
            elif 'social' in value_lower or 'moderate' in value_lower:
                return {'impact': 0.2, 'explanation': 'Social drinker - minimal additional risk'}
            else:
                return {'impact': 0.5, 'explanation': 'Regular drinking - moderate health risks'}
        
        elif lifestyle_type == 'exercise':
            if 'regular' in value_lower or 'daily' in value_lower:
                return {'impact': 0.1, 'explanation': 'Regular exercise - reduced health risks'}
            elif 'sometimes' in value_lower or 'weekly' in value_lower:
                return {'impact': 0.2, 'explanation': 'Moderate exercise - good health maintenance'}
            else:
                return {'impact': 0.4, 'explanation': 'Sedentary lifestyle - increased health risks'}
        
        else:
            return {'impact': 0.3, 'explanation': f'Standard {lifestyle_type} risk assessment'}
    
    def _calculate_income_risk(self, income: str) -> Dict[str, Any]:
        """Calculate income-based risk"""
        # Extract numeric value if possible
        import re
        numbers = re.findall(r'\d+', income.replace(',', ''))
        
        if numbers:
            income_amount = int(numbers[0])
            if income_amount < 200000:  # Less than 2 lakhs
                return {'impact': 0.4, 'explanation': 'Lower income - potential affordability concerns'}
            elif income_amount < 500000:  # 2-5 lakhs
                return {'impact': 0.2, 'explanation': 'Middle income - standard risk profile'}
            else:  # Above 5 lakhs
                return {'impact': 0.1, 'explanation': 'Higher income - good financial stability'}
        else:
            return {'impact': 0.3, 'explanation': 'Standard income risk assessment'}
    
    def _calculate_budget_risk(self, budget: str) -> Dict[str, Any]:
        """Calculate budget-based risk"""
        budget_lower = budget.lower()
        
        if 'low' in budget_lower or 'tight' in budget_lower:
            return {'impact': 0.3, 'explanation': 'Limited budget - may affect coverage choices'}
        elif 'flexible' in budget_lower or 'high' in budget_lower:
            return {'impact': 0.1, 'explanation': 'Flexible budget - allows comprehensive coverage'}
        else:
            return {'impact': 0.2, 'explanation': 'Standard budget considerations'}
    
    def _calculate_risk_tolerance_impact(self, risk_tolerance: str) -> Dict[str, Any]:
        """Calculate risk tolerance impact"""
        tolerance_lower = risk_tolerance.lower()
        
        if 'low' in tolerance_lower or 'conservative' in tolerance_lower:
            return {'impact': 0.1, 'explanation': 'Conservative approach - prefers comprehensive coverage'}
        elif 'high' in tolerance_lower or 'aggressive' in tolerance_lower:
            return {'impact': 0.3, 'explanation': 'High risk tolerance - may choose minimal coverage'}
        else:
            return {'impact': 0.2, 'explanation': 'Moderate risk tolerance'}
    
    # Additional helper methods
    def _load_actuarial_models(self) -> Dict[str, ActuarialModel]:
        """Load actuarial models (would come from database)"""
        return {
            'demographic_model': ActuarialModel(
                model_name="Demographic Risk Model v2.1",
                model_type="demographic",
                base_rate=0.05,
                risk_multiplier=1.0,
                confidence_interval=(0.95, 0.99),
                last_updated=datetime.now()
            ),
            'health_model': ActuarialModel(
                model_name="Health Risk Assessment Model v1.8",
                model_type="health",
                base_rate=0.08,
                risk_multiplier=1.2,
                confidence_interval=(0.92, 0.98),
                last_updated=datetime.now()
            )
        }
    
    def _load_risk_weights(self) -> Dict[str, float]:
        """Load risk factor weights"""
        return {
            'age': 0.25,
            'health': 0.30,
            'occupation': 0.20,
            'location': 0.10,
            'lifestyle': 0.15
        }
    
    def _create_default_assessment(self, customer_profile) -> RiskAssessment:
        """Create default assessment in case of errors"""
        return RiskAssessment(
            customer_id=getattr(customer_profile, 'user_id', 'unknown'),
            overall_risk_level=RiskLevel.MEDIUM,
            risk_score=50.0,
            risk_factors=[],
            actuarial_models_used=[],
            recommended_premium_multiplier=1.0,
            confidence_level=0.5,
            explanation="Risk assessment completed with limited data",
            external_data_used=[],
            assessment_timestamp=datetime.now(),
            requires_manual_review=True
        )
    
    # Placeholder methods for external data integration
    async def _fetch_credit_data(self, customer_profile) -> ExternalDataResult:
        """Placeholder for credit score API integration"""
        return ExternalDataResult(
            source="credit_score",
            data_type="financial",
            status="SUCCESS",
            data={"score": 750, "status": "good"},
            risk_impact=0.3
        )
    
    async def _fetch_medical_data(self, customer_profile) -> ExternalDataResult:
        """Placeholder for medical records API integration"""
        return ExternalDataResult(
            source="medical_records",
            data_type="health",
            status="SUCCESS",
            data={"status": "no_major_conditions"},
            risk_impact=0.2
        )
    
    async def _fetch_government_data(self, customer_profile) -> ExternalDataResult:
        """Placeholder for government verification API integration"""
        return ExternalDataResult(
            source="government_verification",
            data_type="identity",
            status="SUCCESS",
            data={"verified": True},
            risk_impact=0.1
        )
    
    def _initialize_external_apis(self):
        """Initialize external API connections"""
        return {}
    
    def _load_demographic_tables(self):
        """Load demographic actuarial tables"""
        return {}

# Factory function for integration with main system
def create_risk_assessment_agent():
    """Factory function to create the risk assessment agent"""
    return ActuarialRiskAssessmentAgent()

# Example usage
async def test_risk_assessment(customer_profile, historical_patterns=None):
    """Test function for the risk assessment agent"""
    agent = create_risk_assessment_agent()
    assessment = await agent.assess_customer_risk(customer_profile, historical_patterns)
    
    print(f"Risk Assessment Results:")
    print(f"Overall Risk: {assessment.overall_risk_level.value} ({assessment.risk_score:.1f}/100)")
    print(f"Premium Multiplier: {assessment.recommended_premium_multiplier:.2f}")
    print(f"Confidence: {assessment.confidence_level:.2f}")
    print(f"Manual Review Required: {assessment.requires_manual_review}")
    print(f"\nTop Risk Factors:")
    
    sorted_factors = sorted(assessment.risk_factors, key=lambda x: x.impact_score * x.weight, reverse=True)
    for factor in sorted_factors[:3]:
        print(f"  • {factor.factor_name}: {factor.impact_score:.2f} impact")
    
    return assessment
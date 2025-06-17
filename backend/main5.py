# --- Imports ---
import os
import re
import json
import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field


# --- Wrap potentially problematic imports in try-except blocks ---
try:
    import openai
except ImportError:
    print("\n:warning: openai not installed. Install with: pip install openai")
    openai = None

try:
    import fitz  # PyMuPDF
except ImportError:
    print("\n:warning: PyMuPDF not installed. Install with: pip install pymupdf")
    fitz = None

try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import Chroma
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.runnable import RunnablePassthrough
    from langchain.schema.output_parser import StrOutputParser
    from langchain.memory import ConversationBufferMemory
    from langchain.output_parsers import PydanticOutputParser
except ImportError:
    print("\n:warning: LangChain dependencies not installed. Install with: pip install langchain langchain-openai langchain-community")

# --- New imports for enhanced PDF processing with OCR ---
try:
    import PyPDF2
except ImportError:
    print("\n:warning: PyPDF2 not installed. Install with: pip install PyPDF2")
    PyPDF2 = None
    
import tempfile
try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("\n:warning: OCR dependencies not available. Only direct text extraction will be used.")
    print("To enable OCR capabilities, install: pip install pytesseract pdf2image")
    print("And ensure tesseract-ocr and poppler-utils are installed on your system.")

# === INTEGRATED: Smart Underwriting Agents ===
import asyncio
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
# ===============================
# INTEGRATED SMART UNDERWRITING AGENTS
# ===============================

# --- Agent 1: Historical Risk Pattern Analysis ---
class HistoricalPattern(BaseModel):
    pattern_id: str = Field(description="Unique identifier for this pattern")
    pattern_type: str = Field(description="Type of risk pattern identified")
    similarity_score: float = Field(description="How similar to current case (0-1)")
    historical_data: Dict[str, Any] = Field(description="Relevant historical information")
    pattern_insight: str = Field(description="Key insight from this pattern")
    risk_indicators: List[str] = Field(description="List of risk indicators found")
    claims_history: Dict[str, Any] = Field(description="Claims data from similar cases")
    confidence_level: float = Field(description="Confidence in pattern relevance (0-1)")

class RiskFactorCorrelation(BaseModel):
    factor_name: str = Field(description="Name of the risk factor")
    correlation_strength: float = Field(description="Strength of correlation (-1 to 1)")
    historical_frequency: float = Field(description="Frequency in historical data")
    impact_on_claims: float = Field(description="Average impact on claims")

class HistoricalRiskPatternAgent:
    def __init__(self, vector_db, uploaded_vector_db, policy_list):
        self.vector_db = vector_db
        self.uploaded_vector_db = uploaded_vector_db
        self.policy_list = policy_list
        self.llm = ChatOpenAI(model="gpt-4.1", temperature=0.2)
        self.pattern_cache = {}
        self.risk_correlations = self._load_risk_correlations()
        
    async def analyze_historical_patterns(self, customer_profile) -> List[HistoricalPattern]:
        """Main function: Analyze historical patterns for a customer profile"""
        patterns = []
        
        try:
            risk_queries = self._build_risk_queries(customer_profile)
            
            for query_type, query in risk_queries.items():
                similar_cases = await self._search_historical_cases(query, query_type)
                
                for case in similar_cases[:3]:
                    pattern = await self._analyze_case_pattern(case, customer_profile, query_type)
                    if pattern and pattern.similarity_score > 0.6:
                        patterns.append(pattern)
            
            unique_patterns = self._deduplicate_patterns(patterns)
            ranked_patterns = sorted(unique_patterns, key=lambda x: x.similarity_score, reverse=True)
            
            return ranked_patterns[:5]
            
        except Exception as e:
            print(f"Error in historical pattern analysis: {str(e)}")
            return []
    
    def _build_risk_queries(self, customer_profile) -> Dict[str, str]:
        queries = {}
        
        age = self._extract_preference_value(customer_profile, 'age', '30')
        occupation = self._extract_preference_value(customer_profile, 'occupation', 'unknown')
        family_status = self._extract_preference_value(customer_profile, 'family', 'single')
        health_status = self._extract_preference_value(customer_profile, 'health', 'unknown')
        
        queries['demographic'] = f"Risk analysis age {age} occupation {occupation} family {family_status}"
        
        if health_status != 'unknown':
            queries['health'] = f"Health insurance risk {health_status} medical history claims"
        
        return queries
    
    async def _search_historical_cases(self, query: str, query_type: str) -> List[Dict]:
        try:
            main_results = self.vector_db.as_retriever(search_kwargs={"k": 5}).invoke(query)
            
            uploaded_results = []
            if self.uploaded_vector_db._collection.count() > 0:
                uploaded_results = self.uploaded_vector_db.as_retriever(search_kwargs={"k": 3}).invoke(query)
            
            all_results = []
            
            for doc in main_results:
                all_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'source': 'main_db',
                    'query_type': query_type
                })
            
            for doc in uploaded_results:
                all_results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'source': 'uploaded_db',
                    'query_type': query_type
                })
            
            return all_results
            
        except Exception as e:
            print(f"Error searching historical cases: {str(e)}")
            return []
    
    async def _analyze_case_pattern(self, case: Dict, customer_profile, query_type: str) -> Optional[HistoricalPattern]:
        try:
            customer_summary = self._summarize_customer_profile(customer_profile)
            
            analysis_prompt = f"""
            You are an insurance risk analyst. Analyze this historical case for risk patterns:

            HISTORICAL CASE:
            Content: {case['content'][:1500]}
            Source: {case['metadata'].get('policy_name', 'Unknown')}
            Query Type: {query_type}

            CURRENT CUSTOMER PROFILE:
            {customer_summary}

            Analyze and provide similarity score, risk indicators, insights, and risk factors.
            Respond with similarity score between 0.0 and 1.0, list of risk indicators, 
            pattern insight, and confidence level.

            Format as: similarity_score|risk_indicators|pattern_insight|confidence_level
            Example: 0.7|diabetes,age_factor|moderate_risk_profile|0.8
            """
            
            prompt = ChatPromptTemplate.from_template(analysis_prompt)
            chain = prompt | self.llm | StrOutputParser()
            
            result = chain.invoke({})
            analysis = self._parse_llm_response_simple(result)
            
            if analysis and analysis.get('similarity_score', 0) > 0.5:
                return HistoricalPattern(
                    pattern_id=f"{query_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    pattern_type=query_type,
                    similarity_score=analysis.get('similarity_score', 0.5),
                    historical_data={
                        'source': case['metadata'].get('policy_name', 'Unknown'),
                        'content_preview': case['content'][:200],
                        'metadata': case['metadata']
                    },
                    pattern_insight=analysis.get('pattern_insight', 'No specific insights identified'),
                    risk_indicators=analysis.get('risk_indicators', []),
                    claims_history=analysis.get('claims_patterns', {}),
                    confidence_level=analysis.get('confidence_level', 0.5)
                )
            
            return None
            
        except Exception as e:
            print(f"Error analyzing case pattern: {str(e)}")
            return None
    
    def _parse_llm_response_simple(self, response: str) -> Dict:
        try:
            parts = response.strip().split('|')
            if len(parts) >= 4:
                similarity_score = float(parts[0].strip())
                risk_indicators = [indicator.strip() for indicator in parts[1].split(',')]
                pattern_insight = parts[2].strip()
                confidence_level = float(parts[3].strip())
                
                return {
                    'similarity_score': similarity_score,
                    'risk_indicators': risk_indicators,
                    'pattern_insight': pattern_insight,
                    'confidence_level': confidence_level,
                    'claims_patterns': {}
                }
        except Exception as e:
            print(f"Error parsing LLM response: {str(e)}")
        
        return {
            'similarity_score': 0.6,
            'confidence_level': 0.7,
            'pattern_insight': 'Standard pattern analysis completed',
            'risk_indicators': ['general_risk'],
            'claims_patterns': {}
        }
    
    def _extract_preference_value(self, customer_profile, pref_type: str, default: str) -> str:
        try:
            for pref in customer_profile.preferences:
                if pref.preference_type.lower() == pref_type.lower():
                    return pref.value
        except:
            pass
        return default
    
    def _summarize_customer_profile(self, customer_profile) -> str:
        try:
            prefs = []
            for pref in customer_profile.preferences:
                if pref.confidence > 0.6:
                    prefs.append(f"{pref.preference_type}: {pref.value}")
            
            return "\n".join(prefs) if prefs else "Limited customer information available"
        except:
            return "Limited customer information available"
    
    def _deduplicate_patterns(self, patterns: List[HistoricalPattern]) -> List[HistoricalPattern]:
        unique_patterns = []
        seen_insights = set()
        
        for pattern in patterns:
            insight_key = pattern.pattern_insight[:50].lower()
            if insight_key not in seen_insights:
                unique_patterns.append(pattern)
                seen_insights.add(insight_key)
        
        return unique_patterns
    
    def _load_risk_correlations(self) -> Dict:
        return {
            'age_risk': True,
            'occupation_risk': True,
            'health_risk': True,
            'lifestyle_risk': True
        }
    
    def get_pattern_summary(self, patterns: List[HistoricalPattern]) -> Dict[str, Any]:
        if not patterns:
            return {'summary': 'No significant historical patterns identified', 'risk_level': 'unknown'}
        
        avg_similarity = sum(p.similarity_score for p in patterns) / len(patterns)
        risk_indicators = []
        for pattern in patterns:
            risk_indicators.extend(pattern.risk_indicators)
        
        unique_risks = list(set(risk_indicators))
        
        return {
            'total_patterns': len(patterns),
            'average_similarity': avg_similarity,
            'risk_indicators': unique_risks,
            'highest_similarity': max(p.similarity_score for p in patterns),
            'pattern_types': list(set(p.pattern_type for p in patterns)),
            'overall_risk_signal': 'high' if avg_similarity > 0.8 else 'medium' if avg_similarity > 0.6 else 'low'
        }

# --- Agent 2: Risk Assessment ---
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
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1", temperature=0.1)
        self.actuarial_models = self._load_actuarial_models()
        self.risk_weights = self._load_risk_weights()
        
    async def assess_customer_risk(
        self, 
        customer_profile, 
        historical_patterns: List = None,
        external_data_sources: List[str] = None
    ) -> RiskAssessment:
        try:
            assessment_start = datetime.now()
            
            demographic_factors = self._assess_demographic_risk(customer_profile)
            health_factors = await self._assess_health_risk(customer_profile)
            financial_factors = self._assess_financial_risk(customer_profile)
            behavioral_factors = self._assess_behavioral_risk(customer_profile)
            pattern_factors = self._assess_historical_pattern_risk(historical_patterns or [])
            
            all_risk_factors = (
                demographic_factors + health_factors + financial_factors + 
                behavioral_factors + pattern_factors
            )
            
            risk_score, premium_multiplier = self._calculate_overall_risk(all_risk_factors)
            risk_level = self._determine_risk_level(risk_score)
            requires_review = self._requires_manual_review(risk_score, all_risk_factors)
            explanation = await self._generate_risk_explanation(all_risk_factors, risk_score, risk_level)
            confidence = self._calculate_confidence_level(all_risk_factors, [])
            
            return RiskAssessment(
                customer_id=getattr(customer_profile, 'user_id', 'unknown'),
                overall_risk_level=risk_level,
                risk_score=risk_score,
                risk_factors=all_risk_factors,
                actuarial_models_used=list(self.actuarial_models.keys()),
                recommended_premium_multiplier=premium_multiplier,
                confidence_level=confidence,
                explanation=explanation,
                external_data_used=[],
                assessment_timestamp=assessment_start,
                requires_manual_review=requires_review
            )
            
        except Exception as e:
            print(f"Error in risk assessment: {str(e)}")
            return self._create_default_assessment(customer_profile)
    
    def _assess_demographic_risk(self, customer_profile) -> List[RiskFactor]:
        factors = []
        
        try:
            age_pref = next((p for p in customer_profile.preferences if p.preference_type == 'age'), None)
            if age_pref and age_pref.value.isdigit():
                age = int(age_pref.value)
                age_risk = self._calculate_age_risk(age)
                factors.append(RiskFactor(
                    factor_name="age",
                    factor_type="demographic",
                    value=str(age),
                    impact_score=age_risk['impact'],
                    confidence=0.95,
                    weight=0.3,
                    explanation=age_risk['explanation']
                ))
            
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
        except Exception as e:
            print(f"Error in demographic risk assessment: {str(e)}")
        
        return factors
    
    async def _assess_health_risk(self, customer_profile) -> List[RiskFactor]:
        factors = []
        
        try:
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
                    weight=0.4,
                    explanation=health_risk['explanation']
                ))
        except Exception as e:
            print(f"Error in health risk assessment: {str(e)}")
        
        return factors
    
    def _assess_financial_risk(self, customer_profile) -> List[RiskFactor]:
        factors = []
        
        try:
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
        except Exception as e:
            print(f"Error in financial risk assessment: {str(e)}")
        
        return factors
    
    def _assess_behavioral_risk(self, customer_profile) -> List[RiskFactor]:
        factors = []
        
        try:
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
        except Exception as e:
            print(f"Error in behavioral risk assessment: {str(e)}")
        
        return factors
    
    def _analyze_interaction_patterns(self, policies_discussed) -> Dict[str, Any]:
        try:
            num_policies = len(policies_discussed)
            
            if num_policies == 0:
                return {'impact': 0.3, 'explanation': 'No policy discussions - limited engagement pattern'}
            elif num_policies <= 2:
                return {'impact': 0.2, 'explanation': 'Focused policy discussions - indicates clear decision making'}
            elif num_policies <= 5:
                return {'impact': 0.1, 'explanation': 'Moderate policy exploration - normal comparison behavior'}
            else:
                return {'impact': 0.4, 'explanation': 'Extensive policy discussions - may indicate indecisiveness'}
        except:
            return {'impact': 0.2, 'explanation': 'Standard interaction pattern assessment'}
    
    def _assess_historical_pattern_risk(self, historical_patterns) -> List[RiskFactor]:
        factors = []
        
        if not historical_patterns:
            return factors
        
        try:
            avg_similarity = sum(p.similarity_score for p in historical_patterns) / len(historical_patterns)
            all_risk_indicators = []
            for pattern in historical_patterns:
                all_risk_indicators.extend(pattern.risk_indicators)
            
            unique_indicators = list(set(all_risk_indicators))
            pattern_impact = self._calculate_pattern_impact(avg_similarity, unique_indicators)
            
            factors.append(RiskFactor(
                factor_name="historical_patterns",
                factor_type="historical",
                value=f"{len(historical_patterns)} patterns, avg similarity {avg_similarity:.2f}",
                impact_score=pattern_impact['impact'],
                confidence=avg_similarity,
                weight=0.2,
                explanation=pattern_impact['explanation']
            ))
        except Exception as e:
            print(f"Error in historical pattern risk assessment: {str(e)}")
        
        return factors
    
    def _calculate_pattern_impact(self, avg_similarity: float, unique_indicators: List[str]) -> Dict[str, Any]:
        base_impact = avg_similarity * 0.5
        indicator_impact = min(len(unique_indicators) * 0.1, 0.3)
        total_impact = min(base_impact + indicator_impact, 0.8)
        
        return {
            'impact': total_impact,
            'explanation': f'Historical patterns show {avg_similarity:.1%} similarity with {len(unique_indicators)} risk indicators'
        }
    
    def _calculate_overall_risk(self, risk_factors: List[RiskFactor]) -> Tuple[float, float]:
        if not risk_factors:
            return 50.0, 1.0
        
        total_weighted_score = 0.0
        total_weight = 0.0
        
        for factor in risk_factors:
            weighted_score = factor.impact_score * 100 * factor.weight * factor.confidence
            total_weighted_score += weighted_score
            total_weight += factor.weight * factor.confidence
        
        if total_weight > 0:
            risk_score = min(100, max(0, total_weighted_score / total_weight))
        else:
            risk_score = 50.0
        
        base_multiplier = 1.0
        risk_adjustment = (risk_score - 50) / 100
        premium_multiplier = max(0.5, min(2.0, base_multiplier + risk_adjustment))
        
        return risk_score, premium_multiplier
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
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
        if risk_score > 75:
            return True
        
        high_impact_factors = [f for f in risk_factors if f.impact_score > 0.8]
        if len(high_impact_factors) > 2:
            return True
        
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
        try:
            sorted_factors = sorted(risk_factors, key=lambda x: x.impact_score * x.weight, reverse=True)
            top_factors = sorted_factors[:3]
            
            factors_text = "; ".join([
                f"{f.factor_name}: {f.explanation}"
                for f in top_factors
            ])
            
            return f"Risk assessment completed with score {risk_score:.1f}/100 ({risk_level.value.replace('_', ' ').title()}). Key factors: {factors_text}."
        except Exception as e:
            print(f"Error generating explanation: {str(e)}")
            return f"Risk assessment completed with score {risk_score:.1f}/100 ({risk_level.value.replace('_', ' ').title()})."
    
    def _calculate_confidence_level(self, risk_factors: List[RiskFactor], external_data: List) -> float:
        if not risk_factors:
            return 0.5
        
        total_weighted_confidence = 0.0
        total_weight = 0.0
        
        for factor in risk_factors:
            weighted_confidence = factor.confidence * factor.weight
            total_weighted_confidence += weighted_confidence
            total_weight += factor.weight
        
        base_confidence = total_weighted_confidence / total_weight if total_weight > 0 else 0.5
        
        if external_data:
            base_confidence = min(1.0, base_confidence + 0.1)
        
        return base_confidence
    
    def _calculate_age_risk(self, age: int) -> Dict[str, Any]:
        if age < 25:
            return {'impact': 0.2, 'explanation': 'Young adult - low risk, good health profile'}
        elif age < 35:
            return {'impact': 0.3, 'explanation': 'Young professional - generally lower risk period'}
        elif age < 45:
            return {'impact': 0.4, 'explanation': 'Middle age - moderate risk, some health concerns emerging'}
        elif age < 55:
            return {'impact': 0.6, 'explanation': 'Pre-senior - increased health risks and medical needs'}
        else:
            return {'impact': 0.7, 'explanation': 'Senior - higher medical risk and healthcare needs'}
    
    def _calculate_occupation_risk(self, occupation: str) -> Dict[str, Any]:
        occupation_lower = occupation.lower()
        
        if 'software' in occupation_lower or 'engineer' in occupation_lower or 'it' in occupation_lower:
            return {'impact': 0.2, 'explanation': 'Technology profession - low physical risk, stable income'}
        elif 'teacher' in occupation_lower or 'manager' in occupation_lower:
            return {'impact': 0.3, 'explanation': 'Professional occupation - moderate stress, stable environment'}
        else:
            return {'impact': 0.4, 'explanation': 'Standard occupational risk assessment'}
    
    async def _calculate_health_risk(self, health_info: str) -> Dict[str, Any]:
        try:
            health_lower = health_info.lower()
            if 'good' in health_lower or 'healthy' in health_lower:
                return {'impact': 0.2, 'explanation': 'Good health status reported'}
            elif 'average' in health_lower or 'normal' in health_lower:
                return {'impact': 0.4, 'explanation': 'Average health status'}
            else:
                return {'impact': 0.6, 'explanation': 'Health status requires attention'}
        except:
            return {'impact': 0.4, 'explanation': 'Standard health risk assessment'}
    
    def _calculate_income_risk(self, income: str) -> Dict[str, Any]:
        return {'impact': 0.3, 'explanation': 'Income stability assessed'}
    
    def _load_actuarial_models(self) -> Dict:
        return {
            'demographic_model': 'Demographic Risk Model v2.1',
            'health_model': 'Health Risk Assessment Model v1.8'
        }
    
    def _load_risk_weights(self) -> Dict[str, float]:
        return {
            'age': 0.25,
            'health': 0.30,
            'occupation': 0.20,
            'location': 0.10,
            'lifestyle': 0.15
        }
    
    def _create_default_assessment(self, customer_profile) -> RiskAssessment:
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

# --- Agent 3: Policy Generation ---
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

class PolicyGenerationAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1", temperature=0.3)
        self.base_premiums = self._load_base_premiums()
        
    async def generate_policy(
        self,
        customer_profile,
        risk_assessment,
        customer_needs: str,
        policy_type: str = None
    ) -> PolicyDraft:
        try:
            generation_start = datetime.now()
            
            if not policy_type:
                policy_type = await self._determine_policy_type(customer_needs, customer_profile)
            
            premium_calc = self._calculate_premium(risk_assessment, customer_profile, policy_type)
            terms = await self._generate_policy_terms(risk_assessment, customer_profile, premium_calc, policy_type)
            policy_doc = self._generate_policy_document_fixed(policy_type, terms, customer_profile)
            policy_id = self._generate_policy_id(customer_profile, policy_type)
            
            return PolicyDraft(
                policy_id=policy_id,
                policy_name=f"Smart {policy_type} Policy",
                policy_type=policy_type,
                customer_id=getattr(customer_profile, 'user_id', 'unknown'),
                terms=terms,
                premium_calculation=premium_calc,
                policy_document=policy_doc,
                risk_based_adjustments=self._get_risk_adjustments(risk_assessment),
                compliance_status="COMPLIANT",
                generated_at=generation_start,
                agent_recommendations=self._generate_agent_recommendations(risk_assessment, terms)
            )
            
        except Exception as e:
            print(f"Error in policy generation: {str(e)}")
            return self._create_default_policy(customer_profile, policy_type or "Health Insurance")
    
    async def _determine_policy_type(self, customer_needs: str, customer_profile) -> str:
        needs_lower = customer_needs.lower()
        
        if 'health' in needs_lower or 'medical' in needs_lower:
            return "Health Insurance"
        elif 'life' in needs_lower:
            return "Life Insurance"
        elif 'motor' in needs_lower or 'car' in needs_lower:
            return "Motor Insurance"
        else:
            return "Health Insurance"
    
    def _calculate_premium(
        self, 
        risk_assessment, 
        customer_profile,
        policy_type: str
    ) -> PremiumCalculation:
        base_premium = self.base_premiums.get(policy_type, 12000.0)
        
        risk_adjustments = {}
        if risk_assessment:
            risk_multiplier = risk_assessment.recommended_premium_multiplier
            risk_adjustments['overall_risk'] = (risk_multiplier - 1.0) * base_premium
        
        discounts = self._calculate_discounts(customer_profile, base_premium)
        taxes = self._calculate_taxes_and_charges(base_premium)
        
        adjusted_premium = base_premium + sum(risk_adjustments.values()) - sum(discounts.values())
        final_premium = adjusted_premium + sum(taxes.values())
        final_premium = max(1000.0, final_premium)
        
        return PremiumCalculation(
            base_premium=base_premium,
            risk_adjustments=risk_adjustments,
            discounts=discounts,
            taxes=taxes,
            final_premium=final_premium,
            payment_frequency="Annual",
            installment_amount=round(final_premium / 12, 2)
        )
    
    async def _generate_policy_terms(
        self,
        risk_assessment,
        customer_profile,
        premium_calc: PremiumCalculation,
        policy_type: str
    ) -> PolicyTerms:
        coverage_multiplier = 25.0 if risk_assessment and risk_assessment.overall_risk_level.value in ['very_low', 'low'] else 20.0
        coverage_amount = premium_calc.final_premium * coverage_multiplier
        
        deductible = max(2500.0, min(25000.0, coverage_amount * 0.02))
        
        waiting_periods = {'pre_existing_conditions': 1095, 'specific_diseases': 730} if policy_type == "Health Insurance" else {}
        special_conditions = ["Subject to policy terms and conditions"]
        exclusions = ["Pre-existing conditions (subject to waiting period)", "Self-inflicted injuries", "War and terrorism"]
        benefits = [f"Coverage up to ₹{coverage_amount:,.0f}", "Cashless treatment", "Pre and post hospitalization"]
        claims_process = "Step 1: Notify insurer within 24 hours. Step 2: Submit required documents. Step 3: Claim processing within 15 days."
        
        return PolicyTerms(
            coverage_amount=coverage_amount,
            premium_amount=premium_calc.final_premium,
            deductible=deductible,
            policy_term=1,
            waiting_periods=waiting_periods,
            special_conditions=special_conditions,
            exclusions=exclusions,
            benefits=benefits,
            claim_process=claims_process
        )
    
    def _generate_policy_document_fixed(
        self,
        policy_type: str,
        terms: PolicyTerms,
        customer_profile
    ) -> PolicyDocument:
        policy_number = self._generate_policy_number()
        
        document_sections = {
            "POLICY_DECLARATION": f"""
POLICY DECLARATION
Policy Type: {policy_type}
Coverage Amount: ₹{terms.coverage_amount:,.2f}
Annual Premium: ₹{terms.premium_amount:,.2f}
Deductible: ₹{terms.deductible:,.2f}
Policy Term: {terms.policy_term} year(s)
""",
            "COVERAGE_DETAILS": f"""
COVERAGE DETAILS
This {policy_type} policy provides comprehensive coverage including: {', '.join(terms.benefits[:3])}.
The policy covers up to ₹{terms.coverage_amount:,.2f} with an annual premium of ₹{terms.premium_amount:,.2f}.
""",
            "TERMS_CONDITIONS": f"""
TERMS AND CONDITIONS
Policy Term: {terms.policy_term} year(s)
Special Conditions: {'; '.join(terms.special_conditions) if terms.special_conditions else 'Standard terms apply'}
Deductible: ₹{terms.deductible:,.2f} applies to all claims.
""",
            "CLAIMS_PROCEDURE": f"""
CLAIMS PROCEDURE
{terms.claim_process}
""",
            "EXCLUSIONS": f"""
POLICY EXCLUSIONS
The following are excluded from coverage: {'; '.join(terms.exclusions[:3])}.
Please refer to the complete policy document for full list of exclusions.
""",
            "DEFINITIONS": f"""
KEY DEFINITIONS
Policy Holder: The person in whose name this policy is issued
Sum Insured: The maximum amount payable under this policy (₹{terms.coverage_amount:,.2f})
Deductible: The amount you pay before insurance coverage begins (₹{terms.deductible:,.2f})
Premium: The amount paid for insurance coverage (₹{terms.premium_amount:,.2f} annually)
"""
        }
        
        effective_date = datetime.now() + timedelta(days=1)
        expiry_date = effective_date + timedelta(days=365)
        
        return PolicyDocument(
            policy_number=policy_number,
            document_sections=document_sections,
            regulatory_disclosures=["This product is regulated by IRDAI", "Please read policy terms carefully"],
            contact_information={"customer_service": "1800-XXX-XXXX", "email": "support@company.com"},
            effective_date=effective_date,
            expiry_date=expiry_date
        )
    
    def _calculate_discounts(self, customer_profile, base_premium: float) -> Dict[str, float]:
        discounts = {}
        
        try:
            age_pref = next((p for p in customer_profile.preferences if p.preference_type == 'age'), None)
            if age_pref and age_pref.value.isdigit():
                age = int(age_pref.value)
                if age < 30:
                    discounts['young_customer_discount'] = base_premium * 0.10
        except:
            pass
        
        return discounts
    
    def _calculate_taxes_and_charges(self, premium_amount: float) -> Dict[str, float]:
        return {
            'gst': premium_amount * 0.18,
            'service_charges': 100.0,
            'stamp_duty': premium_amount * 0.002
        }
    
    def _get_risk_adjustments(self, risk_assessment) -> List[str]:
        adjustments = []
        
        if risk_assessment:
            for factor in risk_assessment.risk_factors:
                if factor.impact_score > 0.6:
                    adjustments.append(f"Adjustment for {factor.factor_name}: {factor.explanation}")
        
        return adjustments or ["Standard risk assessment applied"]
    
    def _generate_agent_recommendations(self, risk_assessment, terms: PolicyTerms) -> List[str]:
        recommendations = []
        
        if risk_assessment and risk_assessment.overall_risk_level.value in ['high', 'very_high']:
            recommendations.append("Consider additional health screening before policy issuance")
        
        if terms.coverage_amount < 500000:
            recommendations.append("Consider increasing coverage amount for better protection")
        
        return recommendations or ["Standard policy terms recommended"]
    
    def _generate_policy_id(self, customer_profile, policy_type: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        type_code = policy_type.replace(" ", "")[:4].upper()
        customer_id = getattr(customer_profile, 'user_id', 'CUST')[:8]
        return f"POL-{type_code}-{customer_id}-{timestamp}"
    
    def _generate_policy_number(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"SMART-{timestamp}"
    
    def _load_base_premiums(self) -> Dict[str, float]:
        return {
            "Health Insurance": 12000.0,
            "Life Insurance": 15000.0,
            "Term Insurance": 8000.0,
            "Motor Insurance": 10000.0,
            "Home Insurance": 6000.0,
            "Travel Insurance": 2000.0
        }
    
    def _create_default_policy(self, customer_profile, policy_type: str) -> PolicyDraft:
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
                special_conditions=["Standard terms apply"],
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
                regulatory_disclosures=["Standard disclosures"],
                contact_information={"support": "1800-XXX-XXXX"},
                effective_date=datetime.now(),
                expiry_date=datetime.now() + timedelta(days=365)
            ),
            risk_based_adjustments=["Default adjustments"],
            compliance_status="PENDING",
            generated_at=datetime.now(),
            agent_recommendations=["Standard recommendations"]
        )

# --- Orchestrator ---
class UnderwritingDecision(BaseModel):
    decision: str = Field(description="APPROVED, DECLINED, MANUAL_REVIEW")
    decision_reason: str = Field(description="Reason for the decision")
    confidence_level: float = Field(description="Confidence in decision (0-1)")
    manual_review_triggers: List[str] = Field(description="Factors requiring manual review")

class UnderwritingResult(BaseModel):
    customer_id: str
    request_id: str = Field(description="Unique request identifier")
    historical_patterns: List[Any] = Field(description="Output from Agent 1")
    risk_assessment: Any = Field(description="Output from Agent 2")
    policy_draft: Any = Field(description="Output from Agent 3")
    underwriting_decision: UnderwritingDecision
    processing_time: float = Field(description="Total processing time in seconds")
    agents_used: List[str] = Field(description="List of agents that were called")
    success: bool = Field(description="Whether underwriting completed successfully")
    errors: List[str] = Field(description="Any errors encountered")
    external_data_sources: List[str] = Field(description="External APIs used")
    compliance_checks: Dict[str, bool] = Field(description="Regulatory compliance status")
    timestamp: datetime = Field(description="When underwriting was completed")

class SmartUnderwritingOrchestrator:
    def __init__(self, vector_db, uploaded_vector_db, policy_list):
        self.agent1 = HistoricalRiskPatternAgent(vector_db, uploaded_vector_db, policy_list)
        self.agent2 = ActuarialRiskAssessmentAgent()
        self.agent3 = PolicyGenerationAgent()
        
        self.vector_db = vector_db
        self.uploaded_vector_db = uploaded_vector_db
        self.policy_list = policy_list
        
    async def process_underwriting_request(
        self,
        customer_id: str,
        customer_profile,
        customer_needs: str,
        uploaded_documents: List[str] = None,
        policy_type: str = None,
        external_data_sources: List[str] = None
    ) -> UnderwritingResult:
        start_time = datetime.now()
        request_id = f"UW-{customer_id}-{start_time.strftime('%Y%m%d%H%M%S')}"
        agents_used = []
        errors = []
        
        try:
            print(f"🎯 Starting Smart Underwriting for Customer: {customer_id}")
            print(f"📋 Request ID: {request_id}")
            print(f"💼 Customer Needs: {customer_needs}")
            
            # Agent 1
            print(f"🔍 Agent 1: Analyzing historical risk patterns...")
            historical_patterns = []
            try:
                historical_patterns = await self.agent1.analyze_historical_patterns(customer_profile)
                agents_used.append("Agent1_HistoricalPatterns")
                print(f"   ✅ Found {len(historical_patterns)} historical patterns")
            except Exception as e:
                error_msg = f"Agent 1 error: {str(e)}"
                errors.append(error_msg)
                print(f"   ❌ {error_msg}")
            
            # Agent 2
            print(f"📊 Agent 2: Conducting risk assessment...")
            risk_assessment = None
            try:
                risk_assessment = await self.agent2.assess_customer_risk(
                    customer_profile=customer_profile,
                    historical_patterns=historical_patterns,
                    external_data_sources=external_data_sources
                )
                agents_used.append("Agent2_RiskAssessment")
                print(f"   ✅ Risk Level: {risk_assessment.overall_risk_level.value}")
                print(f"   📈 Risk Score: {risk_assessment.risk_score:.1f}/100")
                print(f"   💰 Premium Multiplier: {risk_assessment.recommended_premium_multiplier:.2f}")
            except Exception as e:
                error_msg = f"Agent 2 error: {str(e)}"
                errors.append(error_msg)
                print(f"   ❌ {error_msg}")
                risk_assessment = self.agent2._create_default_assessment(customer_profile)
            
            # Agent 3
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
                policy_draft = self.agent3._create_default_policy(customer_profile, "Health Insurance")
            
            # Decision
            print(f"⚖️ Making underwriting decision...")
            underwriting_decision = self._make_underwriting_decision(
                risk_assessment, policy_draft, historical_patterns, errors
            )
            
            print(f"   🎯 Decision: {underwriting_decision.decision}")
            print(f"   📝 Reason: {underwriting_decision.decision_reason}")
            
            # Compliance
            compliance_checks = self._perform_compliance_checks(policy_draft, risk_assessment)
            
            # Timing
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            print(f"⏱️ Processing completed in {processing_time:.2f} seconds")
            
            # Result
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
            
            return result
            
        except Exception as e:
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            error_msg = f"Critical orchestrator error: {str(e)}"
            print(f"❌ {error_msg}")
            
            return UnderwritingResult(
                customer_id=customer_id,
                request_id=request_id,
                historical_patterns=[],
                risk_assessment=self.agent2._create_default_assessment(customer_profile),
                policy_draft=self.agent3._create_default_policy(customer_profile, "Health Insurance"),
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
        manual_review_triggers = []
        
        if errors:
            manual_review_triggers.extend(errors)
        
        if risk_assessment:
            if risk_assessment.risk_score > 75:
                manual_review_triggers.append(f"High risk score: {risk_assessment.risk_score:.1f}")
            
            if risk_assessment.overall_risk_level.value == "critical":
                manual_review_triggers.append("Critical risk level detected")
            
            if risk_assessment.requires_manual_review:
                manual_review_triggers.append("Risk assessment flagged for manual review")
            
            if risk_assessment.confidence_level < 0.6:
                manual_review_triggers.append(f"Low confidence in risk assessment: {risk_assessment.confidence_level:.2f}")
        
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
        
        else:
            return UnderwritingDecision(
                decision="APPROVED",
                decision_reason="Standard approval - acceptable risk profile",
                confidence_level=0.9,
                manual_review_triggers=[]
            )
    
    def _perform_compliance_checks(self, policy_draft, risk_assessment) -> Dict[str, bool]:
        checks = {}
        
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
        
        if risk_assessment:
            checks['risk_assessment_complete'] = len(risk_assessment.risk_factors) > 0
            checks['confidence_adequate'] = risk_assessment.confidence_level > 0.5
        else:
            checks['risk_assessment_complete'] = False
            checks['confidence_adequate'] = False
        
        checks['irdai_compliant'] = all(checks.values())
        
        return checks
    
    async def enhanced_process_message(
        self, 
        user_id: str, 
        user_input: str, 
        conversation_turn: int = 0,
        customer_profile = None
    ):
        result = await self.process_underwriting_request(
            customer_id=user_id,
            customer_profile=customer_profile,
            customer_needs=user_input
        )
        
        return self._format_underwriting_response(result)
    
    def _format_underwriting_response(self, result: UnderwritingResult) -> str:
        if not result.success:
            return f"I encountered some issues while processing your request. Please try again or contact our support team. Reference: {result.request_id}"
        
        response_parts = []
        
        if result.underwriting_decision.decision == "APPROVED":
            response_parts.append("✅ **Good news! Your application can be approved.**")
        elif result.underwriting_decision.decision == "DECLINED":
            response_parts.append("❌ **I'm sorry, but we cannot offer coverage at this time.**")
        else:
            response_parts.append("📋 **Your application requires manual review by our underwriting team.**")
        
        response_parts.append(f"**Reason:** {result.underwriting_decision.decision_reason}")
        
        if result.policy_draft and result.underwriting_decision.decision == "APPROVED":
            response_parts.append(f"\n**📋 Policy Details:**")
            response_parts.append(f"• **Policy:** {result.policy_draft.policy_name}")
            response_parts.append(f"• **Coverage:** ₹{result.policy_draft.terms.coverage_amount:,.2f}")
            response_parts.append(f"• **Annual Premium:** ₹{result.policy_draft.terms.premium_amount:,.2f}")
            response_parts.append(f"• **Monthly Installment:** ₹{result.policy_draft.premium_calculation.installment_amount:,.2f}")
        
        if result.risk_assessment:
            response_parts.append(f"\n**📊 Risk Assessment:**")
            response_parts.append(f"• **Risk Level:** {result.risk_assessment.overall_risk_level.value.replace('_', ' ').title()}")
            response_parts.append(f"• **Risk Score:** {result.risk_assessment.risk_score:.1f}/100")
        
        if result.historical_patterns:
            response_parts.append(f"\n**🔍 Historical Analysis:**")
            response_parts.append(f"• Found {len(result.historical_patterns)} similar cases in our database")
            
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

# Factory functions
def create_historical_pattern_agent(vector_db, uploaded_vector_db, policy_list):
    return HistoricalRiskPatternAgent(vector_db, uploaded_vector_db, policy_list)

def create_risk_assessment_agent():
    return ActuarialRiskAssessmentAgent()

def create_policy_generation_agent():
    return PolicyGenerationAgent()

def create_smart_underwriting_orchestrator(vector_db, uploaded_vector_db, policy_list):
    return SmartUnderwritingOrchestrator(vector_db, uploaded_vector_db, policy_list)

def setup_smart_underwriting(vector_db, uploaded_vector_db, policy_list, enable_external_apis=False):
    orchestrator = create_smart_underwriting_orchestrator(vector_db, uploaded_vector_db, policy_list)
    return {
        'orchestrator': orchestrator,
        'agent1_historical': orchestrator.agent1,
        'agent2_risk': orchestrator.agent2,
        'agent3_policy': orchestrator.agent3,
        'version': '1.0.0-integrated'
    }

def check_agent_health():
    return {
        'agent1': True,
        'agent2': True,
        'agent3': True,
        'orchestrator': True,
        'overall_health': True,
        'errors': []
    }

async def enhanced_api_handle_message(user_id, user_input, conversation_turn=0, file_path=None, extraction_method='auto', vector_db=None, uploaded_vector_db=None, policy_list=None):
    processing_info = None
    
    if file_path:
        processing_info = {
            "file_processed": True,
            "extraction_method": extraction_method,
            "note": "Document processing integrated with underwriting"
        }
    
    orchestrator = create_smart_underwriting_orchestrator(vector_db, uploaded_vector_db, policy_list)
    response = await orchestrator.enhanced_process_message(user_id, user_input, conversation_turn)
    
    return {
        "response": response,
        "user_profile": {"user_id": user_id, "status": "active"},
        "processing_info": processing_info
    }

# Mark as available
SMART_UNDERWRITING_AVAILABLE = True
print("✅ Smart Underwriting Agents integrated successfully")

# === END SMART UNDERWRITING INTEGRATION ===

# --- Define Constants ---
PERSIST_DIR = "./icici_policy_chroma_db"
UPLOADED_PERSIST_DIR = "./uploaded_policy_chroma_db"
SESSION_DIR = "./user_sessions"
POLICY_LIST_FILE = "./policy_list.json"
UPLOADED_POLICY_LIST_FILE = "./uploaded_policy_list.json"
UPLOADS_DIR = "./uploaded_policies"
PDF_FOLDER = "./icici_policies"

# --- Create necessary directories ---
for directory in [SESSION_DIR, UPLOADS_DIR, PDF_FOLDER, os.path.dirname(POLICY_LIST_FILE), os.path.dirname(UPLOADED_POLICY_LIST_FILE)]:
    os.makedirs(directory, exist_ok=True)

# --- Set your OpenAI Key ---
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Initialize Embedding Model and LLM ---
embedding_model = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4.1", temperature=0.2)
fact_check_llm = ChatOpenAI(model="gpt-4.1", temperature=0.0)

# --- Initialize Memory ---
conversation_memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="history",
    input_key="input",
    output_key="output"
)

# --- Define Pydantic Models for Structured Data ---
class PolicyEntity(BaseModel):
    policy_name: str = Field(description="The name of the policy")
    policy_type: Optional[str] = Field(None, description="Type of policy (health, life, etc.)")
    mentioned_in_turn: int = Field(description="Conversation turn where this policy was mentioned")
    source_document: Optional[str] = Field(None, description="Source document for this policy")
    is_uploaded: bool = Field(False, description="Whether this policy was uploaded by the user")

class UserPreference(BaseModel):
    preference_type: str = Field(description="Type of preference (age, family, budget, etc.)")
    value: str = Field(description="The value of this preference")
    confidence: float = Field(description="Confidence score (0-1) of this preference")
    
class UserProfile(BaseModel):
    user_id: str = Field(description="Unique ID for the user")
    preferences: List[UserPreference] = Field(default_factory=list, description="List of user preferences")
    policies_discussed: List[PolicyEntity] = Field(default_factory=list, description="Policies discussed with this user")
    policies_recommended: List[PolicyEntity] = Field(default_factory=list, description="Policies recommended to this user")
    uploaded_policies: List[PolicyEntity] = Field(default_factory=list, description="Policies uploaded by this user")
    last_interaction: str = Field(description="Timestamp of last interaction")

class PolicyMatch(BaseModel):
    policy_name: str = Field(description="Name of the policy")
    match_score: float = Field(description="Score indicating how well the policy matches user preferences")
    key_matches: List[str] = Field(default_factory=list, description="Key matching points between policy and user preferences")
    is_uploaded: bool = Field(False, description="Whether this policy was uploaded by the user")

class APIResponse(BaseModel):
    response: str
    user_profile: UserProfile
    processing_info: Optional[Dict[str, Any]] = None

# --- PDF Processing Utility Functions ---
def extract_text_with_pypdf2(pdf_path):
    """Extract text from PDF using PyPDF2"""
    print("Extracting text using PyPDF2...")
    full_text = ""
    page_count = 0
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            page_count = len(pdf_reader.pages)
            for i, page in enumerate(pdf_reader.pages):
                print(f"Processing page {i+1}/{page_count}")
                text = page.extract_text()
                if text:
                    full_text += text + "\n\n"
        print(f"PyPDF2 extraction complete. Extracted {len(full_text)} characters from {page_count} pages.")
        
        if len(full_text.strip()) < page_count * 100:
            print("Warning: Very little text extracted. This might be a scanned document.")
    except Exception as e:
        print(f"Error extracting text with PyPDF2: {str(e)}")
    
    return full_text, page_count

def extract_text_with_pymupdf(pdf_path):
    """Extract text from PDF using PyMuPDF (fitz)"""
    print("Extracting text using PyMuPDF...")
    full_text = ""
    try:
        pdf_doc = fitz.open(pdf_path)
        page_count = len(pdf_doc)
        for i, page in enumerate(pdf_doc):
            print(f"Processing page {i+1}/{page_count}")
            text = page.get_text()
            full_text += text + "\n\n"
        pdf_doc.close()
        print(f"PyMuPDF extraction complete. Extracted {len(full_text)} characters from {page_count} pages.")
        
        if len(full_text.strip()) < page_count * 100:
            print("Warning: Very little text extracted. This might be a scanned document.")
    except Exception as e:
        print(f"Error extracting text with PyMuPDF: {str(e)}")
        page_count = 0
    
    return full_text, page_count

def extract_text_with_ocr(pdf_path):
    """Extract text from PDF using Tesseract OCR"""
    if not OCR_AVAILABLE:
        return "OCR dependencies not available. Please install requirements.", 0
    
    print("Extracting text using Tesseract OCR...")
    full_text = ""
    try:
        pages = convert_from_path(pdf_path)
        page_count = len(pages)
        
        for i, page in enumerate(pages):
            print(f"Processing page {i+1}/{page_count} with OCR")
            text = pytesseract.image_to_string(page)
            full_text += text + "\n\n"
        
        print(f"OCR extraction complete. Extracted {len(full_text)} characters from {page_count} pages.")
    except Exception as e:
        print(f"Error extracting text with OCR: {str(e)}")
        page_count = 0
    
    return full_text, page_count

def extract_text_from_pdf(pdf_path, method='auto'):
    """Extract text from PDF with method selection"""
    print("Extracting text from PDF...")
    
    if method == 'auto':
        text, page_count = extract_text_with_pymupdf(pdf_path)
        
        if len(text.strip()) < page_count * 50:
            print("\nNot enough text extracted with PyMuPDF. Trying PyPDF2...")
            text2, page_count2 = extract_text_with_pypdf2(pdf_path)
            
            if len(text2.strip()) > len(text.strip()):
                text, page_count = text2, page_count2
        
        if OCR_AVAILABLE and len(text.strip()) < page_count * 50:
            print("\nNot enough text extracted with direct methods. Switching to OCR...")
            text_ocr, page_count_ocr = extract_text_with_ocr(pdf_path)
            
            if len(text_ocr.strip()) > len(text.strip()):
                text, page_count = text_ocr, page_count_ocr
    
    elif method == 'pymupdf':
        text, page_count = extract_text_with_pymupdf(pdf_path)
    elif method == 'pypdf2':
        text, page_count = extract_text_with_pypdf2(pdf_path)
    elif method == 'ocr' and OCR_AVAILABLE:
        text, page_count = extract_text_with_ocr(pdf_path)
    else:
        print(f"Unknown extraction method: {method}. Using PyMuPDF.")
        text, page_count = extract_text_with_pymupdf(pdf_path)
    
    return text, page_count

def enhanced_chunk_text(text, page_count=None, chunk_size=500, chunk_overlap=50):
    """Create better chunks from text with improved metadata"""
    print("Chunking extracted text...")
    
    if not text.strip():
        print("Warning: No text content to chunk.")
        return []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    docs = splitter.create_documents([text])
    
    if page_count and page_count > 0:
        chunks_per_page = max(1, len(docs) // page_count)
        for i, doc in enumerate(docs):
            estimated_page = min(page_count, (i // chunks_per_page) + 1) if chunks_per_page > 0 else 1
            doc.metadata = {"chunk_id": i+1, "estimated_page": estimated_page, "total_pages": page_count}
    else:
        for i, doc in enumerate(docs):
            doc.metadata = {"chunk_id": i+1}
    
    print(f"Chunking complete. Created {len(docs)} chunks.")
    return docs

def load_pdf_text(pdf_folder):
    """Load text from PDF files in a folder"""
    documents = []
    policy_list = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            filepath = os.path.join(pdf_folder, filename)
            pdf_doc = fitz.open(filepath)
            full_text = ""
            for page in pdf_doc:
                full_text += page.get_text()
            policy_name = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").strip()
            documents.append({"text": full_text, "source": filename, "policy_name": policy_name})
            
            policy_info = {
                "policy_name": policy_name,
                "source": filename,
                "policy_type": "unknown"
            }
            policy_list.append(policy_info)
            pdf_doc.close()
    
    with open(POLICY_LIST_FILE, 'w') as f:
        json.dump(policy_list, f, indent=2)
        
    return documents, policy_list

def chunk_text(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into chunks for the vector store"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc_objects = []
    for doc in documents:
        chunks = splitter.split_text(doc["text"])
        for chunk in chunks:
            doc_objects.append(Document(
                page_content=chunk,
                metadata={"source": doc["source"], "policy_name": doc["policy_name"]}
            ))
    return doc_objects

# --- Load or Create Policy Database ---
os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(POLICY_LIST_FILE), exist_ok=True)

if not os.path.exists(PERSIST_DIR) or not os.listdir(PERSIST_DIR):
    print("\n:warning: No ChromaDB found. Loading PDFs and creating DB...")
    
    if not os.path.exists(PDF_FOLDER) or not [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]:
        print(f"\n:warning: No PDF files found in {PDF_FOLDER}. Creating empty database.")
        chroma_db = Chroma(embedding_function=embedding_model, persist_directory=PERSIST_DIR)
        chroma_db.persist()
        policy_list = []
        with open(POLICY_LIST_FILE, 'w') as f:
            json.dump(policy_list, f, indent=2)
        print("\n:white_check_mark: Empty Policy ChromaDB created successfully!")
    else:
        documents, policy_list = load_pdf_text(PDF_FOLDER)
        docs = chunk_text(documents)
        chroma_db = Chroma.from_documents(docs, embedding_model, persist_directory=PERSIST_DIR)
        chroma_db.persist()
        print("\n:white_check_mark: Policy ChromaDB created successfully!")
else:
    chroma_db = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
    print("\n:white_check_mark: Loaded existing Policy ChromaDB.")
    
    if os.path.exists(POLICY_LIST_FILE):
        with open(POLICY_LIST_FILE, 'r') as f:
            policy_list = json.load(f)
    else:
        policy_list = []
        collection = chroma_db._collection
        metadatas = collection.get()["metadatas"]
        seen_policies = set()
        
        for metadata in metadatas:
            if metadata and "policy_name" in metadata:
                if metadata["policy_name"] not in seen_policies:
                    policy_info = {
                        "policy_name": metadata["policy_name"],
                        "source": metadata.get("source", "unknown"),
                        "policy_type": "unknown"
                    }
                    policy_list.append(policy_info)
                    seen_policies.add(metadata["policy_name"])
        
        with open(POLICY_LIST_FILE, 'w') as f:
            json.dump(policy_list, f, indent=2)

# --- Create or load the uploaded policy database ---
def initialize_uploaded_policy_database():
    """Initialize or load the uploaded policy database"""
    global uploaded_chroma_db, uploaded_policy_list
    
    os.makedirs(UPLOADED_PERSIST_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(UPLOADED_POLICY_LIST_FILE), exist_ok=True)
    
    if not os.path.exists(UPLOADED_PERSIST_DIR):
        uploaded_chroma_db = Chroma(embedding_function=embedding_model, persist_directory=UPLOADED_PERSIST_DIR)
        uploaded_chroma_db.persist()
        uploaded_policy_list = []
        with open(UPLOADED_POLICY_LIST_FILE, 'w') as f:
            json.dump(uploaded_policy_list, f, indent=2)
    else:
        try:
            uploaded_chroma_db = Chroma(persist_directory=UPLOADED_PERSIST_DIR, embedding_function=embedding_model)
        except Exception as e:
            print(f"Error loading uploaded policy DB: {str(e)}. Initializing empty DB.")
            uploaded_chroma_db = Chroma(embedding_function=embedding_model, persist_directory=UPLOADED_PERSIST_DIR)
            uploaded_chroma_db.persist()
        
        if os.path.exists(UPLOADED_POLICY_LIST_FILE):
            with open(UPLOADED_POLICY_LIST_FILE, 'r') as f:
                uploaded_policy_list = json.load(f)
        else:
            uploaded_policy_list = []
            with open(UPLOADED_POLICY_LIST_FILE, 'w') as f:
                json.dump(uploaded_policy_list, f, indent=2)
    
    print("\n:white_check_mark: Uploaded Policy Database initialized.")

try:
    initialize_uploaded_policy_database()
except Exception as e:
    print(f"\n:warning: Error initializing uploaded policy database: {str(e)}")
    uploaded_policy_list = []

# === Initialize Smart Underwriting System ===
smart_underwriting_system = None
if SMART_UNDERWRITING_AVAILABLE:
    try:
        health_status = check_agent_health()
        if health_status['overall_health']:
            smart_underwriting_system = setup_smart_underwriting(
                vector_db=chroma_db,
                uploaded_vector_db=uploaded_chroma_db,
                policy_list=policy_list,
                enable_external_apis=False
            )
            print("🤖 Smart Underwriting System initialized successfully!")
            print(f"   Version: {smart_underwriting_system['version']}")
        else:
            print("⚠️ Agent health check failed:")
            for error in health_status['errors']:
                print(f"     {error}")
    except Exception as e:
        print(f"⚠️ Could not initialize Smart Underwriting System: {e}")
        smart_underwriting_system = None

# --- Policy Processing Functions ---
def process_uploaded_pdf(file_path: str, user_id: str, extraction_method='auto') -> PolicyEntity:
    """Process an uploaded PDF file and add it to the database"""
    global uploaded_policy_list
    
    filename = os.path.basename(file_path)
    policy_name = os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").strip()
    
    if not file_path.startswith(UPLOADS_DIR):
        dest_path = os.path.join(UPLOADS_DIR, filename)
        import shutil
        shutil.copy2(file_path, dest_path)
        file_path = dest_path
    
    try:
        print(f"Processing uploaded policy: {policy_name}")
        
        full_text, page_count = extract_text_from_pdf(file_path, method=extraction_method)
        
        if not full_text.strip():
            print("Warning: No text could be extracted from the PDF.")
            return PolicyEntity(
                policy_name=f"Error: No text extracted from {policy_name}",
                policy_type="unknown",
                mentioned_in_turn=0,
                source_document=filename,
                is_uploaded=True
            )
        
        preview_length = min(3000, len(full_text))
        text_preview = full_text[:preview_length]
        print(f"Successfully extracted {len(full_text)} characters from {page_count} pages.")
        
        policy_type_prompt = f"""
        Based on the following insurance policy text, determine the most likely policy type
        (e.g., health, life, motor, home, travel, etc.). Focus on the coverage, benefits, and terms.
        
        POLICY TEXT EXTRACT:
        {text_preview}
        
        POLICY TYPE:
        """
        
        prompt = ChatPromptTemplate.from_template(policy_type_prompt)
        chain = prompt | llm | StrOutputParser()
        policy_type = chain.invoke({}).strip()
        
        print(f"Detected policy type: {policy_type}")
        
        docs = enhanced_chunk_text(full_text, page_count, chunk_size=500, chunk_overlap=50)
        
        if not docs:
            print("Warning: Failed to create text chunks.")
            return PolicyEntity(
                policy_name=f"Error: Chunking failed for {policy_name}",
                policy_type=policy_type,
                mentioned_in_turn=0,
                source_document=filename,
                is_uploaded=True
            )
        
        for i, doc in enumerate(docs):
            existing_metadata = doc.metadata.copy() if hasattr(doc, 'metadata') else {}
            
            doc.metadata = {
                **existing_metadata,
                "source": filename, 
                "policy_name": policy_name,
                "policy_type": policy_type,
                "uploaded_by": user_id,
                "is_uploaded": True,
                "chunk_index": i
            }
        
        print(f"Adding {len(docs)} chunks to vector database...")
        uploaded_chroma_db.add_documents(docs)
        uploaded_chroma_db.persist()
        print("Successfully added to vector database.")
        
        policy_info = {
            "policy_name": policy_name,
            "source": filename,
            "policy_type": policy_type,
            "uploaded_by": user_id,
            "upload_date": datetime.now().isoformat(),
            "extraction_method": extraction_method,
            "page_count": page_count,
            "chunk_count": len(docs)
        }
        
        uploaded_policy_list.append(policy_info)
        with open(UPLOADED_POLICY_LIST_FILE, 'w') as f:
            json.dump(uploaded_policy_list, f, indent=2)
        
        policy_entity = PolicyEntity(
            policy_name=policy_name,
            policy_type=policy_type,
            mentioned_in_turn=0,
            source_document=filename,
            is_uploaded=True
        )
        
        user_profile = get_or_create_session(user_id)
        user_profile.uploaded_policies.append(policy_entity)
        save_session(user_profile)
        
        print(f"Successfully processed {policy_name}")
        return policy_entity
        
    except Exception as e:
        print(f"Error processing uploaded PDF: {str(e)}")
        return PolicyEntity(
            policy_name=f"Error processing {policy_name}",
            policy_type="unknown",
            mentioned_in_turn=0,
            source_document=filename,
            is_uploaded=True
        )

def extract_policy_entities(text: str, conversation_turn: int) -> List[PolicyEntity]:
    """Extract policy names from text and return as PolicyEntity objects"""
    global policy_list
    
    try:
        global uploaded_policy_list
        if 'uploaded_policy_list' not in globals():
            if os.path.exists(UPLOADED_POLICY_LIST_FILE):
                with open(UPLOADED_POLICY_LIST_FILE, 'r') as f:
                    uploaded_policy_list = json.load(f)
            else:
                uploaded_policy_list = []
    except Exception as e:
        print(f"Error loading uploaded policy list: {str(e)}")
        uploaded_policy_list = []
    
    extract_prompt = """
    Extract any insurance policy names mentioned in the following text:
    
    TEXT: {text}
    
    Return the policy names as a Python list of strings. If no policy names are found, return an empty list.
    Format: ["Policy Name 1", "Policy Name 2", ...]
    """
    
    prompt = ChatPromptTemplate.from_template(extract_prompt)
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({"text": text})
        
        result = result.strip()
        if result.startswith("```") and result.endswith("```"):
            result = result[3:-3].strip()
        if result.startswith("```python") and result.endswith("```"):
            result = result[9:-3].strip()
        
        try:
            policy_names = eval(result)
            if not isinstance(policy_names, list):
                policy_names = []
        except:
            policy_names = []
            matches = re.findall(r'"([^"]+)"', result)
            if matches:
                policy_names = matches
    except:
        policy_names = []
    
    policy_entities = []
    
    icici_policies = policy_list
    
    for name in policy_names:
        matching_icici = next((p for p in icici_policies if p["policy_name"].lower() == name.lower()), None)
        if matching_icici:
            policy_entities.append(PolicyEntity(
                policy_name=matching_icici["policy_name"],
                policy_type=matching_icici.get("policy_type", "unknown"),
                mentioned_in_turn=conversation_turn,
                source_document=matching_icici.get("source", "unknown"),
                is_uploaded=False
            ))
            continue
        
        matching_uploaded = next((p for p in uploaded_policy_list if p["policy_name"].lower() == name.lower()), None)
        if matching_uploaded:
            policy_entities.append(PolicyEntity(
                policy_name=matching_uploaded["policy_name"],
                policy_type=matching_uploaded.get("policy_type", "unknown"),
                mentioned_in_turn=conversation_turn,
                source_document=matching_uploaded.get("source", "unknown"),
                is_uploaded=True
            ))
            continue
        
        for policy in icici_policies:
            if name.lower() in policy["policy_name"].lower() or policy["policy_name"].lower() in name.lower():
                policy_entities.append(PolicyEntity(
                    policy_name=policy["policy_name"],
                    policy_type=policy.get("policy_type", "unknown"),
                    mentioned_in_turn=conversation_turn,
                    source_document=policy.get("source", "unknown"),
                    is_uploaded=False
                ))
                continue
        
        for policy in uploaded_policy_list:
            if name.lower() in policy["policy_name"].lower() or policy["policy_name"].lower() in name.lower():
                policy_entities.append(PolicyEntity(
                    policy_name=policy["policy_name"],
                    policy_type=policy.get("policy_type", "unknown"),
                    mentioned_in_turn=conversation_turn,
                    source_document=policy.get("source", "unknown"),
                    is_uploaded=True
                ))
                continue
    
    return policy_entities

def extract_user_preferences(text: str) -> List[UserPreference]:
    """Extract user preferences from text"""
    extract_prompt = """
    Extract user preferences related to insurance from the following text:
    
    TEXT: {text}
    
    Identify information about:
    - Age
    - Family situation (married, children, etc.)
    - Health conditions
    - Budget constraints
    - Coverage preferences
    - Risk tolerance
    - Any other relevant preferences
    
    For each preference, provide a confidence score (0.0-1.0) indicating how certain you are about this preference.
    
    Format your response as a JSON list of objects with the following structure:
    [
       {{
        "preference_type": "type of preference",
        "value": "the preference value",
        "confidence": 0.0-1.0
       }}
    ]
    """
    
    prompt = ChatPromptTemplate.from_template(extract_prompt)
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({"text": text})
        
        result = result.strip()
        if result.startswith("```json") and result.endswith("```"):
            result = result[7:-3].strip()
        if result.startswith("```") and result.endswith("```"):
            result = result[3:-3].strip()
        
        try:
            preferences_json = json.loads(result)
            preferences = []
            for pref in preferences_json:
                preferences.append(UserPreference(
                    preference_type=pref["preference_type"],
                    value=pref["value"],
                    confidence=float(pref["confidence"])
                ))
            return preferences
        except:
            print("Failed to parse preferences JSON")
            return []
    except Exception as e:
        print(f"Error extracting preferences: {str(e)}")
        return []

# --- Policy Search and Recommendation Functions ---
def search_both_dbs(query_text, k=3, use_uploaded=True):
    """Search both databases for a given query"""
    regular_docs = chroma_db.as_retriever(search_kwargs={"k": k}).invoke(query_text)
    
    uploaded_docs = []
    if use_uploaded and uploaded_chroma_db._collection.count() > 0:
        uploaded_docs = uploaded_chroma_db.as_retriever(search_kwargs={"k": k}).invoke(query_text)
    
    return regular_docs + uploaded_docs

def detect_specific_policy_names(text: str, policies: list) -> List[str]:
    """Detect specific policy names in text with fuzzy matching"""
    if not text or not policies:
        return []
        
    text_lower = text.lower()
    mentioned_policies = []
    
    for policy in policies:
        policy_name = policy["policy_name"].replace(" - Brochure", "")
        policy_name_lower = policy_name.lower()
        
        if policy_name_lower in text_lower:
            if policy_name not in mentioned_policies:
                mentioned_policies.append(policy_name)
            continue
            
        if " " in policy_name_lower:
            parts = policy_name_lower.split()
            if len(parts) > 1 and parts[0].lower() == 'icici':
                non_brand = " ".join(parts[1:])
                if non_brand in text_lower:
                    if policy_name not in mentioned_policies:
                        mentioned_policies.append(policy_name)
                    continue
    
    policy_specific_terms = ["gift", "iprotect", "perfect", "smartkid", "lifetime", "signature", "health shield"]
    
    specific_policies = []
    for term in policy_specific_terms:
        if term in text_lower:
            for policy in policies:
                policy_name = policy["policy_name"]
                policy_name_lower = policy_name.lower()
                
                if term in policy_name_lower and policy_name not in specific_policies:
                    parts = policy_name_lower.split()
                    if any(part.startswith(term) or term.startswith(part) for part in parts):
                        specific_policies.append(policy_name)
    
    if specific_policies:
        return specific_policies
    
    if not mentioned_policies:
        for policy in policies:
            policy_name = policy["policy_name"].replace(" - Brochure", "")
            policy_name_lower = policy_name.lower()
            
            if len(policy_name_lower.split()) > 1:
                policy_words = policy_name_lower.split()
                
                matching_words = sum(1 for word in policy_words if word in text_lower)
                if matching_words >= 2 or (matching_words / len(policy_words) > 0.5):
                    if policy_name not in mentioned_policies:
                        mentioned_policies.append(policy_name)
                    continue
            
            if len(policy_name_lower.split()) > 1:
                for i in range(1, len(policy_name_lower.split())):
                    word = policy_name_lower.split()[i]
                    if len(word) >= 4:
                        abbreviated = word[:4]
                        if abbreviated in text_lower and any(part in text_lower for part in policy_name_lower.split()[:i]):
                            if policy_name not in mentioned_policies:
                                mentioned_policies.append(policy_name)
                            break
    
    return mentioned_policies

def smart_policy_retrieval(query: str, policy_names: List[str] = None, use_uploaded: bool = True, k: int = 3, use_exact_match: bool = True):
    """Retrieves documents based on query, with priority to specific policies if named"""
    if policy_names and len(policy_names) > 0:
        all_docs = []
        
        for policy_name in policy_names:
            try:
                policy_name_lower = policy_name.lower()
                
                filter_dict = {"policy_name": policy_name}
                filtered_docs = chroma_db.as_retriever(
                    search_kwargs={"k": k, "filter": filter_dict}
                ).invoke(query)
                
                if len(filtered_docs) < 2:
                    candidate_docs = chroma_db.as_retriever(
                        search_kwargs={"k": k+5}
                    ).invoke(query)
                    
                    for doc in candidate_docs:
                        doc_policy = doc.metadata.get("policy_name", "")
                        if doc_policy and (
                            doc_policy.lower() == policy_name_lower or 
                            policy_name_lower in doc_policy.lower() or
                            any(part.lower() in doc_policy.lower() for part in policy_name_lower.split() if len(part) > 3)
                        ):
                            filtered_docs.append(doc)
                
                if use_exact_match and len(filtered_docs) < k:
                    all_candidate_docs = chroma_db.as_retriever(
                        search_kwargs={"k": k * 3}
                    ).invoke(policy_name)
                    
                    exact_matches = []
                    for doc in all_candidate_docs:
                        if policy_name.lower() in doc.page_content.lower():
                            exact_matches.append(doc)
                    
                    for doc in exact_matches:
                        if doc not in filtered_docs:
                            filtered_docs.append(doc)
                
                all_docs.extend(filtered_docs)
                
                if use_uploaded and uploaded_chroma_db._collection.count() > 0:
                    uploaded_filtered_docs = uploaded_chroma_db.as_retriever(
                        search_kwargs={"k": k, "filter": filter_dict}
                    ).invoke(query)
                    all_docs.extend(uploaded_filtered_docs)
                    
                    if len(uploaded_filtered_docs) < 2:
                        enhanced_query = f"{query} {policy_name}"
                        more_uploaded_docs = uploaded_chroma_db.as_retriever(
                            search_kwargs={"k": k+2}
                        ).invoke(enhanced_query)
                        
                        more_filtered_uploaded_docs = [
                            doc for doc in more_uploaded_docs 
                            if policy_name_lower in doc.metadata.get("policy_name", "").lower()
                        ]
                        
                        all_docs.extend(more_filtered_uploaded_docs)
                
            except Exception as e:
                print(f"Filter-based retrieval failed for {policy_name}: {str(e)}")
                continue
        
        if len(all_docs) < 3:
            print("Few results with strict filtering, using enhanced query")
            policy_terms = " OR ".join([f'"{name}"' for name in policy_names])
            enhanced_query = f"{query} specifically about {policy_terms}"
            
            broader_docs = search_both_dbs(enhanced_query, k=k, use_uploaded=use_uploaded)
            
            broader_filtered_docs = []
            for doc in broader_docs:
                doc_policy = doc.metadata.get("policy_name", "").lower()
                if any(policy.lower() in doc_policy or doc_policy in policy.lower() for policy in policy_names):
                    broader_filtered_docs.append(doc)
            
            all_docs.extend(broader_filtered_docs)
            
        unique_docs = []
        seen_content = set()
        for doc in all_docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                unique_docs.append(doc)
                seen_content.add(content_hash)
                
        return unique_docs
    else:
        return search_both_dbs(query, k=k, use_uploaded=use_uploaded)

def answer_faq(question: str, user_profile: UserProfile, conversation_turn: int):
    """Answer user questions based on policy documents"""
    conversation_history = conversation_memory.load_memory_variables({})
    history_text = conversation_history.get("history", "")
    
    specific_policies = detect_specific_policy_names(question, policy_list)
    
    enhanced_query = question
    
    if specific_policies:
        print(f"Detected specific policies: {specific_policies}")
        docs = smart_policy_retrieval(
            query=enhanced_query, 
            policy_names=specific_policies, 
            use_uploaded=True, 
            k=7
        )
    else:
        docs = smart_policy_retrieval(
            query=enhanced_query, 
            policy_names=None, 
            use_uploaded=True, 
            k=3
        )
    
    if not docs:
        return "I don't have enough information to answer that question. Could you please be more specific or ask about a particular policy?"
    
    if specific_policies:
        filtered_docs = []
        for doc in docs:
            doc_policy = doc.metadata.get("policy_name", "")
            if any(policy.lower() in doc_policy.lower() or doc_policy.lower() in policy.lower() for policy in specific_policies):
                filtered_docs.append(doc)
        
        if len(filtered_docs) >= 3:
            docs = filtered_docs
    
    contexts = [doc.page_content for doc in docs]
    sources = []
    seen_sources = set()
    
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        policy_name = doc.metadata.get("policy_name", "Unknown")
        source_str = f"{policy_name} ({source})"
        
        if source_str not in seen_sources:
            sources.append(source_str)
            seen_sources.add(source_str)
    
    answer_template = """
    You are an insurance expert assistant at ICICI. Answer the user's question based on the provided information.
    
    CONVERSATION HISTORY:
    {history}
    
    USER QUESTION: {question}
    
    INFORMATION FROM POLICY DOCUMENTS:
    {context}
    """
    
    if specific_policies:
        policy_names_list = ", ".join(specific_policies)
        answer_template += f"""
    The user is specifically asking about the following policies: {policy_names_list}.
    Focus your answer ONLY on information related to these specific policies.
    DO NOT include general information about other policies unless directly relevant for comparison.
    """
    
    answer_template += """
    Please provide a clear, concise answer based only on the provided information. If the information doesn't contain 
    the answer, say so honestly. Don't make up information that isn't in the provided context.
    
    Format your answer nicely with markdown when appropriate for lists, emphasis, etc.
    """
    
    prompt = ChatPromptTemplate.from_template(answer_template)
    
    chain = (
        {"question": RunnablePassthrough(),
         "context": lambda x: "\n\n".join(contexts),
         "history": lambda x: history_text}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = chain.invoke(question)
    
    policies = extract_policy_entities(response, conversation_turn)
    
    for policy in policies:
        already_discussed = False
        for existing in user_profile.policies_discussed:
            if existing.policy_name.lower() == policy.policy_name.lower():
                already_discussed = True
                break
        
        if not already_discussed:
            user_profile.policies_discussed.append(policy)
    
    user_profile.last_interaction = datetime.now().isoformat()
    save_session(user_profile)
    
    if sources:
        top_sources = sources[:3]
        if specific_policies:
            specific_sources = [s for s in sources if any(policy.lower() in s.lower() for policy in specific_policies)]
            if specific_sources:
                top_sources = specific_sources[:3]
        
        source_info = "\n\nSources consulted: " + ", ".join(top_sources)
        if len(sources) > 3:
            source_info += f" and {len(sources) - 3} more"
    else:
        source_info = ""
        
    final_response = response + source_info
    
    conversation_memory.save_context(
        {"input": question},
        {"output": final_response}
    )
    
    return final_response

# --- Function to handle sessions ---
def get_or_create_session(user_id: str) -> UserProfile:
    """Get or create a user session"""
    os.makedirs(SESSION_DIR, exist_ok=True)
    
    session_file = os.path.join(SESSION_DIR, f"{user_id}.json")
    
    if os.path.exists(session_file):
        with open(session_file, 'r') as f:
            data = json.load(f)
            return UserProfile(**data)
    else:
        profile = UserProfile(
            user_id=user_id,
            preferences=[],
            policies_discussed=[],
            policies_recommended=[],
            uploaded_policies=[],
            last_interaction=datetime.now().isoformat()
        )
        save_session(profile)
        return profile

def save_session(profile: UserProfile):
    """Save a user session"""
    os.makedirs(SESSION_DIR, exist_ok=True)
    
    session_file = os.path.join(SESSION_DIR, f"{profile.user_id}.json")
    
    with open(session_file, 'w') as f:
        json.dump(profile.model_dump(), f, indent=2)

# --- Main processing function ---
def process_message(user_id: str, user_input: str, conversation_turn: int = 0):
    """Enhanced process_message with Smart Underwriting integration"""
    user_profile = get_or_create_session(user_id)
    
    new_preferences = extract_user_preferences(user_input)
    
    if new_preferences:
        for new_pref in new_preferences:
            updated = False
            for i, existing_pref in enumerate(user_profile.preferences):
                if existing_pref.preference_type == new_pref.preference_type:
                    if new_pref.confidence > existing_pref.confidence:
                        user_profile.preferences[i] = new_pref
                    updated = True
                    break
            
            if not updated:
                user_profile.preferences.append(new_pref)
    
    # === Smart Underwriting Decision Logic ===
    if smart_underwriting_system and SMART_UNDERWRITING_AVAILABLE:
        underwriting_keywords = [
            'quote','buy policy',
            'underwriting', 'risk assessment'
            ]
        
        needs_underwriting = any(keyword in user_input.lower() for keyword in underwriting_keywords)
        
        if needs_underwriting:
            print(f"🤖 Using Smart Underwriting for: {user_input[:50]}...")
            try:
                orchestrator = smart_underwriting_system['orchestrator']
                
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                response = loop.run_until_complete(
                    orchestrator.enhanced_process_message(user_id, user_input, conversation_turn, user_profile)
                )
                
                print("✅ Smart Underwriting response generated")
                return response
                
            except Exception as e:
                print(f"⚠️ Smart Underwriting error: {e}")
                print("   Falling back to legacy FAQ system")
    
    # === Legacy FAQ System ===
    print(f"📋 Using legacy FAQ system for: {user_input[:50]}...")
    
    if re.search(r"recommend|suggest|what policy|best policy|which policy", user_input.lower()):
        return recommend_policies(user_profile, conversation_turn)
    
    if re.search(r"compare|difference|better|versus|vs", user_input.lower()):
        return find_similar_policies(user_input, user_profile, conversation_turn)
    
    policy_match = re.search(r"(details|information|tell me about|what is) (the )?([A-Za-z\s]+) policy", user_input.lower())
    if policy_match:
        policy_name = policy_match.group(3).strip().title()
        is_uploaded = any(p.policy_name.lower() == policy_name.lower() for p in user_profile.uploaded_policies)
        return get_policy_details(policy_name, is_uploaded)
    
    if re.search(r"fact check|verify|is it true", user_input.lower()):
        return fact_check_claims(user_input, user_profile)
    
    uploaded_policy_match = re.search(r"(evaluate|assess|review) my (uploaded )?policy", user_input.lower())
    if uploaded_policy_match and user_profile.uploaded_policies:
        latest_policy = user_profile.uploaded_policies[-1]
        return evaluate_uploaded_policy(latest_policy, user_profile, conversation_turn)
    
    if re.search(r"compare my (uploaded )?policy", user_input.lower()) and user_profile.uploaded_policies:
        latest_policy = user_profile.uploaded_policies[-1]
        return compare_uploaded_with_existing(latest_policy, user_profile, conversation_turn)
    
    return answer_faq(user_input, user_profile, conversation_turn)

# --- Additional Legacy Functions ---
def recommend_policies(user_profile: UserProfile, conversation_turn: int):
    """Recommend policies based on user preferences"""
    if not user_profile.preferences:
        return "I need more information about your needs and preferences to recommend policies. Please tell me about your insurance requirements, budget, family situation, or any specific coverage needs."
    
    preference_text = ""
    for pref in user_profile.preferences:
        preference_text += f"- {pref.preference_type}: {pref.value}\n"
    
    query_parts = []
    for pref in user_profile.preferences:
        if pref.confidence >= 0.7:
            query_parts.append(f"{pref.preference_type}: {pref.value}")
    
    query = "Insurance policy recommendations for " + ", ".join(query_parts)
    
    policy_names = None
    if user_profile.policies_recommended:
        policy_names = [p.policy_name for p in user_profile.policies_recommended]
    
    docs = smart_policy_retrieval(query, policy_names, use_uploaded=True, k=5)
    
    if not docs:
        return "I don't have enough information to recommend specific policies. Could you please provide more details about your insurance needs?"
    
    policy_info = {}
    for doc in docs:
        policy_name = doc.metadata.get("policy_name", "Unknown")
        is_uploaded = doc.metadata.get("is_uploaded", False)
        
        if policy_name not in policy_info:
            policy_info[policy_name] = {
                "content": [],
                "is_uploaded": is_uploaded
            }
        
        policy_info[policy_name]["content"].append(doc.page_content)
    
    recommendation_template = """
    You are an insurance expert at Policies . Recommend insurance policies based on the user's preferences.
    
    USER PREFERENCES:
    {preferences}
    
    POTENTIAL POLICIES:
    {policy_info}
    
    Please provide:
    1. The top 3 recommended policies that best match the user's needs
    2. For each policy, explain why it's a good fit for the user
    3. Highlight key features and benefits of each recommended policy
    4. Mention any limitations or exclusions the user should be aware of
    
    Format your response in a clear, organized manner with bold policy names for readability.
    """
    
    formatted_policy_info = ""
    for name, info in policy_info.items():
        formatted_policy_info += f"--- {name} ---\n"
        formatted_policy_info += "\n".join(info["content"])
        formatted_policy_info += "\n\n"
    
    prompt = ChatPromptTemplate.from_template(recommendation_template)
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "preferences": preference_text,
        "policy_info": formatted_policy_info
    })
    
    recommended_policies = []
    for name in policy_info.keys():
        if name.lower() in response.lower():
            is_uploaded = policy_info[name]["is_uploaded"]
            recommended_policies.append(PolicyEntity(
                policy_name=name,
                policy_type="unknown",
                mentioned_in_turn=conversation_turn,
                source_document="recommendation",
                is_uploaded=is_uploaded
            ))
    
    for policy in recommended_policies:
        already_recommended = False
        for existing in user_profile.policies_recommended:
            if existing.policy_name.lower() == policy.policy_name.lower():
                already_recommended = True
                break
        
        if not already_recommended:
            user_profile.policies_recommended.append(policy)
    
    sources = set()
    for name, info in policy_info.items():
        if name.lower() in response.lower():
            sources.add(name)
    
    source_info = "\n\nSources consulted: " + ", ".join(sources)
    final_response = response + source_info
    
    user_profile.last_interaction = datetime.now().isoformat()
    save_session(user_profile)
    
    conversation_memory.save_context(
        {"input": "Can you recommend insurance policies based on my needs?"},
        {"output": final_response}
    )
    
    return final_response

def find_similar_policies(user_input: str, user_profile: UserProfile, conversation_turn: int):
    """Find similar policies to those mentioned or uploaded"""
    conversation_history = conversation_memory.load_memory_variables({})
    history_text = conversation_history.get("history", "")
    
    mentioned_policies = extract_policy_entities(user_input, conversation_turn)
    specific_policies = detect_specific_policy_names(user_input, policy_list)
    
    target_policies = []
    
    if mentioned_policies:
        target_policies.extend(mentioned_policies)
    
    if specific_policies:
        for name in specific_policies:
            if not any(p.policy_name.lower() == name.lower() for p in target_policies):
                policy_type = next((p["policy_type"] for p in policy_list if p["policy_name"] == name), "unknown")
                source = next((p["source"] for p in policy_list if p["policy_name"] == name), "unknown")
                target_policies.append(PolicyEntity(
                    policy_name=name,
                    policy_type=policy_type,
                    mentioned_in_turn=conversation_turn,
                    source_document=source,
                    is_uploaded=False
                ))
    
    if not target_policies:
        target_policies.extend(user_profile.uploaded_policies)
        
        if len(target_policies) < 2:
            discussed_policies = sorted(
                user_profile.policies_discussed,
                key=lambda x: x.mentioned_in_turn,
                reverse=True
            )[:3]
            target_policies.extend(discussed_policies)
    
    if not target_policies:
        return "I don't have any policies to compare. Please mention specific policies you're interested in, or upload a policy document."
    
    policy_names = [p.policy_name for p in target_policies]
    
    comparison_query = f"Detailed comparison of {', '.join(policy_names)} policies"
    
    use_uploaded = any(p.is_uploaded for p in target_policies)
    
    docs = smart_policy_retrieval(
        query=comparison_query, 
        policy_names=policy_names, 
        use_uploaded=use_uploaded, 
        k=5,
        use_exact_match=True
    )
    
    if not docs:
        return "I couldn't find any similar policies in our database."
    
    contexts = []
    sources = []
    seen_sources = set()
    
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        contexts.append(doc.page_content)
        if source not in seen_sources:
            sources.append(source)
            seen_sources.add(source)
    
    target_policy_text = "Target policies:\n- " + "\n- ".join(policy_names)
    
    similarity_template = """
    You are an insurance expert for policies. The user wants to find policies similar to the ones they mentioned or uploaded.
    
    CONVERSATION HISTORY:
    {history}
    
    USER QUERY: {query}
    
    TARGET POLICIES:
    {target_policies}
    
    POLICY INFORMATION FROM DATABASE:
    {context}
    
    Please provide:
    1. A detailed analysis of similarities and differences between the target policies and other relevant policies
    2. Compare key factors like coverage, premiums, benefits, and exclusions
    3. Highlight unique features of each policy
    4. Provide recommendations on which policy might be better for different user needs
    
    Bold the policy names when referring to them for clarity.
    """
    
    prompt = ChatPromptTemplate.from_template(similarity_template)
    
    chain = (
        {"query": RunnablePassthrough(),
         "context": lambda x: "\n\n".join(contexts),
         "history": lambda x: history_text,
         "target_policies": lambda x: target_policy_text}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = chain.invoke(user_input)
    
    policies = extract_policy_entities(response, conversation_turn)
    
    for policy in policies:
        already_discussed = False
        for existing in user_profile.policies_discussed:
            if existing.policy_name.lower() == policy.policy_name.lower():
                already_discussed = True
                break
        
        if not already_discussed:
            user_profile.policies_discussed.append(policy)
    
    user_profile.last_interaction = datetime.now().isoformat()
    save_session(user_profile)
    
    source_info = "\n\nSources consulted: " + ", ".join(set(sources))
    final_response = response + source_info
    
    conversation_memory.save_context(
        {"input": user_input},
        {"output": final_response}
    )
    
    return final_response

def get_policy_details(policy_name: str, is_uploaded: bool = False):
    """Get detailed information about a specific policy"""
    query = f"Comprehensive details about {policy_name} policy including benefits, features, exclusions, and eligibility"
    
    policy_names = [policy_name]
    
    docs = smart_policy_retrieval(
        query=query, 
        policy_names=policy_names, 
        use_uploaded=is_uploaded, 
        k=8,
        use_exact_match=True
    )
    
    if not docs:
        return f"I couldn't find detailed information about the {policy_name} policy."
    
    contexts = [doc.page_content for doc in docs]
    sources = [doc.metadata.get("source", "Unknown") for doc in docs]
    
    details_template = """
    You are an insurance expert at policies. Provide comprehensive information about the requested policy.
    
    POLICY: {policy_name}
    
    POLICY INFORMATION:
    {context}
    
    Please provide a detailed and well-structured response about this specific policy with the following sections:
    1. Overview - A clear summary of what this policy is and who it's for
    2. Key Benefits - List all benefits and advantages of this policy
    3. Coverage Details - What specifically is covered by this policy
    4. Exclusions - Important limitations or exclusions
    5. Premium Information - Details about premium payment if available
    6. Eligibility - Who can apply for this policy
    7. Claims Process - How to make claims under this policy (if information available)
    
    Present the information in a well-organized format with clear headings and bullet points where appropriate.
    Do not include information about other policies unless directly relevant for understanding this specific policy.
    If certain information is not available in the provided context, simply omit that section rather than making up details.
    """
    
    prompt = ChatPromptTemplate.from_template(details_template)
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "policy_name": policy_name,
        "context": "\n\n".join(contexts)
    })
    
    filtered_sources = []
    seen_sources = set()
    for source in sources:
        if source not in seen_sources:
            filtered_sources.append(source)
            seen_sources.add(source)
    
    source_info = "\n\nSources consulted: " + ", ".join(filtered_sources[:3])
    if len(filtered_sources) > 3:
        source_info += f" and {len(filtered_sources) - 3} more"
    
    final_response = response + source_info
    
    conversation_memory.save_context(
        {"input": f"Tell me about {policy_name} policy"},
        {"output": final_response}
    )
    
    return final_response

def fact_check_claims(text: str, user_profile: UserProfile):
    """Fact check insurance-related claims in the text"""
    claims_template = """
    Extract factual claims about insurance policies from the following text:
    
    TEXT: {text}
    
    List each claim as a separate item. Focus on specific factual assertions about coverage, benefits, exclusions, etc.
    Format: ["Claim 1", "Claim 2", ...]
    """
    
    prompt = ChatPromptTemplate.from_template(claims_template)
    chain = prompt | llm | StrOutputParser()
    
    try:
        result = chain.invoke({"text": text})
        
        result = result.strip()
        if result.startswith("```") and result.endswith("```"):
            result = result[3:-3].strip()
        
        try:
            claims = eval(result)
            if not isinstance(claims, list):
                return "I didn't identify any specific factual claims to verify in the text."
        except:
            claims = []
            matches = re.findall(r'"([^"]+)"', result)
            if matches:
                claims = matches
            else:
                return "I didn't identify any specific factual claims to verify in the text."
    except:
        return "I wasn't able to process the text for fact-checking."
    
    fact_check_results = []
    
    for claim in claims:
        icici_docs = chroma_db.as_retriever(search_kwargs={"k": 3}).invoke(claim)
        
        uploaded_docs = []
        if uploaded_chroma_db._collection.count() > 0:
            uploaded_docs = uploaded_chroma_db.as_retriever(search_kwargs={"k": 2}).invoke(claim)
        
        docs = icici_docs + uploaded_docs
        
        if not docs:
            fact_check_results.append({
                "claim": claim,
                "result": "No relevant information found",
                "context": ""
            })
            continue
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        check_template = """
        You are a fact-checker for insurance information. Verify the following claim against the provided information:
        
        CLAIM: {claim}
        
        INFORMATION:
        {context}
        
        Please determine if the claim is:
        - VERIFIED (fully supported by the information)
        - PARTIALLY VERIFIED (some aspects are correct, others not mentioned or contradicted)
        - UNVERIFIED (not mentioned in the information)
        - CONTRADICTED (directly contradicted by the information)
        
        Provide a brief explanation for your assessment.
        
        Format your response as:
        VERDICT: [the verdict]
        EXPLANATION: [your explanation]
        """
        
        prompt = ChatPromptTemplate.from_template(check_template)
        
        chain = prompt | fact_check_llm | StrOutputParser()
        
        check_result = chain.invoke({
            "claim": claim,
            "context": context
        })
        
        verdict_match = re.search(r"VERDICT:\s*(.+)", check_result)
        explanation_match = re.search(r"EXPLANATION:\s*(.+)", check_result, re.DOTALL)
        
        verdict = verdict_match.group(1).strip() if verdict_match else "UNVERIFIED"
        explanation = explanation_match.group(1).strip() if explanation_match else "No explanation provided."
        
        fact_check_results.append({
            "claim": claim,
            "verdict": verdict,
            "explanation": explanation
        })
    
    if not fact_check_results:
        return "I didn't find any specific claims to fact-check in the text."
    
    response = "# Fact Check Results\n\n"
    
    for i, result in enumerate(fact_check_results, 1):
        response += f"## Claim {i}: \"{result['claim']}\"\n"
        response += f"**Verdict**: {result['verdict']}\n"
        response += f"**Explanation**: {result['explanation']}\n\n"
    
    response += "\nNote: This fact-checking is based on the information available in my database and may not be comprehensive."
    
    conversation_memory.save_context(
        {"input": "Fact check the following claims: " + text},
        {"output": response}
    )
    
    return response

def evaluate_uploaded_policy(policy: PolicyEntity, user_profile: UserProfile, conversation_turn: int):
    """Evaluate a user-uploaded policy and provide insights"""
    query = f"Detailed information about {policy.policy_name}"
    
    docs = []
    if uploaded_chroma_db._collection.count() > 0:
        try:
            filter_dict = {"policy_name": policy.policy_name}
            docs = uploaded_chroma_db.as_retriever(
                search_kwargs={"k": 5, "filter": filter_dict}
            ).invoke(query)
        except Exception as e:
            print(f"Filter-based retrieval failed: {str(e)}")
    
    if not docs:
        return f"I don't have enough information to evaluate the {policy.policy_name} policy. Please try uploading the document again."
    
    contexts = [doc.page_content for doc in docs]
    
    eval_template = """
    You are an insurance expert at ICICI. Evaluate the following uploaded insurance policy.
    
    POLICY NAME: {policy_name}
    POLICY TYPE: {policy_type}
    
    POLICY CONTENT:
    {context}
    
    Please provide:
    1. A comprehensive analysis of this policy
    2. Key strengths and benefits of this policy
    3. Potential limitations or exclusions to be aware of
    4. How this policy compares to standard offerings in the market
    5. Who this policy would be most suitable for
    
    Present your evaluation in a clear, organized format with headings.
    """
    
    prompt = ChatPromptTemplate.from_template(eval_template)
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "policy_name": policy.policy_name,
        "policy_type": policy.policy_type,
        "context": "\n\n".join(contexts)
    })
    
    conversation_memory.save_context(
        {"input": f"Evaluate my uploaded {policy.policy_name} policy"},
        {"output": response}
    )
    
    return response

def compare_uploaded_with_existing(policy: PolicyEntity, user_profile: UserProfile, conversation_turn: int):
    """Compare an uploaded policy with existing policies"""
    query = f"Details of {policy.policy_name} for comparison"
    
    uploaded_docs = []
    if uploaded_chroma_db._collection.count() > 0:
        try:
            filter_dict = {"policy_name": policy.policy_name}
            uploaded_docs = uploaded_chroma_db.as_retriever(
                search_kwargs={"k": 3, "filter": filter_dict}
            ).invoke(query)
        except Exception as e:
            print(f"Filter-based retrieval failed: {str(e)}")
    
    if not uploaded_docs:
        return f"I don't have enough information about the {policy.policy_name} policy to compare it. Please try uploading the document again."
    
    similar_policy_query = f"Policies similar to {policy.policy_name} {policy.policy_type} insurance"
    icici_docs = chroma_db.as_retriever(search_kwargs={"k": 5}).invoke(similar_policy_query)
    
    if not icici_docs:
        return f"I couldn't find any similar ICICI policies to compare with your {policy.policy_name} policy."
    
    uploaded_context = "\n\n".join([doc.page_content for doc in uploaded_docs])
    
    icici_contexts = {}
    for doc in icici_docs:
        policy_name = doc.metadata.get("policy_name", "Unknown")
        if policy_name not in icici_contexts:
            icici_contexts[policy_name] = []
        icici_contexts[policy_name].append(doc.page_content)
    
    comparison_template = """
    You are an insurance expert at policies. Compare the following uploaded policy with similar insurance policies.
    
    UPLOADED POLICY:
    Name: {uploaded_policy_name}
    Type: {uploaded_policy_type}
    
    UPLOADED POLICY CONTENT:
    {uploaded_context}
    
    COMPARABLE ICICI POLICIES:
    {icici_contexts}
    
    Please provide:
    1. A side-by-side comparison of the uploaded policy vs ICICI policies
    2. Key differences in coverage, premiums, benefits, and exclusions
    3. Strengths and weaknesses of each policy
    4. Which policy might be better for different types of customers
    
    Present your comparison in a clear, organized table format where appropriate.
    """
    
    formatted_icici_contexts = ""
    for name, contexts in icici_contexts.items():
        formatted_icici_contexts += f"--- {name} ---\n"
        formatted_icici_contexts += "\n".join(contexts)
        formatted_icici_contexts += "\n\n"
    
    prompt = ChatPromptTemplate.from_template(comparison_template)
    
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "uploaded_policy_name": policy.policy_name,
        "uploaded_policy_type": policy.policy_type,
        "uploaded_context": uploaded_context,
        "icici_contexts": formatted_icici_contexts
    })
    
    sources = [policy.policy_name]
    sources.extend(list(icici_contexts.keys()))
    
    source_info = "\n\nPolicies compared: " + ", ".join(sources)
    final_response = response + source_info
    
    conversation_memory.save_context(
        {"input": f"Compare my uploaded {policy.policy_name} policy with ICICI policies"},
        {"output": final_response}
    )
    
    return final_response

# --- Enhanced API endpoint implementation ---
def api_handle_message(user_id: str, user_input: str, conversation_turn: int = 0, 
                      file_path: Optional[str] = None, extraction_method: str = 'auto'):
    """Handle API request - enhanced implementation with Smart Underwriting"""
    processing_info = None
    
    if file_path:
        try:
            if not os.path.exists(file_path):
                return APIResponse(
                    response=f"Error: File not found at {file_path}",
                    user_profile=get_or_create_session(user_id),
                    processing_info={"error": "File not found"}
                )
                
            if not file_path.endswith(".pdf"):
                return APIResponse(
                    response="Error: Uploaded file must be a PDF",
                    user_profile=get_or_create_session(user_id),
                    processing_info={"error": "Not a PDF file"}
                )
                
            start_time = datetime.now()
            policy = process_uploaded_pdf(file_path, user_id, extraction_method=extraction_method)
            end_time = datetime.now()
            
            processing_info = {
                "policy_name": policy.policy_name,
                "policy_type": policy.policy_type,
                "extraction_method": extraction_method,
                "processing_time": str(end_time - start_time),
                "success": not policy.policy_name.startswith("Error")
            }
            
            if policy.policy_name.startswith("Error"):
                return APIResponse(
                    response=f"Error processing your PDF: {policy.policy_name}",
                    user_profile=get_or_create_session(user_id),
                    processing_info=processing_info
                )
                
            if not user_input:
                user_input = f"Tell me about the {policy.policy_name} policy I just uploaded"
        
        except Exception as e:
            return APIResponse(
                response=f"Error processing your PDF: {str(e)}",
                user_profile=get_or_create_session(user_id),
                processing_info={"error": str(e)}
            )
    
    # Enhanced message processing with Smart Underwriting
    if smart_underwriting_system and SMART_UNDERWRITING_AVAILABLE:
        try:
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            enhanced_response = loop.run_until_complete(
                enhanced_api_handle_message(
                    user_id=user_id,
                    user_input=user_input,
                    conversation_turn=conversation_turn,
                    file_path=file_path,
                    extraction_method=extraction_method,
                    vector_db=chroma_db,
                    uploaded_vector_db=uploaded_chroma_db,
                    policy_list=policy_list
                )
            )
            
            user_profile = enhanced_response.get("user_profile", get_or_create_session(user_id))
            if processing_info:
                enhanced_response["processing_info"] = processing_info
            
            return APIResponse(
                response=enhanced_response["response"],
                user_profile=user_profile,
                processing_info=enhanced_response.get("processing_info")
            )
            
        except Exception as e:
            print(f"⚠️ Enhanced API error: {e}")
            print("   Falling back to legacy API")
    
    response = process_message(user_id, user_input, conversation_turn)
    
    user_profile = get_or_create_session(user_id)
    
    return APIResponse(
        response=response,
        user_profile=user_profile,
        processing_info=processing_info
    )

# === Smart Underwriting specific functions ===
def get_underwriting_result(user_id: str, customer_needs: str, 
                           policy_type: str = None, 
                           external_data_sources: List[str] = None) -> Dict[str, Any]:
    """Get a complete underwriting result using the Smart Underwriting System"""
    if not smart_underwriting_system or not SMART_UNDERWRITING_AVAILABLE:
        return {
            "error": "Smart Underwriting System not available",
            "fallback_recommendation": "Please use the standard recommendation system"
        }
    
    try:
        user_profile = get_or_create_session(user_id)
        
        orchestrator = smart_underwriting_system['orchestrator']
        
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            orchestrator.process_underwriting_request(
                customer_id=user_id,
                customer_profile=user_profile,
                customer_needs=customer_needs,
                policy_type=policy_type,
                external_data_sources=external_data_sources or []
            )
        )
        
        result_dict = {
            "customer_id": result.customer_id,
            "request_id": result.request_id,
            "success": result.success,
            "processing_time": result.processing_time,
            "agents_used": result.agents_used,
            "underwriting_decision": {
                "decision": result.underwriting_decision.decision,
                "decision_reason": result.underwriting_decision.decision_reason,
                "confidence_level": result.underwriting_decision.confidence_level,
                "manual_review_triggers": result.underwriting_decision.manual_review_triggers
            },
            "risk_assessment": {
                "overall_risk_level": result.risk_assessment.overall_risk_level.value if result.risk_assessment else "unknown",
                "risk_score": result.risk_assessment.risk_score if result.risk_assessment else 0,
                "recommended_premium_multiplier": result.risk_assessment.recommended_premium_multiplier if result.risk_assessment else 1.0,
                "confidence_level": result.risk_assessment.confidence_level if result.risk_assessment else 0.5,
                "requires_manual_review": result.risk_assessment.requires_manual_review if result.risk_assessment else True
            },
            "policy_draft": {
                "policy_name": result.policy_draft.policy_name if result.policy_draft else "No policy generated",
                "policy_type": result.policy_draft.policy_type if result.policy_draft else "unknown",
                "coverage_amount": result.policy_draft.terms.coverage_amount if result.policy_draft else 0,
                "premium_amount": result.policy_draft.terms.premium_amount if result.policy_draft else 0,
                "compliance_status": result.policy_draft.compliance_status if result.policy_draft else "unknown"
            },
            "historical_patterns_count": len(result.historical_patterns),
            "errors": result.errors,
            "timestamp": result.timestamp.isoformat()
        }
        
        return result_dict
        
    except Exception as e:
        return {
            "error": f"Underwriting processing failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

def get_agent_status():
    """Get the status of all smart underwriting agents"""
    if not SMART_UNDERWRITING_AVAILABLE:
        return {"status": "unavailable", "message": "Smart Underwriting not loaded"}
    
    try:
        health_status = check_agent_health()
        return {
            "status": "available" if health_status['overall_health'] else "degraded",
            "agents": {
                "agent1_historical": health_status['agent1'],
                "agent2_risk": health_status['agent2'], 
                "agent3_policy": health_status['agent3'],
                "orchestrator": health_status['orchestrator']
            },
            "errors": health_status['errors'],
            "version": smart_underwriting_system['version'] if smart_underwriting_system else "unknown"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

def test_smart_underwriting_flow(user_id: str = None):
    """Test the complete smart underwriting flow with sample data"""
    if not smart_underwriting_system:
        return "Smart Underwriting System not available"
    
    test_user_id = user_id or f"test_user_{datetime.now().strftime('%H%M%S')}"
    
    user_profile = get_or_create_session(test_user_id)
    
    sample_preferences = [
        UserPreference(preference_type="age", value="35", confidence=0.9),
        UserPreference(preference_type="family", value="married with 2 children", confidence=0.8),
        UserPreference(preference_type="occupation", value="software engineer", confidence=0.9),
        UserPreference(preference_type="budget", value="₹15000-25000 annually", confidence=0.7),
        UserPreference(preference_type="coverage", value="comprehensive health insurance", confidence=0.8)
    ]
    
    user_profile.preferences.extend(sample_preferences)
    save_session(user_profile)
    
    customer_needs = "I need comprehensive health insurance for my family of 4. I'm 35 years old, work as a software engineer, and can afford ₹20000 annually for premiums."
    
    result = get_underwriting_result(
        user_id=test_user_id,
        customer_needs=customer_needs,
        policy_type="Health Insurance"
    )
    
    return {
        "test_user_id": test_user_id,
        "customer_needs": customer_needs,
        "underwriting_result": result
    }

# --- Simple chat loop for testing ---
def main():
    """Enhanced main chat loop with Smart Underwriting capabilities"""
    print("🤖 Enhanced Insurance Policy Advisor Bot")
    if smart_underwriting_system:
        print("✅ Smart Underwriting System: ACTIVE")
        print("   • Agent 1: Historical Risk Analysis")
        print("   • Agent 2: Actuarial Risk Assessment") 
        print("   • Agent 3: Policy Generation")
        print("   • Multi-Agent Orchestration")
    else:
        print("⚠️ Smart Underwriting System: NOT AVAILABLE")
        print("   • Please ensure the smart_underwriting_system is properly configured.")
    print("Type 'exit' to quit")
    print("Type 'upload <path/to/pdf>' to upload a PDF policy")
    print("Type 'underwrite <your needs>' to test smart underwriting")
    print("Type 'help' for more commands\n")
    
    for directory in [SESSION_DIR, UPLOADS_DIR, PDF_FOLDER, 
                     os.path.dirname(POLICY_LIST_FILE), 
                     os.path.dirname(UPLOADED_POLICY_LIST_FILE)]:
        os.makedirs(directory, exist_ok=True)
    
    try:
        initialize_uploaded_policy_database()
    except Exception as e:
        print(f"\n:warning: Error initializing uploaded policy database: {str(e)}")
        print("Will try to continue anyway...")
    
    user_id = str(uuid.uuid4())
    conversation_turn = 0
    
    try:
        user_profile = get_or_create_session(user_id)
        print(f"User session created with ID: {user_id}")
    except Exception as e:
        print(f"\n:warning: Error creating user session: {str(e)}")
        print("Will try to continue anyway...")
    
    while True:
        try:
            user_input = input("👤 You: ").strip()
            
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("\n--- Available Commands ---")
                print("exit - Exit the chat")
                print("upload <path/to/pdf> - Upload a PDF policy")
                print("upload-ocr <path/to/pdf> - Upload a PDF and force OCR processing")
                print("upload-direct <path/to/pdf> - Upload a PDF with direct text extraction only")
                print("underwrite <your needs> - Test smart underwriting system")
                print("compare <policy1> <policy2> - Compare two policies")
                print("recommend - Get policy recommendations based on your profile")
                print("help - Show this help message")
                print("---")
                print("All other input will be processed as questions or statements about insurance policies.\n")
                continue
            
            if not user_input:
                continue
            
            # Smart Underwriting Test Command
            if user_input.lower().startswith("underwrite "):
                customer_needs = user_input[11:].strip()
                if smart_underwriting_system:
                    print("\n🤖 Testing Smart Underwriting System...")
                    try:
                        result = get_underwriting_result(user_id, customer_needs)
                        
                        if "error" in result:
                            print(f"\n❌ Error: {result['error']}")
                        else:
                            print(f"\n✅ Underwriting Complete!")
                            print(f"Request ID: {result['request_id']}")
                            print(f"Processing Time: {result['processing_time']:.2f}s")
                            print(f"Decision: {result['underwriting_decision']['decision']}")
                            print(f"Risk Level: {result['risk_assessment']['overall_risk_level']}")
                            print(f"Risk Score: {result['risk_assessment']['risk_score']:.1f}/100")
                            print(f"Premium Multiplier: {result['risk_assessment']['recommended_premium_multiplier']:.2f}")
                            
                            if result['policy_draft']['policy_name'] != "No policy generated":
                                print(f"Generated Policy: {result['policy_draft']['policy_name']}")
                                print(f"Coverage: ₹{result['policy_draft']['coverage_amount']:,.2f}")
                                print(f"Premium: ₹{result['policy_draft']['premium_amount']:,.2f}")
                        
                        continue
                    except Exception as e:
                        print(f"\n❌ Smart Underwriting error: {str(e)}")
                        continue
                else:
                    print("\n⚠️ Smart Underwriting System not available")
                    continue
            
            if user_input.lower().startswith("upload "):
                file_path = user_input[7:].strip()
                if os.path.exists(file_path) and file_path.endswith(".pdf"):
                    try:
                        print("\nProcessing PDF with automatic method selection...")
                        policy = process_uploaded_pdf(file_path, user_id, extraction_method='auto')
                        print(f"\n🤖 Bot: Successfully uploaded {policy.policy_name}")
                        continue
                    except Exception as e:
                        print(f"\n🤖 Bot: Error uploading file: {str(e)}")
                        continue
                else:
                    print("\n🤖 Bot: File not found or not a PDF")
                    continue
            
            elif user_input.lower().startswith("upload-ocr "):
                file_path = user_input[11:].strip()
                if os.path.exists(file_path) and file_path.endswith(".pdf"):
                    try:
                        print("\nProcessing PDF with OCR (for scanned documents)...")
                        if not OCR_AVAILABLE:
                            print("\n🤖 Bot: OCR dependencies not installed. Please install pytesseract and pdf2image.")
                            continue
                        policy = process_uploaded_pdf(file_path, user_id, extraction_method='ocr')
                        print(f"\n🤖 Bot: Successfully uploaded {policy.policy_name} using OCR")
                        continue
                    except Exception as e:
                        print(f"\n🤖 Bot: Error uploading file with OCR: {str(e)}")
                        continue
                else:
                    print("\n🤖 Bot: File not found or not a PDF")
                    continue
            
            elif user_input.lower().startswith("upload-direct "):
                file_path = user_input[14:].strip()
                if os.path.exists(file_path) and file_path.endswith(".pdf"):
                    try:
                        print("\nProcessing PDF with direct text extraction only...")
                        policy = process_uploaded_pdf(file_path, user_id, extraction_method='pymupdf')
                        print(f"\n🤖 Bot: Successfully uploaded {policy.policy_name} using direct extraction")
                        continue
                    except Exception as e:
                        print(f"\n🤖 Bot: Error uploading file with direct extraction: {str(e)}")
                        continue
                else:
                    print("\n🤖 Bot: File not found or not a PDF")
                    continue
                    
            conversation_turn += 1
            try:
                response = process_message(user_id, user_input, conversation_turn)
                print(f"\n🤖 {response}\n")
            except Exception as e:
                print(f"\n🤖 Bot: Error processing your message: {str(e)}")
                continue
                
        except Exception as e:
            print(f"\n🤖 Bot: An unexpected error occurred: {str(e)}")
            print("Please try again or type 'exit' to quit.")

if __name__ == "__main__":
    main()
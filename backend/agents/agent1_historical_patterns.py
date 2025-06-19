#!/usr/bin/env python3
"""
Agent 1: Historical Risk Pattern Analysis
Analyzes historical data to identify risk patterns similar to current customer
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# Import models from main system (these would be imported from your main_new2.py)
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
    """
    Agent 1: Analyzes historical risk patterns and identifies similar cases
    Builds on your existing vector database and search capabilities
    """
    
    def __init__(self, vector_db, uploaded_vector_db, policy_list):
        self.vector_db = vector_db
        self.uploaded_vector_db = uploaded_vector_db
        self.policy_list = policy_list
        self.llm = ChatOpenAI(model="gpt-4.1", temperature=0.2)
        self.pattern_cache = {}
        self.risk_correlations = self._load_risk_correlations()
        
    async def analyze_historical_patterns(self, customer_profile) -> List[HistoricalPattern]:
        """
        Main function: Analyze historical patterns for a customer profile
        Enhanced version of your existing policy search focused on risk patterns
        """
        patterns = []
        
        try:
            # Step 1: Build risk-focused search queries
            risk_queries = self._build_risk_queries(customer_profile)
            
            # Step 2: Search historical data for similar patterns
            for query_type, query in risk_queries.items():
                similar_cases = await self._search_historical_cases(query, query_type)
                
                # Step 3: Analyze each case for risk patterns
                for case in similar_cases[:3]:  # Top 3 per query type
                    pattern = await self._analyze_case_pattern(case, customer_profile, query_type)
                    if pattern and pattern.similarity_score > 0.6:
                        patterns.append(pattern)
            
            # Step 4: Deduplicate and rank patterns
            unique_patterns = self._deduplicate_patterns(patterns)
            ranked_patterns = sorted(unique_patterns, key=lambda x: x.similarity_score, reverse=True)
            
            return ranked_patterns[:5]  # Return top 5 patterns
            
        except Exception as e:
            print(f"Error in historical pattern analysis: {str(e)}")
            return []
    
    def _build_risk_queries(self, customer_profile) -> Dict[str, str]:
        """
        Build multiple risk-focused search queries based on customer profile
        Uses your existing preference extraction system
        """
        queries = {}
        
        # Extract key risk factors from user preferences
        age = self._extract_preference_value(customer_profile, 'age', '30')
        occupation = self._extract_preference_value(customer_profile, 'occupation', 'unknown')
        family_status = self._extract_preference_value(customer_profile, 'family', 'single')
        health_status = self._extract_preference_value(customer_profile, 'health', 'unknown')
        
        # Build demographic risk query
        queries['demographic'] = f"Risk analysis age {age} occupation {occupation} family {family_status}"
        
        # Build health risk query
        if health_status != 'unknown':
            queries['health'] = f"Health insurance risk {health_status} medical history claims"
        
        # Build behavioral risk query
        lifestyle_prefs = [p for p in customer_profile.preferences 
                          if p.preference_type in ['lifestyle', 'smoking', 'drinking', 'exercise']]
        if lifestyle_prefs:
            lifestyle_terms = " ".join([p.value for p in lifestyle_prefs])
            queries['lifestyle'] = f"Lifestyle risk factors {lifestyle_terms} claims patterns"
        
        # Build policy-specific risk query
        discussed_policies = getattr(customer_profile, 'policies_discussed', [])
        if discussed_policies:
            policy_names = " ".join([p.policy_name for p in discussed_policies[-2:]])
            queries['policy'] = f"Risk patterns {policy_names} similar policies claims history"
        
        return queries
    
    async def _search_historical_cases(self, query: str, query_type: str) -> List[Dict]:
        """
        Search historical cases using your existing vector database system
        Enhanced version of your smart_policy_retrieval function
        """
        try:
            # Search in main policy database (your existing chroma_db)
            main_results = self.vector_db.as_retriever(search_kwargs={"k": 5}).invoke(query)
            
            # Search in uploaded policies if available
            uploaded_results = []
            if self.uploaded_vector_db._collection.count() > 0:
                uploaded_results = self.uploaded_vector_db.as_retriever(search_kwargs={"k": 3}).invoke(query)
            
            # Combine and format results
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
        """
        Analyze a historical case to extract risk patterns
        Uses your existing LLM integration
        """
        try:
            # Build analysis prompt
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
            
            # Use your existing LLM
            prompt = ChatPromptTemplate.from_template(analysis_prompt)
            chain = prompt | self.llm | StrOutputParser()
            
            result = chain.invoke({})
            
            # Parse LLM response
            analysis = self._parse_llm_response(result)
            
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
    
    def calculate_risk_correlations(self, customer_profile) -> List[RiskFactorCorrelation]:
        """
        Calculate risk factor correlations based on historical data
        """
        correlations = []
        
        # Analyze each preference as a potential risk factor
        for preference in customer_profile.preferences:
            if preference.confidence > 0.7:  # Only high-confidence preferences
                correlation = self._calculate_factor_correlation(
                    preference.preference_type, 
                    preference.value
                )
                if correlation:
                    correlations.append(correlation)
        
        return sorted(correlations, key=lambda x: abs(x.correlation_strength), reverse=True)
    
    def _calculate_factor_correlation(self, factor_type: str, factor_value: str) -> Optional[RiskFactorCorrelation]:
        """
        Calculate correlation for a specific risk factor
        """
        # This would ideally use actual historical claims data
        # For now, using heuristic-based correlations
        
        risk_mappings = {
            'age': {
                'ranges': [(0, 25, 0.3), (25, 35, -0.2), (35, 50, 0.1), (50, 65, 0.4), (65, 100, 0.6)],
                'impact': 0.7
            },
            'occupation': {
                'high_risk': ['mining', 'construction', 'racing', 'aviation'],
                'low_risk': ['office', 'teacher', 'engineer', 'manager'],
                'impact': 0.5
            },
            'health': {
                'conditions': ['diabetes', 'hypertension', 'heart', 'cancer'],
                'impact': 0.8
            }
        }
        
        try:
            if factor_type == 'age' and factor_value.isdigit():
                age = int(factor_value)
                for min_age, max_age, correlation in risk_mappings['age']['ranges']:
                    if min_age <= age < max_age:
                        return RiskFactorCorrelation(
                            factor_name=f"age_{age}",
                            correlation_strength=correlation,
                            historical_frequency=0.8,  # High frequency factor
                            impact_on_claims=risk_mappings['age']['impact']
                        )
            
            elif factor_type == 'occupation':
                occupation_lower = factor_value.lower()
                if any(risk in occupation_lower for risk in risk_mappings['occupation']['high_risk']):
                    return RiskFactorCorrelation(
                        factor_name=f"occupation_{factor_value}",
                        correlation_strength=0.6,
                        historical_frequency=0.3,
                        impact_on_claims=risk_mappings['occupation']['impact']
                    )
                elif any(risk in occupation_lower for risk in risk_mappings['occupation']['low_risk']):
                    return RiskFactorCorrelation(
                        factor_name=f"occupation_{factor_value}",
                        correlation_strength=-0.2,
                        historical_frequency=0.7,
                        impact_on_claims=risk_mappings['occupation']['impact']
                    )
        
        except Exception as e:
            print(f"Error calculating correlation for {factor_type}: {str(e)}")
        
        return None
    
    def _extract_preference_value(self, customer_profile, pref_type: str, default: str) -> str:
        """
        Extract preference value from customer profile (uses your existing UserPreference system)
        """
        for pref in customer_profile.preferences:
            if pref.preference_type.lower() == pref_type.lower():
                return pref.value
        return default
    
    def _summarize_customer_profile(self, customer_profile) -> str:
        """
        Create a summary of customer profile for LLM analysis
        """
        prefs = []
        for pref in customer_profile.preferences:
            if pref.confidence > 0.6:
                prefs.append(f"{pref.preference_type}: {pref.value}")
        
        return "\n".join(prefs) if prefs else "Limited customer information available"
    
    def _parse_llm_response(self, response: str) -> Dict:
        """
        Parse LLM JSON response with error handling
        """
        try:
            # Clean response
            cleaned = response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:-3]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:-3]
            
            return json.loads(cleaned)
        except:
            # Fallback parsing
            try:
                # Extract key values using regex if JSON fails
                import re
                similarity_match = re.search(r'similarity_score["\s]*:[\s]*([0-9.]+)', response)
                confidence_match = re.search(r'confidence_level["\s]*:[\s]*([0-9.]+)', response)
                
                return {
                    'similarity_score': float(similarity_match.group(1)) if similarity_match else 0.5,
                    'confidence_level': float(confidence_match.group(1)) if confidence_match else 0.5,
                    'pattern_insight': 'Analysis completed with limited parsing',
                    'risk_indicators': ['pattern_identified'],
                    'risk_factors': ['correlation_found']
                }
            except:
                return None
    
    def _deduplicate_patterns(self, patterns: List[HistoricalPattern]) -> List[HistoricalPattern]:
        """
        Remove duplicate patterns based on similarity
        """
        unique_patterns = []
        seen_insights = set()
        
        for pattern in patterns:
            # Simple deduplication based on insight content
            insight_key = pattern.pattern_insight[:50].lower()
            if insight_key not in seen_insights:
                unique_patterns.append(pattern)
                seen_insights.add(insight_key)
        
        return unique_patterns
    
    def _load_risk_correlations(self) -> Dict:
        """
        Load risk correlation data (would come from historical analysis)
        """
        # This would ideally load from a database of historical correlations
        return {
            'age_risk': True,
            'occupation_risk': True,
            'health_risk': True,
            'lifestyle_risk': True
        }
    
    def get_pattern_summary(self, patterns: List[HistoricalPattern]) -> Dict[str, Any]:
        """
        Generate a summary of all identified patterns
        """
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

# Utility functions for integration with main system
def create_historical_pattern_agent(vector_db, uploaded_vector_db, policy_list):
    """
    Factory function to create the agent with your existing database connections
    """
    return HistoricalRiskPatternAgent(vector_db, uploaded_vector_db, policy_list)

# Example usage function
async def test_historical_patterns(customer_profile, vector_db, uploaded_vector_db, policy_list):
    """
    Test function to demonstrate the agent
    """
    agent = create_historical_pattern_agent(vector_db, uploaded_vector_db, policy_list)
    patterns = await agent.analyze_historical_patterns(customer_profile)
    
    print(f"Found {len(patterns)} historical patterns:")
    for i, pattern in enumerate(patterns, 1):
        print(f"{i}. {pattern.pattern_type}: {pattern.similarity_score:.2f} similarity")
        print(f"   Insight: {pattern.pattern_insight}")
        print(f"   Risk indicators: {', '.join(pattern.risk_indicators)}")
    
    return patterns
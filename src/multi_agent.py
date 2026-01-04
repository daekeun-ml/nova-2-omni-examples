"""
Multi-Agent Multimodal Reasoning Classes
"""
import json
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from .common import get_bedrock_runtime, get_current_model_id

# Tool schemas
class SafetyAssessmentResult(BaseModel):
    """Schema for safety assessment results."""
    identified_hazards: list[str] = Field(description="List of hazards identified")
    risk_level: Literal["low", "medium", "high", "critical"] = Field(description="Overall risk level")
    recommended_actions: list[str] = Field(description="Recommended safety actions")

@tool(args_schema=SafetyAssessmentResult)
def submit_safety_assessment(identified_hazards: list[str], risk_level: str, recommended_actions: list[str]) -> dict:
    """Submit safety assessment results."""
    return {
        "agent": "safety_analyzer",
        "hazards": identified_hazards,
        "risk_level": risk_level,
        "actions": recommended_actions
    }

class ComprehensiveReport(BaseModel):
    """Schema for comprehensive report."""
    summary: str = Field(description="Overall summary of findings")
    key_insights: list[str] = Field(description="Key insights from all agents")
    recommendations: list[str] = Field(description="Final recommendations")

@tool(args_schema=ComprehensiveReport)
def submit_comprehensive_report(summary: str, key_insights: list[str], recommendations: list[str]) -> dict:
    """Submit comprehensive report synthesizing all agent findings."""
    return {
        "agent": "coordinator",
        "summary": summary,
        "insights": key_insights,
        "recommendations": recommendations
    }

def langchain_tool_to_bedrock(lc_tool):
    """Convert LangChain tool to Bedrock format."""
    schema = lc_tool.args_schema.model_json_schema()
    return {
        "toolSpec": {
            "name": lc_tool.name,
            "description": lc_tool.description,
            "inputSchema": {"json": schema}
        }
    }

class MultimodalAgent:
    """Agent using direct boto3 calls with reasoning."""
    
    def __init__(self, name: str, system_prompt: str, tools: list, reasoning_effort: str = "medium"):
        self.name = name
        self.system_prompt = system_prompt
        self.bedrock_tools = [langchain_tool_to_bedrock(t) for t in tools]
        self.reasoning_effort = reasoning_effort
        self.bedrock = get_bedrock_runtime()
    
    def analyze(self, content: list, temperature: float = 0.7, max_tokens: int = 1000, top_p: float = 1.0) -> dict:
        """Analyze content and return results."""
        request = {
            "modelId": get_current_model_id(),
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "system": [{"text": self.system_prompt}],
            "toolConfig": {"tools": self.bedrock_tools},
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens,
                "topP": top_p
            },
            "additionalModelRequestFields": {
                "reasoningConfig": {
                    "type": "enabled",
                    "maxReasoningEffort": self.reasoning_effort
                }
            }
        }
        
        try:
            response = self.bedrock.converse(**request)
            print(f"[{self.name}] Response received: {response.get('output', {}).get('message', {}).get('content', [])}")
            return response
        except Exception as e:
            print(f"[{self.name}] Error: {e}")
            return {"error": str(e)}

class MultiAgentOrchestrator:
    """Orchestrates multiple agents."""
    
    def __init__(self, agents: dict, coordinator: MultimodalAgent):
        self.agents = agents
        self.coordinator = coordinator
    
    def run(self, tasks: dict, temperature: float = 0.7, max_tokens: int = 1000, top_p: float = 1.0) -> dict:
        """Run agents and synthesize results."""
        agent_results = {}
        
        # Run each agent
        for agent_name, task_content in tasks.items():
            if agent_name in self.agents:
                print(f"\n=== Running {agent_name} ===")
                agent = self.agents[agent_name]
                response = agent.analyze(task_content, temperature, max_tokens, top_p)
                
                # Extract tool use results
                if "output" in response and "message" in response["output"]:
                    for content in response["output"]["message"]["content"]:
                        if "toolUse" in content:
                            agent_results[agent_name] = content["toolUse"]["input"]
                            print(f"Results: {json.dumps(agent_results[agent_name], indent=2)}")
                            break
        
        # Check if we have any results
        if not agent_results:
            return {
                "summary": "No agent results were generated. Please check agent configuration and tool usage.",
                "key_insights": ["No insights available - agents did not produce results"],
                "recommendations": ["Verify agent setup and tool configuration"]
            }
        
        # Run coordinator
        print("\n=== Running Coordinator ===")
        synthesis_prompt = f"""Synthesize these detailed analyses from specialized agents:

{json.dumps(agent_results, indent=2)}

Create a comprehensive report that:
1. Integrates all agent findings
2. Provides detailed insights with supporting evidence
3. Offers specific, actionable recommendations
4. Assesses overall risk levels and priorities

Provide a thorough analysis that goes beyond simple summarization."""
        
        coordinator_response = self.coordinator.analyze([{"text": synthesis_prompt}], temperature, max_tokens, top_p)
        
        # Extract coordinator results
        if "output" in coordinator_response and "message" in coordinator_response["output"]:
            for content in coordinator_response["output"]["message"]["content"]:
                if "toolUse" in content:
                    final_result = content["toolUse"]["input"]
                    print(f"Final Answer: {json.dumps(final_result, indent=2)}")
                    return final_result
        
        # Fallback if coordinator doesn't use tools
        return {
            "summary": f"Analysis completed with {len(agent_results)} agent(s). Raw results: {json.dumps(agent_results, indent=2)}",
            "key_insights": [f"Agent {name} provided: {result}" for name, result in agent_results.items()],
            "recommendations": ["Review individual agent findings for detailed recommendations"]
        }

def create_safety_agent(reasoning_effort: str = "medium"):
    """Create safety analysis agent."""
    return MultimodalAgent(
        name="SafetyAnalyzer",
        system_prompt="""You are a comprehensive safety assessment expert. Analyze images thoroughly and ALWAYS use the submit_safety_assessment tool to report your findings.

Your analysis must include:
1. **Immediate Hazards**: Identify all visible safety risks, dangerous objects, unsafe conditions
2. **Environmental Risks**: Assess lighting, weather, terrain, traffic, crowd conditions  
3. **Behavioral Analysis**: Evaluate human actions, posture, attention, protective equipment usage
4. **Risk Severity**: Categorize risks as low/medium/high with detailed justification
5. **Prevention Strategies**: Provide specific, actionable safety recommendations

IMPORTANT: You MUST use the submit_safety_assessment tool with your findings. Do not just provide text analysis.""",
        tools=[submit_safety_assessment],
        reasoning_effort=reasoning_effort
    )

def create_coordinator_agent(reasoning_effort: str = "medium"):
    """Create coordinator agent."""
    return MultimodalAgent(
        name="Coordinator",
        system_prompt="""You are a comprehensive analysis coordinator. You MUST synthesize findings from multiple specialized agents and use the submit_comprehensive_report tool.

Your responsibilities:
1. **Integration**: Combine insights from all agents into coherent analysis
2. **Prioritization**: Rank findings by importance and urgency
3. **Contextualization**: Provide broader context and implications
4. **Actionability**: Ensure recommendations are specific and implementable

IMPORTANT: You MUST use the submit_comprehensive_report tool with your synthesis. Do not just provide text analysis.""",
        tools=[submit_comprehensive_report],
        reasoning_effort=reasoning_effort
    )

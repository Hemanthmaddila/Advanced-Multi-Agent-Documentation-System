#!/usr/bin/env python3
"""
Simple test for the advanced multi-agent system
"""
import asyncio
import sys
import os
sys.path.append('src')

from src.agent_framework import AgentOrchestrator, AgentTask, AgentRole
from src.specialized_agents import AdvancedCodeAnalyzer, EnhancedDocstringWriter, DocstringReviewer
from src.docstring_generator import DocstringGenerator

async def test_individual_agents():
    """Test each agent individually to isolate issues"""
    print("🧪 TESTING INDIVIDUAL AGENTS")
    print("=" * 40)
    
    # Load model
    print("📥 Loading documentation engine...")
    model = DocstringGenerator()
    
    test_code = '''
def fibonacci_sequence(n: int, memo: dict = None) -> int:
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci_sequence(n-1, memo) + fibonacci_sequence(n-2, memo)
    return memo[n]
'''
    
    # Test Analyzer
    print("\n🔍 Testing AdvancedCodeAnalyzer...")
    analyzer = AdvancedCodeAnalyzer(model)
    
    analysis_task = AgentTask(
        task_id="test_analysis",
        agent_role=AgentRole.ANALYZER,
        input_data={
            'source_code': test_code,
            'function_name': 'fibonacci_sequence',
            'analysis_depth': 'deep'
        }
    )
    
    analysis_result = await analyzer.execute(analysis_task)
    print(f"Analysis success: {analysis_result.success}")
    if analysis_result.success:
        print(f"Analysis data keys: {list(analysis_result.data.keys())}")
        complexity = analysis_result.data.get('complexity_analysis', {})
        print(f"Complexity rating: {complexity.get('complexity_rating', 'unknown')}")
    else:
        print(f"Analysis error: {analysis_result.error_message}")
    
    # Test Writer (with analysis data)
    if analysis_result.success:
        print("\nTesting EnhancedDocstringWriter...")
        writer = EnhancedDocstringWriter(model)
        
        writing_task = AgentTask(
            task_id="test_writing",
            agent_role=AgentRole.WRITER,
            input_data={
                'source_code': test_code,
                'function_name': 'fibonacci_sequence',
                'analysis_data': analysis_result.data
            }
        )
        
        writing_result = await writer.execute(writing_task)
        print(f"Writing success: {writing_result.success}")
        if writing_result.success:
            docstring = writing_result.data.get('docstring', '')
            print(f"Generated docstring length: {len(docstring)}")
            print(f"Docstring preview: {docstring[:100]}...")
        else:
            print(f"Writing error: {writing_result.error_message}")
        
        # Test Reviewer (with writing data)
        if writing_result.success:
            print("\nTesting DocstringReviewer...")
            reviewer = DocstringReviewer(model)
            
            review_task = AgentTask(
                task_id="test_review",
                agent_role=AgentRole.REVIEWER,
                input_data={
                    'source_code': test_code,
                    'docstring': writing_result.data.get('docstring', ''),
                    'analysis_data': analysis_result.data
                }
            )
            
            review_result = await reviewer.execute(review_task)
            print(f"Review success: {review_result.success}")
            if review_result.success:
                quality_score = review_result.data.get('final_score', 0.0)
                grade = review_result.data.get('quality_assessment', {}).get('grade', 'N/A')
                print(f"Quality score: {quality_score:.2f}")
                print(f"Quality grade: {grade}")
            else:
                print(f"Review error: {review_result.error_message}")

async def test_orchestrator():
    """Test the orchestrator system"""
    print("\n\n🎭 TESTING ORCHESTRATOR")
    print("=" * 40)
    
    # Create orchestrator
    orchestrator = AgentOrchestrator()
    model = DocstringGenerator()
    
    # Register agents
    orchestrator.register_agent(AgentRole.ANALYZER, AdvancedCodeAnalyzer(model))
    orchestrator.register_agent(AgentRole.WRITER, EnhancedDocstringWriter(model))
    orchestrator.register_agent(AgentRole.REVIEWER, DocstringReviewer(model))
    
    test_code = '''
def calculate_bmi(weight, height):
    if height <= 0:
        raise ValueError("Height must be positive")
    return weight / (height ** 2)
'''
    
    # Create simple task
    task = AgentTask(
        task_id="orchestrator_test",
        agent_role=AgentRole.ANALYZER,
        input_data={
            'source_code': test_code,
            'function_name': 'calculate_bmi',
            'analysis_depth': 'basic'
        }
    )
    
    # Execute single task
    results = await orchestrator.execute_workflow([task])
    
    print(f"Orchestrator results: {len(results)} tasks completed")
    for task_id, result in results.items():
        print(f"Task {task_id}: {'✅ Success' if result.success else '❌ Failed'}")
        if not result.success:
            print(f"  Error: {result.error_message}")

async def main():
    """Run all tests"""
    print("🚀 ADVANCED MULTI-AGENT SYSTEM DIAGNOSTICS")
    print("=" * 50)
    
    try:
        await test_individual_agents()
        await test_orchestrator()
        
        print("\n✅ DIAGNOSTICS COMPLETE!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
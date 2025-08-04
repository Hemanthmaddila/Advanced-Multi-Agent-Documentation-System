# src/agent_framework.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

class AgentRole(Enum):
    ANALYZER = "analyzer"
    CONTEXT = "context"
    WRITER = "writer"
    REVIEWER = "reviewer"

@dataclass
class AgentTask:
    """Represents a task for an agent"""
    task_id: str
    agent_role: AgentRole
    input_data: Dict[str, Any]
    dependencies: List[str] = None
    priority: int = 1
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class AgentResult:
    """Result from an agent execution"""
    task_id: str
    agent_role: AgentRole
    success: bool
    data: Dict[str, Any]
    execution_time: float
    error_message: str = None
    quality_score: float = None

class AIAgent(ABC):
    """Base class for all specialized agents"""
    
    def __init__(self, name: str, model_instance=None):
        self.name = name
        self.model = model_instance
        self.execution_count = 0
        self.total_execution_time = 0.0
        
    @abstractmethod
    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute the agent's main functionality"""
        pass
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for this agent"""
        avg_time = self.total_execution_time / max(1, self.execution_count)
        return {
            "name": self.name,
            "executions": self.execution_count,
            "total_time": self.total_execution_time,
            "avg_time": avg_time
        }

class AgentOrchestrator:
    """Orchestrates multiple specialized agents for complex workflows"""
    
    def __init__(self):
        self.agents: Dict[AgentRole, AIAgent] = {}
        self.task_queue: List[AgentTask] = []
        self.results_cache: Dict[str, AgentResult] = {}
        self.execution_history: List[AgentResult] = []
        
    def register_agent(self, role: AgentRole, agent: AIAgent):
        """Register an agent for a specific role"""
        self.agents[role] = agent
        print(f"✅ Registered {agent.name} for role: {role.value}")
    
    async def execute_workflow(self, workflow_tasks: List[AgentTask]) -> Dict[str, AgentResult]:
        """Execute a complete workflow with dependency management"""
        print(f"🚀 Starting workflow with {len(workflow_tasks)} tasks")
        
        # Sort tasks by dependencies and priority
        sorted_tasks = self._sort_tasks_by_dependencies(workflow_tasks)
        results = {}
        
        for task in sorted_tasks:
            # Check if dependencies are satisfied
            if not self._dependencies_satisfied(task, results):
                continue
                
            # Execute the task
            if task.agent_role in self.agents:
                agent = self.agents[task.agent_role]
                start_time = time.time()
                
                try:
                    result = await agent.execute(task)
                    execution_time = time.time() - start_time
                    
                    # Update agent stats
                    agent.execution_count += 1
                    agent.total_execution_time += execution_time
                    
                    results[task.task_id] = result
                    self.execution_history.append(result)
                    
                    print(f"✅ Completed {task.agent_role.value} task in {execution_time:.2f}s")
                    
                except Exception as e:
                    error_result = AgentResult(
                        task_id=task.task_id,
                        agent_role=task.agent_role,
                        success=False,
                        data={},
                        execution_time=time.time() - start_time,
                        error_message=str(e)
                    )
                    results[task.task_id] = error_result
                    print(f"❌ Failed {task.agent_role.value} task: {e}")
        
        return results
    
    def _sort_tasks_by_dependencies(self, tasks: List[AgentTask]) -> List[AgentTask]:
        """Sort tasks to respect dependencies"""
        # Simple topological sort
        sorted_tasks = []
        remaining_tasks = tasks.copy()
        
        while remaining_tasks:
            # Find tasks with no unresolved dependencies
            ready_tasks = []
            for task in remaining_tasks:
                if all(dep in [t.task_id for t in sorted_tasks] for dep in task.dependencies):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # If no ready tasks, add remaining (potential circular dependency)
                ready_tasks = remaining_tasks
            
            # Sort by priority and add to sorted list
            ready_tasks.sort(key=lambda x: x.priority, reverse=True)
            sorted_tasks.extend(ready_tasks)
            
            # Remove from remaining
            for task in ready_tasks:
                remaining_tasks.remove(task)
        
        return sorted_tasks
    
    def _dependencies_satisfied(self, task: AgentTask, completed_results: Dict[str, AgentResult]) -> bool:
        """Check if all dependencies for a task are satisfied"""
        return all(dep in completed_results for dep in task.dependencies)
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get summary of workflow execution"""
        if not self.execution_history:
            return {"message": "No workflows executed yet"}
        
        total_tasks = len(self.execution_history)
        successful_tasks = sum(1 for r in self.execution_history if r.success)
        total_time = sum(r.execution_time for r in self.execution_history)
        
        agent_stats = {}
        for role, agent in self.agents.items():
            agent_stats[role.value] = agent.get_performance_stats()
        
        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "total_execution_time": total_time,
            "average_task_time": total_time / total_tasks if total_tasks > 0 else 0,
            "agent_performance": agent_stats
        }
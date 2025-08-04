# src/specialized_agents.py
import ast
import networkx as nx
from typing import Dict, Any, List, Set, Optional, Tuple
import time
import hashlib
import re
try:
    from .agent_framework import AIAgent, AgentTask, AgentResult, AgentRole
    from .utils import extract_function_info
except ImportError:
    from agent_framework import AIAgent, AgentTask, AgentResult, AgentRole
    from utils import extract_function_info

class AdvancedCodeAnalyzer(AIAgent):
    """
    Advanced code analysis agent with dependency tracking, 
    complexity analysis, and pattern recognition.
    """
    
    def __init__(self, model_instance=None):
        super().__init__("AdvancedCodeAnalyzer", model_instance)
        self.analysis_cache = {}  # Cache for performance
        self.dependency_graph = nx.DiGraph()  # Code dependency graph
        
    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute advanced code analysis"""
        start_time = time.time()
        
        try:
            source_code = task.input_data.get('source_code', '')
            function_name = task.input_data.get('function_name', '')
            analysis_depth = task.input_data.get('analysis_depth', 'deep')
            
            # Generate cache key
            cache_key = self._generate_cache_key(source_code, function_name, analysis_depth)
            
            # Check cache first
            if cache_key in self.analysis_cache:
                print(f"📋 Using cached analysis for {function_name}")
                return AgentResult(
                    task_id=task.task_id,
                    agent_role=AgentRole.ANALYZER,
                    success=True,
                    data=self.analysis_cache[cache_key],
                    execution_time=time.time() - start_time,
                    quality_score=1.0
                )
            
            # Perform advanced analysis
            analysis_result = {
                'basic_info': self._extract_basic_info(source_code, function_name),
                'complexity_analysis': self._analyze_complexity(source_code, function_name),
                'dependency_analysis': self._analyze_dependencies(source_code, function_name),
                'pattern_analysis': self._analyze_patterns(source_code, function_name),
                'security_analysis': self._analyze_security(source_code, function_name),
                'performance_hints': self._analyze_performance(source_code, function_name),
                'docstring_requirements': self._determine_docstring_requirements(source_code, function_name)
            }
            
            # Cache the result
            self.analysis_cache[cache_key] = analysis_result
            
            # Update dependency graph
            self._update_dependency_graph(analysis_result)
            
            return AgentResult(
                task_id=task.task_id,
                agent_role=AgentRole.ANALYZER,
                success=True,
                data=analysis_result,
                execution_time=time.time() - start_time,
                quality_score=self._calculate_analysis_quality(analysis_result)
            )
            
        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                agent_role=AgentRole.ANALYZER,
                success=False,
                data={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _generate_cache_key(self, source_code: str, function_name: str, analysis_depth: str) -> str:
        """Generate cache key for analysis results"""
        content = f"{source_code}{function_name}{analysis_depth}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _extract_basic_info(self, source_code: str, function_name: str) -> Dict[str, Any]:
        """Extract basic function information"""
        basic_info = extract_function_info(source_code, function_name)
        if not basic_info:
            return {}
        
        # Enhanced analysis
        tree = ast.parse(source_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                # Count lines
                total_lines = node.end_lineno - node.lineno + 1 if node.end_lineno else 1
                code_lines = len([line for line in source_code.split('\n')[node.lineno-1:node.end_lineno] 
                                if line.strip() and not line.strip().startswith('#')])
                
                # Function characteristics
                basic_info.update({
                    'total_lines': total_lines,
                    'code_lines': code_lines,
                    'has_decorators': len(node.decorator_list) > 0,
                    'decorators': [ast.unparse(dec) for dec in node.decorator_list],
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'has_nested_functions': any(isinstance(child, ast.FunctionDef) 
                                              for child in ast.walk(node)),
                })
                break
        
        return basic_info
    
    def _analyze_complexity(self, source_code: str, function_name: str) -> Dict[str, Any]:
        """Analyze code complexity metrics"""
        tree = ast.parse(source_code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                # Cyclomatic complexity
                complexity = self._calculate_cyclomatic_complexity(node)
                
                # Other metrics
                depth = self._calculate_nesting_depth(node)
                branches = self._count_branches(node)
                loops = self._count_loops(node)
                
                return {
                    'cyclomatic_complexity': complexity,
                    'nesting_depth': depth,
                    'branch_count': branches,
                    'loop_count': loops,
                    'complexity_rating': self._rate_complexity(complexity),
                    'maintainability_score': self._calculate_maintainability(complexity, depth, branches)
                }
        
        return {}
    
    def _analyze_dependencies(self, source_code: str, function_name: str) -> Dict[str, Any]:
        """Analyze function dependencies and imports"""
        tree = ast.parse(source_code)
        
        # Find imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                imports.extend([f"{module}.{alias.name}" for alias in node.names])
        
        # Find function calls within the target function
        function_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            function_calls.append(child.func.id)
                        elif isinstance(child.func, ast.Attribute):
                            function_calls.append(ast.unparse(child.func))
        
        return {
            'imports': imports,
            'external_dependencies': [imp for imp in imports if not imp.startswith('.')],
            'function_calls': list(set(function_calls)),
            'dependency_count': len(set(imports + function_calls)),
            'complexity_score': len(set(imports)) * 0.5 + len(set(function_calls)) * 0.3
        }
    
    def _analyze_patterns(self, source_code: str, function_name: str) -> Dict[str, Any]:
        """Identify common programming patterns"""
        tree = ast.parse(source_code)
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                # Pattern detection
                if self._has_guard_clauses(node):
                    patterns.append("guard_clauses")
                if self._has_factory_pattern(node):
                    patterns.append("factory_pattern")
                if self._has_iterator_pattern(node):
                    patterns.append("iterator_pattern")
                if self._has_validation_pattern(node):
                    patterns.append("validation_pattern")
                if self._has_caching_pattern(node):
                    patterns.append("caching_pattern")
                if self._has_error_handling(node):
                    patterns.append("error_handling")
                
                return {
                    'detected_patterns': patterns,
                    'pattern_count': len(patterns),
                    'design_quality': len(patterns) * 10,  # Simple scoring
                    'recommended_patterns': self._suggest_patterns(node)
                }
        
        return {}
    
    def _analyze_security(self, source_code: str, function_name: str) -> Dict[str, Any]:
        """Basic security analysis"""
        security_issues = []
        recommendations = []
        
        # Simple pattern matching for common issues
        if 'eval(' in source_code:
            security_issues.append("potential_code_injection")
            recommendations.append("Avoid using eval() - use ast.literal_eval() for safe evaluation")
        
        if 'exec(' in source_code:
            security_issues.append("dynamic_code_execution")
            recommendations.append("Avoid exec() - consider alternative approaches")
        
        if re.search(r'open\([\'"][^\'"]', source_code):
            security_issues.append("file_path_handling")
            recommendations.append("Ensure proper path validation and sanitization")
        
        return {
            'security_issues': security_issues,
            'security_score': max(0, 100 - len(security_issues) * 20),
            'recommendations': recommendations
        }
    
    def _analyze_performance(self, source_code: str, function_name: str) -> Dict[str, Any]:
        """Analyze potential performance issues"""
        tree = ast.parse(source_code)
        performance_hints = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                # Check for performance patterns
                if self._has_nested_loops(node):
                    performance_hints.append("Consider optimizing nested loops")
                
                if self._has_string_concatenation_in_loop(node):
                    performance_hints.append("Use join() instead of string concatenation in loops")
                
                if self._has_inefficient_search(node):
                    performance_hints.append("Consider using sets or dictionaries for faster lookups")
        
        return {
            'performance_hints': performance_hints,
            'optimization_potential': len(performance_hints),
            'estimated_complexity': self._estimate_time_complexity(source_code, function_name)
        }
    
    def _determine_docstring_requirements(self, source_code: str, function_name: str) -> Dict[str, Any]:
        """Determine what the docstring should include based on code analysis"""
        basic_info = self._extract_basic_info(source_code, function_name)
        complexity = self._analyze_complexity(source_code, function_name)
        dependencies = self._analyze_dependencies(source_code, function_name)
        
        requirements = {
            'needs_detailed_description': complexity.get('complexity_rating', 'low') in ['high', 'very_high'],
            'needs_examples': complexity.get('cyclomatic_complexity', 0) > 5,
            'needs_parameter_details': len(basic_info.get('parameters', [])) > 2,
            'needs_exception_docs': self._has_exception_handling(source_code, function_name),
            'needs_performance_notes': dependencies.get('complexity_score', 0) > 3,
            'suggested_sections': []
        }
        
        # Build suggested sections
        if requirements['needs_detailed_description']:
            requirements['suggested_sections'].append('detailed_description')
        if requirements['needs_examples']:
            requirements['suggested_sections'].append('examples')
        if requirements['needs_exception_docs']:
            requirements['suggested_sections'].append('raises')
        if requirements['needs_performance_notes']:
            requirements['suggested_sections'].append('note')
        
        return requirements

    # Helper methods for pattern detection
    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.With, ast.AsyncWith)):
                complexity += 1
        
        return complexity
    
    def _calculate_nesting_depth(self, node: ast.FunctionDef) -> int:
        """Calculate maximum nesting depth"""
        def get_depth(node, current_depth=0):
            max_depth = current_depth
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.With, ast.Try)):
                    depth = get_depth(child, current_depth + 1)
                    max_depth = max(max_depth, depth)
                else:
                    depth = get_depth(child, current_depth)
                    max_depth = max(max_depth, depth)
            return max_depth
        
        return get_depth(node)
    
    def _count_branches(self, node: ast.FunctionDef) -> int:
        """Count decision branches"""
        return len([n for n in ast.walk(node) if isinstance(n, (ast.If, ast.IfExp))])
    
    def _count_loops(self, node: ast.FunctionDef) -> int:
        """Count loops"""
        return len([n for n in ast.walk(node) if isinstance(n, (ast.For, ast.While, ast.AsyncFor))])
    
    def _rate_complexity(self, complexity: int) -> str:
        """Rate complexity level"""
        if complexity <= 5:
            return "low"
        elif complexity <= 10:
            return "moderate"
        elif complexity <= 15:
            return "high"
        else:
            return "very_high"
    
    def _calculate_maintainability(self, complexity: int, depth: int, branches: int) -> float:
        """Calculate maintainability score (0-100)"""
        score = 100
        score -= complexity * 2
        score -= depth * 5
        score -= branches * 3
        return max(0, min(100, score))
    
    def _has_guard_clauses(self, node: ast.FunctionDef) -> bool:
        """Check if function uses guard clauses"""
        # Simple check for early returns
        for child in node.body[:3]:  # Check first few statements
            if isinstance(child, ast.If):
                # Check if if-block has return/raise
                for stmt in child.body:
                    if isinstance(stmt, (ast.Return, ast.Raise)):
                        return True
        return False
    
    def _has_factory_pattern(self, node: ast.FunctionDef) -> bool:
        """Check for factory pattern"""
        return 'create' in node.name.lower() or 'build' in node.name.lower()
    
    def _has_iterator_pattern(self, node: ast.FunctionDef) -> bool:
        """Check for iterator pattern"""
        for child in ast.walk(node):
            if isinstance(child, ast.Yield):
                return True
        return False
    
    def _has_validation_pattern(self, node: ast.FunctionDef) -> bool:
        """Check for validation pattern"""
        for child in ast.walk(node):
            if isinstance(child, ast.Raise):
                return True
        return False
    
    def _has_caching_pattern(self, node: ast.FunctionDef) -> bool:
        """Check for caching pattern"""
        # Simple check for cache-related variable names
        source = ast.unparse(node)
        return any(word in source.lower() for word in ['cache', 'memo', 'remember'])
    
    def _has_error_handling(self, node: ast.FunctionDef) -> bool:
        """Check for error handling"""
        for child in ast.walk(node):
            if isinstance(child, ast.Try):
                return True
        return False
    
    def _suggest_patterns(self, node: ast.FunctionDef) -> List[str]:
        """Suggest beneficial patterns"""
        suggestions = []
        
        complexity = self._calculate_cyclomatic_complexity(node)
        if complexity > 5:
            suggestions.append("Consider breaking into smaller functions")
        
        if not self._has_error_handling(node):
            suggestions.append("Consider adding error handling")
        
        return suggestions
    
    def _has_nested_loops(self, node: ast.FunctionDef) -> bool:
        """Check for nested loops"""
        for outer in ast.walk(node):
            if isinstance(outer, (ast.For, ast.While)):
                for inner in ast.walk(outer):
                    if isinstance(inner, (ast.For, ast.While)) and inner != outer:
                        return True
        return False
    
    def _has_string_concatenation_in_loop(self, node: ast.FunctionDef) -> bool:
        """Check for string concatenation in loops"""
        for loop in ast.walk(node):
            if isinstance(loop, (ast.For, ast.While)):
                for child in ast.walk(loop):
                    if isinstance(child, ast.AugAssign) and isinstance(child.op, ast.Add):
                        return True
        return False
    
    def _has_inefficient_search(self, node: ast.FunctionDef) -> bool:
        """Check for inefficient search patterns"""
        source = ast.unparse(node)
        return ' in ' in source and 'list' in source.lower()
    
    def _estimate_time_complexity(self, source_code: str, function_name: str) -> str:
        """Estimate time complexity"""
        if self._has_nested_loops(ast.parse(source_code)):
            return "O(n²) or higher"
        elif 'for' in source_code or 'while' in source_code:
            return "O(n)"
        else:
            return "O(1)"
    
    def _has_exception_handling(self, source_code: str, function_name: str) -> bool:
        """Check if function has exception handling"""
        return 'raise' in source_code or 'try:' in source_code
    
    def _update_dependency_graph(self, analysis_result: Dict[str, Any]) -> None:
        """Update the global dependency graph"""
        # Add nodes and edges to track function relationships
        function_name = analysis_result.get('basic_info', {}).get('name', '')
        if function_name:
            self.dependency_graph.add_node(function_name, **analysis_result)
    
    def _calculate_analysis_quality(self, analysis_result: Dict[str, Any]) -> float:
        """Calculate quality score for the analysis"""
        score = 0.0
        
        # Check completeness of analysis
        if analysis_result.get('basic_info'):
            score += 0.2
        if analysis_result.get('complexity_analysis'):
            score += 0.2
        if analysis_result.get('dependency_analysis'):
            score += 0.2
        if analysis_result.get('pattern_analysis'):
            score += 0.2
        if analysis_result.get('docstring_requirements'):
            score += 0.2
        
        return score

class EnhancedDocstringWriter(AIAgent):
    """
    Enhanced docstring writer that uses detailed analysis 
    to create context-aware, high-quality documentation.
    """
    
    def __init__(self, model_instance):
        super().__init__("EnhancedDocstringWriter", model_instance)
        self.prompt_templates = self._load_prompt_templates()
        self.example_cache = {}
        
    async def execute(self, task: AgentTask) -> AgentResult:
        """Generate enhanced docstring using analysis context"""
        start_time = time.time()
        
        try:
            # Get inputs
            source_code = task.input_data.get('source_code', '')
            function_name = task.input_data.get('function_name', '')
            analysis_data = task.input_data.get('analysis_data', {})
            
            print(f"Writer agent processing: {function_name}")
            
            # Create enhanced prompt using analysis
            try:
                prompt = self._create_enhanced_prompt(source_code, function_name, analysis_data)
                print(f"Prompt created successfully, length: {len(prompt) if prompt else 0}")
            except Exception as e:
                print(f"Error creating prompt: {e}")
                prompt = f"Generate a docstring for the function {function_name}"
            
            # Generate docstring with specialized parameters
            try:
                docstring = await self._generate_with_analysis_context(prompt, analysis_data)
                print(f"Docstring generated, length: {len(docstring) if docstring else 0}")
            except Exception as e:
                print(f"Error generating docstring: {e}")
                docstring = f'"""Basic docstring for {function_name}."""'
            
            return AgentResult(
                task_id=task.task_id,
                agent_role=AgentRole.WRITER,
                success=True,
                data={
                    'docstring': docstring,
                    'prompt_used': (prompt[:200] + "..." if prompt and len(prompt) > 200 else prompt) if prompt else "No prompt",  # First 200 chars for debugging
                    'generation_metadata': {
                        'complexity_adapted': True,
                        'pattern_aware': True,
                        'context_enhanced': True
                    }
                },
                execution_time=time.time() - start_time,
                quality_score=self._estimate_docstring_quality(docstring, analysis_data)
            )
            
        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                agent_role=AgentRole.WRITER,
                success=False,
                data={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _create_enhanced_prompt(self, source_code: str, function_name: str, analysis_data: Dict[str, Any]) -> str:
        """Create context-aware prompt based on analysis"""
        
        # Extract analysis insights
        complexity = analysis_data.get('complexity_analysis', {})
        patterns = analysis_data.get('pattern_analysis', {})
        requirements = analysis_data.get('docstring_requirements', {})
        basic_info = analysis_data.get('basic_info', {})
        performance = analysis_data.get('performance_hints', [])
        
        # Choose prompt template based on complexity
        complexity_level = complexity.get('complexity_rating', 'low')
        template = self.prompt_templates.get(complexity_level, self.prompt_templates['default'])
        
        # Build context sections
        context_sections = []
        
        # Add complexity context
        if complexity.get('cyclomatic_complexity', 0) > 5:
            context_sections.append(f"This function has moderate complexity (cyclomatic complexity: {complexity.get('cyclomatic_complexity')}). Provide clear explanation of the logic flow.")
        
        # Add pattern context
        detected_patterns = patterns.get('detected_patterns', [])
        if detected_patterns:
            context_sections.append(f"This function implements these patterns: {', '.join(detected_patterns)}. Mention these in the docstring.")
        
        # Add performance context
        if performance:
            context_sections.append(f"Performance considerations: {'; '.join(performance[:2])}")
        
        # Build parameter details
        parameters = basic_info.get('parameters', [])
        param_details = []
        for param in parameters:
            param_type = param.get('type_annotation', 'Any')
            param_name = param.get('name', '')
            default_val = param.get('default_value')
            
            if default_val:
                param_details.append(f"        {param_name} ({param_type}, optional): [DETAILED_DESCRIPTION]. Defaults to {default_val}.")
            else:
                param_details.append(f"        {param_name} ({param_type}): [DETAILED_DESCRIPTION]")
        
        # Format the enhanced prompt
        context_str = "\n".join(context_sections) if context_sections else "Standard function documentation needed."
        param_str = "\n".join(param_details) if param_details else "        None"
        
        return template.format(
            source_code=source_code,
            function_name=function_name,
            context=context_str,
            parameters=param_str,
            return_type=basic_info.get('return_annotation', 'Unknown'),
            complexity_notes=self._get_complexity_notes(complexity),
            examples_needed="Yes" if requirements.get('needs_examples', False) else "No"
        )
    
    async def _generate_with_analysis_context(self, prompt: str, analysis_data: Dict[str, Any]) -> str:
        """Generate docstring with analysis-optimized parameters"""
        
        # Adjust generation parameters based on analysis
        complexity = analysis_data.get('complexity_analysis', {})
        max_length = 150  # Base length
        
        # Increase length for complex functions
        if complexity.get('complexity_rating') in ['high', 'very_high']:
            max_length = 250
        
        # Increase temperature for creative examples
        temperature = 0.7
        if analysis_data.get('docstring_requirements', {}).get('needs_examples', False):
            temperature = 0.8
        
        # Generate using your model
        try:
            # Use the DocstringGenerator's internal generator and tokenizer
            response = self.model.generator(
                prompt,
                max_length=len(prompt.split()) + max_length,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.model.tokenizer.eos_token_id
            )
            
            generated_text = response[0]['generated_text']
            docstring = generated_text[len(prompt):].strip()
            
            # Clean and format
            return self._clean_and_format_docstring(docstring)
            
        except Exception as e:
            # Fallback to basic generation
            print(f"⚠️ Advanced generation failed, using fallback: {e}")
            return f'"""Generated with fallback method.\n\nFunction: {prompt.split("def ")[1].split("(")[0] if "def " in prompt else "unknown"}\nNote: Advanced generation unavailable."""'
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load different prompt templates for different complexity levels"""
        return {
            'low': """You are an expert Python developer. Generate a concise, professional Google-style docstring.

Function to document:
```python
{source_code}
```

Context: {context}

Generate a clean, professional docstring:
- Brief description of functionality
- Args section with parameter descriptions
- Returns section if applicable
- Keep it concise but informative

Args:
{parameters}

Returns:
    {return_type}: [DESCRIPTION]

Generate the docstring:""",
            
            'moderate': """You are an expert Python developer. Generate a detailed Google-style docstring for this moderately complex function.

Function to document:
```python
{source_code}
```

Analysis Context: {context}

Generate a comprehensive docstring:
- Clear description of what the function does
- Detailed explanation of the algorithm/approach
- Complete Args section with type hints and descriptions
- Returns section with detailed return value description
- Examples if the function has moderate complexity
- Any relevant notes about performance or usage

Complexity Notes: {complexity_notes}
Include Examples: {examples_needed}

Args:
{parameters}

Returns:
    {return_type}: [DETAILED_DESCRIPTION]

Generate the docstring:""",
            
            'high': """You are a senior Python developer documenting a complex function. Generate an extensive Google-style docstring.

Function to document:
```python
{source_code}
```

Analysis Context: {context}

This is a complex function requiring detailed documentation:
- Comprehensive description of functionality and purpose
- Step-by-step explanation of the algorithm
- Detailed Args section with type hints, constraints, and examples
- Comprehensive Returns section
- Examples demonstrating usage
- Raises section for potential exceptions
- Note section for performance considerations and caveats

Complexity Analysis: {complexity_notes}
Examples Required: {examples_needed}

Args:
{parameters}

Returns:
    {return_type}: [COMPREHENSIVE_DESCRIPTION]

Examples:
    >>> # Include usage examples

Note:
    Include performance considerations and important usage notes.

Generate the docstring:""",
            
            'default': """You are an expert Python developer. Generate a professional Google-style docstring.

Function to document:
```python
{source_code}
```

Context: {context}

Args:
{parameters}

Returns:
    {return_type}: Description of return value.

Generate the docstring:"""
        }
    
    def _get_complexity_notes(self, complexity: Dict[str, Any]) -> str:
        """Generate complexity notes for prompt"""
        notes = []
        
        cyclomatic = complexity.get('cyclomatic_complexity', 0)
        if cyclomatic > 10:
            notes.append(f"High cyclomatic complexity ({cyclomatic})")
        
        nesting = complexity.get('nesting_depth', 0)
        if nesting > 3:
            notes.append(f"Deep nesting ({nesting} levels)")
        
        return "; ".join(notes) if notes else "Standard complexity"
    
    def _clean_and_format_docstring(self, docstring: str) -> str:
        """Clean and format the generated docstring"""
        # Remove any extra whitespace
        cleaned = docstring.strip()
        
        # Ensure proper triple quote formatting
        if not cleaned.startswith('"""'):
            cleaned = '"""' + cleaned
        if not cleaned.endswith('"""'):
            cleaned = cleaned + '"""'
        
        # Remove duplicate quotes
        cleaned = cleaned.replace('""""""', '"""')
        
        # Basic formatting fixes
        cleaned = cleaned.replace('"""```python', '"""')
        cleaned = cleaned.replace('```"""', '"""')
        
        return cleaned
    
    def _estimate_docstring_quality(self, docstring: str, analysis_data: Dict[str, Any]) -> float:
        """Estimate the quality of generated docstring"""
        score = 0.5  # Base score
        
        # Check for required sections
        if 'Args:' in docstring:
            score += 0.2
        if 'Returns:' in docstring:
            score += 0.1
        if len(docstring) > 100:  # Sufficient detail
            score += 0.1
        if 'Examples:' in docstring and analysis_data.get('docstring_requirements', {}).get('needs_examples'):
            score += 0.1
        
        return min(1.0, score)


class DocstringReviewer(AIAgent):
    """
    Quality assessment and improvement agent for generated docstrings.
    """
    
    def __init__(self, model_instance):
        super().__init__("DocstringReviewer", model_instance)
        self.quality_metrics = self._load_quality_metrics()
        
    async def execute(self, task: AgentTask) -> AgentResult:
        """Review and improve docstring quality"""
        start_time = time.time()
        
        try:
            docstring = task.input_data.get('docstring', '')
            analysis_data = task.input_data.get('analysis_data', {})
            source_code = task.input_data.get('source_code', '')
            
            # Perform quality assessment
            quality_assessment = self._assess_quality(docstring, analysis_data, source_code)
            
            # Generate improvements if needed
            improvements = []
            if quality_assessment['overall_score'] < 0.8:
                improvements = await self._generate_improvements(docstring, quality_assessment, analysis_data)
            
            return AgentResult(
                task_id=task.task_id,
                agent_role=AgentRole.REVIEWER,
                success=True,
                data={
                    'quality_assessment': quality_assessment,
                    'improvements': improvements,
                    'final_score': quality_assessment['overall_score'],
                    'recommendations': self._generate_recommendations(quality_assessment)
                },
                execution_time=time.time() - start_time,
                quality_score=quality_assessment['overall_score']
            )
            
        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                agent_role=AgentRole.REVIEWER,
                success=False,
                data={},
                execution_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _assess_quality(self, docstring: str, analysis_data: Dict[str, Any], source_code: str) -> Dict[str, Any]:
        """Comprehensive quality assessment"""
        
        scores = {}
        
        # Completeness check
        scores['completeness'] = self._check_completeness(docstring, analysis_data)
        
        # Accuracy check
        scores['accuracy'] = self._check_accuracy(docstring, source_code)
        
        # Clarity check
        scores['clarity'] = self._check_clarity(docstring)
        
        # Format check
        scores['formatting'] = self._check_formatting(docstring)
        
        # Calculate overall score
        weights = {'completeness': 0.4, 'accuracy': 0.3, 'clarity': 0.2, 'formatting': 0.1}
        overall_score = sum(scores[metric] * weights[metric] for metric in scores)
        
        return {
            'scores': scores,
            'overall_score': overall_score,
            'grade': self._assign_grade(overall_score),
            'word_count': len(docstring.split()),
            'has_examples': 'Examples:' in docstring,
            'has_raises': 'Raises:' in docstring
        }
    
    def _check_completeness(self, docstring: str, analysis_data: Dict[str, Any]) -> float:
        """Check if docstring includes all required sections"""
        score = 0.0
        requirements = analysis_data.get('docstring_requirements', {})
        
        # Basic requirements
        if len(docstring) > 50:  # Minimum length
            score += 0.2
        
        # Parameter documentation
        params = analysis_data.get('basic_info', {}).get('parameters', [])
        if params and 'Args:' in docstring:
            score += 0.3
        elif not params:
            score += 0.3  # No parameters to document
        
        # Return documentation
        if 'Returns:' in docstring:
            score += 0.2
        
        # Examples if needed
        if requirements.get('needs_examples', False):
            if 'Examples:' in docstring:
                score += 0.2
            else:
                score -= 0.1  # Penalty for missing required examples
        else:
            score += 0.1  # Bonus for not having unnecessary examples
        
        # Exception documentation
        if requirements.get('needs_exception_docs', False):
            if 'Raises:' in docstring:
                score += 0.1
        else:
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _check_accuracy(self, docstring: str, source_code: str) -> float:
        """Check if docstring accurately describes the code"""
        # Simple heuristic checks
        score = 0.8  # Assume good by default
        
        # Check for obvious mismatches
        if 'return' in source_code.lower() and 'Returns:' not in docstring:
            score -= 0.2
        
        if 'raise' in source_code.lower() and 'Raises:' not in docstring:
            score -= 0.1
        
        return max(0.0, score)
    
    def _check_clarity(self, docstring: str) -> float:
        """Check clarity and readability"""
        score = 0.8  # Base score
        
        # Check for clear structure
        if docstring.count('\n') < 2:  # Too brief
            score -= 0.2
        
        # Check for proper capitalization
        lines = docstring.split('\n')
        for line in lines[:3]:  # Check first few lines
            if line.strip() and not line.strip()[0].isupper():
                score -= 0.1
                break
        
        return max(0.0, score)
    
    def _check_formatting(self, docstring: str) -> float:
        """Check Google-style formatting"""
        score = 1.0
        
        # Check triple quotes
        if not (docstring.startswith('"""') and docstring.endswith('"""')):
            score -= 0.3
        
        # Check section formatting
        sections = ['Args:', 'Returns:', 'Raises:', 'Examples:', 'Note:']
        for section in sections:
            if section in docstring:
                # Check if section is followed by newline and indentation
                if f"{section}\n    " not in docstring and f"{section}\n        " not in docstring:
                    score -= 0.1
        
        return max(0.0, score)
    
    def _assign_grade(self, score: float) -> str:
        """Assign letter grade to quality score"""
        if score >= 0.9:
            return "A"
        elif score >= 0.8:
            return "B"
        elif score >= 0.7:
            return "C"
        elif score >= 0.6:
            return "D"
        else:
            return "F"
    
    async def _generate_improvements(self, docstring: str, quality_assessment: Dict[str, Any], analysis_data: Dict[str, Any]) -> List[str]:
        """Generate specific improvement suggestions"""
        improvements = []
        
        scores = quality_assessment['scores']
        
        if scores['completeness'] < 0.7:
            improvements.append("Add missing sections (Args, Returns, or Examples)")
        
        if scores['accuracy'] < 0.7:
            improvements.append("Ensure docstring accurately describes the function behavior")
        
        if scores['clarity'] < 0.7:
            improvements.append("Improve clarity and add more detailed explanations")
        
        if scores['formatting'] < 0.7:
            improvements.append("Fix Google-style formatting issues")
        
        return improvements
    
    def _generate_recommendations(self, quality_assessment: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if quality_assessment['overall_score'] < 0.6:
            recommendations.append("Consider rewriting the docstring with more detail")
        elif quality_assessment['overall_score'] < 0.8:
            recommendations.append("Make minor improvements to reach high quality")
        else:
            recommendations.append("Docstring meets high quality standards")
        
        return recommendations
    
    def _load_quality_metrics(self) -> Dict[str, Any]:
        """Load quality assessment metrics"""
        return {
            'min_length': 50,
            'max_length': 1000,
            'required_sections': ['description'],
            'optional_sections': ['Args', 'Returns', 'Raises', 'Examples', 'Note']
        }
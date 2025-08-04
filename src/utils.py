import ast
import inspect
from typing import Dict, List, Optional, Any

def extract_function_info(source_code: str, function_name: str) -> Optional[Dict[str, Any]]:
    """
    Extract detailed information about a Python function from source code.
    
    Args:
        source_code: Python source code as string
        function_name: Name of function to analyze
        
    Returns:
        Dictionary with function details or None if function not found
    """
    try:
        # Parse the source code into an Abstract Syntax Tree
        tree = ast.parse(source_code)
        
        # Find the specific function we want
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return {
                    'name': node.name,
                    'parameters': _extract_parameters(node),
                    'return_annotation': _get_return_annotation(node),
                    'source_lines': _get_function_source(source_code, node),
                    'has_docstring': _has_docstring(node)
                }
        
        return None  # Function not found
        
    except SyntaxError:
        print(f"Error: Invalid Python syntax in source code")
        return None

def _extract_parameters(func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
    """Extract parameter information from a function node."""
    parameters = []
    
    # Get argument names
    for arg in func_node.args.args:
        param_info = {
            'name': arg.arg,
            'type_annotation': None,
            'default_value': None
        }
        
        # Check if parameter has type annotation
        if arg.annotation:
            param_info['type_annotation'] = ast.unparse(arg.annotation)
        
        parameters.append(param_info)
    
    # Handle default values (they're stored separately from names)
    defaults = func_node.args.defaults
    if defaults:
        # Default values apply to the last N parameters
        for i, default in enumerate(defaults):
            param_index = len(parameters) - len(defaults) + i
            parameters[param_index]['default_value'] = ast.unparse(default)
    
    return parameters

def _get_return_annotation(func_node: ast.FunctionDef) -> Optional[str]:
    """Get the return type annotation if it exists."""
    if func_node.returns:
        return ast.unparse(func_node.returns)
    return None

def _get_function_source(source_code: str, func_node: ast.FunctionDef) -> str:
    """Extract the source code lines for the function."""
    lines = source_code.split('\n')
    
    # Get the function definition and body
    start_line = func_node.lineno - 1  # AST line numbers are 1-based
    end_line = func_node.end_lineno if func_node.end_lineno else start_line + 1
    
    function_lines = lines[start_line:end_line]
    return '\n'.join(function_lines)

def _has_docstring(func_node: ast.FunctionDef) -> bool:
    """Check if the function already has a docstring."""
    if func_node.body and isinstance(func_node.body[0], ast.Expr):
        return isinstance(func_node.body[0].value, ast.Constant) and isinstance(func_node.body[0].value.value, str)
    return False

def test_extraction():
    """Test function to verify our extraction works."""
    test_code = '''
def calculate_area(length: float, width: float = 10.0) -> float:
    return length * width

def greet(name):
    print(f"Hello, {name}!")
    '''
    
    result = extract_function_info(test_code, "calculate_area")
    if result:
        print("✅ Function analysis working!")
        print(f"Function: {result['name']}")
        print(f"Parameters: {result['parameters']}")
        print(f"Return type: {result['return_annotation']}")
    else:
        print("❌ Function analysis failed")


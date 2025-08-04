import argparse
import sys
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from .utils import extract_function_info

class DocstringGenerator:
    """Automated docstring generator for Python functions."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"):
        """Initialize the generator with your chosen documentation model."""
        print(f"Loading documentation model: {model_name}")
        
        try:
            # Load tokenizer (converts text to numbers for the model)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load the language model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",  # Use optimal data type
                device_map="auto"    # Use GPU if available, else CPU
            )
            
            # Create a text generation pipeline for easy use
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=1024,     # Maximum output length
                temperature=0.7,     # Creativity level
                do_sample=True       # Use sampling for varied outputs
            )
            
            print("✅ Documentation model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 This might take a few minutes on first run (downloading model)")
            raise

    def create_prompt(self, function_info: Dict[str, Any]) -> str:
        """Create a well-engineered prompt for the language model."""
        
        # Extract function details
        func_name = function_info['name']
        parameters = function_info['parameters']
        return_type = function_info.get('return_annotation', 'Unknown')
        source_code = function_info['source_lines']
        
        # Build parameter descriptions
        param_descriptions = []
        for param in parameters:
            param_name = param['name']
            param_type = param.get('type_annotation', 'Any')
            default_val = param.get('default_value')
            
            if default_val:
                param_descriptions.append(f"        {param_name} ({param_type}, optional): [DESCRIPTION]. Defaults to {default_val}.")
            else:
                param_descriptions.append(f"        {param_name} ({param_type}): [DESCRIPTION]")
        
        param_section = '\n'.join(param_descriptions) if param_descriptions else "        None"
        
        # Create the prompt using Google-style docstring format
        prompt = f"""You are an expert Python developer. Generate a professional Google-style docstring for this function.

Function to document:
```python
{source_code}
```

Requirements:
1. Write a clear, concise description of what the function does
2. Use Google-style docstring format
3. Include Args section with proper descriptions
4. Include Returns section if the function returns something
5. Include Raises section if the function can raise exceptions
6. Do NOT include the function definition, only the docstring content
7. Start with triple quotes and end with triple quotes

Example format:
\"\"\"
Brief description of what the function does.

More detailed description if needed.

Args:
{param_section}

Returns:
    {return_type}: Description of what is returned.

Raises:
    ExceptionType: Description of when this exception is raised.
\"\"\"

Generate the docstring:"""

        return prompt
        
    def generate_docstring(self, source_code: str, function_name: str) -> Optional[str]:
        """Generate a docstring for the specified function."""
        
        # Step 1: Extract function information using your analyzer
        print(f"🔍 Analyzing function: {function_name}")
        function_info = extract_function_info(source_code, function_name)
        
        if not function_info:
            print(f"❌ Function '{function_name}' not found in source code")
            return None
        
        # Skip if function already has a docstring
        if function_info['has_docstring']:
            print(f"⚠️  Function '{function_name}' already has a docstring")
            return None
        
        # Step 2: Create documentation prompt using prompt engineer
        print("📝 Creating documentation prompt...")
        prompt = self.create_prompt(function_info)
        
        # Step 3: Generate docstring using language model
        print("🤖 Generating docstring...")
        try:
            response = self.generator(
                prompt,
                max_length=len(prompt.split()) + 200,  # Prompt + response space
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract the generated text
            generated_text = response[0]['generated_text']
            
            # Remove the prompt from the response to get just the docstring
            docstring = generated_text[len(prompt):].strip()
            
            # Clean up the response
            docstring = self._clean_docstring(docstring)
            
            print("✅ Docstring generated successfully!")
            return docstring
            
        except Exception as e:
            print(f"❌ Error generating docstring: {e}")
            return None

    def _clean_docstring(self, raw_docstring: str) -> str:
        """Clean and format the generated docstring."""
        
        # Remove any extra whitespace
        cleaned = raw_docstring.strip()
        
        # Ensure it starts and ends with triple quotes
        if not cleaned.startswith('"""'):
            cleaned = '"""' + cleaned
        if not cleaned.endswith('"""'):
            cleaned = cleaned + '"""'
        
        # Remove any duplicate triple quotes
        cleaned = cleaned.replace('""""""', '"""')
        
        return cleaned

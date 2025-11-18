"""
Python script demonstrating LLM tool calling using Ollama with OpenAI-compatible API.
Uses qwen3:8b model running locally via Ollama.
"""

import json
import requests
from openai import OpenAI

# Configure OpenAI client to use local Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",  # Ollama's OpenAI-compatible endpoint
    api_key="ollama",  # Required but unused by Ollama
)

# Define available tools
def calculate(operation: str, x: float, y: float) -> float:
    """
    Perform basic mathematical operations.
    
    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide'
        x: First number
        y: Second number
    
    Returns:
        Result of the calculation
    """
    operations = {
        "add": lambda a, b: a + b,
        "subtract": lambda a, b: a - b,
        "multiply": lambda a, b: a * b,
        "divide": lambda a, b: a / b if b != 0 else "Error: Division by zero"
    }
    
    if operation not in operations:
        return f"Error: Unknown operation '{operation}'"
    
    result = operations[operation](x, y)
    print(f"🔧 Tool called: calculate({operation}, {x}, {y}) = {result}")
    return result


def get_random_fact() -> dict:
    """
    Get a random interesting fact using a public API.
    
    Returns:
        Dictionary with a random fact
    """
    try:
        # Use uselessfacts.jsph.pl API (free, no API key needed)
        url = "https://uselessfacts.jsph.pl/api/v2/facts/random"
        print(f"🔧 Tool called: get_random_fact()")
        
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            result = {
                "fact": data["text"],
                "source": data.get("source", "Unknown"),
            }
            print(f"   Fact: {result['fact'][:60]}...")
            return result
        else:
            return {"error": f"Could not fetch random fact. Status: {response.status_code}"}
    
    except Exception as e:
        return {"error": f"Error fetching fact: {str(e)}"}


# Define tool schemas for the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform basic mathematical operations (add, subtract, multiply, divide)",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "The mathematical operation to perform"
                    },
                    "x": {
                        "type": "number",
                        "description": "The first number"
                    },
                    "y": {
                        "type": "number",
                        "description": "The second number"
                    }
                },
                "required": ["operation", "x", "y"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_random_fact",
            "description": "Get a random interesting fact from the internet",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

# Map function names to actual Python functions
available_functions = {
    "calculate": calculate,
    "get_random_fact": get_random_fact,
}


def run_agent(user_message: str):
    """
    Run the agent with tool calling capabilities.
    
    Args:
        user_message: The user's query
    """
    print(f"\n{'='*60}")
    print(f"👤 User: {user_message}")
    print(f"{'='*60}\n")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools. Use them when needed to answer user questions accurately."},
        {"role": "user", "content": user_message}
    ]
    
    # First API call - LLM decides whether to use tools
    print("🤖 Calling LLM...")
    response = client.chat.completions.create(
        model="qwen3:8b",
        messages=messages,
        tools=tools,
        tool_choice="auto",  # Let the model decide
    )
    
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    
    # Check if the model wants to call any tools
    if tool_calls:
        print(f"🔍 LLM wants to use {len(tool_calls)} tool(s)\n")
        
        # Add the assistant's response to messages
        messages.append(response_message)
        
        # Execute each tool call
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"📞 Calling tool: {function_name}")
            print(f"   Arguments: {function_args}")
            
            # Call the actual function
            function_to_call = available_functions[function_name]
            function_response = function_to_call(**function_args)
            
            # Add tool response to messages
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(function_response) if isinstance(function_response, dict) else str(function_response),
            })
        
        # Second API call - get final response with tool results
        print("\n🤖 Getting final response from LLM...\n")
        final_response = client.chat.completions.create(
            model="qwen3:8b",
            messages=messages,
        )
        
        final_message = final_response.choices[0].message.content
    else:
        # No tools needed, use the initial response
        print("ℹ️  LLM answered directly without using tools\n")
        final_message = response_message.content
    
    print(f"✅ Assistant: {final_message}")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Example queries that will trigger different tools
    
    print("\n🚀 Starting LLM Agent with Tool Calling (Ollama + OpenAI API)")
    print("Using model: qwen3:8b (local)")
    
    # Test 1: Calculator tool
    run_agent("What is 11 multiplied by 11?")
    
    # Test 2: Random fact API
    run_agent("Tell me an interesting random fact")
    
    # Test 3: No tool
    run_agent("Tell me some animal that has 4 legs.")
    
    # Test 4: Multiple tools
    run_agent("Calculate 50 plus 25, and also tell me a random fact")



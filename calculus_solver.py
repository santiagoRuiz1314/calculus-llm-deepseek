from dotenv import load_dotenv
import json
import requests
import os

class OllamaCalculusSolver:
    def __init__(self, model_name="deepseek-r1", base_url="http://ollama:11434"):
        """Initialize the Ollama Calculus Solver with model name and base URL."""
        load_dotenv()
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/chat"
        
        # Test connection to Ollama server
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code != 200:
                print(f"Warning: Could not connect to Ollama server at {self.base_url}")
        except requests.exceptions.ConnectionError:
            print(f"Warning: Could not connect to Ollama server at {self.base_url}")
        
        # Define the examples file path
        self.examples_file = os.environ.get("EXAMPLES_FILE", "calculus_examples.json")
        
        # Load example calculus problems and solutions
        try:
            self.load_examples(self.examples_file)
        except:
            # If file doesn't exist, use default examples
            self.examples = [
                {
                    "problem": "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3",
                    "solution": "To find the derivative of f(x) = x^3 + 2x^2 - 5x + 3, we apply the power rule and linearity of differentiation:\n\nf'(x) = 3x^2 + 4x - 5\n\nStep-by-step:\n1. For x^3: The derivative is 3x^2\n2. For 2x^2: The derivative is 4x\n3. For -5x: The derivative is -5\n4. For constant 3: The derivative is 0\n\nCombining all terms: f'(x) = 3x^2 + 4x - 5"
                },
                {
                    "problem": "Calculate the integral of g(x) = 2x + e^x",
                    "solution": "To calculate the indefinite integral of g(x) = 2x + e^x, we apply integration rules:\n\n∫g(x)dx = ∫(2x + e^x)dx = ∫2x dx + ∫e^x dx = x^2 + e^x + C\n\nStep-by-step:\n1. For ∫2x dx: The integral is 2(x^2/2) = x^2\n2. For ∫e^x dx: The integral is e^x\n3. Add constant of integration C\n\nFinal answer: ∫g(x)dx = x^2 + e^x + C"
                },
                {
                    "problem": "Find the local extrema of h(x) = x^3 - 6x^2 + 9x + 1",
                    "solution": "To find the local extrema of h(x) = x^3 - 6x^2 + 9x + 1:\n\nStep 1: Find h'(x)\nh'(x) = 3x^2 - 12x + 9\n\nStep 2: Set h'(x) = 0 and solve\n3x^2 - 12x + 9 = 0\n3(x^2 - 4x + 3) = 0\n3(x - 1)(x - 3) = 0\nx = 1 or x = 3\n\nStep 3: Calculate h''(x)\nh''(x) = 6x - 12\n\nStep 4: Evaluate h''(x) at critical points\nh''(1) = 6(1) - 12 = -6 < 0, so x = 1 is a local maximum\nh''(3) = 6(3) - 12 = 6 > 0, so x = 3 is a local minimum\n\nStep 5: Calculate function values\nh(1) = 1^3 - 6(1)^2 + 9(1) + 1 = 1 - 6 + 9 + 1 = 5\nh(3) = 3^3 - 6(3)^2 + 9(3) + 1 = 27 - 54 + 27 + 1 = 1\n\nFinal answer: Local maximum at x = 1 with h(1) = 5, and local minimum at x = 3 with h(3) = 1"
                }
            ]
    
    def create_system_prompt(self):
        """Create a system prompt for the model to solve calculus problems."""
        return (
            "You are a specialized calculus tutor that helps students solve college-level calculus problems. "
            "Your answers should follow this structure:\n"
            "1. Give the final answer clearly and concisely\n"
            "2. Provide a detailed step-by-step solution showing all work\n"
            "3. Explain any calculus concepts or rules being applied\n"
            "4. Highlight any common mistakes to avoid\n\n"
            "Your solution should be mathematically rigorous but also easy to understand "
            "for college students. Use proper mathematical notation."
        )
    
    def format_examples(self):
        """Format the examples as messages for the model."""
        formatted_examples = []
        for example in self.examples:
            formatted_examples.extend([
                {"role": "user", "content": example["problem"]},
                {"role": "assistant", "content": example["solution"]}
            ])
        return formatted_examples
    
    def solve_problem(self, problem, temperature=0.7):
        """Send a calculus problem to Ollama and get the solution."""
        # Prepare the messages with system prompt, examples for few-shot learning, and the user's problem
        messages = [{"role": "system", "content": self.create_system_prompt()}]
        messages.extend(self.format_examples())
        messages.append({"role": "user", "content": problem})
        
        try:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "options": {
                    "temperature": temperature
                },
                "stream": False
            }
            
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            try:
                result = response.json()
                
                if "message" in result:
                    return result["message"]["content"]
                else:
                    return f"Error: Unexpected response format from Ollama: {result}"
            except json.JSONDecodeError as e:
                return f"Error parsing response: {str(e)}\nResponse text: {response.text[:200]}..."
                
        except requests.exceptions.RequestException as e:
            return f"Error: {str(e)}"
    
    def add_example(self, problem, solution):
        """Add a new example to the training data."""
        self.examples.append({"problem": problem, "solution": solution})
        # Auto-save when adding examples
        self.save_examples()
    
    def save_examples(self, filename=None):
        """Save the examples to a JSON file."""
        if filename is None:
            filename = self.examples_file
            
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        
        with open(filename, "w") as f:
            json.dump(self.examples, f, indent=4)
    
    def load_examples(self, filename=None):
        """Load examples from a JSON file."""
        if filename is None:
            filename = self.examples_file
            
        with open(filename, "r") as f:
            self.examples = json.load(f)
        return True
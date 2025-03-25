from dotenv import load_dotenv
import os
import json
import argparse
import requests

class OllamaCalculusSolver:
    def __init__(self, model_name="llama3", base_url="http://localhost:11434"):
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
                print("Make sure Ollama is running with the command: 'ollama serve'")
            else:
                models = response.json().get("models", [])
                model_names = [model.get("name") for model in models]
                if self.model_name not in model_names and models:
                    print(f"Warning: Model '{self.model_name}' not found in available models: {model_names}")
                    print(f"You may need to run: 'ollama pull {self.model_name}'")
        except requests.exceptions.ConnectionError:
            print(f"Warning: Could not connect to Ollama server at {self.base_url}")
            print("Make sure Ollama is running with the command: 'ollama serve'")
        
        # Load example calculus problems and solutions for fine-tuning
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
                }
            }
            
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if "message" in result:
                return result["message"]["content"]
            else:
                return f"Error: Unexpected response format from Ollama: {result}"
        except requests.exceptions.RequestException as e:
            return f"Error: {str(e)}"
    
    def add_example(self, problem, solution):
        """Add a new example to the training data."""
        self.examples.append({"problem": problem, "solution": solution})
    
    def save_examples(self, filename="calculus_examples.json"):
        """Save the examples to a JSON file."""
        with open(filename, "w") as f:
            json.dump(self.examples, f, indent=4)
    
    def load_examples(self, filename="calculus_examples.json"):
        """Load examples from a JSON file."""
        try:
            with open(filename, "r") as f:
                self.examples = json.load(f)
            return True
        except FileNotFoundError:
            return False


def main():
    parser = argparse.ArgumentParser(description="Ollama Calculus Problem Solver")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Solve command
    solve_parser = subparsers.add_parser("solve", help="Solve a calculus problem")
    solve_parser.add_argument("problem", help="The calculus problem to solve")
    solve_parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for response generation")
    solve_parser.add_argument("--model", default="llama3", help="Model name to use (default: llama3)")
    solve_parser.add_argument("--url", default="http://localhost:11434", help="Ollama server URL")
    
    # Add example command
    add_parser = subparsers.add_parser("add", help="Add a new example with problem and solution")
    add_parser.add_argument("--problem", required=True, help="The calculus problem")
    add_parser.add_argument("--solution", required=True, help="The solution to the problem")
    
    # List examples command
    subparsers.add_parser("list", help="List all examples in the training set")
    
    # List models command
    subparsers.add_parser("models", help="List available models in Ollama")
    
    # Save examples command
    save_parser = subparsers.add_parser("save", help="Save examples to a file")
    save_parser.add_argument("--filename", default="calculus_examples.json", help="Filename to save examples")
    
    # Load examples command
    load_parser = subparsers.add_parser("load", help="Load examples from a file")
    load_parser.add_argument("--filename", default="calculus_examples.json", help="Filename to load examples from")
    
    args = parser.parse_args()
    
    try:
        if args.command == "solve":
            solver = OllamaCalculusSolver(model_name=args.model, base_url=args.url)
            solution = solver.solve_problem(args.problem, args.temperature)
            print("\n=== Calculus Problem ===")
            print(args.problem)
            print("\n=== Solution ===")
            print(solution)
        
        elif args.command == "models":
            base_url = args.url if hasattr(args, 'url') else "http://localhost:11434"
            try:
                response = requests.get(f"{base_url}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
                print("\n=== Available Ollama Models ===")
                if models:
                    for model in models:
                        print(f"- {model.get('name')}")
                else:
                    print("No models found. You can pull models with 'ollama pull <model-name>'")
                    print("Example: ollama pull llama3")
            except requests.exceptions.RequestException as e:
                print(f"Error connecting to Ollama server: {str(e)}")
                print("Make sure Ollama is running with the command: 'ollama serve'")
        
        elif args.command in ["add", "list", "save", "load"]:
            solver = OllamaCalculusSolver()
            
            if args.command == "add":
                solver.add_example(args.problem, args.solution)
                print(f"Added new example. Total examples: {len(solver.examples)}")
                
            elif args.command == "list":
                print("\n=== Calculus Examples ===")
                for i, example in enumerate(solver.examples, 1):
                    print(f"\nExample {i}:")
                    print(f"Problem: {example['problem']}")
                    print(f"Solution: {example['solution']}")
                    
            elif args.command == "save":
                solver.save_examples(args.filename)
                print(f"Saved {len(solver.examples)} examples to {args.filename}")
                
            elif args.command == "load":
                success = solver.load_examples(args.filename)
                if success:
                    print(f"Loaded {len(solver.examples)} examples from {args.filename}")
                else:
                    print(f"Could not find file {args.filename}")
        
        else:
            parser.print_help()
            
    except ValueError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    main()
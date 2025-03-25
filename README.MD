# Ollama Calculus Solver

Un solucionador de problemas de cálculo basado en LLM que utiliza Ollama con el modelo DeepSeek-R1 para resolver problemas matemáticos de nivel universitario con soluciones paso a paso.

## Descripción

Ollama Calculus Solver es una herramienta diseñada para estudiantes y profesores que necesitan resolver problemas de cálculo. Utiliza modelos de lenguaje avanzados para proporcionar soluciones detalladas y explicaciones paso a paso, haciendo que el aprendizaje del cálculo sea más accesible.

La herramienta se conecta a Ollama (que ejecuta localmente el modelo DeepSeek-R1) y proporciona:
- Soluciones detalladas paso a paso
- Explicaciones de conceptos de cálculo
- Identificación de errores comunes
- Notación matemática adecuada

## Características

- Resuelve problemas de derivadas, integrales, límites y extremos locales
- Proporciona explicaciones paso a paso con rigor matemático
- Compatible con diferentes modelos de LLM a través de Ollama
- Ajustes de temperatura para controlar la creatividad de las respuestas
- Interfaz de línea de comandos simple y fácil de usar

## Requisitos Previos

Para utilizar esta herramienta, necesitarás:

- Python 3.8 o superior
- [Ollama](https://ollama.ai/) instalado y configurado
- Modelo DeepSeek-R1 (u otro compatible) disponible en tu instalación de Ollama

##  Instalación

1. Crea un entorno virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

2. Clona este repositorio:
```bash
git clone https://github.com/tuusuario/ollama-calculus-solver.git
cd ollama-calculus-solver
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

4. Asegúrate de tener Ollama instalado y el modelo DeepSeek-R1 descargado:
```bash
ollama pull deepseek-r1
```

## Uso

Para resolver un problema de cálculo, utiliza el siguiente comando:

```
python calculus_solver.py solve "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3"
```

```
python calculus_solver.py solve "Calculate the integral of g(x) = 2x + e^x"
```

```
python calculus_solver.py solve "Find the local extrema of h(x) = x^3 - 6x^2 + 9x + 1"
```

```
python calculus_solver.py solve "Evaluate the limit as x approaches 0 of (sin x)/x"
```

```
python calculus_solver.py solve "Find the derivative of f(x) = sin(x)cos(x)"
```

```
python calculus_solver.py solve "Calculate the definite integral of x^2 from 0 to 3"
```

```
python calculus_solver.py solve "Find all critical points of f(x) = x^4 - 4x^2 + 2"
```

```
python calculus_solver.py solve "Calculate the derivative of g(x) = ln(x^2 + 1)"
```

```
python calculus_solver.py solve "Find the area between the curves y = x^2 and y = x^3 from x = 0 to x = 1"
```

```
python calculus_solver.py solve "Compute the Taylor series expansion of e^x around x = 0 up to the 4th term"
```

```
python calculus_solver.py solve "Find the volume of the solid obtained by rotating y = sin(x) from x = 0 to x = π around the x-axis"
```

```
python calculus_solver.py solve "Calculate the derivative of h(x) = (x^2 + 3x)^4 using the chain rule"
```

```
python calculus_solver.py solve "Find the solution to the differential equation dy/dx = y with y(0) = 1"
```

```
python calculus_solver.py solve "Evaluate the improper integral of 1/x^2 from 1 to infinity"
```

```
python calculus_solver.py solve "Calculate the partial derivatives of f(x,y) = x^2y + y^3"
```

You can also test with different models or parameters:

```
python calculus_solver.py solve "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3" --model llama3
```

```
python calculus_solver.py solve "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3" --temperature 0.2
```

And for debugging purposes:

```
python calculus_solver.py test --model deepseek-r1
```

```
python calculus_solver.py models


### Opciones adicionales:

Puedes especificar un modelo diferente o ajustar la temperatura:

```bash
python calculus_solver.py solve "Tu problema" --model llama3
```

```bash
python calculus_solver.py solve "Tu problema" --temperature 0.2
```

### Comandos de utilidad:

Para probar la conexión:
```bash
python calculus_solver.py test --model deepseek-r1
```

Para listar los modelos disponibles:
```bash
python calculus_solver.py models
```

## Solución de problemas

Si encuentras problemas de conexión, asegúrate de que:
1. Ollama esté instalado y en ejecución
2. El modelo DeepSeek-R1 (u otro especificado) esté disponible
3. El servicio de Ollama esté accesible en http://localhost:11434

## Estructura del proyecto

```
ollama-calculus-solver/
├── calculus_solver.py       # Script principal
├── requirements.txt         # Dependencias del proyecto
├── README.md                # Este archivo
└── examples/                # Ejemplos de problemas y soluciones
```


## Agradecimientos

- Proyecto Ollama por proporcionar una forma sencilla de ejecutar LLMs localmente
- Equipo de DeepSeek por su excelente modelo DeepSeek-R1

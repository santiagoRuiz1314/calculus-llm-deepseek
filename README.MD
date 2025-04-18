# Ollama Calculus Solver

Un solucionador de problemas de cálculo basado en LLM que utiliza Ollama con el modelo DeepSeek-R1 para resolver problemas matemáticos de nivel universitario con soluciones paso a paso, ahora disponible con interfaz web a través de OpenWebUI y despliegue sencillo mediante Docker.

## Descripción

Ollama Calculus Solver es una herramienta diseñada para estudiantes y profesores que necesitan resolver problemas de cálculo. Utiliza modelos de lenguaje avanzados para proporcionar soluciones detalladas y explicaciones paso a paso, haciendo que el aprendizaje del cálculo sea más accesible.

La herramienta se conecta a Ollama (que ejecuta localmente el modelo DeepSeek-R1) y proporciona:
- Soluciones detalladas paso a paso
- Explicaciones de conceptos de cálculo
- Identificación de errores comunes
- Notación matemática adecuada
- Interfaz web amigable mediante OpenWebUI
- API RESTful para integraciones con otros sistemas

## Características

- Resuelve problemas de derivadas, integrales, límites y extremos locales
- Proporciona explicaciones paso a paso con rigor matemático
- Compatible con diferentes modelos de LLM a través de Ollama
- Ajustes de temperatura para controlar la creatividad de las respuestas
- Interfaz web intuitiva para consultas mediante chat
- Despliegue rápido con Docker y Docker Compose
- API basada en FastAPI para integración con aplicaciones externas

## Requisitos Previos

Para utilizar esta herramienta, necesitarás:

- Docker y Docker Compose instalados
- GPU NVIDIA para mejor rendimiento (opcional)
- NVIDIA Container Toolkit si utilizas GPU

## Instalación

1. Clona este repositorio:
```bash
git clone https://github.com/tuusuario/ollama-calculus-solver.git
cd ollama-calculus-solver
```

2. Descarga el modelo DeepSeek-R1 (opcional, pero recomendado):
```bash
docker run --rm -v ollama_data:/root/.ollama ollama/ollama pull deepseek-r1
```

3. Inicia los servicios con Docker Compose:
```bash
docker-compose up -d
```

4. Accede a las interfaces:
   - Interfaz web: http://localhost:3000
   - Documentación API: http://localhost:8000/docs

## Uso

### A través de la interfaz web:

1. Abre tu navegador y navega a http://localhost:3000
2. Crea un nuevo chat y selecciona el modelo DeepSeek-R1
3. Escribe tu problema de cálculo en la interfaz
4. Recibe soluciones detalladas con explicaciones paso a paso

### Ejemplos de problemas para probar:

- Find the derivative of f(x) = x^3 + 2x^2 - 5x + 3
- Calculate the integral of g(x) = 2x + e^x
- Find the local extrema of h(x) = x^3 - 6x^2 + 9x + 1
- Evaluate the limit as x approaches 0 of (sin x)/x
- Find the derivative of f(x) = sin(x)cos(x)
- Calculate the definite integral of x^2 from 0 to 3
- Find all critical points of f(x) = x^4 - 4x^2 + 2
- Calculate the derivative of g(x) = ln(x^2 + 1)
- Find the area between the curves y = x^2 and y = x^3 from x = 0 to x = 1
- Compute the Taylor series expansion of e^x around x = 0 up to the 4th term
- Find the volume of the solid obtained by rotating y = sin(x) from x = 0 to x = π around the x-axis
- Calculate the derivative of h(x) = (x^2 + 3x)^4 using the chain rule
- Find the solution to the differential equation dy/dx = y with y(0) = 1
- Evaluate the improper integral of 1/x^2 from 1 to infinity
- Calculate the partial derivatives of f(x,y) = x^2y + y^3

### A través de la API REST:

Puedes interactuar con el solucionador mediante peticiones HTTP:

```bash
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{"problem": "Find the derivative of x^2 + 3x", "temperature": 0.7, "model": "deepseek-r1"}'
```

## Personalización de ejemplos

Puedes añadir tus propios ejemplos para mejorar las capacidades del solucionador:

```bash
curl -X POST http://localhost:8000/examples/add \
  -H "Content-Type: application/json" \
  -d '{"problem": "Find the derivative of f(x) = e^(2x) * cos(x)", "solution": "La solución detallada paso a paso..."}'
```

## Estructura del proyecto

```
ollama-calculus-solver/
├── app.py                    # API FastAPI
├── calculus_solver.py        # Núcleo del solucionador de cálculo
├── Dockerfile                # Configuración Docker
├── docker-compose.yml        # Orquestación de servicios
├── requirements.txt          # Dependencias Python
├── calculus_examples.json    # Ejemplos para entrenamiento
└── README.md                 # Este archivo
```

## Solución de problemas

Si encuentras problemas con la instalación, revisa los logs:
```bash
docker-compose logs -f calculus-api
docker-compose logs -f ollama
docker-compose logs -f open-webui
```

Asegúrate de que:
1. Docker y Docker Compose estén instalados correctamente
2. El modelo DeepSeek-R1 esté disponible
3. Los puertos 3000, 8000 y 11434 no estén siendo utilizados por otras aplicaciones

Si Ollama no puede cargar el modelo, intenta descargarlo manualmente:
```bash
docker exec -it ollama ollama pull deepseek-r1
```

## Configuración avanzada

Para habilitar la autenticación en la interfaz web, modifica la variable `WEBUI_AUTH=true` en el archivo docker-compose.yml. Las credenciales predeterminadas son:
- Usuario: admin
- Contraseña: password

Es recomendable cambiar estas credenciales después del primer inicio de sesión.

## Agradecimientos

- Proyecto Ollama por proporcionar una forma sencilla de ejecutar LLMs localmente
- Equipo de DeepSeek por su excelente modelo DeepSeek-R1
- Proyecto OpenWebUI por la interfaz web
- FastAPI por el framework de API
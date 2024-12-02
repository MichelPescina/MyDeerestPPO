# MyDeeerestPPO (Español)
Los contenidos de este repositorio son fruto de un proyecto en el que participe durante una estancia académica en el IPICyT. El objetivo de este proyecto fue entrenar un agente basado en redes neuronales para poder recorrer un camino donde señales con formas geométricas le indicaran a donde ir a continuación con un robot movil omnidireccional. Una de las limitantes fue el poder de cómputo, ya que el robot disponía solamente de una Raspberry Pi 4 para correr el agente, así que el diseño de la arquitectura de las redes estuvo orientado a eso.

El codigo presentado aquí contiene la implementación del algoritmo de Optimización por Políticas Próximas así como el código necesario para entrenar el agente utilizando el entorno FigFollowerEnv-v1, cuyo código e instrucciones de instalación se pueden encontrar en el repositorio [FigureFollowerEnv](https://github.com/MichelPescina/FigureFollowerEnv).

## Uso
El archivo FigFollowerPPO.py contiene el código para entrenar el agente. Dentro de este se puede encontrar el ciclo principal que va cambiando entre la fase de recolección de datos, dónde el agente interactua con el ambiente, y la fase de aprendizaje, dónde los pesos de las redes neuronales se actualizan con el fin de aprender. El código para la fase de aprendizaje se encuentra dentro del archivo MyDeerestPPO/PPO.py ó PPO_Recurrent.py según sea el tipo de agente que estés utilizando. También se van haciendo respaldos del agente en diferentes partes del entrenamiento para poder evaluarlo después.

El script EvaluateRAgent.py evalua al agente tras el entrenamiento en diferentes puntos de tiempo utilizando los respaldos almacenados dentro de Training. Aparte de esto, su así se desea, se puede ver como es la interacción del agente con su entrno utilizando una opción que provee el script. Al finalizar genera un gráfico dónde se muestra la recompensa obtenida por el agente utilizando las distintas políticas, así también el porcentaje de recompensa positiva que obtuvo en comparación de toda la disponible.

## Referencias
1. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
2. [Towards Delivering a Coherent Self-Contained Explanation of Proximal Policy Optimization](https://fse.studenttheses.ub.rug.nl/25709/)
3. [The 37 Implementation Details of Proximal Policy Optimization](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

## Agradecimientos
El código en este repositorio fue resultado de un proyecto en el que participé durante mi estancia en el IPICyT
Quiero agradecer al IPICyT y CONAHCYT por haberme permitido participar en este proyecto así como por el apoyo que me brindaron. También quiero agradecer al Dr. Juan Gonzalo Barajas Ramírez, a mis compañeros del laboratorio y personal de la institución por todo.
¡Muchas gracias!

Esto lo pongo de mi propia voluntad, simplemente me nació.
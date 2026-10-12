# Campaña de retención Starbucks Rewards: a quién enviar el cupón de $5

**Para:** Presidencia de Marketing
**De:** Grupo 12 — Marketing Analytics
**Asunto:** Recomendación de alcance para la campaña de cupón de $5

---

## Recomendación

Enviar el cupón al **20% de la base**, seleccionado por uplift estimado con un modelo T-learner
de Random Forest. Son 8,000 de los 40,000 clientes. Sobre el conjunto de prueba del experimento
esa política genera **$4,434 de utilidad neta**; llevada a la base completa, del orden de
**$14,800 por campaña**.

## El costo de mandarle el cupón a todos

Enviar el cupón a toda la base no es una decisión neutral: destruye valor. En el experimento,
contactar al 100% de los clientes perdió **$20,236** de utilidad neta sobre el conjunto de
prueba, equivalente a unos **$67,000** si se proyecta a los 40,000 clientes.

La razón está en la composición de la base. El **48.9%** de los clientes compra con o sin
cupón. Para ellos el descuento no genera una venta adicional: la subsidia. Cada uno de esos
clientes cuesta en promedio **$3.95** cuando se le envía el cupón. A eso se suma un **7.2%** de
clientes cuya respuesta al contacto es negativa, con un costo promedio de **$3.45**. Solo el
**22.4%** de la base responde al cupón con una compra que no habría hecho de otra forma, y ese
grupo aporta **+$2.26** por cliente.

La campaña masiva funciona entonces como una transferencia: el margen que generan los
persuadables se consume subsidiando a quienes ya eran clientes.

## Por qué un modelo de churn no resuelve el problema

La respuesta intuitiva es targetear a los clientes en riesgo de perderse. Lo probamos. Un
modelo de churn aplicado al 30% de mayor riesgo deja **$245** — positivo, pero marginal: siete
centavos por cliente contactado.

El diagnóstico es claro al revisar a quién eligió: ese 30% está compuesto en un **71% por
clientes que no compran bajo ninguna circunstancia**. El modelo hizo exactamente lo que se le
pidió — encontrar a quienes no van a comprar — pero esa no es la pregunta del negocio. La
pregunta no es quién se va, sino a quién le cambia la decisión el cupón. Medida contra el
uplift verdadero, la probabilidad de churn tiene una correlación de apenas **0.227**.

## Qué aporta el T-learner

El T-learner entrena dos modelos, uno con los clientes tratados y otro con los de control, y
estima la diferencia. Ordenar por esa diferencia es ordenar por la pregunta correcta.

Comparamos varios modelos base manteniendo todo lo demás constante:

| Modelo dentro del T-learner | Correlación con el uplift real | Qini | % óptimo | Valor total |
|---|---|---|---|---|
| Logística (la del notebook) | 0.619 | +0.142 | 25% | $2,454 |
| Árbol sin límite | 0.204 | +0.066 | 5% | -$342 |
| Árbol max_depth=2 | 0.605 | +0.112 | 25% | $2,340 |
| Árbol max_depth=4, min_samples_leaf=200 | 0.748 | +0.143 | 20% | $3,569 |
| **Random Forest (300, depth=6, leaf=100)** | **0.857** | **+0.157** | **20%** | **$4,434** |
| Gradient Boosting (depth=3, lr=0.05) | 0.838 | +0.156 | 20% | $4,480 |

El Random Forest y el Gradient Boosting quedan prácticamente empatados y ambos superan a la
logística por un margen amplio: casi duplican el valor recuperado. El árbol sin podar es el
peor de todos, por sobreajuste — memoriza el ruido del entrenamiento, y como el uplift se
obtiene restando dos modelos, ese ruido se amplifica. Seleccionamos Random Forest por tener la
mejor correlación con la verdad y por ser más estable ante cambios en la muestra.

## Qué tan cerca estuvo el modelo de la realidad

El dataset del experimento incluye una columna que en una campaña real no existe: el tipo
verdadero de cada cliente y su uplift real. Al compararla contra nuestras estimaciones:

- La correlación entre el uplift estimado y el verdadero es **0.857**.
- El 20% que el modelo selecciona está compuesto en un **95.8% por persuadables**, frente al
  22.4% que representan en la base general.
- El valor real de contacto promedio en ese grupo es **+$2.01** por cliente, contra **-$1.73**
  de la base completa.

El modelo no es perfecto, pero acierta en lo que importa para la decisión: concentra el
presupuesto casi exclusivamente en el único segmento donde el cupón genera margen incremental.

## Por qué 20% y no otro número

El valor por cliente es más alto en cortes pequeños — al 10% de la base el promedio es mayor —
pero el valor total es menor porque se contacta a menos gente. Conforme se amplía el corte, el
promedio cae y el total sube, hasta que el cliente marginal deja de aportar. Ese cruce ocurre en
el 20%. Al 40% la política ya es negativa, y de ahí en adelante cada punto porcentual adicional
resta.

La decisión de negocio es el total, no el promedio, y el total se maximiza en 20%.

## Riesgos y siguiente paso

Dos consideraciones antes de ejecutar. Primero, el 20% seleccionado todavía incluye **42
clientes de respuesta negativa** (1.8% del grupo); el modelo los reduce de forma importante
pero no los elimina. Segundo, estas cifras provienen de una sola campaña, y el comportamiento de
respuesta cambia con la estacionalidad.

Recomendamos ejecutar la próxima campaña al 20% manteniendo un grupo de control aleatorio del
10%, lo que permite validar el modelo en producción y reentrenarlo con datos frescos sin perder
la capacidad de medir el efecto incremental.

---

*Resultados obtenidos sobre la campaña completa de 40,000 clientes, conjunto de prueba de 12,000
clientes no vistos durante el entrenamiento. Detalle metodológico en `laboratorio_final.ipynb`.*

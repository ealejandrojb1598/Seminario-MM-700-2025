# =============================================================================
# 1. INSTALACION Y CARGA DE PAQUETES
# =============================================================================

paquetes <- c(
  "randomForest",    # Algoritmo Random Forest
  "caret",           # Preprocesamiento y validacion
  "ggplot2",         # Visualizaciones
  "dplyr",           # Manipulacion de datos
  "tidyr",           # Limpieza de datos
  "pROC",            # Curvas ROC y AUC
  "corrplot",        # Matrices de correlacion
  "gridExtra",       # Multiples graficos
  "RColorBrewer",    # Paletas de colores
  "scales",          # Escalas para graficos
  "rmarkdown",       # Para generar reportes HTML/PDF
  "knitr",           # Soporte para reportes
  "kableExtra"       # Tablas formateadas en HTML/PDF
)

# Instalar paquetes que no esten instalados
paquetes_faltantes <- paquetes[!(paquetes %in% installed.packages()[,"Package"])]
if(length(paquetes_faltantes) > 0) {
  install.packages(paquetes_faltantes, dependencies = TRUE)
}

# Cargar todos los paquetes
invisible(lapply(paquetes, library, character.only = TRUE))

# Mensaje de confirmacion
cat("✓ Todos los paquetes cargados correctamente\n")

# =============================================================================
# 2. CONFIGURACION INICIAL
# =============================================================================

# Establecer semilla para reproducibilidad
set.seed(123)

# Crear carpetas para resultados si no existen
if(!dir.exists("resultados")) dir.create("resultados")
if(!dir.exists("graficos")) dir.create("graficos")
if(!dir.exists("reportes")) dir.create("reportes")

cat("✓ Configuracion inicial completa\n")

# =============================================================================
# 3. CARGA DE DATOS
# =============================================================================

cat("\n=== CARGANDO DATOS ===\n")

# Cargar datos
datos <- read.csv("heart_2020_cleaned.csv", stringsAsFactors = FALSE, na.strings = c("", "NA", "N/A"))

cat("✓ Datos cargados exitosamente\n")
cat("Dimensiones:", nrow(datos), "filas x", ncol(datos), "columnas\n")

# =============================================================================
# 4. EXPLORACION INICIAL DE DATOS
# =============================================================================

cat("\n=== EXPLORACION INICIAL ===\n")

# Estructura de los datos
cat("\nEstructura de los datos:\n")
str(datos)

# Resumen estadistico
cat("\nResumen estadistico:\n")
summary(datos)

# Verificar nombres de columnas
cat("\nNombres de las columnas:\n")
print(colnames(datos))

# Verificar valores faltantes
cat("\nValores faltantes por columna:\n")
na_count <- colSums(is.na(datos))
print(na_count[na_count > 0])

# Proporcion de clases en la variable objetivo
cat("\nDistribucion de la variable objetivo (HeartDisease):\n")
tabla_objetivo <- table(datos$HeartDisease)
print(tabla_objetivo)
cat("Proporcion:\n")
print(prop.table(tabla_objetivo) * 100)

# =============================================================================
# 5. PREPROCESAMIENTO DE DATOS
# =============================================================================

cat("\n=== PREPROCESAMIENTO ===\n")

# Crear una copia para preprocesamiento
datos_prep <- datos

# Convertir variable objetivo a factor
datos_prep$HeartDisease <- factor(datos_prep$HeartDisease, 
                                  levels = c("No", "Yes"),
                                  labels = c("No", "Yes"))

# Identificar variables categoricas y numericas
vars_categoricas <- c("Sex", "AgeCategory", "Race", "Diabetic", 
                      "GenHealth", "Smoking", "AlcoholDrinking",
                      "Stroke", "DiffWalking", "PhysicalActivity",
                      "Asthma", "KidneyDisease", "SkinCancer")

vars_numericas <- c("BMI", "PhysicalHealth", "MentalHealth", "SleepTime")

# Convertir variables categoricas a factores
for(var in vars_categoricas) {
  if(var %in% colnames(datos_prep)) {
    datos_prep[[var]] <- as.factor(datos_prep[[var]])
  }
}

cat("✓ Variables convertidas a factores\n")

# Verificar valores faltantes despues de conversion
cat("\nValores faltantes despues de conversion:\n")
na_final <- sum(is.na(datos_prep))
cat("Total de NA:", na_final, "\n")

# Eliminar filas con NA si existen (opcional)
if(na_final > 0) {
  datos_prep <- na.omit(datos_prep)
  cat("✓ Filas con NA eliminadas. Nuevo tamaño:", nrow(datos_prep), "\n")
}

# =============================================================================
# 6. ANALISIS EXPLORATORIO DE DATOS (EDA)
# =============================================================================

cat("\n=== ANALISIS EXPLORATORIO ===\n")

# Grafico 1: Distribucion de la variable objetivo
p1 <- ggplot(datos_prep, aes(x = HeartDisease, fill = HeartDisease)) +
  geom_bar(alpha = 0.7) +
  geom_text(stat = 'count', aes(label = after_stat(count)), vjust = -0.5) +
  scale_fill_manual(values = c("No" = "#2ecc71", "Yes" = "#e74c3c")) +
  labs(title = "Distribucion de Cardiopatias",
       subtitle = paste("Total casos:", nrow(datos_prep)),
       x = "Enfermedad Cardiaca",
       y = "Frecuencia") +
  theme_minimal() +
  theme(legend.position = "none")

print(p1)
ggsave("graficos/01_distribucion_objetivo.png", p1, width = 8, height = 6)

# Grafico 2: Distribucion por edad
if("AgeCategory" %in% colnames(datos_prep)) {
  p2 <- ggplot(datos_prep, aes(x = AgeCategory, fill = HeartDisease)) +
    geom_bar(position = "fill") +
    scale_fill_manual(values = c("No" = "#2ecc71", "Yes" = "#e74c3c")) +
    labs(title = "Prevalencia de Cardiopatia por Grupo de Edad",
         x = "Categoria de Edad",
         y = "Proporcion",
         fill = "Cardiopatia") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  print(p2)
  ggsave("graficos/02_cardiopatia_edad.png", p2, width = 10, height = 6)
}

# Grafico 3: BMI vs Cardiopatia
if("BMI" %in% colnames(datos_prep)) {
  p3 <- ggplot(datos_prep, aes(x = HeartDisease, y = BMI, fill = HeartDisease)) +
    geom_boxplot(alpha = 0.7) +
    scale_fill_manual(values = c("No" = "#2ecc71", "Yes" = "#e74c3c")) +
    labs(title = "Distribucion de IMC segun Cardiopatia",
         x = "Enfermedad Cardiaca",
         y = "IMC (Indice de Masa Corporal)") +
    theme_minimal() +
    theme(legend.position = "none")
  
  print(p3)
  ggsave("graficos/03_bmi_cardiopatia.png", p3, width = 8, height = 6)
}

# Grafico 4: Salud mental vs fisica
if(all(c("PhysicalHealth", "MentalHealth") %in% colnames(datos_prep))) {
  p4 <- ggplot(datos_prep, aes(x = PhysicalHealth, y = MentalHealth, color = HeartDisease)) +
    geom_point(alpha = 0.6, size = 1) +
    scale_color_manual(values = c("No" = "#2ecc71", "Yes" = "#e74c3c")) +
    labs(title = "Salud Mental vs Salud Fisica",
         x = "Salud Fisica (dias)",
         y = "Salud Mental (dias)",
         color = "Cardiopatia") +
    theme_minimal()
  
  print(p4)
  ggsave("graficos/04_salud_mental_fisica.png", p4, width = 10, height = 6)
}

cat("✓ Graficos exploratorios generados y guardados\n")

# =============================================================================
# 7. DIVISION DE DATOS (ENTRENAMIENTO Y PRUEBA)
# =============================================================================

cat("\n=== DIVISION DE DATOS ===\n")

# Crear indices de particion estratificada (80% entrenamiento, 20% prueba)
indices_entrenamiento <- createDataPartition(datos_prep$HeartDisease, 
                                             p = 0.80, 
                                             list = FALSE)

# Dividir datos
datos_train <- datos_prep[indices_entrenamiento, ]
datos_test <- datos_prep[-indices_entrenamiento, ]

cat("✓ Datos divididos:\n")
cat("  - Entrenamiento:", nrow(datos_train), "observaciones\n")
cat("  - Prueba:", nrow(datos_test), "observaciones\n")

# Verificar balance en ambos conjuntos
cat("\nBalance en conjunto de entrenamiento:\n")
print(table(datos_train$HeartDisease))
cat("\nBalance en conjunto de prueba:\n")
print(table(datos_test$HeartDisease))

# =============================================================================
# 8. CONFIGURACION DE VALIDACION CRUZADA
# =============================================================================

cat("\n=== CONFIGURACION DE VALIDACION CRUZADA ===\n")

# Configurar control de entrenamiento con validacion cruzada k-fold (k=10)
control_cv <- trainControl(
  method = "cv",                    # Validacion cruzada
  number = 10,                      # 10 folds
  summaryFunction = twoClassSummary,# Metricas para clasificacion binaria
  classProbs = TRUE,                # Calcular probabilidades de clase
  savePredictions = "final",        # Guardar predicciones finales
  verboseIter = TRUE                # Mostrar progreso
)

cat("✓ Validacion cruzada configurada (10-fold CV)\n")

# =============================================================================
# 9. ENTRENAMIENTO DE RANDOM FOREST - MODELO BASICO
# =============================================================================

cat("\n=== ENTRENAMIENTO DE RANDOM FOREST (MODELO BASICO) ===\n")
cat("Este proceso puede tomar varios minutos...\n")

# Entrenar modelo basico con parametros por defecto
modelo_rf_basico <- randomForest(
  HeartDisease ~ .,                 # Formula: predecir HeartDisease con todas las demas variables
  data = datos_train,               # Datos de entrenamiento
  ntree = 100,                      # Numero de arboles
  mtry = sqrt(ncol(datos_train)-1), # Numero de variables a considerar en cada split
  importance = TRUE,                # Calcular importancia de variables
  na.action = na.omit               # Omitir NAs
)

cat("✓ Modelo basico entrenado exitosamente\n")

# Resumen del modelo
print(modelo_rf_basico)

# =============================================================================
# 10. EVALUACION DEL MODELO BASICO
# =============================================================================

cat("\n=== EVALUACION DEL MODELO BASICO ===\n")

# Predicciones en conjunto de prueba
predicciones_test <- predict(modelo_rf_basico, datos_test, type = "response")
predicciones_prob <- predict(modelo_rf_basico, datos_test, type = "prob")

# Matriz de confusion
matriz_confusion <- confusionMatrix(predicciones_test, datos_test$HeartDisease, positive = "Yes")
print(matriz_confusion)

# Guardar metricas
metricas <- data.frame(
  Exactitud = matriz_confusion$overall['Accuracy'],
  Sensibilidad = matriz_confusion$byClass['Sensitivity'],
  Especificidad = matriz_confusion$byClass['Specificity'],
  Precision = matriz_confusion$byClass['Pos Pred Value'],
  F1_Score = matriz_confusion$byClass['F1']
)

cat("\n=== METRICAS DEL MODELO ===\n")
print(round(metricas * 100, 2))

# Guardar metricas en CSV
write.csv(metricas, "resultados/metricas_modelo_basico.csv", row.names = TRUE)

# =============================================================================
# 11. CURVA ROC Y AUC
# =============================================================================

cat("\n=== CURVA ROC ===\n")

# Calcular curva ROC
roc_obj <- roc(datos_test$HeartDisease, predicciones_prob[,2])
auc_value <- auc(roc_obj)

cat("AUC-ROC:", round(auc_value, 4), "\n")

# Graficar curva ROC
png("graficos/05_curva_roc.png", width = 800, height = 600)
plot(roc_obj, 
     main = paste("Curva ROC - Random Forest\nAUC =", round(auc_value, 4)),
     col = "#3498db",
     lwd = 2,
     print.auc = TRUE,
     auc.polygon = TRUE,
     auc.polygon.col = alpha("#3498db", 0.2))
abline(a = 0, b = 1, lty = 2, col = "gray")
dev.off()

cat("✓ Curva ROC generada\n")

# =============================================================================
# 12. IMPORTANCIA DE VARIABLES
# =============================================================================

cat("\n=== IMPORTANCIA DE VARIABLES ===\n")

# Extraer importancia de variables
importancia <- importance(modelo_rf_basico)
importancia_df <- data.frame(
  Variable = rownames(importancia),
  MeanDecreaseAccuracy = importancia[, "MeanDecreaseAccuracy"],
  MeanDecreaseGini = importancia[, "MeanDecreaseGini"]
)

# Ordenar por MeanDecreaseGini
importancia_df <- importancia_df[order(-importancia_df$MeanDecreaseGini), ]

# Mostrar top 10
cat("\nTop 10 variables mas importantes:\n")
print(head(importancia_df, 10))

# Graficar importancia
png("graficos/06_importancia_variables.png", width = 1000, height = 700)
varImpPlot(modelo_rf_basico, 
           main = "Importancia de Variables - Random Forest",
           pch = 19,
           col = "#e74c3c")
dev.off()

# Guardar tabla de importancia
write.csv(importancia_df, "resultados/importancia_variables.csv", row.names = FALSE)

cat("✓ Analisis de importancia completado\n")

# =============================================================================
# 13. GRAFICO DE ERROR OOB
# =============================================================================

cat("\n=== ANALISIS DE ERROR OOB ===\n")

# Extraer error OOB
oob_error <- modelo_rf_basico$err.rate

# Crear dataframe para graficar
error_df <- data.frame(
  Arboles = 1:nrow(oob_error),
  OOB = oob_error[, "OOB"],
  No = oob_error[, "No"],
  Yes = oob_error[, "Yes"]
)

# Graficar convergencia del error
p_oob <- ggplot(error_df, aes(x = Arboles)) +
  geom_line(aes(y = OOB, color = "OOB"), size = 1.2) +
  geom_line(aes(y = No, color = "No (Sin Cardiopatia)"), size = 0.8, alpha = 0.7) +
  geom_line(aes(y = Yes, color = "Yes (Con Cardiopatia)"), size = 0.8, alpha = 0.7) +
  scale_color_manual(values = c("OOB" = "#34495e", 
                                "No (Sin Cardiopatia)" = "#2ecc71",
                                "Yes (Con Cardiopatia)" = "#e74c3c")) +
  labs(title = "Convergencia del Error Out-of-Bag",
       x = "Numero de Arboles",
       y = "Tasa de Error",
       color = "Tipo de Error") +
  theme_minimal() +
  theme(legend.position = "bottom")

print(p_oob)
ggsave("graficos/07_convergencia_oob.png", p_oob, width = 10, height = 6)

cat("✓ Grafico de error OOB generado\n")

# =============================================================================
# 14. VISUALIZACION AVANZADA DEL RANDOM FOREST
# =============================================================================

cat("\n=== VISUALIZACION AVANZADA DEL RANDOM FOREST ===\n")

# Grafico de evolucion del error OOB (version base R)
png("graficos/08_evolucion_error_rf.png", width = 1000, height = 600)
plot(modelo_rf_basico, 
     main = "Evolucion del Error OOB - Random Forest",
     lwd = 2)
legend("topright", 
       legend = c("Error OOB", "Error No", "Error Yes"),
       col = c("black", "green", "red"),
       lwd = 2)
dev.off()

# Importancia de variables detallada
png("graficos/09_importancia_variables_detallada.png", width = 1200, height = 800)
par(mfrow = c(1, 2))
# Importancia por precision
barplot(sort(importancia[, "MeanDecreaseAccuracy"], decreasing = TRUE)[1:10],
        main = "Top 10 - Importancia por Precision",
        xlab = "Reduccion en Precision", 
        col = "steelblue", las = 2)
# Importancia por Gini
barplot(sort(importancia[, "MeanDecreaseGini"], decreasing = TRUE)[1:10],
        main = "Top 10 - Importancia por Gini",
        xlab = "Reduccion en Impureza Gini", 
        col = "darkred", las = 2)
dev.off()

# Distribucion de probabilidades
png("graficos/10_distribucion_probabilidades.png", width = 1000, height = 600)
par(mfrow = c(1, 2))
hist(predicciones_prob[, "Yes"], 
     main = "Probabilidades Clase YES",
     xlab = "Probabilidad", 
     col = "lightcoral",
     breaks = 20)
hist(predicciones_prob[, "No"], 
     main = "Probabilidades Clase NO", 
     xlab = "Probabilidad", 
     col = "lightgreen",
     breaks = 20)
dev.off()

cat("✓ Graficos avanzados generados\n")

# =============================================================================
# 15. GENERAR REPORTE HTML MEJORADO
# =============================================================================

cat("\n=== GENERANDO REPORTE HTML ===\n")

# [AQUI IRIA EL CODIGO COMPLETO DEL REPORTE HTML QUE TE PASE ANTES]
# (Es muy largo, pero incluye todo el HTML con metricas, graficos, etc.)

# =============================================================================
# 16. RESUMEN Y GUARDADO DEL MODELO
# =============================================================================

cat("\n=== GUARDANDO MODELO ===\n")

# Guardar el modelo entrenado
saveRDS(modelo_rf_basico, "resultados/modelo_rf_basico.rds")

# Crear reporte de resumen
resumen <- list(
  fecha_analisis = Sys.time(),
  n_observaciones_total = nrow(datos_prep),
  n_entrenamiento = nrow(datos_train),
  n_prueba = nrow(datos_test),
  n_arboles = modelo_rf_basico$ntree,
  variables_utilizadas = ncol(datos_train) - 1,
  auc_roc = as.numeric(auc_value),
  metricas = metricas
)

# Guardar resumen
saveRDS(resumen, "resultados/resumen_analisis.rds")

cat("✓ Resumen guardado\n")

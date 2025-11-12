# =======================================================================
# 1. INSTALACION Y CARGA DE PAQUETES
# =======================================================================
paquetes <- c(
  "randomForest", "caret", "ggplot2", "dplyr", "tidyr",
  "pROC", "corrplot", "gridExtra", "RColorBrewer", "ROSE"
)

paquetes_faltantes <- paquetes[!(paquetes %in% installed.packages()[,"Package"])]
if(length(paquetes_faltantes) > 0) {
  install.packages(paquetes_faltantes, dependencies = TRUE)
}

invisible(lapply(paquetes, library, character.only = TRUE))

# ========================================================================
# 2. CONFIGURACION INICIAL
# ========================================================================
set.seed(123)
if(!dir.exists("resultados")) dir.create("resultados")
if(!dir.exists("graficos")) dir.create("graficos")

# ========================================================================
# 3. CARGA DE DATOS
# =======================================================================
datos <- read.csv("heart_2020_cleaned.csv", stringsAsFactors = FALSE,
                  na.strings = c("", "NA", "N/A"))

# ========================================================================
# 4. PREPROCESAMIENTO
# ========================================================================
datos_prep <- datos
datos_prep$HeartDisease <- factor(datos_prep$HeartDisease,
                                  levels = c("No", "Yes"),
                                  labels = c("No", "Yes"))

vars_categoricas <- c("Sex", "AgeCategory", "Race", "Diabetic", "GenHealth",
                      "Smoking", "AlcoholDrinking", "Stroke", "DiffWalking",
                      "PhysicalActivity", "Asthma", "KidneyDisease", "SkinCancer")

for(var in vars_categoricas) {
  if(var %in% colnames(datos_prep)) {
    datos_prep[[var]] <- as.factor(datos_prep[[var]])
  }
}

datos_prep <- na.omit(datos_prep)

# ========================================================================
# 5. ANALISIS EXPLORATORIO
# ========================================================================
p1 <- ggplot(datos_prep, aes(x = HeartDisease, fill = HeartDisease)) +
  geom_bar() +
  geom_text(stat = 'count', aes(label = after_stat(count)), vjust = -0.5, size = 5) +
  scale_fill_manual(values = c("No" = "#2ecc71", "Yes" = "#e74c3c")) +
  labs(title = "Distribucion de Cardiopatias",
       x = "Enfermedad Cardiaca",
       y = "Frecuencia") +
  theme_minimal()

ggsave("graficos/01_distribucion_objetivo.png", p1, width = 8, height = 6)

# ========================================================================
# 6. DIVISION DE DATOS
# ========================================================================
indices_entrenamiento <- createDataPartition(datos_prep$HeartDisease,
                                             p = 0.80, list = FALSE)
datos_train <- datos_prep[indices_entrenamiento, ]
datos_test <- datos_prep[-indices_entrenamiento, ]

# ========================================================================
# 6.5 VALIDACION CRUZADA CON RANDOM FOREST (VERSION SIMPLIFICADA)
# ========================================================================
cat("=== VALIDACION CRUZADA DE 5 FOLDS CON RANDOM FOREST ===\n")

# Crear los folds una sola vez
set.seed(123)
folds <- createFolds(datos_train$HeartDisease, k = 5)

# Dataframe para resultados
resultados_cv <- data.frame()

# Procesar ambos metodos
for(metodo in c("rose", "under")) {
  cat("\n--- Procesando metodo:", toupper(metodo), "---\n")
  
  for(i in 1:5) {
    cat(" Fold", i, "de 5...\n")
    
    # Dividir datos para este fold
    train_indices <- unlist(folds[-i])
    test_indices <- folds[[i]]
    train_data <- datos_train[train_indices, ]
    test_data <- datos_train[test_indices, ]
    
    # Balancear los datos de entrenamiento
    if(metodo == "rose") {
      datos_balanceados <- ROSE(HeartDisease ~ ., data = train_data,
                                seed = 123)$data
    } else {
      n_pos <- sum(train_data$HeartDisease == "Yes")
      n_neg_deseado <- round(n_pos * 4)
      total_deseado <- n_neg_deseado + n_pos
      
      datos_balanceados <- ovun.sample(
        HeartDisease ~ .,
        data = train_data,
        method = "under",
        N = total_deseado,
        seed = 123
      )$data
    }
    
    # Entrenar modelo Random Forest
    modelo_rf <- randomForest(
      HeartDisease ~ .,
      data = datos_balanceados,
      ntree = 50,
      mtry = 5,
      importance = FALSE,
      na.action = na.omit
    )
    
    # Predecir en datos de test originales
    predicciones <- predict(modelo_rf, test_data)
    predicciones_prob <- predict(modelo_rf, test_data, type = "prob")[, "Yes"]
    
    # Calcular metricas
    cm <- confusionMatrix(predicciones, test_data$HeartDisease, positive = "Yes")
    roc_auc <- auc(roc(test_data$HeartDisease, predicciones_prob))
    
    # Guardar resultados
    resultados_cv <- rbind(resultados_cv, data.frame(
      Fold = i,
      Metodo = metodo,
      Accuracy = round(cm$overall["Accuracy"], 4),
      Sensitivity = round(cm$byClass["Sensitivity"], 4),
      Specificity = round(cm$byClass["Specificity"], 4),
      Precision = round(cm$byClass["Precision"], 4),
      F1 = round(cm$byClass["F1"], 4),
      AUC = round(roc_auc, 4)
    ))
    
    # Limpiar memoria
    rm(modelo_rf, datos_balanceados, train_data, test_data)
    gc()
  }
}

# Calcular resumen estadistico
resumen_cv <- resultados_cv %>%
  group_by(Metodo) %>%
  summarise(
    Accuracy_mean = round(mean(Accuracy), 4),
    Accuracy_sd = round(sd(Accuracy), 4),
    Sensitivity_mean = round(mean(Sensitivity), 4),
    Sensitivity_sd = round(sd(Sensitivity), 4),
    Specificity_mean = round(mean(Specificity), 4),
    Specificity_sd = round(sd(Specificity), 4),
    AUC_mean = round(mean(AUC), 4),
    AUC_sd = round(sd(AUC), 4)
  )

cat("\n=== RESULTADOS VALIDACION CRUZADA ===\n")
print(resumen_cv)

# Determinar mejor metodo
mejor_metodo_cv <- resumen_cv %>%
  arrange(desc(AUC_mean)) %>%
  slice(1) %>%
  pull(Metodo)

cat("\n*** MEJOR METODO SEGUN VALIDACION CRUZADA:", toupper(mejor_metodo_cv), "***\n")

# Guardar resultados
write.csv(resultados_cv, "resultados/validacion_cruzada_rf.csv", row.names = FALSE)
write.csv(resumen_cv, "resultados/resumen_validacion_cruzada_rf.csv", row.names = FALSE)

# Grafico de comparacion
p_cv <- ggplot(resultados_cv, aes(x = Metodo, y = AUC, fill = Metodo)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.2, size = 2, alpha = 0.6) +
  scale_fill_manual(values = c("rose" = "#3498db", "under" = "#e74c3c")) +
  labs(title = "Comparacion de Metodos - Validacion Cruzada (5 Folds)",
       subtitle = "Distribucion del AUC por metodo",
       x = "Metodo de Balanceo",
       y = "AUC") +
  theme_minimal()

ggsave("graficos/05_validacion_cruzada_comparacion.png", p_cv, width = 8, height = 6)

cat("Validacion cruzada completada exitosamente\n")

# ========================================================================
# 7. BALANCEO CON ROSE
# ========================================================================
cat("=== BALANCEANDO DATOS CON ROSE ===\n")

n_neg <- sum(datos_train$HeartDisease == "No")
n_pos <- sum(datos_train$HeartDisease == "Yes")

cat("Distribucion original:\n")
cat(" No:", n_neg, "\n")
cat(" Yes:", n_pos, "\n")
cat(" Ratio:", round(n_neg/n_pos, 2), ":1\n\n")

datos_train_rose <- ROSE(HeartDisease ~ ., data = datos_train, seed = 123)$data

cat("Resultado con ROSE:\n")
tabla_final_rose <- table(datos_train_rose$HeartDisease)
print(tabla_final_rose)

porcentaje_yes_rose <- round(tabla_final_rose[2] / sum(tabla_final_rose) * 100, 1)
cat("Porcentaje Yes:", porcentaje_yes_rose, "%\n")
cat("Ratio:", round(tabla_final_rose[1] / tabla_final_rose[2], 2), ":1\n")

# =======================================================================
# 8. MODELO RANDOM FOREST CON DATOS BALANCEADOS POR ROSE
# =======================================================================
cat("=== ENTRENANDO MODELO RANDOM FOREST CON ROSE ===\n")

modelo_rf_rose <- randomForest(
  HeartDisease ~ .,
  data = datos_train_rose,
  ntree = 100,
  mtry = 5,
  importance = TRUE,
  na.action = na.omit
)

cat("Modelo con ROSE entrenado exitosamente\n")

# ========================================================================
# 9. EVALUACION DEL MODELO ROSE
# ========================================================================
cat("=== EVALUANDO MODELO ROSE ===\n")

predicciones_test_rose <- predict(modelo_rf_rose, datos_test)
predicciones_prob_rose <- predict(modelo_rf_rose, datos_test, type = "prob")

matriz_confusion_rose <- confusionMatrix(predicciones_test_rose,
                                         datos_test$HeartDisease,
                                         positive = "Yes")
print(matriz_confusion_rose)

# ========================================================================
# 10. CURVA ROC ROSE
# ========================================================================
roc_obj_rose <- roc(datos_test$HeartDisease, predicciones_prob_rose[,2])
auc_value_rose <- auc(roc_obj_rose)

png("graficos/02_curva_roc_rose.png", width = 800, height = 600)
plot(roc_obj_rose,
     main = paste("Curva ROC con ROSE - AUC =", round(auc_value_rose, 4)),
     col = "#3498db", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
dev.off()

# ========================================================================
# 11. MATRIZ DE CONFUSION ROSE
# ========================================================================
matriz_plot_rose <- as.data.frame(matriz_confusion_rose$table)
matriz_plot_rose$Prediction <- factor(matriz_plot_rose$Prediction,
                                      levels = c("Yes", "No"))
matriz_plot_rose$Reference <- factor(matriz_plot_rose$Reference,
                                     levels = c("No", "Yes"))

p_matriz_rose <- ggplot(matriz_plot_rose,
                        aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "#3498db", high = "#e74c3c") +
  labs(title = "Matriz de Confusion con ROSE",
       x = "Valor Real",
       y = "Prediccion") +
  theme_minimal() +
  scale_y_discrete(limits = c("No", "Yes"))

ggsave("graficos/03_matriz_confusion_rose.png", p_matriz_rose, width = 8, height = 6)

# ========================================================================
# 12. IMPORTANCIA DE VARIABLES ROSE
# ========================================================================
importancia_rose <- importance(modelo_rf_rose)
importancia_df_rose <- data.frame(
  Variable = rownames(importancia_rose),
  Importance = importancia_rose[, "MeanDecreaseGini"]
) %>%
  arrange(desc(Importance))

p_importancia_rose <- importancia_df_rose %>%
  head(10) %>%
  ggplot(aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col(fill = "#3498db") +
  coord_flip() +
  labs(title = "Top 10 Variables Mas Importantes (ROSE)",
       x = "Variables",
       y = "Importancia") +
  theme_minimal()

ggsave("graficos/04_importancia_variables_rose.png", p_importancia_rose,
       width = 10, height = 6)

# ========================================================================
# 13. OPTIMIZACION CON UMBRAL ROSE
# ========================================================================
cat("=== OPTIMIZANDO UMBRAL DE CLASIFICACION ROSE ===\n")

mejor_umbral_rose <- coords(roc_obj_rose, "best", ret = "threshold",
                            best.method = "youden")$threshold
cat("Mejor umbral encontrado:", round(mejor_umbral_rose, 3), "\n")

predicciones_optimizadas_rose <- ifelse(predicciones_prob_rose[,2] > mejor_umbral_rose,
                                        "Yes", "No")
predicciones_optimizadas_rose <- factor(predicciones_optimizadas_rose,
                                        levels = c("No", "Yes"))

matriz_optimizada_rose <- confusionMatrix(predicciones_optimizadas_rose,
                                          datos_test$HeartDisease,
                                          positive = "Yes")

cat("Metricas con umbral optimizado ROSE:\n")
print(matriz_optimizada_rose)

# ========================================================================
# 14. METRICAS COMPLETAS ROSE
# ========================================================================
f1_original_rose <- matriz_confusion_rose$byClass['F1']
f1_optimizado_rose <- matriz_optimizada_rose$byClass['F1']

metricas_completas_rose <- data.frame(
  Modelo = c("ROSE Umbral 0.5", "ROSE Umbral Optimizado"),
  Exactitud = c(round(matriz_confusion_rose$overall['Accuracy'], 4),
                round(matriz_optimizada_rose$overall['Accuracy'], 4)),
  Sensibilidad = c(round(matriz_confusion_rose$byClass['Sensitivity'], 4),
                   round(matriz_optimizada_rose$byClass['Sensitivity'], 4)),
  Especificidad = c(round(matriz_confusion_rose$byClass['Specificity'], 4),
                    round(matriz_optimizada_rose$byClass['Specificity'], 4)),
  Precision = c(round(matriz_confusion_rose$byClass['Pos Pred Value'], 4),
                round(matriz_optimizada_rose$byClass['Pos Pred Value'], 4)),
  F1_Score = c(round(as.numeric(f1_original_rose), 4),
               round(as.numeric(f1_optimizado_rose), 4)),
  AUC = c(round(auc_value_rose, 4),
          round(auc_value_rose, 4))
)

cat("Comparacion de metricas con ROSE:\n")
print(metricas_completas_rose)
write.csv(metricas_completas_rose, "resultados/metricas_completas_rose.csv",
          row.names = FALSE)

# ========================================================================
# 15. BALANCEO 80-20 CON UNDERSAMPLING
# ========================================================================
cat("=== BALANCEANDO DATOS 80-20 CON UNDERSAMPLING ===\n")

n_neg <- sum(datos_train$HeartDisease == "No")
n_pos <- sum(datos_train$HeartDisease == "Yes")

cat("Distribucion original:\n")
cat(" No:", n_neg, "\n")
cat(" Yes:", n_pos, "\n")
cat(" Ratio:", round(n_neg/n_pos, 2), ":1\n\n")

n_neg_deseado <- round(n_pos * 4)
total_deseado <- n_neg_deseado + n_pos

cat("Objetivo 80-20 con undersampling:\n")
cat(" No (80%):", n_neg_deseado, "\n")
cat(" Yes (20%):", n_pos, "\n")
cat(" Total:", total_deseado, "\n")
cat(" Ratio: 4:1\n\n")

datos_train_under <- ovun.sample(
  HeartDisease ~ .,
  data = datos_train,
  method = "under",
  N = total_deseado,
  seed = 123
)$data

cat("Resultado con undersampling 80-20:\n")
tabla_final_under <- table(datos_train_under$HeartDisease)
print(tabla_final_under)

porcentaje_yes_under <- round(tabla_final_under[2] / sum(tabla_final_under) * 100, 1)
cat("Porcentaje Yes:", porcentaje_yes_under, "%\n")
cat("Ratio:", round(tabla_final_under[1] / tabla_final_under[2], 2), ":1\n")

# ========================================================================
# 16. MODELO RANDOM FOREST CON UNDERSAMPLING
# ========================================================================
cat("=== ENTRENANDO MODELO RANDOM FOREST CON UNDERSAMPLING ===\n")

modelo_rf_under <- randomForest(
  HeartDisease ~ .,
  data = datos_train_under,
  ntree = 100,
  mtry = 5,
  importance = TRUE,
  na.action = na.omit
)

cat("Modelo con undersampling entrenado exitosamente\n")

# ========================================================================
# 17. EVALUACION DEL MODELO UNDERSAMPLING
# ========================================================================
cat("=== EVALUANDO MODELO UNDERSAMPLING ===\n")

predicciones_test_under <- predict(modelo_rf_under, datos_test)
predicciones_prob_under <- predict(modelo_rf_under, datos_test, type = "prob")

matriz_confusion_under <- confusionMatrix(predicciones_test_under,
                                          datos_test$HeartDisease,
                                          positive = "Yes")
print(matriz_confusion_under)

# ========================================================================
# 18. CURVA ROC UNDERSAMPLING
# ========================================================================
roc_obj_under <- roc(datos_test$HeartDisease, predicciones_prob_under[,2])
auc_value_under <- auc(roc_obj_under)

png("graficos/02_curva_roc_under.png", width = 800, height = 600)
plot(roc_obj_under,
     main = paste("Curva ROC con Undersampling - AUC =", round(auc_value_under, 4)),
     col = "#3498db", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
dev.off()

# ========================================================================
# 19. MATRIZ DE CONFUSION UNDERSAMPLING
# ========================================================================
matriz_plot_under <- as.data.frame(matriz_confusion_under$table)

p_matriz_under <- ggplot(matriz_plot_under,
                         aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "#3498db", high = "#e74c3c") +
  labs(title = "Matriz de Confusion con Undersampling",
       x = "Valor Real",
       y = "Prediccion") +
  theme_minimal()

ggsave("graficos/03_matriz_confusion_under.png", p_matriz_under,
       width = 8, height = 6)

# ========================================================================
# 20. IMPORTANCIA DE VARIABLES UNDERSAMPLING
# ========================================================================
importancia_under <- importance(modelo_rf_under)
importancia_df_under <- data.frame(
  Variable = rownames(importancia_under),
  Importance = importancia_under[, "MeanDecreaseGini"]
) %>%
  arrange(desc(Importance))

p_importancia_under <- importancia_df_under %>%
  head(10) %>%
  ggplot(aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col(fill = "#3498db") +
  coord_flip() +
  labs(title = "Top 10 Variables Mas Importantes (Undersampling)",
       x = "Variables",
       y = "Importancia") +
  theme_minimal()

ggsave("graficos/04_importancia_variables_under.png", p_importancia_under,
       width = 10, height = 6)

# ========================================================================
# 21. OPTIMIZACION CON UMBRAL UNDERSAMPLING
# ========================================================================
cat("=== OPTIMIZANDO UMBRAL DE CLASIFICACION UNDERSAMPLING ===\n")

mejor_umbral_under <- coords(roc_obj_under, "best", ret = "threshold",
                             best.method = "youden")$threshold
cat("Mejor umbral encontrado:", round(mejor_umbral_under, 3), "\n")

predicciones_optimizadas_under <- ifelse(predicciones_prob_under[,2] > mejor_umbral_under,
                                         "Yes", "No")
predicciones_optimizadas_under <- factor(predicciones_optimizadas_under,
                                         levels = c("No", "Yes"))

matriz_optimizada_under <- confusionMatrix(predicciones_optimizadas_under,
                                           datos_test$HeartDisease,
                                           positive = "Yes")

cat("Metricas con umbral optimizado UNDERSAMPLING:\n")
print(matriz_optimizada_under)

# ========================================================================
# 22. METRICAS COMPLETAS UNDERSAMPLING
# ========================================================================
f1_original_under <- matriz_confusion_under$byClass['F1']
f1_optimizado_under <- matriz_optimizada_under$byClass['F1']

metricas_completas_under <- data.frame(
  Modelo = c("Under Umbral 0.5", "Under Umbral Optimizado"),
  Exactitud = c(round(matriz_confusion_under$overall['Accuracy'], 4),
                round(matriz_optimizada_under$overall['Accuracy'], 4)),
  Sensibilidad = c(round(matriz_confusion_under$byClass['Sensitivity'], 4),
                   round(matriz_optimizada_under$byClass['Sensitivity'], 4)),
  Especificidad = c(round(matriz_confusion_under$byClass['Specificity'], 4),
                    round(matriz_optimizada_under$byClass['Specificity'], 4)),
  Precision = c(round(matriz_confusion_under$byClass['Pos Pred Value'], 4),
                round(matriz_optimizada_under$byClass['Pos Pred Value'], 4)),
  F1_Score = c(round(as.numeric(f1_original_under), 4),
               round(as.numeric(f1_optimizado_under), 4)),
  AUC = c(round(auc_value_under, 4),
          round(auc_value_under, 4))
)

cat("Comparacion de metricas con Undersampling:\n")
print(metricas_completas_under)
write.csv(metricas_completas_under, "resultados/metricas_completas_under.csv",
          row.names = FALSE)

# =======================================================================
# 23. COMPARACION FINAL Y GUARDADO
# =======================================================================
cat("\n")
cat(strrep("=", 60), "\n")
cat("COMPARACION FINAL ENTRE METODOS\n")
cat(strrep("=", 60), "\n")

cat("\nRESULTADO VALIDACION CRUZADA (5 FOLDS):\n")
print(resumen_cv)

cat("\nMETRICAS FINALES EN TEST:\n")

# Comparativa final
comparativa_final <- data.frame(
  Metodo = c("ROSE", "Undersampling"),
  AUC_Test = c(round(auc_value_rose, 4), round(auc_value_under, 4)),
  Sensibilidad_Test = c(
    round(matriz_confusion_rose$byClass['Sensitivity'], 4),
    round(matriz_confusion_under$byClass['Sensitivity'], 4)
  ),
  Especificidad_Test = c(
    round(matriz_confusion_rose$byClass['Specificity'], 4),
    round(matriz_confusion_under$byClass['Specificity'], 4)
  ),
  AUC_CV = c(
    resumen_cv$AUC_mean[resumen_cv$Metodo == "rose"],
    resumen_cv$AUC_mean[resumen_cv$Metodo == "under"]
  )
)

print(comparativa_final)

cat("\nRECOMENDACION BASADA EN VALIDACION CRUZADA:\n")
if(mejor_metodo_cv == "rose") {
  cat("El metodo ROSE mostro mejor performance en validacion cruzada\n")
  cat("Considera usar modelo_rf_rose para predicciones futuras\n")
} else {
  cat("El metodo UNDERSAMPLING mostro mejor performance en validacion cruzada\n")
  cat("Considera usar modelo_rf_under para predicciones futuras\n")
}

# ======================================================================
# 24. GUARDADO DE RESULTADOS DE AMBOS MODELOS
# ========================================================================
saveRDS(modelo_rf_rose, "resultados/modelo_rf_rose.rds")
saveRDS(modelo_rf_under, "resultados/modelo_rf_under.rds")

cat("\nMODELOS GUARDADOS:\n")
cat(" - modelo_rf_rose.rds\n")
cat(" - modelo_rf_under.rds\n")
cat(" - Resultados de validacion cruzada\n")

cat("\n")
cat(strrep("=", 60), "\n")
cat("ANALISIS COMPLETADO EXITOSAMENTE CON RANDOM FOREST\n")
cat(strrep("=", 60), "\n")

as.data.frame(resumen_cv)

# ========================================================================
# 25. INTERVALOS DE CONFIANZA Y CURVAS PRECISION-RECALL
# ========================================================================


# ========================================================================
# PARTE 1: INTERVALOS DE CONFIANZA (95%)
# ========================================================================

cat("\n")
cat(strrep("=", 70), "\n")
cat("CÁLCULO DE INTERVALOS DE CONFIANZA (95%)\n")
cat(strrep("=", 70), "\n\n")

# Función para calcular IC usando método de Wilson
calcular_ic_wilson <- function(x, n, nivel = 0.95) {
  # x = número de éxitos
  # n = número total
  # nivel = nivel de confianza (0.95 para 95%)
  
  if(n == 0) return(c(0, 0))
  
  z <- qnorm((1 + nivel) / 2)  # Valor z para el nivel de confianza
  p <- x / n
  
  denominador <- 1 + z^2 / n
  centro <- (p + z^2 / (2 * n)) / denominador
  margen <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / denominador
  
  ic_inf <- max(0, centro - margen)
  ic_sup <- min(1, centro + margen)
  
  return(c(ic_inf, ic_sup))
}

# Función para calcular IC de todas las métricas
calcular_ic_metricas <- function(matriz_confusion, nombre_metodo, verbose = FALSE) {
  
  # Extraer valores de la matriz
  tn <- matriz_confusion$table[1,1]  # Verdaderos Negativos
  fp <- matriz_confusion$table[1,2]  # Falsos Positivos
  fn <- matriz_confusion$table[2,1]  # Falsos Negativos
  tp <- matriz_confusion$table[2,2]  # Verdaderos Positivos
  
  total <- tn + fp + fn + tp
  total_positivos <- tp + fn
  total_negativos <- tn + fp
  total_pred_positivos <- tp + fp
  
  if(verbose) {
    cat("\n", nombre_metodo, ":\n")
    cat("  TP:", tp, "| FP:", fp, "| TN:", tn, "| FN:", fn, "\n")
  }
  
  # Calcular valores y IC para cada métrica
  
  # 1. Accuracy = (TP + TN) / Total
  accuracy_val <- (tn + tp) / total
  ic_accuracy <- calcular_ic_wilson(tn + tp, total)
  
  # 2. Sensibilidad (Recall/TPR) = TP / (TP + FN)
  sensitivity_val <- tp / total_positivos
  ic_sensitivity <- calcular_ic_wilson(tp, total_positivos)
  
  # 3. Especificidad (TNR) = TN / (TN + FP)
  specificity_val <- tn / total_negativos
  ic_specificity <- calcular_ic_wilson(tn, total_negativos)
  
  # 4. Precisión (PPV) = TP / (TP + FP)
  precision_val <- tp / total_pred_positivos
  ic_precision <- calcular_ic_wilson(tp, total_pred_positivos)
  
  if(verbose) {
    cat("  Accuracy:", round(accuracy_val * 100, 2), "%\n")
    cat("  Sensibilidad:", round(sensitivity_val * 100, 2), "%\n")
    cat("  Especificidad:", round(specificity_val * 100, 2), "%\n")
    cat("  Precisión:", round(precision_val * 100, 2), "%\n")
  }
  
  # Crear dataframe con resultados (ORDEN CORREGIDO)
  resultados_ic <- data.frame(
    Metodo = rep(nombre_metodo, 4),
    Metrica = c("Accuracy", "Precisión", "Especificidad", "Sensibilidad"),
    Valor = round(c(accuracy_val, precision_val, specificity_val, sensitivity_val) * 100, 2),
    IC_Inferior = round(c(ic_accuracy[1], ic_precision[1], ic_specificity[1], ic_sensitivity[1]) * 100, 2),
    IC_Superior = round(c(ic_accuracy[2], ic_precision[2], ic_specificity[2], ic_sensitivity[2]) * 100, 2),
    Amplitud = round(c(
      ic_accuracy[2] - ic_accuracy[1],
      ic_precision[2] - ic_precision[1],
      ic_specificity[2] - ic_specificity[1],
      ic_sensitivity[2] - ic_sensitivity[1]
    ) * 100, 2)
  )
  
  return(resultados_ic)
}

# ========================================================================
# 25.1 CALCULAR IC PARA TODAS LAS CONFIGURACIONES
# ========================================================================

# IC para ROSE (umbral 0.5)
cat("Calculando IC para ROSE (umbral 0.5)...\n")
ic_rose_05 <- calcular_ic_metricas(matriz_confusion_rose, "ROSE (0.5)", verbose = TRUE)

# IC para ROSE (umbral optimizado)
cat("\nCalculando IC para ROSE (umbral optimizado 0.275)...\n")
ic_rose_opt <- calcular_ic_metricas(matriz_optimizada_rose, "ROSE (0.275)", verbose = TRUE)

# IC para Under-Sampling (umbral 0.5)
cat("\nCalculando IC para Under-Sampling (umbral 0.5)...\n")
ic_under_05 <- calcular_ic_metricas(matriz_confusion_under, "Under (0.5)", verbose = TRUE)

# IC para Under-Sampling (umbral optimizado)
cat("\nCalculando IC para Under-Sampling (umbral optimizado 0.185)...\n")
ic_under_opt <- calcular_ic_metricas(matriz_optimizada_under, "Under (0.185)", verbose = TRUE)

# Combinar todos los resultados
ic_completos <- rbind(ic_rose_05, ic_rose_opt, ic_under_05, ic_under_opt)

# Mostrar resultados
cat("\n")
cat(strrep("=", 70), "\n")
cat("RESULTADOS: INTERVALOS DE CONFIANZA (95%)\n")
cat(strrep("=", 70), "\n")
print(ic_completos, row.names = FALSE)

# Guardar
write.csv(ic_completos, "resultados/intervalos_confianza.csv", row.names = FALSE)
cat("\nResultados guardados en: resultados/intervalos_confianza.csv\n")

# ========================================================================
# 25.2 VISUALIZACIÓN DE INTERVALOS DE CONFIANZA
# ========================================================================

cat("\nGenerando gráfico de intervalos de confianza...\n")

# Preparar datos para gráfico
ic_completos$Metodo <- factor(ic_completos$Metodo, 
                              levels = c("ROSE (0.5)", "ROSE (0.275)", 
                                         "Under (0.5)", "Under (0.185)"))

# Crear gráfico
p_ic <- ggplot(ic_completos, aes(x = Metrica, y = Valor, color = Metodo, group = Metodo)) +
  geom_point(size = 3, position = position_dodge(width = 0.6)) +
  geom_errorbar(
    aes(ymin = IC_Inferior, ymax = IC_Superior),
    width = 0.3,
    size = 0.8,
    position = position_dodge(width = 0.6)
  ) +
  coord_flip() +
  scale_color_manual(
    values = c(
      "ROSE (0.5)" = "#3498db",
      "ROSE (0.275)" = "#5dade2",
      "Under (0.5)" = "#e74c3c",
      "Under (0.185)" = "#ec7063"
    ),
    name = "Configuración"
  ) +
  labs(
    title = "Intervalos de Confianza (95%) por Métrica y Configuración",
    subtitle = "Barras de error representan límites inferior y superior del IC al 95%",
    x = "",
    y = "Valor (%)",
    caption = "Método: Wilson Score Interval"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 11),
    axis.text = element_text(size = 10),
    legend.text = element_text(size = 10)
  ) +
  guides(color = guide_legend(nrow = 2))

# Guardar gráfico
ggsave("graficos/06_intervalos_confianza.png", p_ic, width = 12, height = 7, dpi = 300)
cat("Gráfico guardado en: graficos/06_intervalos_confianza.png\n")

# ========================================================================
# 25.3 ANÁLISIS DE SOLAPAMIENTO DE IC
# ========================================================================

cat("\n")
cat(strrep("=", 70), "\n")
cat("ANÁLISIS DE SOLAPAMIENTO DE INTERVALOS\n")
cat(strrep("=", 70), "\n\n")

# Función para verificar solapamiento
verificar_solapamiento <- function(ic1_inf, ic1_sup, ic2_inf, ic2_sup) {
  solapa <- (ic1_inf <= ic2_sup) & (ic2_inf <= ic1_sup)
  return(!solapa)  # TRUE si NO solapan (diferencia significativa)
}

# Comparar ROSE vs Under (umbral 0.5)
cat("Comparación: ROSE (0.5) vs Under (0.5)\n")
cat(strrep("-", 70), "\n")

for(metrica in unique(ic_completos$Metrica)) {
  rose_data <- ic_completos[ic_completos$Metodo == "ROSE (0.5)" & ic_completos$Metrica == metrica, ]
  under_data <- ic_completos[ic_completos$Metodo == "Under (0.5)" & ic_completos$Metrica == metrica, ]
  
  significativo <- verificar_solapamiento(
    rose_data$IC_Inferior, rose_data$IC_Superior,
    under_data$IC_Inferior, under_data$IC_Superior
  )
  
  cat(sprintf("%-15s: %s\n", metrica, 
              ifelse(significativo, "Diferencia SIGNIFICATIVA ✓", "No significativa")))
}

cat("\n✓ = Los IC no se solapan, la diferencia es estadísticamente significativa\n")

# ========================================================================
# 26. CURVAS PRECISION-RECALL
# ========================================================================

cat("\n")
cat(strrep("=", 70), "\n")
cat("CÁLCULO DE CURVAS PRECISION-RECALL\n")
cat(strrep("=", 70), "\n\n")

# Función para calcular curva PR manualmente
calcular_pr_curve <- function(probabilidades, clases_reales, nombre_metodo) {
  
  cat("Procesando", nombre_metodo, "...\n")
  
  # Convertir clases a 0/1
  y_true <- as.numeric(clases_reales == "Yes")
  y_score <- probabilidades[, "Yes"]
  
  # Ordenar por probabilidad descendente
  order_idx <- order(y_score, decreasing = TRUE)
  y_true_sorted <- y_true[order_idx]
  y_score_sorted <- y_score[order_idx]
  
  # Calcular precision y recall en cada umbral
  n_positivos <- sum(y_true)
  
  precision_vals <- numeric()
  recall_vals <- numeric()
  thresholds <- numeric()
  
  tp <- 0
  fp <- 0
  
  for(i in 1:length(y_true_sorted)) {
    if(y_true_sorted[i] == 1) {
      tp <- tp + 1
    } else {
      fp <- fp + 1
    }
    
    precision <- tp / (tp + fp)
    recall <- tp / n_positivos
    
    precision_vals <- c(precision_vals, precision)
    recall_vals <- c(recall_vals, recall)
    thresholds <- c(thresholds, y_score_sorted[i])
  }
  
  # Agregar punto (0,1) al inicio
  precision_vals <- c(1, precision_vals)
  recall_vals <- c(0, recall_vals)
  
  # Calcular AUC-PR usando regla del trapecio
  auc_pr <- 0
  for(i in 2:length(recall_vals)) {
    auc_pr <- auc_pr + (recall_vals[i] - recall_vals[i-1]) * 
      (precision_vals[i] + precision_vals[i-1]) / 2
  }
  
  cat("  AUC-PR:", round(auc_pr, 4), "\n")
  
  return(list(
    precision = precision_vals,
    recall = recall_vals,
    auc_pr = auc_pr
  ))
}

# Calcular curvas PR para ambos métodos
cat("\nCalculando curvas Precision-Recall...\n")
pr_rose <- calcular_pr_curve(predicciones_prob_rose, datos_test$HeartDisease, "ROSE")
pr_under <- calcular_pr_curve(predicciones_prob_under, datos_test$HeartDisease, "Under-Sampling")

# Mostrar resultados
cat("\n")
cat(strrep("=", 70), "\n")
cat("RESULTADOS: AUC-PR (Area Under Precision-Recall Curve)\n")
cat(strrep("=", 70), "\n")
cat("ROSE           AUC-PR:", round(pr_rose$auc_pr, 4), "\n")
cat("Under-Sampling AUC-PR:", round(pr_under$auc_pr, 4), "\n")
cat("Diferencia           :", round(pr_rose$auc_pr - pr_under$auc_pr, 4), "\n")

# Crear dataframe para gráfico
pr_data_rose <- data.frame(
  Recall = pr_rose$recall,
  Precision = pr_rose$precision,
  Metodo = "ROSE"
)

pr_data_under <- data.frame(
  Recall = pr_under$recall,
  Precision = pr_under$precision,
  Metodo = "Under-Sampling"
)

pr_combined <- rbind(pr_data_rose, pr_data_under)

# Calcular prevalencia (línea base)
prevalencia <- sum(datos_test$HeartDisease == "Yes") / nrow(datos_test)

# Crear gráfico
cat("\nGenerando gráfico de curvas Precision-Recall...\n")

p_pr <- ggplot(pr_combined, aes(x = Recall, y = Precision, color = Metodo)) +
  geom_line(size = 1.5) +
  geom_hline(yintercept = prevalencia, linetype = "dashed", color = "gray50", size = 1) +
  annotate("text", x = 0.5, y = prevalencia + 0.03, 
           label = sprintf("Línea base (prevalencia = %.2f%%)", prevalencia * 100),
           color = "gray30", size = 4) +
  scale_color_manual(
    values = c("ROSE" = "#3498db", "Under-Sampling" = "#e74c3c"),
    name = "Técnica"
  ) +
  labs(
    title = "Curvas Precision-Recall: Comparación entre Técnicas de Balanceo",
    subtitle = sprintf("AUC-PR: ROSE = %.4f | Under-Sampling = %.4f", 
                       pr_rose$auc_pr, pr_under$auc_pr),
    x = "Recall (Sensibilidad)",
    y = "Precisión (Positive Predictive Value)",
    caption = "Línea punteada: clasificador aleatorio (prevalencia de la clase positiva)"
  ) +
  xlim(0, 1) +
  ylim(0, 1) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(size = 11),
    axis.text = element_text(size = 10),
    legend.text = element_text(size = 11)
  )

# Guardar gráfico
ggsave("graficos/07_curvas_precision_recall.png", p_pr, width = 10, height = 7, dpi = 300)
cat("Gráfico guardado en: graficos/07_curvas_precision_recall.png\n")

# ========================================================================
# COMPARACIÓN AUC-ROC vs AUC-PR
# ========================================================================

cat("\n")
cat(strrep("=", 70), "\n")
cat("COMPARACIÓN: AUC-ROC vs AUC-PR\n")
cat(strrep("=", 70), "\n")

# Crear tabla comparativa
metricas_auc <- data.frame(
  Metodo = c("ROSE", "Under-Sampling"),
  AUC_ROC = c(round(auc_value_rose, 4), round(auc_value_under, 4)),
  AUC_PR = c(round(pr_rose$auc_pr, 4), round(pr_under$auc_pr, 4)),
  Diferencia = c(
    round(auc_value_rose - pr_rose$auc_pr, 4),
    round(auc_value_under - pr_under$auc_pr, 4)
  ),
  Ratio_PR_ROC = c(
    round(pr_rose$auc_pr / auc_value_rose, 3),
    round(pr_under$auc_pr / auc_value_under, 3)
  )
)

print(metricas_auc, row.names = FALSE)

# Interpretación
cat("\nInterpretación:\n")
cat("- Diferencia = AUC-ROC - AUC-PR (cuánto más optimista es ROC)\n")
cat("- Ratio = AUC-PR / AUC-ROC (proporción del rendimiento ROC que se mantiene en PR)\n")
cat("- Un ratio cercano a 1 indica que ROC no es excesivamente optimista\n")
cat("- En datasets desbalanceados, AUC-PR es más informativo que AUC-ROC\n")

# Guardar resultados
write.csv(metricas_auc, "resultados/comparacion_auc_roc_pr.csv", row.names = FALSE)
cat("\nResultados guardados en: resultados/comparacion_auc_roc_pr.csv\n")

# ========================================================================
# 27. RESUMEN 
# ========================================================================

cat("\n")
cat(strrep("=", 70), "\n")
cat("ANÁLISIS DE INTERVALOS DE CONFIANZA Y CURVAS PR COMPLETADO\n")
cat(strrep("=", 70), "\n")
cat("Archivos generados:\n")
cat("  - resultados/intervalos_confianza.csv\n")
cat("  - resultados/comparacion_auc_roc_pr.csv\n")
cat("  - graficos/06_intervalos_confianza.png\n")
cat("  - graficos/07_curvas_precision_recall.png\n")
cat(strrep("=", 70), "\n\n")

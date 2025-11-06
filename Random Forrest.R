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
# 24. GUARDADO FINAL
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

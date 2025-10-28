# =============================================================================
# 1. INSTALACION Y CARGA DE PAQUETES
# =============================================================================

paquetes <- c(
  "randomForest", "caret", "ggplot2", "dplyr", "tidyr", 
  "pROC", "corrplot", "gridExtra", "RColorBrewer", "ROSE"
)

paquetes_faltantes <- paquetes[!(paquetes %in% installed.packages()[,"Package"])]
if(length(paquetes_faltantes) > 0) {
  install.packages(paquetes_faltantes, dependencies = TRUE)
}

invisible(lapply(paquetes, library, character.only = TRUE))

# =============================================================================
# 2. CONFIGURACION INICIAL
# =============================================================================

set.seed(123)
if(!dir.exists("resultados")) dir.create("resultados")
if(!dir.exists("graficos")) dir.create("graficos")

# =============================================================================
# 3. CARGA DE DATOS
# =============================================================================

datos <- read.csv("heart_2020_cleaned.csv", stringsAsFactors = FALSE, na.strings = c("", "NA", "N/A"))

# =============================================================================
# 4. PREPROCESAMIENTO
# =============================================================================

datos_prep <- datos
datos_prep$HeartDisease <- factor(datos_prep$HeartDisease, levels = c("No", "Yes"), labels = c("No", "Yes"))

vars_categoricas <- c("Sex", "AgeCategory", "Race", "Diabetic", "GenHealth", "Smoking", "AlcoholDrinking",
                      "Stroke", "DiffWalking", "PhysicalActivity", "Asthma", "KidneyDisease", "SkinCancer")

for(var in vars_categoricas) {
  if(var %in% colnames(datos_prep)) {
    datos_prep[[var]] <- as.factor(datos_prep[[var]])
  }
}

datos_prep <- na.omit(datos_prep)

# =============================================================================
# 5. ANALISIS EXPLORATORIO
# =============================================================================

p1 <- ggplot(datos_prep, aes(x = HeartDisease, fill = HeartDisease)) +
  geom_bar() +
  geom_text(stat = 'count', aes(label = after_stat(count)), vjust = -0.5, size = 5) +
  scale_fill_manual(values = c("No" = "#2ecc71", "Yes" = "#e74c3c")) +
  labs(title = "Distribucion de Cardiopatias", x = "Enfermedad Cardiaca", y = "Frecuencia") +
  theme_minimal()

ggsave("graficos/01_distribucion_objetivo.png", p1, width = 8, height = 6)

# =============================================================================
# 6. DIVISION DE DATOS
# =============================================================================

indices_entrenamiento <- createDataPartition(datos_prep$HeartDisease, p = 0.80, list = FALSE)
datos_train <- datos_prep[indices_entrenamiento, ]
datos_test <- datos_prep[-indices_entrenamiento, ]


# =============================================================================
# 7. BALANCEO CON ROSE
# =============================================================================

cat("=== BALANCEANDO DATOS CON ROSE ===\n")

n_neg <- sum(datos_train$HeartDisease == "No")
n_pos <- sum(datos_train$HeartDisease == "Yes")

cat("Distribucion original:\n")
cat("  No:", n_neg, "\n")
cat("  Yes:", n_pos, "\n")
cat("  Ratio:", round(n_neg/n_pos, 2), ":1\n\n")

datos_train_balanced <- ROSE(HeartDisease ~ ., data = datos_train, seed = 123)$data

cat("Resultado con ROSE:\n")
tabla_final <- table(datos_train_balanced$HeartDisease)
print(tabla_final)
porcentaje_yes <- round(tabla_final[2] / sum(tabla_final) * 100, 1)
cat("Porcentaje Yes:", porcentaje_yes, "%\n")
cat("Ratio:", round(tabla_final[1] / tabla_final[2], 2), ":1\n")




# =============================================================================
# 8. MODELO RANDOM FOREST CON DATOS BALANCEADOS POR ROSE
# =============================================================================

cat("=== ENTRENANDO MODELO RANDOM FOREST CON ROSE ===\n")

modelo_rf_rose <- randomForest(
  HeartDisease ~ .,
  data = datos_train_balanced,
  ntree = 100,
  mtry = 5,
  importance = TRUE,
  na.action = na.omit
)

cat("Modelo con ROSE entrenado exitosamente\n")

# =============================================================================
# 9. EVALUACION DEL MODELO ROSE
# =============================================================================

cat("=== EVALUANDO MODELO ROSE ===\n")

predicciones_test <- predict(modelo_rf_rose, datos_test)
predicciones_prob <- predict(modelo_rf_rose, datos_test, type = "prob")

matriz_confusion <- confusionMatrix(predicciones_test, datos_test$HeartDisease, positive = "Yes")
print(matriz_confusion)

# =============================================================================
# 10. CURVA ROC
# =============================================================================

roc_obj <- roc(datos_test$HeartDisease, predicciones_prob[,2])
auc_value <- auc(roc_obj)

png("graficos/02_curva_roc_rose.png", width = 800, height = 600)
plot(roc_obj, main = paste("Curva ROC con ROSE - AUC =", round(auc_value, 4)), col = "#3498db", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
dev.off()

# =============================================================================
# 11. MATRIZ DE CONFUSION 
# =============================================================================

matriz_plot <- as.data.frame(matriz_confusion$table)

# Reordenar los niveles para que coincidan con la matriz estándar
matriz_plot$Prediction <- factor(matriz_plot$Prediction, levels = c("Yes", "No"))
matriz_plot$Reference <- factor(matriz_plot$Reference, levels = c("No", "Yes"))

p_matriz <- ggplot(matriz_plot, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "#3498db", high = "#e74c3c") +
  labs(title = "Matriz de Confusion con ROSE", x = "Valor Real", y = "Prediccion") +
  theme_minimal() +
  scale_y_discrete(limits = c("No", "Yes"))  # Esto fuerza el orden correcto

ggsave("graficos/03_matriz_confusion_rose.png", p_matriz, width = 8, height = 6)
# =============================================================================
# 12. IMPORTANCIA DE VARIABLES
# =============================================================================

importancia <- importance(modelo_rf_rose)
importancia_df <- data.frame(
  Variable = rownames(importancia),
  Importance = importancia[, "MeanDecreaseGini"]
) %>% arrange(desc(Importance))

p_importancia <- importancia_df %>%
  head(10) %>%
  ggplot(aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col(fill = "#3498db") +
  coord_flip() +
  labs(title = "Top 10 Variables Mas Importantes (ROSE)", x = "Variables", y = "Importancia") +
  theme_minimal()

ggsave("graficos/04_importancia_variables_rose.png", p_importancia, width = 10, height = 6)

# =============================================================================
# 13. OPTIMIZACION CON UMBRAL
# =============================================================================

cat("=== OPTIMIZANDO UMBRAL DE CLASIFICACION ===\n")

mejor_umbral <- coords(roc_obj, "best", ret = "threshold", best.method = "youden")$threshold
cat("Mejor umbral encontrado:", round(mejor_umbral, 3), "\n")

predicciones_optimizadas <- ifelse(predicciones_prob[,2] > mejor_umbral, "Yes", "No")
predicciones_optimizadas <- factor(predicciones_optimizadas, levels = c("No", "Yes"))

matriz_optimizada <- confusionMatrix(predicciones_optimizadas, datos_test$HeartDisease, positive = "Yes")
cat("Metricas con umbral optimizado:\n")
print(matriz_optimizada)

# =============================================================================
# 14. METRICAS COMPLETAS
# =============================================================================

f1_original <- matriz_confusion$byClass['F1']
f1_optimizado <- matriz_optimizada$byClass['F1']

metricas_completas <- data.frame(
  Modelo = c("ROSE Umbral 0.5", "ROSE Umbral Optimizado"),
  Exactitud = c(round(matriz_confusion$overall['Accuracy'], 4),
                round(matriz_optimizada$overall['Accuracy'], 4)),
  Sensibilidad = c(round(matriz_confusion$byClass['Sensitivity'], 4),
                   round(matriz_optimizada$byClass['Sensitivity'], 4)),
  Especificidad = c(round(matriz_confusion$byClass['Specificity'], 4),
                    round(matriz_optimizada$byClass['Specificity'], 4)),
  Precision = c(round(matriz_confusion$byClass['Pos Pred Value'], 4),
                round(matriz_optimizada$byClass['Pos Pred Value'], 4)),
  F1_Score = c(round(as.numeric(f1_original), 4),
               round(as.numeric(f1_optimizado), 4)),
  AUC = c(round(auc_value, 4), round(auc_value, 4))
)

cat("Comparacion de metricas con ROSE:\n")
print(metricas_completas)

write.csv(metricas_completas, "resultados/metricas_completas_rose.csv", row.names = FALSE)

# =============================================================================
# 15. GUARDADO
# =============================================================================

saveRDS(modelo_rf_rose, "resultados/modelo_rf_rose.rds")
saveRDS(list(umbral_optimizado = mejor_umbral, predicciones_optimizadas = predicciones_optimizadas), 
        "resultados/optimizacion_umbral_rose.rds")

cat("Analisis con ROSE completado exitosamente\n")
cat("Modelo guardado: modelo_rf_rose.rds\n")
cat("Metricas guardadas: metricas_completas_rose.csv\n")


# =============================================================================
# 16. BALANCEO 80-20 CON UNDERSAMPLING
# =============================================================================

cat("=== BALANCEANDO DATOS 80-20 CON UNDERSAMPLING ===\n")

n_neg <- sum(datos_train$HeartDisease == "No")
n_pos <- sum(datos_train$HeartDisease == "Yes")

cat("Distribucion original:\n")
cat("  No:", n_neg, "\n")
cat("  Yes:", n_pos, "\n")
cat("  Ratio:", round(n_neg/n_pos, 2), ":1\n\n")

n_neg_deseado <- round(n_pos * 4)
total_deseado <- n_neg_deseado + n_pos

cat("Objetivo 80-20 con undersampling:\n")
cat("  No (80%):", n_neg_deseado, "\n")
cat("  Yes (20%):", n_pos, "\n")
cat("  Total:", total_deseado, "\n")
cat("  Ratio: 4:1\n\n")

datos_train_balanced <- ovun.sample(
  HeartDisease ~ ., 
  data = datos_train, 
  method = "under",
  N = total_deseado,
  seed = 123
)$data

cat("Resultado con undersampling 80-20:\n")
tabla_final <- table(datos_train_balanced$HeartDisease)
print(tabla_final)
porcentaje_yes <- round(tabla_final[2] / sum(tabla_final) * 100, 1)
cat("Porcentaje Yes:", porcentaje_yes, "%\n")
cat("Ratio:", round(tabla_final[1] / tabla_final[2], 2), ":1\n")

# =============================================================================
# 17. MODELO RANDOM FOREST CON UNDERSAMPLING
# =============================================================================

cat("=== ENTRENANDO MODELO RANDOM FOREST CON UNDERSAMPLING ===\n")

modelo_rf_under <- randomForest(
  HeartDisease ~ .,
  data = datos_train_balanced,
  ntree = 100,
  mtry = 5,
  importance = TRUE,
  na.action = na.omit
)

cat("Modelo con undersampling entrenado exitosamente\n")

# =============================================================================
# 18. EVALUACION DEL MODELO UNDERSAMPLING
# =============================================================================

cat("=== EVALUANDO MODELO UNDERSAMPLING ===\n")

predicciones_test <- predict(modelo_rf_under, datos_test)
predicciones_prob <- predict(modelo_rf_under, datos_test, type = "prob")

matriz_confusion <- confusionMatrix(predicciones_test, datos_test$HeartDisease, positive = "Yes")
print(matriz_confusion)

# =============================================================================
# 19. CURVA ROC
# =============================================================================

roc_obj <- roc(datos_test$HeartDisease, predicciones_prob[,2])
auc_value <- auc(roc_obj)

png("graficos/02_curva_roc_under.png", width = 800, height = 600)
plot(roc_obj, main = paste("Curva ROC con Undersampling - AUC =", round(auc_value, 4)), col = "#3498db", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
dev.off()

# =============================================================================
# 20. MATRIZ DE CONFUSION
# =============================================================================

matriz_plot <- as.data.frame(matriz_confusion$table)
p_matriz <- ggplot(matriz_plot, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "#3498db", high = "#e74c3c") +
  labs(title = "Matriz de Confusion con Undersampling", x = "Valor Real", y = "Prediccion") +
  theme_minimal()

ggsave("graficos/03_matriz_confusion_under.png", p_matriz, width = 8, height = 6)

# =============================================================================
# 21. IMPORTANCIA DE VARIABLES
# =============================================================================

importancia <- importance(modelo_rf_under)
importancia_df <- data.frame(
  Variable = rownames(importancia),
  Importance = importancia[, "MeanDecreaseGini"]
) %>% arrange(desc(Importance))

p_importancia <- importancia_df %>%
  head(10) %>%
  ggplot(aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col(fill = "#3498db") +
  coord_flip() +
  labs(title = "Top 10 Variables Mas Importantes (Undersampling)", x = "Variables", y = "Importancia") +
  theme_minimal()

ggsave("graficos/04_importancia_variables_under.png", p_importancia, width = 10, height = 6)

# =============================================================================
# 22. OPTIMIZACION CON UMBRAL
# =============================================================================

cat("=== OPTIMIZANDO UMBRAL DE CLASIFICACION ===\n")

mejor_umbral <- coords(roc_obj, "best", ret = "threshold", best.method = "youden")$threshold
cat("Mejor umbral encontrado:", round(mejor_umbral, 3), "\n")

predicciones_optimizadas <- ifelse(predicciones_prob[,2] > mejor_umbral, "Yes", "No")
predicciones_optimizadas <- factor(predicciones_optimizadas, levels = c("No", "Yes"))

matriz_optimizada <- confusionMatrix(predicciones_optimizadas, datos_test$HeartDisease, positive = "Yes")
cat("Metricas con umbral optimizado:\n")
print(matriz_optimizada)

# =============================================================================
# 23. METRICAS COMPLETAS
# =============================================================================

f1_original <- matriz_confusion$byClass['F1']
f1_optimizado <- matriz_optimizada$byClass['F1']

metricas_completas <- data.frame(
  Modelo = c("Under Umbral 0.5", "Under Umbral Optimizado"),
  Exactitud = c(round(matriz_confusion$overall['Accuracy'], 4),
                round(matriz_optimizada$overall['Accuracy'], 4)),
  Sensibilidad = c(round(matriz_confusion$byClass['Sensitivity'], 4),
                   round(matriz_optimizada$byClass['Sensitivity'], 4)),
  Especificidad = c(round(matriz_confusion$byClass['Specificity'], 4),
                    round(matriz_optimizada$byClass['Specificity'], 4)),
  Precision = c(round(matriz_confusion$byClass['Pos Pred Value'], 4),
                round(matriz_optimizada$byClass['Pos Pred Value'], 4)),
  F1_Score = c(round(as.numeric(f1_original), 4),
               round(as.numeric(f1_optimizado), 4)),
  AUC = c(round(auc_value, 4), round(auc_value, 4))
)

cat("Comparacion de metricas con Undersampling:\n")
print(metricas_completas)

write.csv(metricas_completas, "resultados/metricas_completas_under.csv", row.names = FALSE)

# =============================================================================
# 24. GUARDADO
# =============================================================================

saveRDS(modelo_rf_under, "resultados/modelo_rf_under.rds")
saveRDS(list(umbral_optimizado = mejor_umbral, predicciones_optimizadas = predicciones_optimizadas), 
        "resultados/optimizacion_umbral_under.rds")

cat("Analisis con Undersampling completado exitosamente\n")
cat("Modelo guardado: modelo_rf_under.rds\n")
cat("Metricas guardadas: metricas_completas_under.csv\n")


cat("\n=== VERIFICACION DE CONSISTENCIA ===\n")
cat("Tamaño datos_test:", nrow(datos_test), "\n")
cat("Distribución en datos_test:\n")
print(table(datos_test$HeartDisease))


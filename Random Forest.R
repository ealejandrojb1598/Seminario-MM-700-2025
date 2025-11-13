# =======================================================================
# 1. INSTALACION Y CARGA DE PAQUETES
# =======================================================================
paquetes <- c(
  "randomForest", "caret", "ggplot2", "dplyr", "tidyr",
  "pROC", "corrplot", "gridExtra", "RColorBrewer", "ROSE"
)

paquetes_faltantes <- paquetes[!(paquetes %in% installed.packages()[,"Package"])]
if (length(paquetes_faltantes) > 0) {
  install.packages(paquetes_faltantes, dependencies = TRUE)
}

invisible(lapply(paquetes, library, character.only = TRUE))

# =======================================================================
# 2. CONFIGURACION INICIAL
# =======================================================================
set.seed(123)
if (!dir.exists("resultados")) dir.create("resultados")
if (!dir.exists("graficos"))  dir.create("graficos")

# =======================================================================
# 3. CARGA DE DATOS
# =======================================================================
datos <- read.csv("heart_2020_cleaned.csv", stringsAsFactors = FALSE,
                  na.strings = c("", "NA", "N/A"))

# =======================================================================
# 4. PREPROCESAMIENTO
# =======================================================================
datos_prep <- datos

# objetivo (asegura niveles correctos)
datos_prep$HeartDisease <- factor(datos_prep$HeartDisease,
                                  levels = c("No", "Yes"),
                                  labels = c("No", "Yes"))

vars_categoricas <- c("Sex","AgeCategory","Race","Diabetic","GenHealth",
                      "Smoking","AlcoholDrinking","Stroke","DiffWalking",
                      "PhysicalActivity","Asthma","KidneyDisease","SkinCancer")

for (var in vars_categoricas) {
  if (var %in% colnames(datos_prep)) {
    datos_prep[[var]] <- as.factor(datos_prep[[var]])
  }
}

datos_prep <- na.omit(datos_prep)

# =======================================================================
# 5. ANALISIS EXPLORATORIO
# =======================================================================
p1 <- ggplot(datos_prep, aes(x = HeartDisease, fill = HeartDisease)) +
  geom_bar() +
  geom_text(stat = 'count', aes(label = after_stat(count)), vjust = -0.5, size = 5) +
  scale_fill_manual(values = c("No" = "#2ecc71", "Yes" = "#e74c3c")) +
  labs(title = "Distribucion de Cardiopatias",
       x = "Enfermedad Cardiaca", y = "Frecuencia") +
  theme_minimal()
ggsave("graficos/01_distribucion_objetivo.png", p1, width = 8, height = 6, dpi = 300)

# =======================================================================
# 6. DIVISION DE DATOS
# =======================================================================
set.seed(123)
indices_entrenamiento <- createDataPartition(datos_prep$HeartDisease, p = 0.80, list = FALSE)
datos_train <- datos_prep[indices_entrenamiento, ]
datos_test  <- datos_prep[-indices_entrenamiento, ]

# =======================================================================
# 6.5 VALIDACION CRUZADA CON RANDOM FOREST (5 FOLDS)
# =======================================================================
cat("=== VALIDACION CRUZADA DE 5 FOLDS CON RANDOM FOREST ===\n")
set.seed(123)
folds <- createFolds(datos_train$HeartDisease, k = 5)
resultados_cv <- data.frame()

for (metodo in c("rose","under")) {
  cat("\n--- Procesando metodo:", toupper(metodo), "---\n")
  for (i in 1:5) {
    cat(" Fold", i, "de 5...\n")
    
    train_indices <- unlist(folds[-i])
    test_indices  <- folds[[i]]
    train_data <- datos_train[train_indices, ]
    test_data  <- datos_train[test_indices, ]
    
    # Balanceo
    if (metodo == "rose") {
      datos_balanceados <- ROSE(HeartDisease ~ ., data = train_data, seed = 123)$data
    } else {
      n_pos <- sum(train_data$HeartDisease == "Yes")
      n_neg_deseado <- round(n_pos * 4)   # 4:1
      total_deseado  <- n_neg_deseado + n_pos
      datos_balanceados <- ovun.sample(
        HeartDisease ~ ., data = train_data, method = "under",
        N = total_deseado, seed = 123
      )$data
    }
    
    # Modelo (dejamos nodesize por defecto en CV para comparabilidad)
    modelo_rf <- randomForest(
      HeartDisease ~ ., data = datos_balanceados,
      ntree = 50, mtry = 5, importance = FALSE, na.action = na.omit
    )
    
    # Prediccion en fold-test (original)
    pred_clase <- predict(modelo_rf, test_data)
    pred_prob  <- predict(modelo_rf, test_data, type = "prob")[, "Yes"]
    
    # Metricas
    cm <- confusionMatrix(pred_clase, test_data$HeartDisease, positive = "Yes")
    
    # ROC/AUC robusto (fija niveles)
    roc_obj <- roc(response = test_data$HeartDisease,
                   predictor = pred_prob,
                   levels = c("No","Yes"),
                   quiet = TRUE)
    roc_auc <- as.numeric(auc(roc_obj))
    
    resultados_cv <- rbind(resultados_cv, data.frame(
      Fold = i,
      Metodo = metodo,
      Accuracy = round(cm$overall[["Accuracy"]], 4),
      Sensitivity = round(cm$byClass[["Sensitivity"]], 4),
      Specificity = round(cm$byClass[["Specificity"]], 4),
      Precision   = round(cm$byClass[["Precision"]], 4),
      F1          = round(cm$byClass[["F1"]], 4),
      AUC         = round(roc_auc, 4)
    ))
    
    rm(modelo_rf, datos_balanceados, train_data, test_data); gc()
  }
}

# Resumen CV
resumen_cv <- resultados_cv %>%
  group_by(Metodo) %>%
  summarise(
    Accuracy_mean = round(mean(Accuracy), 4),
    Accuracy_sd   = round(sd(Accuracy), 4),
    Sensitivity_mean = round(mean(Sensitivity), 4),
    Sensitivity_sd   = round(sd(Sensitivity), 4),
    Specificity_mean = round(mean(Specificity), 4),
    Specificity_sd   = round(sd(Specificity), 4),
    AUC_mean = round(mean(AUC), 4),
    AUC_sd   = round(sd(AUC), 4),
    .groups = "drop"
  )

cat("\n=== RESULTADOS VALIDACION CRUZADA ===\n"); print(resumen_cv)

mejor_metodo_cv <- resumen_cv %>%
  arrange(desc(AUC_mean)) %>%
  slice(1) %>% pull(Metodo)

cat("\n*** MEJOR METODO SEGUN VALIDACION CRUZADA:", toupper(mejor_metodo_cv), "***\n")

write.csv(resultados_cv, "resultados/validacion_cruzada_rf.csv", row.names = FALSE)
write.csv(resumen_cv, "resultados/resumen_validacion_cruzada_rf.csv", row.names = FALSE)

p_cv <- ggplot(resultados_cv, aes(x = Metodo, y = AUC, fill = Metodo)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.2, size = 2, alpha = 0.6) +
  scale_fill_manual(values = c("rose" = "#3498db", "under" = "#e74c3c")) +
  labs(title = "Comparacion de Metodos - Validacion Cruzada (5 Folds)",
       subtitle = "Distribucion del AUC por metodo",
       x = "Metodo de Balanceo", y = "AUC") +
  theme_minimal()
ggsave("graficos/05_validacion_cruzada_comparacion.png", p_cv, width = 8, height = 6, dpi = 300)
cat("Validacion cruzada completada exitosamente\n")

# =======================================================================
# 7. BALANCEO CON ROSE (ENTRENAMIENTO COMPLETO)
# =======================================================================
cat("=== BALANCEANDO DATOS CON ROSE ===\n")
n_neg <- sum(datos_train$HeartDisease == "No")
n_pos <- sum(datos_train$HeartDisease == "Yes")
cat("Distribucion original:\n No:", n_neg, "\n Yes:", n_pos, "\n Ratio:", round(n_neg/n_pos, 2), ":1\n\n")

datos_train_rose <- ROSE(HeartDisease ~ ., data = datos_train, seed = 123)$data
tabla_final_rose <- table(datos_train_rose$HeartDisease)
print(tabla_final_rose)
porcentaje_yes_rose <- round(tabla_final_rose[["Yes"]] / sum(tabla_final_rose) * 100, 1)
cat("Porcentaje Yes:", porcentaje_yes_rose, "%\n")
cat("Ratio:", round(tabla_final_rose[["No"]] / tabla_final_rose[["Yes"]], 2), ":1\n")

# =======================================================================
# 8. MODELO RANDOM FOREST CON ROSE
# =======================================================================
cat("=== ENTRENANDO MODELO RANDOM FOREST CON ROSE ===\n")
modelo_rf_rose <- randomForest(
  HeartDisease ~ ., data = datos_train_rose,
  ntree = 100, mtry = 5, importance = TRUE, na.action = na.omit,
  nodesize = 1      # === NUEVO === nodesize=1
)
cat("Modelo con ROSE entrenado exitosamente\n")

# === NUEVO: Grafica Error OOB (ROSE) ===
oob_df_rose <- data.frame(Trees = 1:nrow(modelo_rf_rose$err.rate),
                          OOB = modelo_rf_rose$err.rate[, "OOB"])
p_oob_rose <- ggplot(oob_df_rose, aes(x = Trees, y = OOB)) +
  geom_line(size = 1) +
  labs(title = "Error OOB vs Número de Árboles (ROSE)",
       x = "Árboles", y = "Error OOB") +
  theme_minimal()
ggsave("graficos/08_oob_error_rose.png", p_oob_rose, width = 8, height = 6, dpi = 300)

# =======================================================================
# 9. EVALUACION ROSE
# =======================================================================
cat("=== EVALUANDO MODELO ROSE ===\n")
predicciones_test_rose <- predict(modelo_rf_rose, datos_test)
predicciones_prob_rose <- predict(modelo_rf_rose, datos_test, type = "prob")

matriz_confusion_rose <- confusionMatrix(predicciones_test_rose, datos_test$HeartDisease, positive = "Yes")
print(matriz_confusion_rose)

# =======================================================================
# 10. CURVA ROC ROSE
# =======================================================================
roc_obj_rose <- roc(response = datos_test$HeartDisease,
                    predictor = predicciones_prob_rose[, "Yes"],
                    levels = c("No","Yes"),
                    quiet = TRUE)
auc_value_rose <- as.numeric(auc(roc_obj_rose))

png("graficos/02_curva_roc_rose.png", width = 800, height = 600)
plot(roc_obj_rose,
     main = paste("Curva ROC con ROSE - AUC =", round(auc_value_rose, 4)),
     col = "#3498db", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
dev.off()

# =======================================================================
# 11. MATRIZ DE CONFUSION ROSE (HEATMAP)
# =======================================================================
matriz_plot_rose <- as.data.frame(matriz_confusion_rose$table)
matriz_plot_rose$Reference <- factor(matriz_plot_rose$Reference, levels = c("No","Yes"))
matriz_plot_rose$Prediction <- factor(matriz_plot_rose$Prediction, levels = c("No","Yes"))

p_matriz_rose <- ggplot(matriz_plot_rose, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "#3498db", high = "#e74c3c") +
  labs(title = "Matriz de Confusion con ROSE", x = "Valor Real", y = "Prediccion") +
  theme_minimal()
ggsave("graficos/03_matriz_confusion_rose.png", p_matriz_rose, width = 8, height = 6, dpi = 300)

# =======================================================================
# 12. IMPORTANCIA DE VARIABLES ROSE
# =======================================================================
importancia_rose <- importance(modelo_rf_rose)
importancia_df_rose <- data.frame(
  Variable = rownames(importancia_rose),
  MeanDecreaseGini = importancia_rose[, "MeanDecreaseGini"]
) %>% arrange(desc(MeanDecreaseGini))

p_importancia_rose <- importancia_df_rose %>%
  head(10) %>%
  ggplot(aes(x = reorder(Variable, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_col(fill = "#3498db") +
  coord_flip() +
  labs(title = "Top 10 Variables Mas Importantes (ROSE)",
       x = "Variables", y = "Importancia (MeanDecreaseGini)") +
  theme_minimal()
ggsave("graficos/04_importancia_variables_rose.png", p_importancia_rose, width = 10, height = 6, dpi = 300)

# ===  Tabla de importancia en porcentaje (ROSE) ===
total_gini_rose <- sum(importancia_df_rose$MeanDecreaseGini, na.rm = TRUE)
tabla_importancia_rose <- importancia_df_rose %>%
  mutate(Importancia_pct = round(100 * MeanDecreaseGini / total_gini_rose, 2)) %>%
  arrange(desc(Importancia_pct)) %>%
  select(Variable, Importancia_pct) %>%
  head(10)

cat("\nTop 10 Importancia (ROSE) en %:\n"); print(tabla_importancia_rose, row.names = FALSE)
write.csv(tabla_importancia_rose, "resultados/tabla_importancia_rose.csv", row.names = FALSE)

# =======================================================================
# 13. OPTIMIZACION DE UMBRAL ROSE
# =======================================================================
cat("=== OPTIMIZANDO UMBRAL DE CLASIFICACION ROSE ===\n")
mejor_umbral_rose <- as.numeric(coords(roc_obj_rose, "best", ret = "threshold", best.method = "youden"))
cat("Mejor umbral encontrado (ROSE):", round(mejor_umbral_rose, 3), "\n")

predicciones_optimizadas_rose <- ifelse(predicciones_prob_rose[, "Yes"] > mejor_umbral_rose, "Yes", "No")
predicciones_optimizadas_rose <- factor(predicciones_optimizadas_rose, levels = c("No","Yes"))

matriz_optimizada_rose <- confusionMatrix(predicciones_optimizadas_rose, datos_test$HeartDisease, positive = "Yes")
cat("Metricas con umbral optimizado ROSE:\n"); print(matriz_optimizada_rose)

# =======================================================================
# 15. BALANCEO 80-20 CON UNDERSAMPLING
# =======================================================================
cat("=== BALANCEANDO DATOS 80-20 CON UNDERSAMPLING ===\n")
n_neg <- sum(datos_train$HeartDisease == "No")
n_pos <- sum(datos_train$HeartDisease == "Yes")
cat("Distribucion original:\n No:", n_neg, "\n Yes:", n_pos, "\n Ratio:", round(n_neg/n_pos, 2), ":1\n\n")

n_neg_deseado <- round(n_pos * 4)
total_deseado  <- n_neg_deseado + n_pos
cat("Objetivo 80-20 con undersampling:\n No (80%):", n_neg_deseado, "\n Yes (20%):", n_pos, "\n Total:", total_deseado, "\n Ratio: 4:1\n\n")

datos_train_under <- ovun.sample(
  HeartDisease ~ ., data = datos_train, method = "under",
  N = total_deseado, seed = 123
)$data

tabla_final_under <- table(datos_train_under$HeartDisease)
print(tabla_final_under)
porcentaje_yes_under <- round(tabla_final_under[["Yes"]] / sum(tabla_final_under) * 100, 1)
cat("Porcentaje Yes:", porcentaje_yes_under, "%\n")
cat("Ratio:", round(tabla_final_under[["No"]] / tabla_final_under[["Yes"]], 2), ":1\n")

# =======================================================================
# 16. MODELO RANDOM FOREST CON UNDERSAMPLING
# =======================================================================
cat("=== ENTRENANDO MODELO RANDOM FOREST CON UNDERSAMPLING ===\n")
modelo_rf_under <- randomForest(
  HeartDisease ~ ., data = datos_train_under,
  ntree = 100, mtry = 5, importance = TRUE, na.action = na.omit,
  nodesize = 1      # === NUEVO === nodesize=1
)
cat("Modelo con undersampling entrenado exitosamente\n")

# === NUEVO: Grafica Error OOB (UNDER) ===
oob_df_under <- data.frame(Trees = 1:nrow(modelo_rf_under$err.rate),
                           OOB = modelo_rf_under$err.rate[, "OOB"])
p_oob_under <- ggplot(oob_df_under, aes(x = Trees, y = OOB)) +
  geom_line(size = 1) +
  labs(title = "Error OOB vs Número de Árboles (Under-Sampling)",
       x = "Árboles", y = "Error OOB") +
  theme_minimal()
ggsave("graficos/09_oob_error_under.png", p_oob_under, width = 8, height = 6, dpi = 300)

# =======================================================================
# 17-20. EVALUACION, ROC, MATRIZ HEATMAP, IMPORTANCIA (UNDER)
# =======================================================================
cat("=== EVALUANDO MODELO UNDERSAMPLING ===\n")
predicciones_test_under <- predict(modelo_rf_under, datos_test)
predicciones_prob_under <- predict(modelo_rf_under, datos_test, type = "prob")
matriz_confusion_under <- confusionMatrix(predicciones_test_under, datos_test$HeartDisease, positive = "Yes")
print(matriz_confusion_under)

roc_obj_under <- roc(response = datos_test$HeartDisease,
                     predictor = predicciones_prob_under[, "Yes"],
                     levels = c("No","Yes"),
                     quiet = TRUE)
auc_value_under <- as.numeric(auc(roc_obj_under))

png("graficos/02_curva_roc_under.png", width = 800, height = 600)
plot(roc_obj_under,
     main = paste("Curva ROC con Undersampling - AUC =", round(auc_value_under, 4)),
     col = "#3498db", lwd = 2)
abline(a = 0, b = 1, lty = 2, col = "gray")
dev.off()

matriz_plot_under <- as.data.frame(matriz_confusion_under$table)
matriz_plot_under$Reference <- factor(matriz_plot_under$Reference, levels = c("No","Yes"))
matriz_plot_under$Prediction <- factor(matriz_plot_under$Prediction, levels = c("No","Yes"))

p_matriz_under <- ggplot(matriz_plot_under, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "#3498db", high = "#e74c3c") +
  labs(title = "Matriz de Confusion con Undersampling", x = "Valor Real", y = "Prediccion") +
  theme_minimal()
ggsave("graficos/03_matriz_confusion_under.png", p_matriz_under, width = 8, height = 6, dpi = 300)

importancia_under <- importance(modelo_rf_under)
importancia_df_under <- data.frame(
  Variable = rownames(importancia_under),
  MeanDecreaseGini = importancia_under[, "MeanDecreaseGini"]
) %>% arrange(desc(MeanDecreaseGini))

p_importancia_under <- importancia_df_under %>%
  head(10) %>%
  ggplot(aes(x = reorder(Variable, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_col(fill = "#3498db") +
  coord_flip() +
  labs(title = "Top 10 Variables Mas Importantes (Undersampling)",
       x = "Variables", y = "Importancia (MeanDecreaseGini)") +
  theme_minimal()
ggsave("graficos/04_importancia_variables_under.png", p_importancia_under, width = 10, height = 6, dpi = 300)

# ===  Tabla de importancia en porcentaje (UNDER) ===
total_gini_under <- sum(importancia_df_under$MeanDecreaseGini, na.rm = TRUE)
tabla_importancia_under <- importancia_df_under %>%
  mutate(Importancia_pct = round(100 * MeanDecreaseGini / total_gini_under, 2)) %>%
  arrange(desc(Importancia_pct)) %>%
  select(Variable, Importancia_pct) %>%
  head(10)

cat("\nTop 10 Importancia (UNDER) en %:\n"); print(tabla_importancia_under, row.names = FALSE)
write.csv(tabla_importancia_under, "resultados/tabla_importancia_under.csv", row.names = FALSE)

# =======================================================================
# 21. OPTIMIZACION DE UMBRAL (UNDER)
# =======================================================================
cat("=== OPTIMIZANDO UMBRAL DE CLASIFICACION UNDERSAMPLING ===\n")
mejor_umbral_under <- as.numeric(coords(roc_obj_under, "best", ret = "threshold", best.method = "youden"))
cat("Mejor umbral encontrado (UNDER):", round(mejor_umbral_under, 3), "\n")

predicciones_optimizadas_under <- ifelse(predicciones_prob_under[, "Yes"] > mejor_umbral_under, "Yes", "No")
predicciones_optimizadas_under <- factor(predicciones_optimizadas_under, levels = c("No","Yes"))

matriz_optimizada_under <- confusionMatrix(predicciones_optimizadas_under, datos_test$HeartDisease, positive = "Yes")
cat("Metricas con umbral optimizado UNDER:\n"); print(matriz_optimizada_under)

# =======================================================================
# 22-24. COMPARACION FINAL + GUARDADO
# =======================================================================
cat("\n", strrep("=", 60), "\n", "COMPARACION FINAL ENTRE METODOS\n", strrep("=", 60), "\n", sep = "")
cat("\nRESULTADO VALIDACION CRUZADA (5 FOLDS):\n"); print(resumen_cv)

comparativa_final <- data.frame(
  Metodo = c("ROSE", "Undersampling"),
  AUC_Test = c(round(auc_value_rose, 4), round(auc_value_under, 4)),
  Sensibilidad_Test = c(round(matriz_confusion_rose$byClass['Sensitivity'], 4),
                        round(matriz_confusion_under$byClass['Sensitivity'], 4)),
  Especificidad_Test = c(round(matriz_confusion_rose$byClass['Specificity'], 4),
                         round(matriz_confusion_under$byClass['Specificity'], 4)),
  AUC_CV = c(resumen_cv$AUC_mean[resumen_cv$Metodo == "rose"],
             resumen_cv$AUC_mean[resumen_cv$Metodo == "under"])
)
print(comparativa_final)

cat("\nRECOMENDACION BASADA EN VALIDACION CRUZADA:\n")
if (mejor_metodo_cv == "rose") {
  cat("El metodo ROSE mostro mejor performance en validacion cruzada\n")
  cat("Considera usar modelo_rf_rose para predicciones futuras\n")
} else {
  cat("El metodo UNDERSAMPLING mostro mejor performance en validacion cruzada\n")
  cat("Considera usar modelo_rf_under para predicciones futuras\n")
}

saveRDS(modelo_rf_rose,  "resultados/modelo_rf_rose.rds")
saveRDS(modelo_rf_under, "resultados/modelo_rf_under.rds")
cat("\nMODELOS GUARDADOS:\n - modelo_rf_rose.rds\n - modelo_rf_under.rds\n - Resultados de validacion cruzada\n")

# =======================================================================
# 25. INTERVALOS DE CONFIANZA (95%) 
# =======================================================================

cat("\n", strrep("=", 70), "\nCÁLCULO DE INTERVALOS DE CONFIANZA (95%)\n", strrep("=", 70), "\n\n", sep="")

# --- Función IC Wilson (no cambia) ---
calcular_ic_wilson <- function(x, n, nivel = 0.95) {
  if (is.na(n) || n <= 0 || is.na(x) || x < 0) return(c(0, 0))
  z <- qnorm((1 + nivel) / 2)
  p <- x / n
  denom <- 1 + z^2 / n
  centro <- (p + z^2 / (2 * n)) / denom
  margen <- z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / denom
  ic_inf <- max(0, centro - margen)
  ic_sup <- min(1, centro + margen)
  if (ic_inf > ic_sup) { tmp <- ic_inf; ic_inf <- ic_sup; ic_sup <- tmp }
  c(ic_inf, ic_sup)
}

# --- FUNCIÓN PRINCIPAL CORREGIDA ---
calcular_ic_metricas <- function(matriz_confusion, nombre_metodo, verbose = FALSE) {
  # === 1. Extraer TP, TN, FP, FN correctamente ===
  tn <- matriz_confusion$table["No", "No"]
  fp <- matriz_confusion$table["Yes", "No"]
  fn <- matriz_confusion$table["No", "Yes"]
  tp <- matriz_confusion$table["Yes", "Yes"]
  
  if (verbose) {
    cat("\n", nombre_metodo, ":\n")
    cat("  TN:", tn, "| FP:", fp, "| FN:", fn, "| TP:", tp, "\n")
  }
  
  total <- tn + fp + fn + tp
  acc <- (tp + tn) / total
  sen <- tp / (tp + fn)
  esp <- tn / (tn + fp)
  ppv <- tp / (tp + fp)
  
  # === 2. Intervalos de Wilson ===
  ic_acc <- calcular_ic_wilson(tp + tn, total)
  ic_sen <- calcular_ic_wilson(tp, tp + fn)
  ic_esp <- calcular_ic_wilson(tn, tn + fp)
  ic_ppv <- calcular_ic_wilson(tp, tp + fp)
  
  # === 3. Crear tabla ===
  resultados_ic <- data.frame(
    Metodo = rep(nombre_metodo, 4),
    Metrica = c("Accuracy", "Sensibilidad", "Especificidad", "Precisión"),
    Valor = round(c(acc, sen, esp, ppv) * 100, 2),
    IC_Inferior = round(c(ic_acc[1], ic_sen[1], ic_esp[1], ic_ppv[1]) * 100, 2),
    IC_Superior = round(c(ic_acc[2], ic_sen[2], ic_esp[2], ic_ppv[2]) * 100, 2),
    Amplitud = round(c(
      ic_acc[2] - ic_acc[1],
      ic_sen[2] - ic_sen[1],
      ic_esp[2] - ic_esp[1],
      ic_ppv[2] - ic_ppv[1]
    ) * 100, 2)
  )
  
  return(resultados_ic)
}

# --- CÁLCULO PARA CADA CONFIGURACIÓN ---
cat("Calculando IC para ROSE (umbral 0.5)...\n")
ic_rose_05 <- calcular_ic_metricas(matriz_confusion_rose, "ROSE (0.5)", verbose = TRUE)

cat("\nCalculando IC para ROSE (umbral optimizado)...\n")
ic_rose_opt <- calcular_ic_metricas(matriz_optimizada_rose,
                                    paste0("ROSE (", round(mejor_umbral_rose, 3), ")"),
                                    verbose = TRUE)

cat("\nCalculando IC para Under-Sampling (umbral 0.5)...\n")
ic_under_05 <- calcular_ic_metricas(matriz_confusion_under, "Under (0.5)", verbose = TRUE)

cat("\nCalculando IC para Under-Sampling (umbral optimizado)...\n")
ic_under_opt <- calcular_ic_metricas(matriz_optimizada_under,
                                     paste0("Under (", round(mejor_umbral_under, 3), ")"),
                                     verbose = TRUE)

# --- UNIR Y MOSTRAR RESULTADOS ---
ic_completos <- rbind(ic_rose_05, ic_rose_opt, ic_under_05, ic_under_opt)

cat("\n", strrep("=", 70), "\nRESULTADOS: INTERVALOS DE CONFIANZA (95%)\n", strrep("=", 70), "\n", sep="")
print(ic_completos, row.names = FALSE)

write.csv(ic_completos, "resultados/intervalos_confianza.csv", row.names = FALSE)
cat("\nResultados guardados en: resultados/intervalos_confianza.csv\n")

# --- GRAFICO DE IC ---
cat("\nGenerando gráfico de intervalos de confianza...\n")
ic_completos$Metodo <- factor(ic_completos$Metodo, levels = unique(ic_completos$Metodo))

p_ic <- ggplot(ic_completos, aes(x = Metrica, y = Valor, color = Metodo, group = Metodo)) +
  geom_point(size = 3, position = position_dodge(width = 0.6)) +
  geom_errorbar(aes(ymin = IC_Inferior, ymax = IC_Superior),
                width = 0.3, size = 0.8,
                position = position_dodge(width = 0.6)) +
  coord_flip() +
  labs(title = "Intervalos de Confianza (95%) por Métrica y Configuración",
       subtitle = "Barras de error: límites inferior y superior (Wilson)",
       x = "", y = "Valor (%)") +
  theme_minimal() +
  theme(legend.position = "bottom",
        plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(size = 11),
        axis.text = element_text(size = 10),
        legend.text = element_text(size = 10)) +
  guides(color = guide_legend(nrow = 2))

ggsave("graficos/06_intervalos_confianza.png", p_ic, width = 12, height = 7, dpi = 300)
cat("Gráfico guardado en: graficos/06_intervalos_confianza.png\n")

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
# =======================================================================
# 27. RESUMEN
# =======================================================================
cat("\n", strrep("=", 70), "\nANÁLISIS COMPLETADO\n", strrep("=", 70), "\n", sep = "")
cat("Archivos generados:\n",
    "  - resultados/validacion_cruzada_rf.csv\n",
    "  - resultados/resumen_validacion_cruzada_rf.csv\n",
    "  - resultados/intervalos_confianza.csv\n",
    "  - resultados/comparacion_auc_roc_pr.csv\n",
    "  - resultados/tabla_importancia_rose.csv\n",
    "  - resultados/tabla_importancia_under.csv\n",
    "  - graficos/01_distribucion_objetivo.png\n",
    "  - graficos/02_curva_roc_rose.png\n",
    "  - graficos/02_curva_roc_under.png\n",
    "  - graficos/03_matriz_confusion_rose.png\n",
    "  - graficos/03_matriz_confusion_under.png\n",
    "  - graficos/04_importancia_variables_rose.png\n",
    "  - graficos/04_importancia_variables_under.png\n",
    "  - graficos/05_validacion_cruzada_comparacion.png\n",
    "  - graficos/06_intervalos_confianza.png\n",
    "  - graficos/07_curvas_precision_recall.png\n",
    "  - graficos/08_oob_error_rose.png\n",
    "  - graficos/09_oob_error_under.png\n", sep = "")

# =======================================================================
# 28. Interpretacion SHAP al undersampling
# =======================================================================


install.packages("fastshap")
install.packages("data.table")




library(fastshap)
library(data.table)
library(randomForest)
library(ggplot2)



# Matriz de predictores (sin la variable objetivo)
X_test <- datos_test[, colnames(datos_test) != "HeartDisease"]

# Uso de una muestra para mejorar eficiencia en conjuntos grandes
if (nrow(X_test) > 1000) {
  set.seed(123)
  sample_idx <- sample(1:nrow(X_test), 1000)
  X_test <- X_test[sample_idx, ]
  cat("Usando muestra de", nrow(X_test), "observaciones para SHAP\n")
}

# Funcion de prediccion para fastshap
f_pred <- function(object, newdata) {
  predict(object, newdata, type = "prob")[, "Yes"]
}

# Calculo de valores SHAP
set.seed(123)
shap_values <- fastshap::explain(
  object = modelo_rf_under,
  X = X_test,
  pred_wrapper = f_pred,
  nsim = 100,
  adjust = TRUE
)

# Importancia SHAP promedio por variable
shap_importance <- data.frame(
  Variable = colnames(shap_values),
  SHAP = colMeans(abs(shap_values))
)
shap_importance <- shap_importance[order(-shap_importance$SHAP), ]

write.csv(shap_importance, "resultados/shap_importances_under.csv", row.names = FALSE)

# ============================================================
# Grafico 1: Importancia SHAP (Top 10)
# ============================================================
p_shap_importance <- ggplot(shap_importance[1:10, ],
                            aes(x = reorder(Variable, SHAP), y = SHAP)) +
  geom_col(fill = "#e74c3c") +
  coord_flip() +
  labs(title = "Importancia SHAP - Top 10 (Under-Sampling)",
       x = "Variable",
       y = "Valor SHAP Medio") +
  theme_minimal()

ggsave("graficos/11_shap_importances.png",
       p_shap_importance, width = 8, height = 6, dpi = 300)

# ============================================================
# Grafico 2: Distribucion SHAP (Beeswarm, Top 15)
# ============================================================
shap_long <- data.table::melt(
  data.table(shap_values),
  variable.name = "Variable",
  value.name = "SHAP"
)

top_vars <- shap_importance$Variable[1:15]
shap_long_top <- shap_long[Variable %in% top_vars]

p_shap_bee <- ggplot(shap_long_top, aes(x = SHAP, y = Variable)) +
  geom_point(alpha = 0.4, color = "#e74c3c", size = 1.4) +
  geom_vline(xintercept = 0, linetype = 2, color = "gray40") +
  labs(title = "Distribución de valores SHAP - Top 15 (Under-Sampling)",
       x = "Valor SHAP",
       y = "Variable") +
  theme_minimal()

ggsave("graficos/12_shap_beeswarm.png",
       p_shap_bee, width = 9, height = 7, dpi = 300)

# ============================================================
# Grafico 3: Resumen SHAP (Top 10)
# ============================================================
p_shap_summary <- ggplot(shap_importance[1:10, ],
                         aes(x = SHAP, y = reorder(Variable, SHAP))) +
  geom_point(color = "#c0392b", size = 3) +
  labs(title = "Top 10 - Valores SHAP promedio (Under-Sampling)",
       x = "Valor SHAP Medio",
       y = "Variable") +
  theme_minimal()

ggsave("graficos/10_shap_summary.png",
       p_shap_summary, width = 8, height = 6, dpi = 300)

cat("SHAP calculado y graficos generados\n")

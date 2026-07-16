## Installing necessary packages -------------------------------------------
if(!require("pacman")) install.packages("pacman")

pacman::p_load(
  here,
  rio,
  
  skimr,
  janitor,
  lubridate,
  parsedate,
  epikit,
  forcats,
  tidyverse,
  ExclusionTable,
  
  naniar,
  
  tableone,
  flextable,
  
  # machine learning packages
  visdat,

  caret,
  glmnet,
  rocr,
  
  vip,
  broom,
  rocr,
  rpart.plot,
  ranger,
  xgboost,
  
  recipes,
  pdp,
  
  shapviz,
  kernalshap,
  kernlab,
  patchwork,
  
  pROC,
  scales
)

## Importing data ----------------------------------------------------------
data_cnv <- import(here("R", "R datasets", "synthetic_cnv_raw.csv"))

data_cnv <- data_cnv %>%
  mutate(
    disease_status          = as.factor(disease_status),
    sex                     = as.factor(sex),
    ancestry                = as.character(ancestry)
  ) %>%
  mutate(
    across(starts_with("GENE"), as.numeric)
  ) 

## Baseline characteristics ---------------------------------------------------------------
data_cnv_base <- data_cnv %>%
  select("age", "sex", "ancestry", "disease_status")

vars <- dput(names(data_cnv_base))
vars_cat <-names(select(data_cnv_base, c("sex", "disease_status")))

table_one <- CreateTableOne(
  vars            = vars,
  strata          = c("disease_status"),
  data            = data_cnv_base,
  factorVars      = vars_cat,
  testNormal      = oneway.test,
  smd             = TRUE,
  addOverall      = TRUE
)

print(table_one,
      missing        = FALSE,
      test           = FALSE,
      smd            = TRUE,
      showAllLevels  = TRUE
      )

## Data splitting ----------------------------------------------------------
data_cnv_gene <- data_cnv %>%
  select(-c("patient_id", "ancestry"))

set.seed(010101)
index <- createDataPartition(data_cnv_gene$disease_status, p = 0.7,
                             list = FALSE)

data_cnv_train <- data_cnv_gene[index, ]
data_cnv_test <- data_cnv_gene[-index, ]

## Penalized logistic regression -----------------------------------------------------
cv_log <- train(
  disease_status ~ .,
  data = data_cnv_train,
  method = "glmnet",
  family = "binomial",
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = expand.grid(
    alpha = 1,
    lambda = seq(0.001, 0.1, length.out = 20)
  )
)

cat("Best lambda:", cv_log$bestTune$lambda, "\n")

# Using lambda coefficients
best_lambda <- cv_log$bestTune$lambda
lasso_coefs <- coef(cv_log$finalModel, s = best_lambda)

coef_df <- data.frame(
  gene     = rownames(lasso_coefs),
  coef     = as.numeric(lasso_coefs)
) %>%
  filter(coef != 0, gene != "(Intercept)") %>%
  arrange(desc(abs(coef)))

cat("\nGenes selected by Lasso (non-zero coefficients):", nrow(coef_df), "\n")
print(coef_df)

## Random forests ----------------------------------------------------------
n_features <- length(setdiff(names(data_cnv_gene), "disease_status"))

# Train the model
model_rf <- ranger(
  formula                   = disease_status ~ .,
  data                      = data_cnv_train,
  num.trees                 = n_features * 10,
  mtry                      = floor(sqrt(n_features)),
  min.node.size             = 5,
  replace                   = TRUE,
  sample.fraction           = 0.632,
  verbose                   = FALSE,
  respect.unordered.factors = "order",
  seed                      = 1234
)

default_oob <- model_rf$prediction.error
cat("The default OOB error rate is", round(default_oob, 4), "\n")

# Creating and running the hyperparameter grid search
grid_rf <- expand.grid(
  mtry                = floor(n_features * c(.05, .15, .25, .333, .4)),
  min.node.size       = c(1, 3, 5, 10),
  replace             = c(TRUE, FALSE),
  sample.fraction     = c(.5, .63, .8),
  oob_err             = NA
)

for(i in seq_len(nrow(grid_rf))) {
  fit <- ranger(
    formula                    = disease_status ~ .,
    data                       = data_cnv_train,
    num.trees                  = n_features * 10,
    mtry                       = grid_rf$mtry[i],
    min.node.size              = grid_rf$min.node.size[i],
    replace                    = grid_rf$replace[i],
    sample.fraction            = grid_rf$sample.fraction[i],
    verbose                    = FALSE,
    seed                       = 1234,
    respect.unordered.factors  = 'order'
  )
  grid_rf$oob_err[i] <- fit$prediction.error
}

best_params <- grid_rf %>%
  arrange(oob_err) %>%
  mutate(perc_gain = (default_oob - oob_err) / default_oob * 100) %>%
  slice(1)

cat("\nThe best hyperparameters are: \n")
print(best_params)

# Create the final model
vip_rf <- ranger(
  formula                   = disease_status ~ .,
  data                      = data_cnv_train,
  num.trees                 = n_features * 10,
  mtry                      = best_params$mtry,
  min.node.size             = best_params$min.node.size,
  sample.fraction           = best_params$sample.fraction,
  replace                   = best_params$replace,
  importance                = "permutation",
  probability               = TRUE,    
  respect.unordered.factors = "order",
  verbose                   = FALSE,
  seed                      = 1234
)

cat("\nFinal model OOB error rate:", round(vip_rf$prediction.error, 4), "\n")

# Evaluate on the test set
rf_probs <- predict(vip_rf, data = data_cnv_test)$predictions

test_preds <- colnames(rf_probs)[apply(rf_probs, 1, which.max)]
test_preds <- factor(test_preds,
                     levels = levels(data_cnv_test$disease_status))

cm <- confusionMatrix(test_preds,
                      data_cnv_test$disease_status,
                      positive = "1")
print(cm)

# Determining variable importance
plot_rf <- vip(vip_rf, num_features = 25, bar = FALSE) +
  labs(
    title    = "Random Forest — Top 25 Features by Permutation Importance",
    subtitle = "Permutation importance: drop in accuracy when feature is shuffled",
    x        = "Feature",
    y        = "Importance"
  ) +
  theme_minimal()

plot_rf

## XGBOOST -----------------------------------------------------------------
# Variable preparation
xgb_prep <- recipe(disease_status ~ ., data = data_cnv_gene) %>%
  step_integer(all_nominal()) %>%
  prep(training = data_cnv_train, retain = TRUE) %>%
  juice()

x_var <- as.matrix(xgb_prep[setdiff(names(xgb_prep), "disease_status")])
y_var <- as.numeric(xgb_prep$disease_status == "1")
dtrain <- xgb.DMatrix(data = x_var, label = y_var)

xgb_test_prep <- recipe(disease_status ~ ., data = data_cnv_gene) %>%
  step_integer(all_nominal()) %>%
  prep(training = data_cnv_train, retain = TRUE) %>%
  bake(new_data = data_cnv_test)

x_test <- as.matrix(xgb_test_prep[setdiff(names(xgb_test_prep), "disease_status")])
y_test  <- as.numeric(data_cnv_test$disease_status == "1")
dtest   <- xgb.DMatrix(data = x_test, label = y_test)

# Creating hyperparameter grid
set.seed(42)
n_iter <- 20

grid_xgb <- data.frame(
  eta              = 0.01,
  max_depth        = 3,
  min_child_weight = 3,
  subsample        = 0.5,
  colsample_bytree = 0.5,
  gamma            = sample(c(0, 1, 10, 100, 1000),  n_iter, replace = TRUE),
  lambda           = sample(c(0, 0.01, 0.1, 1, 100), n_iter, replace = TRUE),
  alpha            = sample(c(0, 0.01, 0.1, 1, 100), n_iter, replace = TRUE),
  logloss          = 0,
  trees            = 0
)

# Searching the hyperparameter grid
for (i in seq_len(nrow(grid_xgb))) {
  set.seed(123)
  
  cat(sprintf("Running iteration %d of %d\n", i, nrow(grid_xgb))) # This was a long sequence. Adding this in helped me to know where in the loop I was. 
  
  tryCatch({
    
    m <- xgb.cv(
      data                  = dtrain,
      nrounds               = 500,
      early_stopping_rounds = 20,
      nfold                 = 5,
      verbose               = 0,
      params = list(
        objective        = "binary:logistic",
        eval_metric      = "logloss",
        eta              = grid_xgb$eta[i],
        max_depth        = grid_xgb$max_depth[i],
        min_child_weight = grid_xgb$min_child_weight[i],
        subsample        = grid_xgb$subsample[i],
        colsample_bytree = grid_xgb$colsample_bytree[i],
        gamma            = grid_xgb$gamma[i],
        lambda           = grid_xgb$lambda[i],
        alpha            = grid_xgb$alpha[i]
      )
    )
    
    grid_xgb$logloss[i] <- min(m$evaluation_log$test_logloss_mean)
    
    grid_xgb$trees[i] <- if (!is.null(m$best_iteration)) {
      m$best_iteration
    } else {
      nrow(m$evaluation_log)
    }
    
  }, error = function(e) {
    cat(sprintf("Iteration %d failed: %s\n", i, e$message))
  })
}

# Displaying best hyperparameters
best_params_xgb <- grid_xgb %>%
  filter(logloss > 0) %>%
  arrange(logloss) %>%
  slice(1)

cat("\nBest hyperparameters:\n")
glimpse(best_params_xgb)

# Implementing final model
opt_xgb <- list(
  objective        = "binary:logistic",
  eval_metric      = "logloss",
  eta              = best_params_xgb$eta,
  max_depth        = best_params_xgb$max_depth,
  min_child_weight = best_params_xgb$min_child_weight,
  subsample        = best_params_xgb$subsample,
  colsample_bytree = best_params_xgb$colsample_bytree,
  gamma            = best_params_xgb$gamma,
  lambda           = best_params_xgb$lambda,
  alpha            = best_params_xgb$alpha
)

test_xgb <- xgb.train(
  params  = opt_xgb,
  data    = dtest,
  nrounds = best_params_xgb$trees,
  verbose = 0
)

# Evaluating the test set
pred_probs <- predict(test_xgb, newdata = dtest)

pred_class <- factor(ifelse(pred_probs > 0.5, "Case", "Control"),
                     levels = c("Control", "Case"))
true_class <- factor(ifelse(y_test == 1, "Case", "Control"),
                     levels = c("Control", "Case"))

cm <- confusionMatrix(pred_class, true_class, positive = "Case")
print(cm)

# Evaluating feature importance
vip(test_xgb, num_features = 25) +
  geom_col(fill = "#2C7BB6") +
  labs(
    title    = "XGBoost — Top 25 Features by Gain",
    subtitle = "Importance measured by average gain across all splits",
    x        = "Feature",
    y        = "Importance"
  ) +
  theme_minimal()

## Support vector machines -------------------------------------------------
# Pre-processing variables
preprocess_params <- preProcess(
  data_cnv_train[, setdiff(names(data_cnv_train), "disease_status")],
  method = c("center", "scale")
)

data_cnv_train_scaled <- predict(preprocess_params, data_cnv_train)
data_cnv_test_scaled  <- predict(preprocess_params, data_cnv_test)

data_cnv_train_scaled <- data_cnv_train_scaled %>%
  mutate(disease_status = factor(disease_status,
                                 levels = c(0, 1),
                                 labels = c("Control", "Case")))

data_cnv_test_scaled <- data_cnv_test_scaled %>%
  mutate(disease_status = factor(disease_status,
                                 levels = c(0, 1),
                                 labels = c("Control", "Case")))

# Cross validation and creating a small hyperparameter grid
cv_ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

svm_grid <- expand.grid(
  C     = c(0.001, 0.01, 0.1, 1, 10, 100, 1000),
  sigma = c(0.0001, 0.001, 0.01, 0.1, 1)
)

# Running SVM model with grid search
set.seed(42)
svm_model <- train(
  disease_status ~ .,
  data      = data_cnv_train_scaled,
  method    = "svmRadial",
  metric    = "ROC",
  trControl = cv_ctrl,
  tuneGrid  = svm_grid
)

cat("\nBest hyperparameters:\n")
print(svm_model$bestTune)

# Evaluating on the test set
pred_class <- predict(svm_model, newdata = data_cnv_test_scaled)
pred_probs <- predict(svm_model, newdata = data_cnv_test_scaled,
                      type = "prob")[, "Case"]

cm <- confusionMatrix(pred_class,
                      data_cnv_test_scaled$disease_status,
                      positive = "Case")
print(cm)

# Determining permutation importance
predict_fn_vip <- function(object, newdata) {
  predict(object, newdata = newdata, type = "prob")[, "Case"]
}

set.seed(42)
perm_svm <- vi_permute(
  object       = svm_model,
  train        = data_cnv_test_scaled,
  target       = "disease_status",
  metric       = "roc_auc",
  pred_wrapper = predict_fn_vip,
  nsim         = 10
)

p_perm <- vip(perm_svm, num_features = 25) +
  geom_col(fill = "#D7191C") +
  labs(
    title    = "SVM — Top 25 Features by Permutation Importance",
    subtitle = "Drop in ROC-AUC when feature is randomly shuffled",
    x        = "Feature",
    y        = "Permutation Importance"
  ) +
  theme_minimal()

print(p_perm)

## Comparing the performance of all four models ----------------------------
# Creating ROC-AUC
roc_lasso <- roc(data_cnv_test$disease_status,
                 predict(cv_log, newdata = data_cnv_test, type = "prob")[, 2])

roc_rf <- roc(
  data_cnv_test$disease_status,
  rf_probs[, "1"]
)

roc_xgb   <- roc(data_cnv_test$disease_status,
                 predict(test_xgb, newdata = dtest))

roc_svm   <- roc(data_cnv_test_scaled$disease_status,
                 predict(svm_model, newdata = data_cnv_test_scaled, type = "prob")[, 2])

# Creating ROC-AUC plot
plot(roc_lasso,
     col  = "#2C7BB6", lwd = 2,
     main = "ROC Curves — All Four Models",
     legacy.axes = TRUE)
plot(roc_rf,  col = "#D7191C", lwd = 2, add = TRUE)
plot(roc_xgb, col = "#1A9641", lwd = 2, add = TRUE)
plot(roc_svm, col = "#F4A736", lwd = 2, add = TRUE)
abline(a = 0, b = 1, lty = 2, col = "grey60")
legend("bottomright",
       legend = c(
         sprintf("Lasso         AUC = %.4f", auc(roc_lasso)),
         sprintf("Random Forest AUC = %.4f", auc(roc_rf)),
         sprintf("XGBoost       AUC = %.4f", auc(roc_xgb)),
         sprintf("SVM           AUC = %.4f", auc(roc_svm))
       ),
       col = c("#2C7BB6", "#D7191C", "#1A9641", "#F4A736"),
       lwd = 2,
       cex = 0.85)

# Using DeLong test to statistically compare AUCs
pairs <- list(
  c("Lasso",         "Random Forest"),
  c("Lasso",         "XGBoost"),
  c("Lasso",         "SVM"),
  c("Random Forest", "XGBoost"),
  c("Random Forest", "SVM"),
  c("XGBoost",       "SVM")
)

roc_list <- list(
  "Lasso"         = roc_lasso,
  "Random Forest" = roc_rf,
  "XGBoost"       = roc_xgb,
  "SVM"           = roc_svm
)

delong_results <- map_dfr(pairs, function(p) {
  test <- roc.test(roc_list[[p[1]]], roc_list[[p[2]]], method = "delong")
  data.frame(
    model_1   = p[1],
    model_2   = p[2],
    auc_1     = round(as.numeric(auc(roc_list[[p[1]]])), 4),
    auc_2     = round(as.numeric(auc(roc_list[[p[2]]])), 4),
    statistic = round(test$statistic, 3),
    p_value   = round(test$p.value, 4),
    significant = ifelse(test$p.value < 0.05, "Yes", "No")
  )
})

print(delong_results)

# Computing confusion matrices of all models
cm_lasso <- confusionMatrix(
  predict(cv_log, newdata = data_cnv_test),
  data_cnv_test$disease_status,
  positive = "1"
)

cm_rf <- confusionMatrix(test_preds,
                         data_cnv_test$disease_status,
                         positive = "1")

cm_xgb <- confusionMatrix(pred_class, true_class, positive = "Case")

cm_svm <- confusionMatrix(
  predict(svm_model, newdata = data_cnv_test_scaled),
  data_cnv_test_scaled$disease_status,
  positive = "Case"
)

# Compiling into a single comparison table
perf_table <- data.frame(
  Model    = c("Lasso", "Random Forest", "XGBoost", "SVM"),
  AUC      = round(c(auc(roc_lasso), auc(roc_rf),
                     auc(roc_xgb),   auc(roc_svm)), 4),
  Accuracy = round(c(cm_lasso$overall["Accuracy"],
                     cm_rf$overall["Accuracy"],
                     cm_xgb$overall["Accuracy"],
                     cm_svm$overall["Accuracy"]), 4),
  Sensitivity = round(c(cm_lasso$byClass["Sensitivity"],
                        cm_rf$byClass["Sensitivity"],
                        cm_xgb$byClass["Sensitivity"],
                        cm_svm$byClass["Sensitivity"]), 4),
  Specificity = round(c(cm_lasso$byClass["Specificity"],
                        cm_rf$byClass["Specificity"],
                        cm_xgb$byClass["Specificity"],
                        cm_svm$byClass["Specificity"]), 4),
  Bal_Accuracy = round(c(cm_lasso$byClass["Balanced Accuracy"],
                         cm_rf$byClass["Balanced Accuracy"],
                         cm_xgb$byClass["Balanced Accuracy"],
                         cm_svm$byClass["Balanced Accuracy"]), 4)
)

print(perf_table %>% arrange(desc(AUC)))

























library(ggplot2)

dat <- data.frame (`Input.Type` = c("Small", "Small", "Large" , "Large"), 
                    `Time` = c(2.7, .67, 12.15, 4.06),
                    `Execution.Type` = c("Sequential", "Parallel", "Sequential", "Parallel") )

ggplot (dat, aes(x = as.factor(`Input.Type`), y = `Time`)) +
  geom_bar(aes(fill = as.factor(`Execution.Type`)), stat = 'identity', position = 'dodge') +
  labs(x = 'Input Type', fill = 'Execution Type')
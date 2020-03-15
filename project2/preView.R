csv_files <- list.files (path       = ".", 
                         pattern    = "*.csv", 
                         full.names = T)
library (data.table)
library(pastecs)
data.list <- lapply(csv_files, read.csv)
data.cat <- do.call(rbind, data.list)

stat.desc(data.cat)

library(data.table)
library(RTCGA)
library(RTCGA.clinical)

reshape <- function (df){
  colnames<-df[,1]
  tdf<- t(df[,-1])
  colnames(tdf) <- colnames
  return(tdf)
}
merge_normalization_opts <- function(opts = list()) {
  default_opts <- list()
  default_opts$row_center  <- FALSE
  default_opts$col_center <- FALSE
  default_opts$row_normalize <- FALSE
  default_opts$col_normalize <- FALSE
  modifyList(default_opts, opts)
}

normalize_data <- function(X, opts = list()) {
  opts <- merge_normalization_opts(opts)
  if(opts$row_center)
    X <- t(scale(t(X), center = TRUE, scale = FALSE))
  if(opts$row_normalize)
    X <- t(scale(t(X), center = FALSE, scale = TRUE))
  if(opts$col_center)
    X <- scale(X, center = TRUE, scale = FALSE)
  if(opts$col_normalize)
    X <- scale(X, center = FALSE, scale = TRUE)
  return (X)
}

#Download gene expression data:
#Version 2 of the adjusted pan-cancer gene expression data obtained from Synapse: 
#https://www.synapse.org/#!Synapse:syn4976369.2. 

setwd("/Users/breesheyroskams-hieter/Desktop/Uni_Edinburgh/VAEs_MDI_rotation/Qiu_et_al/analysis")
rna<- read.csv("../input_data/EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2-v2.geneExp.tsv",sep="\t")
rna_reshaped<- reshape(rna)
ind <- which(is.na(colSums(rna_reshaped)))
rna_reshaped_rmna<- rna_reshaped[,-ind]
rna_log <- apply(rna_reshaped_rmna+1,2,log)
data<- normalize_data(rna_log,opts = list(col_center=TRUE, col_normalize=TRUE))

# Export cleaned complete data with all NAs removed
if (!file.exists("rna_naremoved_logtransformed_normalized.csv")) {
  write.csv(data, "rna_naremoved_logtransformed_normalized.csv")
}

# Splitting by . and then taking the first three elements as the patient barcode and connecting with -
ps<-function(li){
  b=do.call(paste,as.list(c(li[1:3],sep='-')))
  return(b)
}
samplenames=rownames(data) # Updated this line as previously it did not return a character vector
a=strsplit(samplenames,"\\.")
barcode=sapply(a, ps)

# data variable currently has 11,069 rows (samples) and 17,176 columns (genes)
# data needs to be a dataframe or data table, cannot add column with $ to a matrix
data_df <- as.data.frame(data)
# Check
stopifnot(length(barcode) == nrow(data_df))
data_df$bcr_patient_barcode=barcode

# Extract TCGA Survival information
clin <- survivalTCGA(ACC.clinical,BLCA.clinical,BRCA.clinical,CESC.clinical,CHOL.clinical,
                     COAD.clinical,COADREAD.clinical,DLBC.clinical,ESCA.clinical,
                     FPPP.clinical,GBM.clinical,GBMLGG.clinical,HNSC.clinical,
                     KICH.clinical,KIPAN.clinical,KIRC.clinical,KIRP.clinical,
                     LAML.clinical,LGG.clinical,LIHC.clinical,LUAD.clinical,
                     LUSC.clinical,MESO.clinical,OV.clinical,PAAD.clinical,PCPG.clinical,
                     PRAD.clinical,READ.clinical,SARC.clinical,SKCM.clinical,STAD.clinical,
                     STES.clinical,TGCT.clinical,THCA.clinical,THYM.clinical,UCEC.clinical,
                     UCS.clinical,UVM.clinical,extract.cols="admin.disease_code")
# Merge clinical information with gene expression information by barcode
survdata=merge(data_df,clin, by='bcr_patient_barcode')
survdata=survdata[!duplicated(survdata[,1]),]
survdata=survdata[,c(1,17178:17180,3:17177)]
fwrite(survdata, "pancan_survdata.csv")

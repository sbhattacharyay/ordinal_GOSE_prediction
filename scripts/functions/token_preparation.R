categorizer <- function(curr.data, no.cuts) {
  for (i in 1:ncol(curr.data)){
    if (length(unique(curr.data[[i]])) <= no.cuts){
      curr.data[[i]] <- as.character(curr.data[[i]])
    } else {
      next
    }
  }
  return(curr.data)
}

categorical_tokenizer <- function(curr.data, startColIdx = 2, prefix = ''){
  for (i in startColIdx:ncol(curr.data)){
    pred.name <- names(curr.data)[i]
    curr.data[,i] <- paste0(prefix,pred.name,'_',curr.data[[i]])
  }
  return(curr.data)
}

build_recipe <- function(variable,NUM.CUTS){
  variable.rec <- variable %>%
    recipe(as.formula("~."), data = .) %>%
    update_role(GUPI, new_role = "id variable") %>%
    step_discretize(all_numeric_predictors(), options = list(cuts = NUM.CUTS,min_unique = 1, prefix = 'BIN', keep_na = T, infs = T, na.rm = T)) %>%
    prep(training = variable)
  return(variable.rec)
}

tf_variables <- function(variable,variable.rec,NUM.CUTS){
  tf.variable <- bake(variable.rec, new_data = variable) %>%
    mutate(GUPI = variable$GUPI) %>%
    relocate(GUPI) %>%
    mutate_if(is.factor, as.character) %>%
    categorizer(no.cuts = NUM.CUTS)
  return(tf.variable)
}

add.timestamp.event.tokens <- function(curr.tokens,variable,curr.GUPI){
  curr.variable <- variable %>%
    filter(GUPI == curr.GUPI)
  
  if (nrow(curr.variable) == 0){
    return(curr.tokens)
  }
  
  if (nrow(curr.variable) == 1){
    curr.variable[,!names(curr.variable) %in% c('Timestamp')] <- t(apply(curr.variable[,!names(curr.variable) %in% c('Timestamp')],2,function(x)gsub('\\s+', '',x)))
  } else {
    curr.variable[,!names(curr.variable) %in% c('Timestamp')] <- as.data.frame(apply(curr.variable[,!names(curr.variable) %in% c('Timestamp')],2,function(x)gsub('\\s+', '',x)))
  }
  
  curr.variable$Token <- do.call(paste, c(curr.variable %>% select(-c(GUPI,Timestamp)), sep=" "))
  curr.variable <- curr.variable %>%
    select(GUPI,Timestamp,Token) %>%
    distinct() %>%
    arrange(Timestamp)
  
  for (curr.row.idx in 1:nrow(curr.variable)){
    if (curr.variable$Timestamp[curr.row.idx] < curr.tokens$TimeStampStart[curr.tokens$WindowIdx == 1]){
      curr.tokens$Token[curr.tokens$WindowIdx == 1] <- paste(curr.tokens$Token[curr.tokens$WindowIdx == 1],
                                                             curr.variable$Token[curr.row.idx])
    } else {
      curr.window.idx <- curr.tokens$WindowIdx[which((curr.tokens$TimeStampStart <= curr.variable$Timestamp[curr.row.idx]) &
                                                       (curr.tokens$TimeStampEnd >= curr.variable$Timestamp[curr.row.idx]))]
      curr.tokens$Token[curr.tokens$WindowIdx == curr.window.idx] <- paste(curr.tokens$Token[curr.tokens$WindowIdx == curr.window.idx],
                                                                           curr.variable$Token[curr.row.idx])
    }
  }
  return(curr.tokens)
}

add.date.event.tokens <- function(curr.tokens,variable,curr.GUPI){
  curr.variable <- variable %>%
    filter(GUPI == curr.GUPI)
  
  if (nrow(curr.variable) == 0){
    return(curr.tokens)
  }
  
  if (nrow(curr.variable) == 1){
    curr.variable[,!names(curr.variable) %in% c('Date')] <- t(apply(curr.variable[,!names(curr.variable) %in% c('Date')],2,function(x)gsub('\\s+', '',x)))
  } else {
    curr.variable[,!names(curr.variable) %in% c('Date')] <- as.data.frame(apply(curr.variable[,!names(curr.variable) %in% c('Date')],2,function(x)gsub('\\s+', '',x)))
  }
  
  curr.variable$Token <- do.call(paste, c(curr.variable %>% select(-c(GUPI,Date)), sep=" "))
  curr.variable <- curr.variable %>%
    select(GUPI,Date,Token) %>%
    distinct() %>%
    arrange(Date)
  
  for (curr.row.idx in 1:nrow(curr.variable)){
    if (curr.variable$Date[curr.row.idx] < as.Date(curr.tokens$TimeStampStart[curr.tokens$WindowIdx == 1])){
      curr.tokens$Token[curr.tokens$WindowIdx == 1] <- paste(curr.tokens$Token[curr.tokens$WindowIdx == 1],
                                                             curr.variable$Token[curr.row.idx])
    } else {
      curr.window.idx <- curr.tokens$WindowIdx[which(as.Date(curr.tokens$TimeStampEnd) == (curr.variable$Date[curr.row.idx]))]
      curr.tokens$Token[curr.tokens$WindowIdx %in% curr.window.idx] <- paste(curr.tokens$Token[curr.tokens$WindowIdx %in% curr.window.idx],
                                                                             curr.variable$Token[curr.row.idx])
    }
  }
  
  return(curr.tokens)
}

add.timestamp.interval.tokens <- function(curr.tokens,variable,end.case.variables,curr.GUPI){
  curr.variable <- variable %>%
    filter(GUPI == curr.GUPI)
  
  if (nrow(curr.variable) == 0){
    return(curr.tokens)
  }
  
  if (nrow(curr.variable) == 1){
    curr.variable[,which(!names(curr.variable) %in% c('StartTimestamp','StopTimestamp'))] <- t(apply(curr.variable[,!names(curr.variable) %in% c('StartTimestamp','StopTimestamp')],2,function(x)gsub('\\s+', '',x)))
  } else {
    curr.variable[,!names(curr.variable) %in% c('StartTimestamp','StopTimestamp')] <- as.data.frame(apply(curr.variable[,!names(curr.variable) %in% c('StartTimestamp','StopTimestamp')],2,function(x)gsub('\\s+', '',x)))
  }
  
  curr.variable$Token <- do.call(paste, c(curr.variable[,!names(curr.variable) %in% end.case.variables] %>% select(-c(GUPI,StartTimestamp,StopTimestamp)), sep=" "))
  
  curr.variable <- curr.variable[,names(curr.variable) %in% c('GUPI','StartTimestamp','StopTimestamp','Token',end.case.variables)] %>%
    distinct() %>%
    arrange(StartTimestamp)
  
  for (curr.row.idx in 1:nrow(curr.variable)){
    if (curr.variable$StopTimestamp[curr.row.idx] < curr.tokens$TimeStampStart[curr.tokens$WindowIdx == 1]){
      curr.tokens$Token[curr.tokens$WindowIdx == 1] <- paste(curr.tokens$Token[curr.tokens$WindowIdx == 1],
                                                             curr.variable$Token[curr.row.idx])
    } else {
      curr.window.idx <- curr.tokens$WindowIdx[which((curr.tokens$TimeStampStart <= curr.variable$StopTimestamp[curr.row.idx]) &
                                                       (curr.tokens$TimeStampEnd >= curr.variable$StartTimestamp[curr.row.idx]))]
      curr.tokens$Token[curr.tokens$WindowIdx %in% curr.window.idx] <- paste(curr.tokens$Token[curr.tokens$WindowIdx %in% curr.window.idx],
                                                                             curr.variable$Token[curr.row.idx])
      if (end.case.variables[1] != ''){
        if (length(end.case.variables) == 1){
          curr.tokens$Token[curr.tokens$WindowIdx == as.character(max(as.integer(curr.window.idx)))] <- 
            paste(curr.tokens$Token[curr.tokens$WindowIdx == as.character(max(as.integer(curr.window.idx)))],
                  curr.variable[curr.row.idx,end.case.variables]) 
        } else {
          curr.tokens$Token[curr.tokens$WindowIdx == as.character(max(as.integer(curr.window.idx)))] <- 
            paste(curr.tokens$Token[curr.tokens$WindowIdx == as.character(max(as.integer(curr.window.idx)))],
                  do.call(paste, c(curr.variable[curr.row.idx,end.case.variables], sep=" "))) 
        }
      }
    }
  }
  return(curr.tokens)
}
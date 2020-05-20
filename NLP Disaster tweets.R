library(data.table)
library(tidyverse)
library(tidyr)
library(rex)
library(stringr)
library(ggplot2)
library(tm)
library(SnowballC)
library(wordcloud)
library(RColorBrewer)
library(dplyr)
library(tidytext)
library(keras)
library(tensorflow)
train_i = train
test_i = test
all_data =bind_rows(train_i,test_i)
all_data
all_data = data.table(all_data)
sum(is.na(all_data))
(colSums(sapply(all_data,is.na))/nrow(data))
#text and target doesnot have any missing values however location has 33% of missing values
unique(all_data['location'])
#Location data consists of unreliable data for example 'World wide','Somewhere only we know' etc.It does not appear to be a useful column
unique(data$keyword)#keyword has a slighter proportion of missing data and appears to contain relevant values
all_data[,"location":= NULL]#dropping location
colnames(all_data)
all_data = na.omit(all_data)
#keyword has a unique pattern in the text
pattern = all_data$keyword[grep("%20",data$keyword)]
all_data$keyword[grep('20',all_data$keyword)]
sum(table(pattern))/nrow(!is.na(data))
#15% of data has the pattern
str_remove_all(all_data$keyword,"%20" )#removing the pattern
#text and target doesnot have any na values now we can look to the text data
all_data$text
#Classification distribution
library(plyr)
count(all_data$target == 0)/nrow(all_data)            
ggplot(all_data, aes(fill = target, x = (target == 0),y = (target == 1)))+
  geom_bar(stack = 'count',stat = 'identity')
#higher number of false values 57% and 43% TRue values
wordcloud(all_data$keyword)
text = Corpus(VectorSource(all_data$text))#creates a collection of character vectors
text
inspect(text)
# Convert the text to lower case
text <- tm_map(text, content_transformer(tolower))
#removing characters
toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))
text <- tm_map(text, toSpace, "/")
text <- tm_map(text, toSpace, "@")
text <- tm_map(text, toSpace, "\\|")
text <- tm_map(text, toSpace, "http:*")
text <- tm_map(text, toSpace, "t.co")
text <- tm_map(text, toSpace, "https:*")
#removing stopwords
text <- tm_map(text, removeWords, stopwords("english"))
text <- tm_map(text,removeWords,c('the','like','just','now','get','via','will','get','new','now','can'))


#eliminating white spaces
text = tm_map(text,stripWhitespace)

#eiminate punctuations
text = tm_map(text,removePunctuation)
#stemming
text = tm_map(text,stemDocument)
inspect(text)

#convert to document matrix
dtm <- TermDocumentMatrix(text)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
d_top = head(d,20,sort(d$freq,decreasing = T))
ggplot(data = d_top,aes(x =reorder(word,freq),y = freq))+ geom_col()+coord_flip()+labs(x = "Word",y = 'Frequency of occurence')
#creating a word cloud
set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 1,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))
unnest_tokens(d_top,bigram,text, token = "ngrams", n = 2)
text
d_top
inspect(text)
tail(d,20)
#the least occuring words are gibberish and doesn't add any value
#extracting frequent words(threshold at 80)
#freq_words = d[d$freq >= 80,]
freq_words
#484 negative words and 215 positive words
str(senten)
senten <- d %>%
  inner_join(get_sentiments("bing"))
positive = senten[senten$sentiment == 'positive',]
negative = senten[senten$sentiment == 'negative',]
plotpos = head(positive,10)
plotneg = head(negative,10)
ggplot(plotpos,aes(x = reorder(word,freq),y =freq))+ geom_col(aes(fill = sentiment))+ coord_flip()+
  labs(x = 'Frequency',y = 'Word',title = 'Top 10 positive words')+scale_fill_manual(values = "dark green")
ggplot(plotneg,aes(x = reorder(word,freq),y = freq))+geom_col(aes(fill = sentiment))+ coord_flip()+
  labs(x = 'Frequency',y = 'Word',title = 'Top 10 negative words')+scale_fill_manual(values = "dark red")
#wordclouds
wordcloud(words = positive$word, freq = positive$freq, min.freq = 10,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))
wordcloud(words = negative$word, freq = negative$freq, min.freq = 10,
          max.words=200, random.order=FALSE, rot.per=0.35, colors=brewer.pal(10, "Dark2"))
inspect(text)
length(text)
train_data = text[0:7614]
test_data = text[7614:10876]
library(caret) 
set.seed(123)
#trainIndex <- createDataPartition(train_data,p=0.75,list=FALSE)
fdata = all_data
fdata$text = gsub('https:*',"",fdata$text)
fdata$text = gsub('http:*','',fdata$text)
fdata$text <- gsub('//','',fdata$text)

head(fdata)
text_i = fdata$text
target_i = fdata$target
final_data = data.frame(text_i,target_i)
smp_size <- floor(0.75 * nrow(final_data))
set.seed(123)
train_ind <- sample(seq_len(nrow(final_data)), size = smp_size)
train_train <- final_data[train_ind,]
test_train <- final_data[-train_ind,]

token = text_tokenizer(num_words = 10000)
token %>%
  fit_text_tokenizer(train_train$text_i)
token$word_counts
token$word_index
text_seqs <- texts_to_sequences(token, train_train$text_i) # Tokenize - i.e. convert text into a sequence of integers
text_seqs
x_train <- text_seqs %>% pad_sequences(maxlen = 100, padding = "post")
dim(x_train)
y_train <- train_train$target_i
length(y_train)
# Set parameters:
maxlen <- 120
batch_size <- 32
embedding_dims <- as.integer(16)
hidden_dims <- 16
epochs <- 10
input_dim = 10001
output_dim = 10001

model <- keras_model_sequential() %>% 
  layer_embedding(input_dim,output_dim, input_length = maxlen)%>%
  layer_dropout(0.3)%>%
  layer_global_max_pooling_1d() %>%
  layer_dense(hidden_dims)%>%
  layer_dropout(0.2) %>%
  layer_activation("relu") %>%
  layer_dense(1)%>%
  layer_activation("sigmoid")%>%
  compile(
    loss = "binary_crossentropy",
    optimizer = "adam",
    metrics = "accuracy"
  )

model %>% summary()
token
token %>%
  fit_text_tokenizer(test_train$text_i)
test_seqs <- texts_to_sequences(token,test_train$text_i) # Tokenize - i.e. convert text into a sequence of integers
x_test <- test_seqs %>% pad_sequences(maxlen = maxlen, padding = "post")
results <- model %>% evaluate(x_test,test_train$target_i)
results


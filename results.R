library(tidyverse)

results <- read_csv('D:/Active Research/cnn_lstm/results/results.csv')

results %>%
  group_by(arch,model,n_read,n_batch) %>%
  summarize(elapsed=median(elapsed),n_params=unique(n_params),gb=unique(gb)) %>%
  ggplot(aes(gb,elapsed,color=arch)) +
  geom_line() +
  facet_wrap(~model) +
  theme_classic() +
  labs(x='Memory (GB)',y='Median Second per Batch',color='') +
  theme(aspect.ratio=1)
ggsave('D:/Active Research/cnn_lstm/results/model_gb_v_s.png')

df <- results %>%
  group_by(arch,model,n_read,n_batch) %>%
  summarize(elapsed=median(elapsed)) %>%
  spread(arch,elapsed) 

wilcox.test(df$`/gpu:0`,df$`/cpu:0`,alternative='less',paired=TRUE)

results %>%
  ggplot(aes(arch,elapsed)) +
  geom_boxplot() +
  scale_y_log10() +
  labs(x='Memory (GB)',y='Log10 Second per Batch') +
  annotate('text', x = 2, y = 2, label = '*', color='red', size=10) +
  annotate('text', x = 2, y = 3.5, label = 'p < .001 (Mann-Whitney)', color='black', size=3) +
  theme_classic()
ggsave('D:/Active Research/cnn_lstm/results/arch_boxplot.png')

results %>%
  group_by(arch,model,n_read,n_batch) %>%
  mutate(epoch=row_number()) %>%
  select(epoch,arch,n_read,n_batch,model,elapsed) %>%
  spread(arch,elapsed) %>%
  split(list(.$model,.$n_read,.$n_batch),drop=TRUE) %>%
  map(~wilcox.test(.$`/gpu:0`,.$`/cpu:0`,alternative='less')$p.value) %>%
  as.data.frame() %>%
  gather() %>%
  rename(pvalue=value) %>%
  mutate(padj=p.adjust(pvalue,method='bonf')) %>%
  separate(key,into=c('model','n_reads','n_batches'),sep='\\.') %>%
  write_csv('D:/Active Research/cnn_lstm/results/pvals.csv')


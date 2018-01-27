library(tidyverse)

results <- read_csv('D:/Active Research/cnn_lstm/results/results.csv')

results %>%
  group_by(arch,model,n_read,n_batch) %>%
  summarize(elapsed=median(elapsed),n_params=unique(n_params),gb=unique(gb)) %>%
  ggplot(aes(gb,elapsed,color=arch)) +
  geom_point(alpha=.7) +
  stat_smooth(method='lm') +
  facet_wrap(~model,scales='free') +
  labs(x='Total Memory per Batch (GB)',y='Median Seconds per Epoch',color='') +
  theme(aspect.ratio=1) +
  scale_color_brewer(palette='Set1')
ggsave('D:/Active Research/cnn_lstm/results/model_gb_v_s.png',scale=2)

results %>%
  group_by(arch,model,n_read,n_batch) %>%
  summarize(elapsed=median(elapsed),n_params=unique(n_params),gb=unique(gb)) %>%
  spread(arch,elapsed) %>%
  mutate(ratio=`/cpu:0`/`/gpu:0`) %>%
  ggplot(aes(gb,ratio,color=model)) +
  geom_jitter(size=3,alpha=.7,width=1.1) +
  facet_grid(n_read~n_batch) +
  labs(x='Total Memory per Batch (GB)',y='cpu:gpu (Median Seconds per Epoch)',color='') +
  theme(aspect.ratio=1) +
  scale_color_brewer(palette='Set1')
ggsave('D:/Active Research/cnn_lstm/results/model_dot_ratio.png',scale=2)

df <- results %>%
  group_by(arch,model,n_read,n_batch) %>%
  summarize(elapsed=median(elapsed)) %>%
  spread(arch,elapsed) %>%
  mutate(ratio=`/cpu:0`/`/gpu:0`)
wilcox.test(df$ratio,alternative='greater')

results %>%
  ggplot(aes(arch,elapsed)) +
  geom_boxplot() +
  scale_y_log10() +
  labs(x='Memory (GB)',y='Log10 Second per Batch') +
  annotate('text', 
           x = 2, 
           y = results %>% filter(arch == '/gpu:0') %>% summarize(elapsed=max(elapsed)) %>% unlist() + 1, 
           label = '*', color='red', size=10) +
  annotate('text', 
           x = 2, 
           y = results %>% filter(arch == '/gpu:0') %>% summarize(elapsed=max(elapsed)) %>% unlist() + 3, 
           label = 'p < .0001 (Mann-Whitney)', color='black', size=3) +
  theme_classic()
ggsave('D:/Active Research/cnn_lstm/results/arch_boxplot.png',scale=2)

results %>%
  select(epoch,arch,n_read,n_batch,model,elapsed) %>%
  spread(arch,elapsed) %>%
  mutate(ratio=`/cpu:0`/`/gpu:0`) %>%
  split(list(.$model,.$n_read,.$n_batch),drop=TRUE) %>%
  map(~wilcox.test(.$ratio,alternative='greater')$p.value) %>%
  as.data.frame() %>%
  gather() %>%
  rename(pvalue=value) %>%
  mutate(padj=p.adjust(pvalue,method='bonf')) %>%
  separate(key,into=c('model','n_reads','n_batches'),sep='\\.') %>%
  write_csv('D:/Active Research/cnn_lstm/results/pvals.csv')


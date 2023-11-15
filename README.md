# Detecção de Avaliações Falsas em Português Utilizando Aprendizado de Máquina

## Sobre o Projeto
Este projeto foca na detecção de avaliações falsas (fake reviews) em plataformas de avaliação online, especificamente em português. Utilizamos técnicas avançadas de aprendizado de máquina, processamento de linguagem natural e análise de dados para identificar padrões e características que distinguem avaliações autênticas das fraudulentas.

### Contexto
Avaliações online têm um papel significativo na influência das decisões de compra dos consumidores. No entanto, o surgimento de avaliações falsas pode distorcer a percepção do consumidor e prejudicar a integridade das plataformas online. Este projeto visa abordar essa questão no contexto do idioma português, que é menos explorado na literatura científica sobre o tema.

### Objetivo
O objetivo principal deste projeto é desenvolver e avaliar modelos de aprendizado de máquina que possam identificar eficientemente avaliações falsas em português, contribuindo para a integridade e confiabilidade de plataformas online.

## Dataset
O dataset utilizado neste projeto foi coletado do site [Yelp](https://www.yelp.com/), focando em avaliações de bares e restaurantes no Brasil e em Portugal. 

### Estrutura do Dataset
O dataset inclui várias características, como conteúdo da avaliação, dados comportamentais dos usuários e metafeatures extraídas do texto das avaliações.

## Metodologia
- **Coleta de Dados:** Utilizamos técnicas de web scraping para coletar avaliações.
- **Pré-processamento:** Limpeza e organização dos dados, incluindo a remoção de stopwords, lematização e POS-tagging.
- **Extração de Características:** Exploração de características textuais e comportamentais.
- **Modelagem:** Implementação e teste de vários modelos de aprendizado de máquina, incluindo Random Forest, SVC, XGBoost, entre outros.
- **Validação Cruzada:** Utilização de validação cruzada de 5 folds para avaliar os modelos.

## Resultados
Os resultados mostram a eficácia dos modelos na distinção entre avaliações verdadeiras e falsas, com análises detalhadas disponíveis no repositório.

## Contribuições
Contribuições são sempre bem-vindas! Se você tem sugestões para melhorar este projeto, sinta-se à vontade para abrir um pull request ou uma issue.

## Contato
- **Lucas Percisi** - lucas_percisi@hotmail.com.br

---

Link para o Repositório: [https://github.com/lucaspercisi/yelp-fake-reviews-ptbr](https://github.com/lucaspercisi/yelp-fake-reviews-ptbr)


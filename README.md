# Rede Neural Perceptron Multicamadas

Este projeto implementa uma rede neural perceptron multicamadas com um algoritmo de retropropagação para treinamento. A interface permite ajustar os parâmetros da rede, carregar datasets e avaliar o desempenho por meio de uma matriz de confusão e estatísticas de acurácia.

---

## Funcionalidades

- **Configuração inicial da rede**:
  - Número de neurônios na camada oculta.
  - Valor de erro mínimo.
  - Número máximo de iterações.
  - Taxa de aprendizado.
  - Função de transferência (linear, logística ou hiperbólica).

- **Seleção do dataset**:
  - Carregue um dataset em formato CSV diretamente do seu computador.
  - Escolha entre:
    - **Subconjunto aleatório do dataset**: 70% dos dados são utilizados para treino e 30% para teste.
    - **Arquivo externo**: Use um arquivo separado como dados de teste.

- **Treinamento e Detecção de Platô**:
  - Ao clicar em "Resolver", o algoritmo começa o treinamento.
  - Caso um platô seja detectado, o programa exibe as seguintes opções:
    - **Continuar o treinamento**.
    - **Modificar a taxa de aprendizado** (valores entre 0 e 1).

- **Visualização**:
  - Um gráfico interativo é gerado ao final do treinamento, exibindo:
    - **Eixo X**: Épocas do aprendizado.
    - **Eixo Y**: Erro médio por época.

- **Testar o modelo**:
  - Clique em "Testar" após o treinamento para:
    - Gerar a matriz de confusão.
    - Exibir a acurácia geral e a acurácia por classe.

---

## Requisitos do Sistema

- **Node.js**: Certifique-se de que o [Node.js](https://nodejs.org) está instalado em sua máquina.
- **NPM** ou **Yarn**: Gerenciador de pacotes para instalar dependências.

---

## Instalação

1. Clone este repositório:
   ```bash
   git clone https://github.com/paulomcfm/rna.git
   ```

2. Navegue até o diretório do projeto:
   ```bash
   cd rna
   ```

3. Instale as dependências:
   ```bash
   npm install
   ```
   ou
   ```bash
   yarn install
   ```

---

## Uso

1. Inicie o servidor de desenvolvimento:
   ```bash
   npm start
   ```
   ou
   ```bash
   yarn start
   ```

2. Acesse o aplicativo em seu navegador:
   ```
   http://localhost:3000
   ```

---

## Fluxo de Uso

1. **Configuração**: Ajuste os parâmetros iniciais da rede, como número de neurônios na camada oculta, taxa de aprendizado, etc.
2. **Carregamento do Dataset**: Escolha um arquivo CSV e defina se deseja usar um subconjunto ou arquivo externo para teste.
3. **Treinamento**: Clique em "Resolver" para iniciar o treinamento. Caso detecte um platô, siga as instruções exibidas.
4. **Visualização do Gráfico**: Analise o gráfico gerado para observar a evolução do erro médio por época.
5. **Teste do Modelo**: Clique em "Testar" para verificar a matriz de confusão e as estatísticas de acurácia.

---

## Exemplo de Dataset

O dataset deve estar no formato CSV com cabeçalhos, como no exemplo abaixo:

```csv
X1,X2,X3,X4,X5,X6,classe
0.0286,0.1233,0.2987,0.1493,0.0435,0.0411,CA
0.0586,0.1333,0.3187,0.1393,0.0635,0.0511,CB
```

---
## Tecnologias Utilizadas

- **React.js**: Biblioteca principal para a interface do usuário.
- **React-Bootstrap**: Componentes de UI prontos para uso.
- **Chart.js**: Geração de gráficos interativos.
- **PapaParse**: Parser de arquivos CSV.

---

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

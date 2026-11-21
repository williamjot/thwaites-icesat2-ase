# Fundamentação Teórica, Modelos Físicos e Rastreabilidade Estatística

Este documento reúne a formulação matemática e as referências bibliográficas primárias de todos os modelos físicos e testes estatísticos implementados no pacote `thwaites`.

---

## 1. Modelos Físicos e Altimétricos

### 1.1 Elevação Altimétrica Corrigida e Referenciação Topográfica
Para cada segmento de $40\text{ m}$ do produto ICESat-2 ATL06 r007 (Smith et al., 2019), a elevação superficial corrigida $h_{\text{corr}}$ e o resíduo topográfico estático $h_{\text{res}}$ em relação ao modelo digital de elevação REMA v2 ($32\text{ m}$; Howat et al., 2019) são dados por:

$$h_{\text{corr}}(x, y, t) = h_{\text{LI}}(x, y, t) - \mathbb{I}_{\text{flut}}(x, y)\,\eta_{\text{CATS}}(x, y, t) - \eta_{\text{DAC}}(x, y, t)$$

$$h_{\text{res}}(x, y, t) = h_{\text{corr}}(x, y, t) - h_{\text{REMA}}(x, y)$$

onde $\eta_{\text{CATS}}$ é a maré oceânica predita pelo modelo regional antártico CATS2008 v2023 (Padman et al., 2002; Howard et al., 2024) e $\eta_{\text{DAC}}$ é a correção atmosférica dinâmica. Como $\partial_t h_{\text{REMA}} \equiv 0$, a diferenciação topográfica remove a variância espacial associada à declividade local sem alterar a derivada temporal $\partial_t h_{\text{res}} = \partial_t h_{\text{corr}}$.

### 1.2 Equilíbrio Hidrostático e Flexão Elástica na Zona de Aterramento
Sobre plataformas de gelo em flutuação livre, a altura da superfície acima do nível médio do mar local $h_{\text{mar}}$ relaciona-se à espessura total equivalente à coluna de gelo com ar de firn por:

$$h_{\text{mar}} = h_{\text{LI}} - \left(\eta_{\text{CATS}} + \eta_{\text{DAC}}\right) - \left(N_{\text{EIGEN-6C4}} + \text{MDT}_{\text{DTU15}}\right)$$

$$H = \left(h_{\text{mar}} - \text{FAC}\right) \frac{\rho_w}{\rho_w - \rho_i}, \qquad \frac{\rho_w}{\rho_w - \rho_i} = \frac{1027}{1027 - 917} \approx 9{,}336$$

onde $\text{FAC}$ é o conteúdo de ar do firn (*Firn Air Content*, proveniente do GEMB v1.5 ou IMAU-FDM v1.2A). Na zona de aterramento ($0 \le d \le d^*$), tensões de flexão elástica acoplam a plataforma ao gelo apoiado segundo a equação da viga elástica sobre fundação fluida (Vaughan, 1995; Holdsworth, 1969):

$$D \frac{d^4 w}{dx^4} + \rho_w g w = \rho_w g \eta_{\text{maré}}(t), \qquad D = \frac{E H^3}{12(1 - \nu^2)}$$

A solução para uma viga semi-infinita engastada em $x = 0$ fornece o fator de amortecimento de maré $\beta(x) = w(x)/\eta_{\text{maré}}$:

$$\beta(x) = 1 - e^{-\lambda x}\left(\cos \lambda x + \sin \lambda x\right), \qquad \lambda = \left(\frac{\rho_w g}{4 D}\right)^{1/4}$$

### 1.3 Conservação de Massa Lagrangiana em Plataformas Flutuantes
A equação da continuidade integrada verticalmente para uma coluna de gelo flutuante de espessura $H$ e velocidade horizontal média na coluna $\mathbf{u} = (u, v)$ é:

$$\frac{\partial H}{\partial t} + \nabla\cdot(H\mathbf{u}) = \frac{\partial H}{\partial t} + \mathbf{u}\cdot\nabla H + H(\nabla\cdot\mathbf{u}) = \text{SMB} - m_b$$

Seguindo a trajetória material da parcela $\frac{d\mathbf{X}}{dt} = \mathbf{u}(\mathbf{X}, t)$, a derivada material $DH/Dt = \partial_t H + \mathbf{u}\cdot\nabla H$ já incorpora a advecção de gradientes de espessura $\mathbf{u}\cdot\nabla H$. Portanto, a taxa de derretimento basal $m_b$ (positiva para perda basal) em referencial Lagrangiano envolve exclusivamente o termo de espalhamento $H(\nabla\cdot\mathbf{u})$:

$$m_b = \text{SMB} - \frac{DH}{Dt} - H(\nabla\cdot\mathbf{u})$$

---

## 2. Rastreabilidade Teórica dos Estimadores e Testes Estatísticos

### 2.1 Estimação Espaço-Temporal Local Robusta (`fitsec`)
- **Módulo:** `thwaites.timeseries.dhdt`
- **Modelo:** Dentro de um raio de busca $R = 15\text{ km}$ em torno de cada nó $(x_0, y_0)$, ajusta-se por mínimos quadrados ponderados ($w_i = \sigma_i^{-2}$):
  $$h_{\text{res}}(x_i, y_i, t_i) = \beta_0 + \beta_1(t_i - t_{\text{ref}}) + \frac{1}{2}\beta_2(t_i - t_{\text{ref}})^2 + c_1 \tilde{x}_i + c_2 \tilde{y}_i + c_3 \tilde{x}_i\tilde{y}_i + c_4 \tilde{x}_i^2 + c_5 \tilde{y}_i^2 + \varepsilon_i$$
- **Por que funciona:** A rejeição iterativa de outliers utiliza o desvio absoluto mediano reescalado $\hat{\sigma}_{\text{MAD}} = 1{,}4826 \cdot \text{mediana}(|r_i - \text{mediana}(r)|)$, que possui ponto de ruptura de $50\%$ (Hampel, 1974), impedindo que retornos espúrios de nuvens baixas ou fendas distorçam a inclinação temporal $\beta_1 = \text{dh/dt}$.
- **Referências:** Smith et al. (2019, *Remote Sens. Environ.*, doi:10.1016/j.rse.2019.111352); Smith et al. (2020, *Science*, doi:10.1126/science.aaz5845); Hampel (1974, *JASA*).

### 2.2 Jackknife Agrupado por Ano (*Leave-One-Year-Out*) e por Grupo de Passagens (*DAGJK*)
- **Módulo:** `thwaites.stats.resampling` e `thwaites.timeseries.uncertainty`
- **Formulação:**
  $$\text{Var}_{\text{JK}}(\hat{\theta}) = \frac{k - 1}{k}\sum_{y=1}^{k}\left(\hat{\theta}_{(-y)} - \bar{\theta}_{(\cdot)}\right)^2$$
- **Por que funciona:** O erro-padrão formal de mínimos quadrados assume que os $N \sim 10^4$ segmentos dentro de um nó são independentes, subestimando a incerteza da tendência em $\sim 92\times$ no gelo apoiado e $\sim 13\times$ nas plataformas. Como a alavancagem temporal reside nas $k \approx 7$ épocas anuais (ou nas passagens orbitais `(rgt, cycle)`), remover um cluster temporal inteiro por vez captura toda a covariância intra-época (Tukey, 1958; Efron & Stein, 1981; Kott, 2001).
- **Referências:** Tukey (1958, *Ann. Math. Stat.*); Efron & Stein (1981, *Ann. Stat.*, doi:10.1214/aos/1176345462); Kott (2001, *J. Off. Stat.*).

### 2.3 Testes de Tendência de Mann–Kendall, Mann–Kendall Sazonal, Theil–Sen e Controle FDR
- **Módulo:** `thwaites.stats.trend_tests` e `thwaites.timeseries.trend`
- **Por que funciona:** O teste de Mann–Kendall avalia a soma dos sinais das diferenças par a par $S = \sum_{i<j}\text{sgn}(y_j - y_i)$, não exigindo gaussianidade dos resíduos. A variante sazonal (Hirsch, Slack & Smith, 1982) restringe as comparações a pares do mesmo mês/estação ($\sum_{m=1}^{M} S_m$), removendo a sazonalidade sem viés de fase. O procedimento de Benjamini–Hochberg (1995) controla a taxa esperada de falsas descobertas ($\text{FDR} \le \alpha$) nos $7.231$ nós espaciais sob dependência positiva PRDS (Benjamini & Yekutieli, 2001).
- **Referências:** Mann (1945, *Econometrica*); Kendall (1975); Sen (1968, *JASA*); Hirsch, Slack & Smith (1982, *Water Resour. Res.*); Benjamini & Hochberg (1995, *JRSS-B*); Benjamini & Yekutieli (2001, *Ann. Stat.*).

### 2.4 Seleção Multi-Critério de Aceleração e Block Bootstrap por Equações Normais
- **Módulo:** `thwaites.timeseries.acceleration` e `thwaites.stats.resampling`
- **Por que funciona:** Em registros de 7 anos, anomalias interanuais autocorrelacionadas de precipitação/firn produzem curvatura espúria quando testadas apenas pelo erro formal ou $\text{AIC}_c$. O algoritmo exige simultaneamente: (i) $\Delta\text{AIC}_c \ge 2$ (Hurvich & Tsai, 1989); (ii) redução do erro fora da amostra em validação *leave-one-year-out*; (iii) autocorrelação residual de lag 1 $|r_1| \le 0{,}5$; (iv) intervalo de confiança de $95\%$ por *block bootstrap* de anos inteiros que não inclua zero; e (v) estabilidade de sinal em $\ge 95\%$ das réplicas.
- **Referências:** Hurvich & Tsai (1989, *Biometrika*); Carlstein (1986, *Ann. Stat.*).

### 2.5 Eliminação por Complemento de Schur (Frisch–Waugh–Lovell) e Variância Sanduíche CR1
- **Módulo:** `thwaites.stats.panel_schur` e `pipelines/derretimento_basal/run_tide_response_lagrangian.py`
- **Formulação:** Para o modelo conjunto $h_i = \mathbf{x}_{b(i)}^T\boldsymbol{\alpha}_{b(i)} + \mathbf{z}_{c(i)}^T\boldsymbol{\beta} + \varepsilon_i$, a eliminação analítica da topografia móvel $\boldsymbol{\alpha}_b$ produz o sistema reduzido exato:
  $$\left(\mathbf{Z}^T\mathbf{Z} - \sum_{b}\mathbf{V}_b^T\mathbf{A}_b^{-1}\mathbf{V}_b\right)\hat{\boldsymbol{\beta}} = \mathbf{Z}^T\mathbf{h} - \sum_{b}\mathbf{V}_b^T\mathbf{A}_b^{-1}\mathbf{X}_b^T\mathbf{h}$$
  com variância agrupada por passagem orbital $g \in \{1, \dots, G\}$:
  $$\widehat{\text{Var}}_{\text{CR1}}(\hat{\boldsymbol{\beta}}) = \frac{G}{G - 1}\mathbf{H}^{-1}\left(\sum_{g=1}^{G}\tilde{\mathbf{z}}_g\tilde{\mathbf{z}}_g^T\right)\mathbf{H}^{-1}$$
- **Por que funciona:** Resolver topografia e maré iterativamente (*backfitting*) falha em convergir em plataformas rápidas e atenua $\hat{\beta}$ porque a tendência e o ciclo anual absorvem o alias da maré na órbita de 91 dias. O complemento de Schur resolve o sistema conjunto em passo único exato.
- **Referências:** Frisch & Waugh (1933, *Econometrica*); Lovell (1963, *JASA*); Liang & Zeger (1986, *Biometrika*); Cameron & Miller (2015, *J. Hum. Resour.*).

### 2.6 Validação Cruzada Espacial em Blocos e Comprimento de Correlação do Erro ($L$)
- **Módulo:** `thwaites.stats.spatial_cv`, `thwaites.interp.select` e `thwaites.uncertainty.error_correlation`
- **Por que funciona:** Em campos espaciais autocorrelacionados, o *Spatial Block CV* (Roberts et al., 2017) retém blocos contíguos inteiros, impedindo vazamento por vizinhança imediata e permitindo selecionar objetivamente entre Krigagem Ordinária, Interpolação Ótima/Markov, IDW e Kernels com base no RMSE e na calibração $z_{\text{std}} = \text{SD}[(z_{\text{pred}} - z_{\text{obs}})/\hat{\sigma}] \approx 1$ (Cressie, 1993). A propagação da incerteza para a massa integrada utiliza a área efetiva de correlação $A_{\text{corr}} = \pi L^2$ estimada sobre os resíduos fora da amostra (Rolstad, Haug & Denby, 2009).
- **Referências:** Roberts et al. (2017, *Ecography*, doi:10.1111/ecog.02881); Cressie (1993); Rolstad, Haug & Denby (2009, *J. Glaciol.*, doi:10.3189/002214309788608859).

### 2.7 Derivada Gaussiana Normalizada e Validação Cruzada Anos Pares vs. Ímpares
- **Módulo:** `thwaites.stats.spatial_cv` e `pipelines/derretimento_basal/run_divergence_ase.py`
- **Por que funciona:** A convolução normalizada (Knutsson & Westin, 1993) diferencia o quociente $(G_\sigma * (wf))/(G_\sigma * w)$, evitando artefatos de borda junto a lacunas de velocidade. Como os ruídos de correlação de imagem dos mosaicos anuais ITS_LIVE são independentes entre anos pares e ímpares, minimizar $\text{RMS}(\nabla\cdot\mathbf{u}_{\sigma,\text{par}} - \nabla\cdot\mathbf{u}_{\sigma_0,\text{ímpar}})$ seleciona a escala $\sigma^*$ que minimiza o erro quadrático médio contra o campo verdadeiro sem necessidade de escolha subjetiva.
- **Referências:** Knutsson & Westin (1993, *IEEE CVPR*).

### 2.8 Testes de Variabilidade: $Q/\chi^2$ de Cochran, Graus de Liberdade Efetivos e Substitutas de Fourier
- **Módulo:** `thwaites.stats.time_series_tests`
- **Por que funciona:**
  1. O teste $Q = \sum_{t=1}^K (y_t - \hat{\mu}_w)^2/\sigma_t^2 \sim \chi^2_{K-1}$ (Cochran, 1954) avalia se as médias anuais calculadas sobre janelas disjuntas de 4 estações variam além do erro estatístico de cada ano.
  2. Para séries trimestrais suavizadas em 4 trimestres, a autocorrelação serial reduz severamente os graus de liberdade ($N_{\text{ef}}$ de Bretherton et al., 1999 e Pyper & Peterman, 1998). O teste não-paramétrico com $2.000$ séries substitutas de fase de Fourier aleatória (Theiler et al., 1992; Ebisuzaki, 1997) preserva exatamente a densidade espectral de potência das anomalias ao testar a hipótese nula de ausência de acoplamento com o ENSO (ONI) ou entre plataformas.
- **Referências:** Cochran (1954, *Biometrics*); Bretherton et al. (1999, *J. Climate*); Pyper & Peterman (1998, *Can. J. Fish. Aquat. Sci.*); Theiler et al. (1992, *Physica D*); Ebisuzaki (1997, *J. Climate*).

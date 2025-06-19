#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ================================
# 1. Configuração da Página Streamlit
# ================================
st.set_page_config(page_title="Análise de Consumo de Energia - Indústria Siderúrgica", layout="wide")
st.title("⚙️ Análise de Consumo de Energia na Indústria Siderúrgica")

# ================================
# 2. Carregamento e preparação dos dados
# ================================
df = pd.read_csv('Steel_industry_data.csv')
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df = df.set_index('date')

st.subheader("🔍 Visão Geral dos Dados")
st.dataframe(df.head(5))

# ================================
# 3. Série temporal do consumo de energia
# ================================
st.subheader("📈 Série Temporal do Consumo de Energia")

fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(df['Usage_kWh'], label='Consumo de Energia (kWh)', color='blue')
mean_usage = df['Usage_kWh'].mean()
ax1.axhline(y=mean_usage, color='red', linestyle='--', label='Média de Consumo')
ax1.set_title("Série Temporal do Consumo de Energia com Média")
ax1.set_xlabel("Data")
ax1.set_ylabel("Consumo de Energia (kWh)")
ax1.legend()
st.pyplot(fig1)

# ================================
# 4. Clusterização com K-Means
# ================================
st.subheader("🧪 Clusterização de Perfis de Consumo")

# Colunas com os nomes corretos
features = [
    'Usage_kWh',
    'Lagging_Current_Reactive.Power_kVarh',
    'Leading_Current_Reactive_Power_kVarh',
    'CO2(tCO2)',
    'Lagging_Current_Power_Factor',
    'Leading_Current_Power_Factor'
]

# Verificar se colunas existem
missing_cols = [col for col in features if col not in df.columns]
if missing_cols:
    st.error(f"As seguintes colunas estão faltando no CSV: {missing_cols}")
    st.stop()

# Dados para clusterização
data_cluster = df[features].dropna()

# Normalizar os dados
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_cluster)

# PCA para visualização
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

# Clusterização com KMeans
k = st.slider("Selecione o número de clusters", 2, 6, 3)
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(data_scaled)

# DataFrame com clusters
df_clustered = data_cluster.copy()
df_clustered['Cluster'] = clusters
df_clustered['PCA1'] = data_pca[:, 0]
df_clustered['PCA2'] = data_pca[:, 1]

# Visualização
fig2, ax2 = plt.subplots(figsize=(8, 5))
sb.scatterplot(data=df_clustered, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', ax=ax2)
ax2.set_title("Visualização dos Clusters (PCA)")
st.pyplot(fig2)

# ================================
# 5. Insights por Cluster
# ================================
st.subheader("📊 Insights por Cluster")
cluster_means = df_clustered.groupby('Cluster').mean(numeric_only=True)[features]
st.dataframe(cluster_means.style.format("{:.2f}"))

for c in cluster_means.index:
    st.markdown(f"**🔹 Cluster {c}:**")
    high_vars = cluster_means.loc[c][cluster_means.loc[c] > cluster_means.mean() + cluster_means.std()]
    low_vars = cluster_means.loc[c][cluster_means.loc[c] < cluster_means.mean() - cluster_means.std()]
    
    if not high_vars.empty:
        st.markdown(f"- **Altos valores em**: {', '.join(high_vars.index)}")
    if not low_vars.empty:
        st.markdown(f"- **Baixos valores em**: {', '.join(low_vars.index)}")
    if high_vars.empty and low_vars.empty:
        st.markdown("- Perfil médio, sem variáveis muito discrepantes.")

# ================================
# 6. Detecção de Anomalias
# ================================
st.subheader("🚨 Detecção de Anomalias (Picos de Consumo)")
threshold = mean_usage + 2 * df['Usage_kWh'].std()
anomalias = df[df['Usage_kWh'] > threshold]
st.write(f"Quantidade de anomalias detectadas: **{len(anomalias)}**")
st.dataframe(anomalias[['Usage_kWh']].head(10))

# ================================
# 7. Correlação entre Variáveis
# ================================
st.subheader("📌 Correlação entre Variáveis")
correlacoes = df.corr(numeric_only=True)
fig3, ax3 = plt.subplots(figsize=(10, 6))
sb.heatmap(correlacoes, annot=True, cmap='coolwarm', ax=ax3)
st.pyplot(fig3)

st.markdown("**Correlação com `Usage_kWh`:**")
st.dataframe(correlacoes['Usage_kWh'].sort_values(ascending=False).to_frame())

# ================================
# 8. Conclusões
# ================================
st.subheader("📝 Conclusões e Recomendações")
st.markdown("""
- Clusters permitem identificar grupos distintos de comportamento energético.
- Os dias com consumo anômalo devem ser investigados por possíveis falhas ou excessos operacionais.
- A correlação entre variáveis ajuda a prever o consumo com base em indicadores como o fator de potência ou emissão de CO2.
- Recomenda-se monitorar continuamente os clusters e ajustar operações conforme o perfil de cada grupo.
""")

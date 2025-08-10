#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZVT_DATA_MAPPER.py - Mapeamento e Visualização dos Dados de Ressonâncias
Author: Jefferson M. Okushigue
Date: 2025-08-10
Script independente para gerar mapas e gráficos a partir dos dados salvos
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import os
import json
from datetime import datetime
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import networkx as nx
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")

# Configuração científica para plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Constantes para mapeamento
FUNDAMENTAL_FORCES = {
    'eletromagnetica': 1 / 137.035999084,
    'forte': 0.1185,
    'fraca': 0.0338,
    'gravitacional': 5.906e-39,
    'weinberg_angle': 0.2312,
    'proton_electron': 1836.15267343,
    'euler_mascheroni': 0.5772156649,
    'fermi_coupling': 1.1663787e-5,
    'muon_electron': 206.7682826,
    'tau_electron': 3477.15,
    'neutron_proton': 1.00137841931,
    'dark_energy': 0.6847,
    'dark_matter': 0.2589,
    'baryon_density': 0.0486,
    'hubble_reduced': 0.6736,
    'sigma8': 0.8111,
    'gyromagnetic_proton': 2.7928473508,
    'gyromagnetic_neutron': 1.9130427,
    'magnetic_moment_ratio': 3.1524512605
}

# Categorias físicas para colorização
PHYSICS_CATEGORIES = {
    'Forças Fundamentais': ['eletromagnetica', 'forte', 'fraca', 'gravitacional'],
    'Unificação Eletrofraca': ['weinberg_angle', 'fermi_coupling'],
    'Razões de Massa': ['proton_electron', 'muon_electron', 'tau_electron', 'neutron_proton'],
    'Cosmologia': ['dark_energy', 'dark_matter', 'baryon_density', 'hubble_reduced', 'sigma8'],
    'Magnéticas': ['gyromagnetic_proton', 'gyromagnetic_neutron', 'magnetic_moment_ratio'],
    'Matemáticas': ['euler_mascheroni']
}

# Cores por categoria
CATEGORY_COLORS = {
    'Forças Fundamentais': '#FF6B6B',
    'Unificação Eletrofraca': '#4ECDC4', 
    'Razões de Massa': '#45B7D1',
    'Cosmologia': '#96CEB4',
    'Magnéticas': '#FECA57',
    'Matemáticas': '#DDA0DD'
}

class ZVTDataMapper:
    def __init__(self, cache_file="zeta_zeros_cache.pkl", results_dir="zvt_constants_results"):
        self.cache_file = cache_file
        self.results_dir = results_dir
        self.maps_dir = os.path.join(results_dir, "maps")
        self.data = None
        self.resonances_df = None
        
        # Criar diretório de mapas
        os.makedirs(self.maps_dir, exist_ok=True)
        
        print("🗺️ ZVT DATA MAPPER - Visualização Científica")
        print("=" * 60)
        
    def load_data(self):
        """Carrega dados do cache e constrói DataFrame de ressonâncias"""
        print("📂 Carregando dados...")
        
        # Carregar zeros do cache
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.zeros = pickle.load(f)
            print(f"✅ {len(self.zeros):,} zeros carregados do cache")
        else:
            print("❌ Cache não encontrado!")
            return False
            
        # Construir DataFrame com as melhores ressonâncias
        self.build_resonances_dataframe()
        return True
    
    def build_resonances_dataframe(self):
        """Constrói DataFrame com as melhores ressonâncias encontradas"""
        print("🔬 Construindo DataFrame de ressonâncias...")
        
        resonances_data = []
        
        # Para cada constante, encontrar sua melhor ressonância
        for const_name, const_value in FUNDAMENTAL_FORCES.items():
            best_quality = float('inf')
            best_resonance = None
            
            # Buscar melhor ressonância para esta constante
            for n, gamma in self.zeros:
                mod_val = gamma % const_value
                min_distance = min(mod_val, const_value - mod_val)
                
                if min_distance < best_quality:
                    best_quality = min_distance
                    best_resonance = {
                        'constant': const_name,
                        'constant_value': const_value,
                        'zero_index': n,
                        'gamma': gamma,
                        'quality': min_distance,
                        'error_percent': (min_distance / const_value) * 100,
                        'energy_gev': gamma / 10,
                        'log_quality': np.log10(min_distance),
                        'log_constant': np.log10(const_value),
                        'category': self.get_category(const_name)
                    }
            
            if best_resonance:
                resonances_data.append(best_resonance)
        
        self.resonances_df = pd.DataFrame(resonances_data)
        print(f"✅ DataFrame construído com {len(self.resonances_df)} ressonâncias")
        
    def get_category(self, const_name):
        """Retorna categoria física da constante"""
        for category, constants in PHYSICS_CATEGORIES.items():
            if const_name in constants:
                return category
        return 'Outras'
    
    def map_energy_vs_quality(self):
        """Mapa: Energia vs Qualidade das Ressonâncias"""
        print("🎯 Gerando mapa Energia vs Qualidade...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Gráfico 1: Linear
        for category in PHYSICS_CATEGORIES.keys():
            cat_data = self.resonances_df[self.resonances_df['category'] == category]
            if not cat_data.empty:
                ax1.scatter(cat_data['energy_gev'], cat_data['quality'], 
                           c=CATEGORY_COLORS.get(category, 'gray'), 
                           s=100, alpha=0.7, label=category, edgecolors='black')
        
        ax1.set_xlabel('Energia (GeV)', fontsize=12)
        ax1.set_ylabel('Qualidade da Ressonância', fontsize=12)
        ax1.set_title('Mapa Energia vs Qualidade (Linear)', fontsize=14, fontweight='bold')
        ax1.set_yscale('log')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Log-Log
        for category in PHYSICS_CATEGORIES.keys():
            cat_data = self.resonances_df[self.resonances_df['category'] == category]
            if not cat_data.empty:
                ax2.scatter(cat_data['energy_gev'], cat_data['quality'], 
                           c=CATEGORY_COLORS.get(category, 'gray'), 
                           s=100, alpha=0.7, label=category, edgecolors='black')
        
        ax2.set_xlabel('Energia (GeV)', fontsize=12)
        ax2.set_ylabel('Qualidade da Ressonância', fontsize=12)
        ax2.set_title('Mapa Energia vs Qualidade (Log-Log)', fontsize=14, fontweight='bold')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Adicionar anotações para pontos notáveis
        for _, row in self.resonances_df.head(5).iterrows():
            ax1.annotate(row['constant'], (row['energy_gev'], row['quality']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        filename = os.path.join(self.maps_dir, "energy_vs_quality_map.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Salvo: {filename}")
        
    def map_hierarchy_tree(self):
        """Mapa hierárquico das ressonâncias por qualidade"""
        print("🌳 Gerando mapa hierárquico...")
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Ordenar por qualidade
        sorted_df = self.resonances_df.sort_values('quality')
        
        # Criar gráfico de barras horizontal
        y_pos = np.arange(len(sorted_df))
        colors = [CATEGORY_COLORS.get(cat, 'gray') for cat in sorted_df['category']]
        
        bars = ax.barh(y_pos, -sorted_df['log_quality'], color=colors, alpha=0.8, edgecolor='black')
        
        # Customizar
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_df['constant'], fontsize=10)
        ax.set_xlabel('Log₁₀(Qualidade) - Melhor →', fontsize=12)
        ax.set_title('Hierarquia de Qualidade das Ressonâncias', fontsize=16, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Adicionar valores
        for i, (_, row) in enumerate(sorted_df.iterrows()):
            ax.text(-row['log_quality'] + 0.5, i, f"{row['error_percent']:.2e}%", 
                   va='center', fontsize=8, fontweight='bold')
        
        # Legenda por categoria
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, edgecolor='black') 
                          for color in CATEGORY_COLORS.values()]
        ax.legend(legend_elements, CATEGORY_COLORS.keys(), 
                 loc='lower right', fontsize=10)
        
        plt.tight_layout()
        filename = os.path.join(self.maps_dir, "hierarchy_tree_map.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Salvo: {filename}")
        
    def map_energy_landscape(self):
        """Mapa do landscape energético"""
        print("⚡ Gerando landscape energético...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Gráfico 1: Distribuição de energia
        energies = self.resonances_df['energy_gev']
        
        ax1.hist(energies, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(energies.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Média: {energies.mean():.0f} GeV')
        ax1.axvline(energies.median(), color='orange', linestyle='--', linewidth=2,
                   label=f'Mediana: {energies.median():.0f} GeV')
        
        ax1.set_xlabel('Energia (GeV)', fontsize=12)
        ax1.set_ylabel('Número de Ressonâncias', fontsize=12)
        ax1.set_title('Distribuição de Energias das Ressonâncias', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfico 2: Energia vs Índice do Zero
        colors = [CATEGORY_COLORS.get(cat, 'gray') for cat in self.resonances_df['category']]
        
        ax2.scatter(self.resonances_df['zero_index'], self.resonances_df['energy_gev'], 
                   c=colors, s=100, alpha=0.7, edgecolors='black')
        
        ax2.set_xlabel('Índice do Zero na Sequência', fontsize=12)
        ax2.set_ylabel('Energia (GeV)', fontsize=12)
        ax2.set_title('Energia vs Posição na Sequência de Zeros', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Identificar zona de alta concentração
        high_energy_zone = energies.quantile(0.75)
        ax2.axhline(high_energy_zone, color='red', linestyle=':', alpha=0.7,
                   label=f'Zona Alta Energia: {high_energy_zone:.0f} GeV')
        ax2.legend()
        
        plt.tight_layout()
        filename = os.path.join(self.maps_dir, "energy_landscape_map.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Salvo: {filename}")
        
    def map_physics_network(self):
        """Mapa de rede das relações físicas"""
        print("🌐 Gerando rede de relações físicas...")
        
        fig, ax = plt.subplots(figsize=(14, 14))
        
        # Criar grafo
        G = nx.Graph()
        
        # Adicionar nós (constantes)
        for _, row in self.resonances_df.iterrows():
            G.add_node(row['constant'], 
                      category=row['category'],
                      quality=row['quality'],
                      energy=row['energy_gev'])
        
        # Adicionar arestas baseadas em similaridade de energia
        for i, row1 in self.resonances_df.iterrows():
            for j, row2 in self.resonances_df.iterrows():
                if i < j:  # Evitar duplicatas
                    energy_diff = abs(row1['energy_gev'] - row2['energy_gev'])
                    if energy_diff < 50000:  # Threshold de similaridade
                        weight = 1 / (energy_diff + 1)  # Peso inversamente proporcional à distância
                        G.add_edge(row1['constant'], row2['constant'], weight=weight)
        
        # Layout do grafo
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Desenhar nós por categoria
        for category, color in CATEGORY_COLORS.items():
            nodes_in_cat = [node for node in G.nodes() 
                           if G.nodes[node]['category'] == category]
            if nodes_in_cat:
                nx.draw_networkx_nodes(G, pos, nodelist=nodes_in_cat, 
                                     node_color=color, node_size=1000, 
                                     alpha=0.8, ax=ax)
        
        # Desenhar arestas
        nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', ax=ax)
        
        # Desenhar labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)
        
        # Customizar
        ax.set_title('Rede de Relações entre Constantes Físicas\n(Conectadas por Proximidade Energética)', 
                    fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Legenda
        legend_elements = [plt.scatter([], [], c=color, s=100, alpha=0.8, edgecolors='black') 
                          for color in CATEGORY_COLORS.values()]
        ax.legend(legend_elements, CATEGORY_COLORS.keys(), 
                 loc='upper left', fontsize=10)
        
        plt.tight_layout()
        filename = os.path.join(self.maps_dir, "physics_network_map.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Salvo: {filename}")
        
    def map_clustering_analysis(self):
        """Análise de clustering das ressonâncias"""
        print("🔍 Gerando análise de clustering...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Preparar dados para clustering
        features = ['log_constant', 'log_quality', 'energy_gev']
        X = self.resonances_df[features].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-Means Clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters_kmeans = kmeans.fit_predict(X_scaled)
        
        ax1.scatter(self.resonances_df['energy_gev'], self.resonances_df['quality'], 
                   c=clusters_kmeans, cmap='viridis', s=100, alpha=0.8, edgecolors='black')
        ax1.set_xlabel('Energia (GeV)')
        ax1.set_ylabel('Qualidade')
        ax1.set_yscale('log')
        ax1.set_title('K-Means Clustering (k=4)')
        ax1.grid(True, alpha=0.3)
        
        # DBSCAN Clustering
        dbscan = DBSCAN(eps=0.5, min_samples=2)
        clusters_dbscan = dbscan.fit_predict(X_scaled)
        
        ax2.scatter(self.resonances_df['energy_gev'], self.resonances_df['quality'], 
                   c=clusters_dbscan, cmap='plasma', s=100, alpha=0.8, edgecolors='black')
        ax2.set_xlabel('Energia (GeV)')
        ax2.set_ylabel('Qualidade')
        ax2.set_yscale('log')
        ax2.set_title('DBSCAN Clustering')
        ax2.grid(True, alpha=0.3)
        
        # t-SNE Visualization
        if len(X_scaled) >= 4:  # t-SNE precisa de pelo menos n_components amostras
            # Ajustar perplexity para número pequeno de amostras
            perplexity = min(5, len(X_scaled) - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            X_tsne = tsne.fit_transform(X_scaled)
            
            colors = [CATEGORY_COLORS.get(cat, 'gray') for cat in self.resonances_df['category']]
            ax3.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, s=100, alpha=0.8, edgecolors='black')
            ax3.set_xlabel('t-SNE Dimensão 1')
            ax3.set_ylabel('t-SNE Dimensão 2')
            ax3.set_title(f't-SNE: Redução Dimensional (perplexity={perplexity})')
            ax3.grid(True, alpha=0.3)
            
            # Adicionar labels para identificar pontos
            for i, row in self.resonances_df.iterrows():
                ax3.annotate(row['constant'][:8], (X_tsne[i, 0], X_tsne[i, 1]), 
                           xytext=(3, 3), textcoords='offset points', fontsize=6, alpha=0.7)
        else:
            ax3.text(0.5, 0.5, 'Insuficientes amostras\npara t-SNE', 
                    transform=ax3.transAxes, ha='center', va='center', fontsize=12)
            ax3.set_title('t-SNE: Redução Dimensional')
        
        # Análise de correlação
        corr_matrix = self.resonances_df[['log_constant', 'log_quality', 'energy_gev', 'error_percent']].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax4)
        ax4.set_title('Matriz de Correlação')
        
        plt.tight_layout()
        filename = os.path.join(self.maps_dir, "clustering_analysis_map.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Salvo: {filename}")
        
    def map_statistical_summary(self):
        """Mapa de resumo estatístico"""
        print("📊 Gerando resumo estatístico...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Distribuição por categoria
        category_counts = self.resonances_df['category'].value_counts()
        colors = [CATEGORY_COLORS.get(cat, 'gray') for cat in category_counts.index]
        
        ax1.pie(category_counts.values, labels=category_counts.index, colors=colors, 
               autopct='%1.0f%%', startangle=90)
        ax1.set_title('Distribuição por Categoria Física')
        
        # Box plot de qualidade por categoria
        categories = list(CATEGORY_COLORS.keys())
        qualities_by_cat = [self.resonances_df[self.resonances_df['category'] == cat]['log_quality'].values 
                           for cat in categories]
        
        bp = ax2.boxplot(qualities_by_cat, labels=categories, patch_artist=True)
        for patch, color in zip(bp['boxes'], CATEGORY_COLORS.values()):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_ylabel('Log₁₀(Qualidade)')
        ax2.set_title('Distribuição de Qualidade por Categoria')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Scatter: Constante vs Qualidade
        ax3.scatter(self.resonances_df['log_constant'], self.resonances_df['log_quality'], 
                   c=[CATEGORY_COLORS.get(cat, 'gray') for cat in self.resonances_df['category']], 
                   s=100, alpha=0.8, edgecolors='black')
        ax3.set_xlabel('Log₁₀(Valor da Constante)')
        ax3.set_ylabel('Log₁₀(Qualidade)')
        ax3.set_title('Valor da Constante vs Qualidade')
        ax3.grid(True, alpha=0.3)
        
        # Top 10 melhores ressonâncias
        top_10 = self.resonances_df.nsmallest(10, 'quality')
        colors = [CATEGORY_COLORS.get(cat, 'gray') for cat in top_10['category']]
        
        ax4.barh(range(len(top_10)), -top_10['log_quality'], color=colors, alpha=0.8, edgecolor='black')
        ax4.set_yticks(range(len(top_10)))
        ax4.set_yticklabels(top_10['constant'], fontsize=10)
        ax4.set_xlabel('Log₁₀(Qualidade) - Melhor →')
        ax4.set_title('Top 10 Melhores Ressonâncias')
        ax4.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(self.maps_dir, "statistical_summary_map.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Salvo: {filename}")
        
    def generate_comprehensive_report(self):
        """Gera relatório completo de mapeamento"""
        print("📋 Gerando relatório de mapeamento...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.maps_dir, f"Mapping_Report_{timestamp}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ZVT DATA MAPPER - RELATÓRIO DE MAPEAMENTO\n")
            f.write("="*80 + "\n\n")
            f.write(f"Data: {datetime.now().isoformat()}\n")
            f.write(f"Zeros analisados: {len(self.zeros):,}\n")
            f.write(f"Constantes mapeadas: {len(self.resonances_df)}\n\n")
            
            f.write("ESTATÍSTICAS GERAIS:\n")
            f.write(f"Melhor qualidade: {self.resonances_df['quality'].min():.2e}\n")
            f.write(f"Pior qualidade: {self.resonances_df['quality'].max():.2e}\n")
            f.write(f"Energia média: {self.resonances_df['energy_gev'].mean():.2f} GeV\n")
            f.write(f"Energia mediana: {self.resonances_df['energy_gev'].median():.2f} GeV\n\n")
            
            f.write("TOP 5 MELHORES RESSONÂNCIAS:\n")
            top_5 = self.resonances_df.nsmallest(5, 'quality')
            for i, (_, row) in enumerate(top_5.iterrows(), 1):
                f.write(f"{i}. {row['constant']}: {row['error_percent']:.2e}% erro\n")
                f.write(f"   Zero #{row['zero_index']:,}, γ={row['gamma']:.6f}\n")
            
            f.write(f"\nPOR CATEGORIA:\n")
            for category in PHYSICS_CATEGORIES.keys():
                cat_data = self.resonances_df[self.resonances_df['category'] == category]
                if not cat_data.empty:
                    best_in_cat = cat_data.loc[cat_data['quality'].idxmin()]
                    f.write(f"{category}: Melhor = {best_in_cat['constant']} ({best_in_cat['error_percent']:.2e}%)\n")
            
            f.write(f"\nARQUIVOS GERADOS:\n")
            f.write("- energy_vs_quality_map.png\n")
            f.write("- hierarchy_tree_map.png\n")  
            f.write("- energy_landscape_map.png\n")
            f.write("- physics_network_map.png\n")
            f.write("- clustering_analysis_map.png\n")
            f.write("- statistical_summary_map.png\n")
            f.write("="*80 + "\n")
        
        print(f"📊 Relatório salvo: {report_file}")
        
    def run_all_mappings(self):
        """Executa todos os mapeamentos"""
        print("\n🚀 INICIANDO MAPEAMENTO COMPLETO...")
        print("="*60)
        
        if not self.load_data():
            return
            
        print(f"\n📊 Dataset carregado:")
        print(f"   • {len(self.zeros):,} zeros da função zeta")
        print(f"   • {len(self.resonances_df)} constantes fundamentais")
        print(f"   • {len(PHYSICS_CATEGORIES)} categorias físicas")
        
        print(f"\n🗺️ Gerando mapas...")
        
        # Executar todos os mapeamentos
        self.map_energy_vs_quality()
        self.map_hierarchy_tree()
        self.map_energy_landscape()
        self.map_physics_network()
        self.map_clustering_analysis()
        self.map_statistical_summary()
        
        # Gerar relatório
        self.generate_comprehensive_report()
        
        print(f"\n✅ MAPEAMENTO CONCLUÍDO!")
        print(f"📁 Todos os mapas salvos em: {self.maps_dir}")
        print("="*60)

def main():
    """Função principal"""
    mapper = ZVTDataMapper()
    mapper.run_all_mappings()

if __name__ == "__main__":
    main()

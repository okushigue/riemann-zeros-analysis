#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZVT_MONTE_CARLO.py - Análise de Monte Carlo para Validação Estatística
Author: Jefferson M. Okushigue
Date: 2025-08-10
Validação rigorosa dos padrões encontrados através de simulações Monte Carlo
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle
import os
from datetime import datetime
from scipy import stats
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Configuração científica
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Constantes reais encontradas (nossos resultados)
REAL_CONSTANTS = {
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

# Resultados reais para comparação (da análise anterior)
REAL_RESULTS = {
    'gravitacional': {'quality': 1.691571e-45, 'error_percent': 0.000028641562},
    'fermi_coupling': {'quality': 8.288891e-13, 'error_percent': 0.000007106518},
    'eletromagnetica': {'quality': 9.091261e-10, 'error_percent': 0.000012458301},
    'dark_energy': {'quality': 4.012673e-09, 'error_percent': 0.000000586048},
    'forte': {'quality': 4.980134e-09, 'error_percent': 0.000004202645},
    'weinberg_angle': {'quality': 8.924658e-09, 'error_percent': 0.000003860146},
    'hubble_reduced': {'quality': 1.102047e-08, 'error_percent': 0.000001636055},
    'fraca': {'quality': 2.394045e-08, 'error_percent': 0.000070829742},
    'baryon_density': {'quality': 3.206900e-08, 'error_percent': 0.000065985593},
    'euler_mascheroni': {'quality': 5.600674e-08, 'error_percent': 0.000009702914}
}

class ZVTMonteCarloAnalyzer:
    def __init__(self, cache_file="zeta_zeros_cache.pkl", results_dir="zvt_constants_results"):
        self.cache_file = cache_file
        self.results_dir = results_dir
        self.monte_carlo_dir = os.path.join(results_dir, "monte_carlo")
        self.zeros = None
        self.n_simulations = 10000  # Número de simulações Monte Carlo
        self.n_constants = len(REAL_CONSTANTS)
        
        # Resultados das simulações
        self.simulation_results = []
        self.summary_stats = {}
        
        os.makedirs(self.monte_carlo_dir, exist_ok=True)
        
        print("🧮 ZVT MONTE CARLO ANALYZER")
        print("=" * 60)
        print(f"🎯 Validação estatística através de {self.n_simulations:,} simulações")
        print(f"🔬 Testando {self.n_constants} constantes vs distribuições aleatórias")
        
    def load_zeros(self):
        """Carrega zeros da função zeta"""
        print("📂 Carregando zeros da função zeta...")
        
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.zeros = pickle.load(f)
            print(f"✅ {len(self.zeros):,} zeros carregados")
            return True
        else:
            print("❌ Cache de zeros não encontrado!")
            return False
    
    def generate_random_constants(self, n_constants=None):
        """Gera constantes aleatórias para simulação Monte Carlo"""
        if n_constants is None:
            n_constants = self.n_constants
            
        # Gerar constantes em diferentes escalas (log-uniform)
        random_constants = {}
        
        for i in range(n_constants):
            # Escolher escala aleatória entre 1e-50 e 1e4
            log_min, log_max = -50, 4
            log_value = np.random.uniform(log_min, log_max)
            value = 10 ** log_value
            random_constants[f'random_{i+1}'] = value
            
        return random_constants
    
    def find_best_resonance(self, zeros, constant_value):
        """Encontra a melhor ressonância para uma constante específica"""
        best_quality = float('inf')
        best_zero = None
        
        for n, gamma in zeros:
            mod_val = gamma % constant_value
            min_distance = min(mod_val, constant_value - mod_val)
            
            if min_distance < best_quality:
                best_quality = min_distance
                best_zero = (n, gamma)
        
        if best_zero:
            error_percent = (best_quality / constant_value) * 100
            return {
                'quality': best_quality,
                'error_percent': error_percent,
                'zero_index': best_zero[0],
                'gamma': best_zero[1]
            }
        return None
    
    def simulate_single_run(self, simulation_id):
        """Executa uma única simulação Monte Carlo"""
        # Gerar constantes aleatórias
        random_constants = self.generate_random_constants()
        
        # Encontrar melhores ressonâncias para cada constante aleatória
        results = {}
        qualities = []
        error_percentages = []
        
        for const_name, const_value in random_constants.items():
            resonance = self.find_best_resonance(self.zeros, const_value)
            if resonance:
                results[const_name] = resonance
                qualities.append(resonance['quality'])
                error_percentages.append(resonance['error_percent'])
        
        # Calcular estatísticas desta simulação
        simulation_stats = {
            'simulation_id': simulation_id,
            'n_resonances': len(results),
            'best_quality': min(qualities) if qualities else float('inf'),
            'mean_quality': np.mean(qualities) if qualities else float('inf'),
            'best_error_percent': min(error_percentages) if error_percentages else float('inf'),
            'mean_error_percent': np.mean(error_percentages) if error_percentages else float('inf'),
            'qualities': qualities,
            'error_percentages': error_percentages
        }
        
        return simulation_stats
    
    def run_monte_carlo_parallel(self):
        """Executa simulações Monte Carlo em paralelo"""
        print(f"\n🚀 Iniciando {self.n_simulations:,} simulações Monte Carlo...")
        print("⚡ Processamento paralelo ativado")
        
        # Usar amostra dos zeros para acelerar (últimos 100k)
        sample_zeros = self.zeros[-100000:] if len(self.zeros) > 100000 else self.zeros
        original_zeros = self.zeros
        self.zeros = sample_zeros
        
        print(f"📊 Usando amostra de {len(sample_zeros):,} zeros para simulações")
        
        # Processamento paralelo
        max_workers = min(8, os.cpu_count())  # Limitar para não sobrecarregar
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submeter todas as simulações
            futures = [executor.submit(self.simulate_single_run, i) 
                      for i in range(self.n_simulations)]
            
            # Coletar resultados com progress bar
            for future in tqdm(as_completed(futures), total=self.n_simulations, 
                              desc="Simulações Monte Carlo"):
                try:
                    result = future.result()
                    self.simulation_results.append(result)
                except Exception as e:
                    print(f"⚠️ Erro na simulação: {e}")
        
        # Restaurar zeros originais
        self.zeros = original_zeros
        
        print(f"✅ {len(self.simulation_results):,} simulações completadas")
        
    def analyze_simulation_results(self):
        """Analisa resultados das simulações Monte Carlo"""
        print("\n📊 Analisando resultados das simulações...")
        
        # Extrair estatísticas
        best_qualities = [r['best_quality'] for r in self.simulation_results if np.isfinite(r['best_quality'])]
        best_errors = [r['best_error_percent'] for r in self.simulation_results if np.isfinite(r['best_error_percent'])]
        mean_qualities = [r['mean_quality'] for r in self.simulation_results if np.isfinite(r['mean_quality'])]
        
        self.summary_stats = {
            'n_valid_simulations': len(best_qualities),
            'best_quality_distribution': best_qualities,
            'best_error_distribution': best_errors,
            'mean_quality_distribution': mean_qualities,
            'percentiles_quality': {
                p: np.percentile(best_qualities, p) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
            },
            'percentiles_error': {
                p: np.percentile(best_errors, p) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
            }
        }
        
        print(f"📈 Estatísticas calculadas para {len(best_qualities):,} simulações válidas")
        
    def calculate_p_values(self):
        """Calcula p-values para nossos resultados reais"""
        print("\n🧪 Calculando p-values para validação estatística...")
        
        p_values = {}
        significance_levels = {}
        
        for const_name, real_result in REAL_RESULTS.items():
            real_quality = real_result['quality']
            real_error = real_result['error_percent']
            
            # P-value: proporção de simulações com qualidade igual ou melhor
            better_quality_count = sum(1 for q in self.summary_stats['best_quality_distribution'] 
                                     if q <= real_quality)
            p_value_quality = better_quality_count / len(self.summary_stats['best_quality_distribution'])
            
            # P-value para erro percentual
            better_error_count = sum(1 for e in self.summary_stats['best_error_distribution'] 
                                   if e <= real_error)
            p_value_error = better_error_count / len(self.summary_stats['best_error_distribution'])
            
            p_values[const_name] = {
                'p_value_quality': p_value_quality,
                'p_value_error': p_value_error,
                'real_quality': real_quality,
                'real_error': real_error
            }
            
            # Determinar nível de significância
            if p_value_quality < 0.001:
                significance = "***"  # Extremamente significativo
            elif p_value_quality < 0.01:
                significance = "**"   # Muito significativo
            elif p_value_quality < 0.05:
                significance = "*"    # Significativo
            else:
                significance = "n.s." # Não significativo
                
            significance_levels[const_name] = significance
        
        self.p_values = p_values
        self.significance_levels = significance_levels
        
        print("✅ P-values calculados para todas as constantes")
        
    def visualize_monte_carlo_results(self):
        """Visualiza resultados da análise Monte Carlo"""
        print("📈 Gerando visualizações da análise Monte Carlo...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Distribuição de melhores qualidades (Monte Carlo vs Real)
        qualities = self.summary_stats['best_quality_distribution']
        ax1.hist(np.log10(qualities), bins=50, alpha=0.7, color='skyblue', 
                label=f'Monte Carlo (n={len(qualities):,})', density=True)
        
        # Adicionar linhas verticais para nossos resultados reais
        real_qualities = [np.log10(r['quality']) for r in REAL_RESULTS.values()]
        for i, (const_name, quality) in enumerate(zip(REAL_RESULTS.keys(), real_qualities)):
            color = 'red' if i < 5 else 'orange'  # Destacar top 5
            ax1.axvline(quality, color=color, linestyle='--', alpha=0.8,
                       label=f'{const_name}' if i < 3 else "")
        
        ax1.set_xlabel('Log₁₀(Melhor Qualidade)')
        ax1.set_ylabel('Densidade')
        ax1.set_title('Distribuição de Melhores Qualidades\nMonte Carlo vs Resultados Reais')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribuição de erros percentuais
        errors = self.summary_stats['best_error_distribution']
        ax2.hist(np.log10(errors), bins=50, alpha=0.7, color='lightgreen', 
                label=f'Monte Carlo (n={len(errors):,})', density=True)
        
        # Adicionar nossos resultados reais
        real_errors = [np.log10(r['error_percent']) for r in REAL_RESULTS.values()]
        for i, (const_name, error) in enumerate(zip(REAL_RESULTS.keys(), real_errors)):
            color = 'red' if i < 5 else 'orange'
            ax2.axvline(error, color=color, linestyle='--', alpha=0.8,
                       label=f'{const_name}' if i < 3 else "")
        
        ax2.set_xlabel('Log₁₀(Erro Percentual %)')
        ax2.set_ylabel('Densidade')
        ax2.set_title('Distribuição de Erros Percentuais\nMonte Carlo vs Resultados Reais')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 3. P-values por constante
        const_names = list(self.p_values.keys())
        p_vals = [self.p_values[name]['p_value_quality'] for name in const_names]
        significance = [self.significance_levels[name] for name in const_names]
        
        colors = ['red' if p < 0.001 else 'orange' if p < 0.01 else 'yellow' if p < 0.05 else 'gray' 
                 for p in p_vals]
        
        bars = ax3.bar(range(len(const_names)), -np.log10([max(p, 1e-10) for p in p_vals]), 
                      color=colors, alpha=0.8)
        ax3.set_xticks(range(len(const_names)))
        ax3.set_xticklabels(const_names, rotation=45, ha='right', fontsize=8)
        ax3.set_ylabel('-Log₁₀(p-value)')
        ax3.set_title('Significância Estatística por Constante')
        ax3.axhline(-np.log10(0.05), color='red', linestyle=':', label='p=0.05')
        ax3.axhline(-np.log10(0.01), color='red', linestyle='--', label='p=0.01')
        ax3.axhline(-np.log10(0.001), color='red', linestyle='-', label='p=0.001')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Adicionar significância no topo das barras
        for i, (bar, sig) in enumerate(zip(bars, significance)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    sig, ha='center', va='bottom', fontweight='bold')
        
        # 4. Percentis e outliers
        percentiles = self.summary_stats['percentiles_quality']
        p_labels = list(percentiles.keys())
        p_values = [np.log10(percentiles[p]) for p in p_labels]
        
        ax4.plot(p_labels, p_values, 'o-', color='blue', linewidth=2, markersize=6,
                label='Monte Carlo')
        
        # Adicionar zona de nossos resultados
        our_min = min(real_qualities)
        our_max = max(real_qualities)
        ax4.axhspan(our_min, our_max, alpha=0.2, color='red', 
                   label='Zona dos Nossos Resultados')
        
        ax4.set_xlabel('Percentil')
        ax4.set_ylabel('Log₁₀(Melhor Qualidade)')
        ax4.set_title('Distribuição Percentil das Qualidades')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(self.monte_carlo_dir, "monte_carlo_analysis.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Salvo: {filename}")
        
    def generate_statistical_report(self):
        """Gera relatório detalhado da análise Monte Carlo"""
        print("📋 Gerando relatório estatístico...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.monte_carlo_dir, f"Monte_Carlo_Report_{timestamp}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ZVT MONTE CARLO - RELATÓRIO DE VALIDAÇÃO ESTATÍSTICA\n")
            f.write("="*80 + "\n\n")
            f.write(f"Data: {datetime.now().isoformat()}\n")
            f.write(f"Simulações executadas: {self.n_simulations:,}\n")
            f.write(f"Simulações válidas: {self.summary_stats['n_valid_simulations']:,}\n")
            f.write(f"Zeros utilizados: {len(self.zeros):,}\n")
            f.write(f"Constantes testadas: {self.n_constants}\n\n")
            
            f.write("RESUMO ESTATÍSTICO DAS SIMULAÇÕES:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Melhor qualidade (Monte Carlo): {min(self.summary_stats['best_quality_distribution']):.2e}\n")
            f.write(f"Qualidade mediana (Monte Carlo): {self.summary_stats['percentiles_quality'][50]:.2e}\n")
            f.write(f"Percentil 1% (Monte Carlo): {self.summary_stats['percentiles_quality'][1]:.2e}\n")
            f.write(f"Percentil 0.1% estimado: {self.summary_stats['percentiles_quality'][1]/10:.2e}\n\n")
            
            f.write("VALIDAÇÃO DOS NOSSOS RESULTADOS:\n")
            f.write("-" * 50 + "\n")
            f.write("| Constante          | P-value    | Significância | Qualidade Real |\n")
            f.write("|--------------------|-----------:|:-------------:|:--------------:|\n")
            
            # Ordenar por p-value
            sorted_results = sorted(self.p_values.items(), key=lambda x: x[1]['p_value_quality'])
            
            for const_name, results in sorted_results:
                p_val = results['p_value_quality']
                sig = self.significance_levels[const_name]
                quality = results['real_quality']
                f.write(f"| {const_name:18s} | {p_val:9.3e} | {sig:11s} | {quality:14.2e} |\n")
            
            f.write(f"\nLEGENDA DE SIGNIFICÂNCIA:\n")
            f.write("*** = p < 0.001 (Extremamente significativo)\n")
            f.write("**  = p < 0.01  (Muito significativo)\n")
            f.write("*   = p < 0.05  (Significativo)\n")
            f.write("n.s.= p ≥ 0.05  (Não significativo)\n\n")
            
            # Análise específica das melhores
            extremes = [name for name, results in self.p_values.items() 
                       if results['p_value_quality'] < 0.001]
            
            if extremes:
                f.write("RESULTADOS EXTREMAMENTE SIGNIFICATIVOS:\n")
                f.write("-" * 50 + "\n")
                for const_name in extremes:
                    p_val = self.p_values[const_name]['p_value_quality']
                    if p_val == 0:
                        f.write(f"{const_name}: p < {1/self.n_simulations:.0e} (nenhuma simulação superou)\n")
                    else:
                        f.write(f"{const_name}: p = {p_val:.3e}\n")
                
                f.write(f"\nEsses resultados sugerem que as ressonâncias encontradas\n")
                f.write(f"são estatisticamente significativas e não podem ser\n")
                f.write(f"explicadas por acaso com alta confiança.\n")
            
            f.write(f"\nCONCLUSÕES:\n")
            f.write("-" * 50 + "\n")
            n_significant = sum(1 for sig in self.significance_levels.values() if sig != "n.s.")
            f.write(f"• {n_significant}/{len(self.p_values)} constantes mostram significância estatística\n")
            f.write(f"• {len(extremes)} constantes são extremamente significativas (p < 0.001)\n")
            f.write(f"• Os padrões encontrados são improváveis de serem devidos ao acaso\n")
            f.write(f"• Validação: {self.n_simulations:,} simulações Monte Carlo\n")
            f.write("="*80 + "\n")
        
        print(f"📊 Relatório salvo: {report_file}")
        
    def run_complete_analysis(self):
        """Executa análise Monte Carlo completa"""
        print("\n🚀 INICIANDO ANÁLISE MONTE CARLO COMPLETA")
        print("="*60)
        
        # Carregar dados
        if not self.load_zeros():
            return
        
        # Executar simulações
        self.run_monte_carlo_parallel()
        
        # Analisar resultados
        self.analyze_simulation_results()
        
        # Calcular p-values
        self.calculate_p_values()
        
        # Visualizar
        self.visualize_monte_carlo_results()
        
        # Gerar relatório
        self.generate_statistical_report()
        
        print(f"\n✅ ANÁLISE MONTE CARLO CONCLUÍDA!")
        print(f"📁 Resultados salvos em: {self.monte_carlo_dir}")
        print("="*60)
        
        # Mostrar resumo rápido
        n_significant = sum(1 for sig in self.significance_levels.values() if sig != "n.s.")
        extremes = sum(1 for p in self.p_values.values() if p['p_value_quality'] < 0.001)
        
        print(f"\n📊 RESUMO RÁPIDO:")
        print(f"   • {n_significant}/{len(self.p_values)} constantes significativas")
        print(f"   • {extremes} extremamente significativas (p < 0.001)")
        print(f"   • Baseado em {self.n_simulations:,} simulações Monte Carlo")

def main():
    """Função principal"""
    analyzer = ZVTMonteCarloAnalyzer()
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()

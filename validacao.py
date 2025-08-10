#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZVT_MONTE_CARLO_CORRECTED.py - Análise Monte Carlo Corrigida
Author: Jefferson M. Okushigue
Date: 2025-08-10
Teste correto: pequenas perturbações das constantes físicas reais
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

# Constantes físicas REAIS - valores exatos encontrados
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

# Resultados reais observados (nossas "descobertas")
REAL_RESULTS = {
    'gravitacional': {'quality': 1.691571e-45, 'error_percent': 0.000028641562, 'zero_index': 1593106},
    'fermi_coupling': {'quality': 8.288891e-13, 'error_percent': 0.000007106518, 'zero_index': 1470480},
    'eletromagnetica': {'quality': 9.091261e-10, 'error_percent': 0.000012458301, 'zero_index': 118412},
    'dark_energy': {'quality': 4.012673e-09, 'error_percent': 0.000000586048, 'zero_index': 735953},
    'forte': {'quality': 4.980134e-09, 'error_percent': 0.000004202645, 'zero_index': 1978224},
    'weinberg_angle': {'quality': 8.924658e-09, 'error_percent': 0.000003860146, 'zero_index': 1948828},
    'hubble_reduced': {'quality': 1.102047e-08, 'error_percent': 0.000001636055, 'zero_index': 1680335},
    'fraca': {'quality': 2.394045e-08, 'error_percent': 0.000070829742, 'zero_index': 539638},
    'baryon_density': {'quality': 3.206900e-08, 'error_percent': 0.000065985593, 'zero_index': 1782980},
    'euler_mascheroni': {'quality': 5.600674e-08, 'error_percent': 0.000009702914, 'zero_index': 733625}
}

# Categorias físicas
PHYSICS_CATEGORIES = {
    'Cosmologia': ['dark_energy', 'dark_matter', 'baryon_density', 'hubble_reduced', 'sigma8'],
    'Eletrofraca': ['weinberg_angle', 'fermi_coupling'],
    'Forças': ['eletromagnetica', 'forte', 'fraca', 'gravitacional'],
    'Massas': ['proton_electron', 'muon_electron', 'tau_electron', 'neutron_proton'],
    'Magnéticas': ['gyromagnetic_proton', 'gyromagnetic_neutron', 'magnetic_moment_ratio'],
    'Matemáticas': ['euler_mascheroni']
}

class ZVTMonteCarloCorrected:
    def __init__(self, cache_file="zeta_zeros_cache.pkl", results_dir="zvt_constants_results"):
        self.cache_file = cache_file
        self.results_dir = results_dir
        self.monte_carlo_dir = os.path.join(results_dir, "monte_carlo_corrected")
        self.zeros = None
        self.n_simulations = 5000  # Reduzido para teste mais específico
        
        # Configurações do teste
        self.perturbation_levels = [0.001, 0.01, 0.1, 1.0]  # 0.1%, 1%, 10%, 100%
        
        # Resultados
        self.simulation_results = {}
        self.hierarchy_results = []
        self.uniqueness_results = []
        
        os.makedirs(self.monte_carlo_dir, exist_ok=True)
        
        print("🔬 ZVT MONTE CARLO CORRECTED ANALYZER")
        print("=" * 60)
        print("🎯 TESTANDO A PERGUNTA CERTA:")
        print("   • Os valores EXATOS das constantes são especiais?")
        print("   • Ou valores próximos também dariam ressonâncias?")
        print(f"🧪 {self.n_simulations:,} simulações com perturbações: {self.perturbation_levels}")
        
    def load_zeros(self):
        """Carrega zeros da função zeta"""
        print("\n📂 Carregando zeros da função zeta...")
        
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                self.zeros = pickle.load(f)
            print(f"✅ {len(self.zeros):,} zeros carregados")
            return True
        else:
            print("❌ Cache de zeros não encontrado!")
            return False
    
    def perturb_constants(self, perturbation_percent):
        """Gera versões perturbadas das constantes reais"""
        perturbed = {}
        
        for const_name, real_value in REAL_CONSTANTS.items():
            # Adicionar ruído gaussiano
            noise_factor = np.random.normal(1.0, perturbation_percent/100)
            perturbed_value = real_value * noise_factor
            perturbed[const_name] = perturbed_value
            
        return perturbed
    
    def find_best_resonance_for_constant(self, zeros, constant_value):
        """Encontra a melhor ressonância para uma constante específica"""
        best_quality = float('inf')
        best_result = None
        
        for n, gamma in zeros:
            mod_val = gamma % constant_value
            min_distance = min(mod_val, constant_value - mod_val)
            
            if min_distance < best_quality:
                best_quality = min_distance
                best_result = {
                    'quality': min_distance,
                    'error_percent': (min_distance / constant_value) * 100,
                    'zero_index': n,
                    'gamma': gamma
                }
        
        return best_result
    
    def run_single_perturbation_test(self, perturbation_level, simulation_id):
        """Executa um teste com constantes perturbadas"""
        # Gerar constantes perturbadas
        perturbed_constants = self.perturb_constants(perturbation_level)
        
        # Encontrar melhores ressonâncias
        results = {}
        for const_name, perturbed_value in perturbed_constants.items():
            resonance = self.find_best_resonance_for_constant(self.zeros, perturbed_value)
            if resonance:
                results[const_name] = resonance
        
        # Analisar se mantém padrões observados
        analysis = self.analyze_simulation_patterns(results)
        
        return {
            'simulation_id': simulation_id,
            'perturbation_level': perturbation_level,
            'results': results,
            'analysis': analysis
        }
    
    def analyze_simulation_patterns(self, sim_results):
        """Analisa se a simulação mantém os padrões observados"""
        analysis = {
            'hierarchy_preserved': False,
            'uniqueness_preserved': False,
            'energy_concentration': False,
            'category_patterns': {}
        }
        
        if not sim_results:
            return analysis
        
        # 1. Testar hierarquia por categoria
        category_qualities = {}
        for category, constants in PHYSICS_CATEGORIES.items():
            cat_qualities = []
            for const_name in constants:
                if const_name in sim_results and const_name in REAL_RESULTS:
                    cat_qualities.append(sim_results[const_name]['quality'])
            if cat_qualities:
                category_qualities[category] = np.median(cat_qualities)
        
        # Verificar se cosmologia ainda domina
        if 'Cosmologia' in category_qualities:
            cosmo_quality = category_qualities['Cosmologia']
            other_qualities = [q for cat, q in category_qualities.items() if cat != 'Cosmologia']
            if other_qualities and cosmo_quality <= min(other_qualities):
                analysis['hierarchy_preserved'] = True
        
        # 2. Testar se ainda encontra ressonâncias "únicas" (muito melhores)
        qualities = [r['quality'] for r in sim_results.values()]
        if qualities:
            best_quality = min(qualities)
            mean_quality = np.mean(qualities)
            # Se a melhor é muito melhor que a média (fator 100x)
            if best_quality * 100 < mean_quality:
                analysis['uniqueness_preserved'] = True
        
        # 3. Testar concentração energética (~100 TeV)
        energies = [r['gamma']/10 for r in sim_results.values()]
        if energies:
            # Concentração se >50% estão na faixa 50-150 TeV
            target_range = sum(1 for e in energies if 50000 <= e <= 150000)
            if target_range / len(energies) > 0.5:
                analysis['energy_concentration'] = True
        
        # 4. Análise por categoria
        for category, constants in PHYSICS_CATEGORIES.items():
            cat_results = []
            for const_name in constants:
                if const_name in sim_results and const_name in REAL_RESULTS:
                    real_quality = REAL_RESULTS[const_name]['quality']
                    sim_quality = sim_results[const_name]['quality']
                    # Comparar se simulação chegou perto do resultado real
                    ratio = sim_quality / real_quality
                    cat_results.append(ratio)
            
            if cat_results:
                analysis['category_patterns'][category] = {
                    'mean_ratio': np.mean(cat_results),
                    'median_ratio': np.median(cat_results),
                    'close_matches': sum(1 for r in cat_results if 0.1 <= r <= 10)  # Dentro de 1 ordem de magnitude
                }
        
        return analysis
    
    def run_perturbation_analysis(self):
        """Executa análise completa com perturbações"""
        print("\n🚀 Iniciando análise de perturbações...")
        
        # Usar amostra menor para acelerar
        sample_size = 200000  # 200k zeros para teste mais rápido
        sample_zeros = self.zeros[-sample_size:] if len(self.zeros) > sample_size else self.zeros
        original_zeros = self.zeros
        self.zeros = sample_zeros
        
        print(f"📊 Usando amostra de {len(sample_zeros):,} zeros")
        
        for perturbation_level in self.perturbation_levels:
            print(f"\n🔬 Testando perturbação de {perturbation_level*100:.1f}%...")
            
            level_results = []
            
            # Processo paralelo para cada nível
            max_workers = min(6, os.cpu_count())
            
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submeter simulações para este nível
                futures = [executor.submit(self.run_single_perturbation_test, perturbation_level, i) 
                          for i in range(self.n_simulations)]
                
                # Coletar resultados
                for future in tqdm(as_completed(futures), total=self.n_simulations, 
                                  desc=f"Perturbação {perturbation_level*100:.1f}%"):
                    try:
                        result = future.result()
                        level_results.append(result)
                    except Exception as e:
                        print(f"⚠️ Erro: {e}")
            
            self.simulation_results[perturbation_level] = level_results
            
            # Análise rápida dos resultados deste nível
            self.analyze_perturbation_level(perturbation_level, level_results)
        
        # Restaurar zeros originais
        self.zeros = original_zeros
        
    def analyze_perturbation_level(self, perturbation_level, level_results):
        """Analisa resultados de um nível de perturbação"""
        if not level_results:
            return
        
        # Estatísticas dos padrões preservados
        hierarchy_preserved = sum(1 for r in level_results if r['analysis']['hierarchy_preserved'])
        uniqueness_preserved = sum(1 for r in level_results if r['analysis']['uniqueness_preserved'])
        energy_concentration = sum(1 for r in level_results if r['analysis']['energy_concentration'])
        
        total_sims = len(level_results)
        
        print(f"📊 Resultados para perturbação {perturbation_level*100:.1f}%:")
        print(f"   • Hierarquia preservada: {hierarchy_preserved}/{total_sims} ({hierarchy_preserved/total_sims*100:.1f}%)")
        print(f"   • Unicidade preservada: {uniqueness_preserved}/{total_sims} ({uniqueness_preserved/total_sims*100:.1f}%)")
        print(f"   • Concentração energética: {energy_concentration}/{total_sims} ({energy_concentration/total_sims*100:.1f}%)")
        
        # Armazenar para análise final
        self.hierarchy_results.append({
            'perturbation': perturbation_level,
            'hierarchy_rate': hierarchy_preserved/total_sims,
            'uniqueness_rate': uniqueness_preserved/total_sims,
            'energy_rate': energy_concentration/total_sims
        })
    
    def visualize_corrected_results(self):
        """Visualiza resultados da análise corrigida"""
        print("\n📈 Gerando visualizações da análise corrigida...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Taxa de preservação dos padrões vs perturbação
        perturbations = [r['perturbation']*100 for r in self.hierarchy_results]
        hierarchy_rates = [r['hierarchy_rate']*100 for r in self.hierarchy_results]
        uniqueness_rates = [r['uniqueness_rate']*100 for r in self.hierarchy_results]
        energy_rates = [r['energy_rate']*100 for r in self.hierarchy_results]
        
        ax1.plot(perturbations, hierarchy_rates, 'o-', label='Hierarquia Cosmológica', linewidth=2, markersize=8)
        ax1.plot(perturbations, uniqueness_rates, 's-', label='Unicidade das Ressonâncias', linewidth=2, markersize=8)
        ax1.plot(perturbations, energy_rates, '^-', label='Concentração Energética', linewidth=2, markersize=8)
        
        ax1.set_xlabel('Perturbação das Constantes (%)')
        ax1.set_ylabel('Taxa de Preservação do Padrão (%)')
        ax1.set_title('Robustez dos Padrões vs Perturbação')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Linha de expectativa aleatória
        ax1.axhline(20, color='red', linestyle='--', alpha=0.7, label='Expectativa Aleatória')
        
        # 2. Comparação qualidade: Real vs Perturbado (menor perturbação)
        if 0.001 in self.simulation_results:
            small_pert_results = self.simulation_results[0.001]
            
            real_qualities = []
            pert_qualities_mean = []
            const_names = []
            
            for const_name in REAL_RESULTS.keys():
                if const_name in REAL_CONSTANTS:
                    real_qualities.append(REAL_RESULTS[const_name]['quality'])
                    
                    # Média das qualidades perturbadas para esta constante
                    pert_quals = []
                    for sim in small_pert_results:
                        if const_name in sim['results']:
                            pert_quals.append(sim['results'][const_name]['quality'])
                    
                    if pert_quals:
                        pert_qualities_mean.append(np.mean(pert_quals))
                        const_names.append(const_name)
            
            if real_qualities and pert_qualities_mean:
                ax2.scatter(np.log10(real_qualities), np.log10(pert_qualities_mean), 
                           s=100, alpha=0.7, edgecolors='black')
                
                # Linha 1:1
                min_val = min(min(np.log10(real_qualities)), min(np.log10(pert_qualities_mean)))
                max_val = max(max(np.log10(real_qualities)), max(np.log10(pert_qualities_mean)))
                ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Linha 1:1')
                
                ax2.set_xlabel('Log₁₀(Qualidade Real)')
                ax2.set_ylabel('Log₁₀(Qualidade Perturbada 0.1%)')
                ax2.set_title('Correlação: Real vs Perturbado')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # 3. Distribuição de qualidades por categoria (perturbação 1%)
        if 0.01 in self.simulation_results:
            med_pert_results = self.simulation_results[0.01]
            
            category_data = {cat: [] for cat in PHYSICS_CATEGORIES.keys()}
            
            for sim in med_pert_results[:100]:  # Primeiras 100 simulações
                for category, constants in PHYSICS_CATEGORIES.items():
                    for const_name in constants:
                        if const_name in sim['results']:
                            category_data[category].append(np.log10(sim['results'][const_name]['quality']))
            
            # Box plot por categoria
            valid_categories = [cat for cat, data in category_data.items() if len(data) > 10]
            if valid_categories:
                data_for_box = [category_data[cat] for cat in valid_categories]
                
                bp = ax3.boxplot(data_for_box, labels=valid_categories, patch_artist=True)
                colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
                for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax3.set_ylabel('Log₁₀(Qualidade)')
                ax3.set_title('Distribuição por Categoria (Perturbação 1%)')
                ax3.tick_params(axis='x', rotation=45)
                ax3.grid(True, alpha=0.3)
        
        # 4. Análise de significância estatística
        # Testar se padrões são mantidos além do esperado por acaso
        perturbation_labels = [f"{p*100:.1f}%" for p in perturbations]
        
        # Teste binomial: H0 = taxa deveria ser ~20% por acaso
        expected_rate = 20  # %
        observed_hierarchy = hierarchy_rates
        
        # P-values para cada perturbação
        p_values = []
        for i, obs_rate in enumerate(observed_hierarchy):
            n_trials = len(self.simulation_results[perturbations[i]/100])
            n_successes = int(obs_rate * n_trials / 100)
            
            # Teste binomial
            p_val = 1 - stats.binom.cdf(n_successes-1, n_trials, expected_rate/100)
            p_values.append(p_val)
        
        # Plotar -log10(p-value)
        log_p_values = [-np.log10(max(p, 1e-10)) for p in p_values]
        
        bars = ax4.bar(perturbation_labels, log_p_values, alpha=0.8)
        ax4.set_ylabel('-Log₁₀(p-value)')
        ax4.set_xlabel('Perturbação das Constantes')
        ax4.set_title('Significância da Preservação da Hierarquia')
        ax4.axhline(-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
        ax4.axhline(-np.log10(0.01), color='red', linestyle='-', label='p=0.01')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Colorir barras significativas
        for i, (bar, p_val) in enumerate(zip(bars, p_values)):
            if p_val < 0.01:
                bar.set_color('red')
            elif p_val < 0.05:
                bar.set_color('orange')
            else:
                bar.set_color('gray')
        
        plt.tight_layout()
        filename = os.path.join(self.monte_carlo_dir, "monte_carlo_corrected_analysis.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"💾 Salvo: {filename}")
    
    def generate_corrected_report(self):
        """Gera relatório da análise corrigida"""
        print("\n📋 Gerando relatório da análise corrigida...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.monte_carlo_dir, f"Monte_Carlo_Corrected_Report_{timestamp}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ZVT MONTE CARLO CORRIGIDO - TESTE DAS CONSTANTES REAIS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Data: {datetime.now().isoformat()}\n")
            f.write(f"Simulações por nível: {self.n_simulations:,}\n")
            f.write(f"Níveis de perturbação testados: {self.perturbation_levels}\n")
            f.write(f"Zeros utilizados: {len(self.zeros):,}\n\n")
            
            f.write("PERGUNTA CIENTÍFICA:\n")
            f.write("-" * 50 + "\n")
            f.write("Os valores EXATOS das constantes físicas são especiais?\n")
            f.write("Ou pequenas variações também produziriam padrões similares?\n\n")
            
            f.write("PADRÕES TESTADOS:\n")
            f.write("-" * 50 + "\n")
            f.write("1. Hierarquia Cosmológica: Cosmologia > Eletrofraca > Nuclear\n")
            f.write("2. Unicidade: Uma ressonância claramente melhor por constante\n")
            f.write("3. Concentração Energética: ~100 TeV\n\n")
            
            f.write("RESULTADOS POR NÍVEL DE PERTURBAÇÃO:\n")
            f.write("-" * 50 + "\n")
            f.write("| Perturbação | Hierarquia | Unicidade | Energia | Interpretação |\n")
            f.write("|-------------|------------|-----------|---------|---------------|\n")
            
            for result in self.hierarchy_results:
                pert = result['perturbation'] * 100
                hier = result['hierarchy_rate'] * 100
                uniq = result['uniqueness_rate'] * 100
                ener = result['energy_rate'] * 100
                
                # Interpretação
                if hier > 40 and uniq > 40:
                    interp = "PADRÃO ROBUSTO"
                elif hier > 20 or uniq > 20:
                    interp = "Parcialmente preservado"
                else:
                    interp = "Não preservado"
                
                f.write(f"| {pert:9.1f}% | {hier:8.1f}% | {uniq:7.1f}% | {ener:5.1f}% | {interp:13s} |\n")
            
            f.write(f"\nANÁLISE ESTATÍSTICA:\n")
            f.write("-" * 50 + "\n")
            
            # Análise da robustez
            robust_levels = sum(1 for r in self.hierarchy_results if r['hierarchy_rate'] > 0.4)
            
            if robust_levels > 0:
                f.write(f"✅ EVIDÊNCIA DE ESTRUTURA REAL:\n")
                f.write(f"   • {robust_levels}/{len(self.hierarchy_results)} níveis mantêm padrões (>40%)\n")
                f.write(f"   • Padrões são robustos a pequenas variações\n")
                f.write(f"   • Sugere que valores exatos das constantes SÃO especiais\n\n")
            else:
                f.write(f"❌ PADRÕES NÃO ROBUSTOS:\n")
                f.write(f"   • Nenhum nível mantém padrões consistentemente\n")
                f.write(f"   • Pequenas variações destroem a estrutura\n")
                f.write(f"   • Sugere coincidência estatística\n\n")
            
            f.write("CONCLUSÕES:\n")
            f.write("-" * 50 + "\n")
            
            # Conclusão baseada nos resultados
            avg_hierarchy = np.mean([r['hierarchy_rate'] for r in self.hierarchy_results])
            avg_uniqueness = np.mean([r['uniqueness_rate'] for r in self.hierarchy_results])
            
            if avg_hierarchy > 0.3 and avg_uniqueness > 0.3:
                f.write("🎯 RESULTADO: EVIDÊNCIA DE ESTRUTURA REAL\n")
                f.write("Os padrões observados mostram robustez estatística.\n")
                f.write("Os valores exatos das constantes físicas parecem ter\n")
                f.write("conexões especiais com os zeros da função zeta.\n")
            else:
                f.write("🎯 RESULTADO: PADRÕES NÃO ROBUSTOS\n")
                f.write("Os padrões observados não resistem a pequenas perturbações.\n")
                f.write("Sugere que as 'ressonâncias' são coincidências estatísticas\n")
                f.write("devidas ao grande espaço de busca.\n")
            
            f.write("="*80 + "\n")
        
        print(f"📊 Relatório salvo: {report_file}")
    
    def run_complete_corrected_analysis(self):
        """Executa análise Monte Carlo corrigida completa"""
        print("\n🚀 INICIANDO ANÁLISE MONTE CARLO CORRIGIDA")
        print("="*60)
        print("🎯 TESTANDO SE OS VALORES EXATOS DAS CONSTANTES SÃO ESPECIAIS")
        print("="*60)
        
        # Carregar dados
        if not self.load_zeros():
            return
        
        # Executar análise de perturbações
        self.run_perturbation_analysis()
        
        # Visualizar resultados
        self.visualize_corrected_results()
        
        # Gerar relatório
        self.generate_corrected_report()
        
        print(f"\n✅ ANÁLISE MONTE CARLO CORRIGIDA CONCLUÍDA!")
        print(f"📁 Resultados salvos em: {self.monte_carlo_dir}")
        
        # Mostrar conclusão rápida
        avg_hierarchy = np.mean([r['hierarchy_rate'] for r in self.hierarchy_results])
        avg_uniqueness = np.mean([r['uniqueness_rate'] for r in self.hierarchy_results])
        
        print("\n🎯 CONCLUSÃO RÁPIDA:")
        if avg_hierarchy > 0.3 and avg_uniqueness > 0.3:
            print("   ✅ PADRÕES SÃO ROBUSTOS - Estrutura real detectada!")
            print(f"   📊 Hierarquia mantida: {avg_hierarchy*100:.1f}% das vezes")
            print(f"   📊 Unicidade mantida: {avg_uniqueness*100:.1f}% das vezes")
        else:
            print("   ❌ Padrões não são robustos - Coincidência estatística")
            print(f"   📊 Hierarquia mantida: {avg_hierarchy*100:.1f}% das vezes")
            print(f"   📊 Unicidade mantida: {avg_uniqueness*100:.1f}% das vezes")

def main():
    """Função principal"""
    analyzer = ZVTMonteCarloCorrected()
    analyzer.run_complete_corrected_analysis()

if __name__ == "__main__":
    main()

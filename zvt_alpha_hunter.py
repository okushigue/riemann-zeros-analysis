#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZVT_FINE_STRUCTURE_HUNTER_NO_PLOTS.py - Zeta/α resonance hunter with file loading, no plots
Author: Jefferson M. Okushigue
Date: 2025-08-09
"""

import numpy as np
from mpmath import mp
import time
from concurrent.futures import ProcessPoolExecutor
import pickle
import os
import signal
import sys
from datetime import datetime
from scipy import stats
from scipy.stats import kstest, anderson
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configuration
mp.dps = 50  # High precision
FINE_STRUCTURE = 1 / 137.035999084  # Fine-structure constant (α)
TOLERANCE_LEVELS = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]  # Adjusted for α
DEFAULT_TOLERANCE = 1e-5
INCREMENT = 1000  # Batch size for processing

# Control constants for statistical validation
CONTROL_CONSTANTS = {
    'fine_structure': FINE_STRUCTURE,
    'random_1': 1 / 142.7,
    'random_2': 1 / 129.3,
    'random_3': 1 / 131.2,
    'random_4': 1 / 144.8,
    'golden_ratio': (np.sqrt(5) - 1) / 2,
    'pi_scale': np.pi / 100,
    'e_scale': np.e / 100
}

MAX_WORKERS = os.cpu_count()
CACHE_FILE = "zeta_zeros_cache.pkl"
STATS_FILE = "zvt_stats.txt"
RESULTS_DIR = "zvt_results"
ZEROS_FILE = os.path.expanduser("~/zeta/zero.txt")  # Path to the zeros file

# Parameters for analysis
FRESH_START_ZEROS = 5000
MINIMUM_FOR_STATS = 1000

# Create results directory if it doesn't exist
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Global variable for shutdown control
shutdown_requested = False

# Signal handler for graceful shutdown
def signal_handler(signum, frame):
    global shutdown_requested
    print(f"\n⏸️ Shutdown solicitado. Completando lote atual e salvando...")
    shutdown_requested = True

# Load zeros from a file
def load_zeros_from_file(filename):
    zeros = []
    try:
        print(f"📂 Carregando zeros do arquivo: {filename}")
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, start=1):
                if shutdown_requested:
                    break
                line = line.strip()
                if line:
                    try:
                        zero = float(line)
                        zeros.append((line_num, zero))  # Format: (index, value)
                    except ValueError:
                        print(f"⚠️ Linha inválida {line_num}: '{line}'")
                        continue
        print(f"✅ {len(zeros):,} zeros carregados")
        return zeros
    except Exception as e:
        print(f"❌ Erro ao ler arquivo: {e}")
        return []

# Save zeros to cache
def save_enhanced_cache(zeros, backup=True):
    try:
        if backup and os.path.exists(CACHE_FILE):
            backup_file = f"{CACHE_FILE}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(CACHE_FILE, backup_file)
            print(f"📦 Backup do cache criado: {backup_file}")
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(zeros, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"💾 Cache salvo: {len(zeros)} zeros")
    except Exception as e:
        print(f"❌ Erro ao salvar cache: {e}")

# Load zeros from cache or file
def load_enhanced_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, list) and len(data) > 0:
                    print(f"✅ Cache válido: {len(data):,} zeros carregados")
                    return data
        except:
            print("⚠️ Cache inválido, carregando do arquivo...")
    zeros = load_zeros_from_file(ZEROS_FILE)
    if zeros:
        save_enhanced_cache(zeros)
        return zeros[:FRESH_START_ZEROS] if FRESH_START_ZEROS > 0 else zeros
    return []

# Find resonances at multiple tolerance levels
def find_multi_tolerance_resonances(zeros, constants_dict=None):
    if constants_dict is None:
        constants_dict = {'fine_structure': FINE_STRUCTURE}
    all_results = {}
    for const_name, const_value in constants_dict.items():
        all_results[const_name] = {}
        for tolerance in TOLERANCE_LEVELS:
            resonances = []
            for n, gamma in zeros:
                mod_val = gamma % const_value
                min_distance = min(mod_val, const_value - mod_val)
                if min_distance < tolerance:
                    resonances.append((n, gamma, min_distance, tolerance))
            all_results[const_name][tolerance] = resonances
    return all_results

# Enhanced statistical analysis
def enhanced_statistical_analysis(zeros, resonances, constant_value, tolerance):
    if len(zeros) == 0 or len(resonances) == 0:
        return None
    total_zeros = len(zeros)
    resonant_count = len(resonances)
    expected_random = total_zeros * (2 * tolerance / constant_value)
    results = {
        'basic_stats': {
            'total_zeros': total_zeros,
            'resonant_count': resonant_count,
            'expected_random': expected_random,
            'resonance_rate': resonant_count / total_zeros,
            'significance_factor': resonant_count / expected_random if expected_random > 0 else float('inf')
        }
    }
    if expected_random >= 5:
        chi2_stat = (resonant_count - expected_random)**2 / expected_random
        chi2_pvalue = 1 - stats.chi2.cdf(chi2_stat, df=1)
        results['chi2_test'] = {
            'statistic': chi2_stat,
            'p_value': chi2_pvalue,
            'critical_value_05': 3.841,
            'significant': chi2_stat > 3.841
        }
    p_expected = 2 * tolerance / constant_value
    try:
        binom_result = stats.binomtest(resonant_count, total_zeros, p_expected, alternative='two-sided')
        binom_pvalue = binom_result.pvalue
    except AttributeError:
        try:
            binom_pvalue = stats.binom_test(resonant_count, total_zeros, p_expected, alternative='two-sided')
        except AttributeError:
            from scipy.stats import binom
            binom_pvalue = 2 * min(binom.cdf(resonant_count, total_zeros, p_expected),
                                  1 - binom.cdf(resonant_count - 1, total_zeros, p_expected))
    results['binomial_test'] = {
        'p_value': binom_pvalue,
        'significant': binom_pvalue < 0.05
    }
    poisson_pvalue = 1 - stats.poisson.cdf(resonant_count - 1, expected_random)
    results['poisson_test'] = {
        'p_value': poisson_pvalue,
        'significant': poisson_pvalue < 0.05
    }
    if len(resonances) > 10:
        qualities = [r[2] for r in resonances]
        uniform_expected = stats.uniform(0, tolerance)
        ks_stat, ks_pvalue = kstest(qualities, uniform_expected.cdf)
        results['ks_uniformity'] = {
            'statistic': ks_stat,
            'p_value': ks_pvalue,
            'significant': ks_pvalue < 0.05
        }
        try:
            ad_stat, critical_values, _ = anderson(qualities, dist='norm')
            results['anderson_darling'] = {
                'statistic': ad_stat,
                'critical_values': critical_values,
                'normal_distribution': ad_stat < critical_values[2]
            }
        except:
            results['anderson_darling'] = None
    if len(resonances) > 20:
        indices = [r[0] for r in resonances]
        indices_sorted = sorted(indices)
        gaps = np.diff(indices_sorted)
        median_gap = np.median(gaps)
        runs_sequence = [1 if gap > median_gap else 0 for gap in gaps]
        if len(runs_sequence) > 10:
            try:
                def manual_runs_test(sequence):
                    n1 = sum(sequence)
                    n2 = len(sequence) - n1
                    if n1 == 0 or n2 == 0:
                        return 0, 1.0
                    runs = 1
                    for i in range(1, len(sequence)):
                        if sequence[i] != sequence[i-1]:
                            runs += 1
                    expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
                    variance = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))
                    if variance <= 0:
                        return 0, 1.0
                    z_score = (runs - expected_runs) / np.sqrt(variance)
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                    return z_score, p_value
                runs_stat, runs_pvalue = manual_runs_test(runs_sequence)
                results['runs_test'] = {
                    'statistic': runs_stat,
                    'p_value': runs_pvalue,
                    'random': runs_pvalue > 0.05
                }
            except Exception as e:
                print(f"    Aviso: Teste de corridas falhou: {e}")
                results['runs_test'] = None
    if len(resonances) > 30:
        qualities = [r[2] for r in sorted(resonances, key=lambda x: x[0])]
        autocorr = np.correlate(qualities, qualities, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]
        n = len(qualities)
        if n > 3:
            lag1_autocorr = autocorr[1] if len(autocorr) > 1 else 0
            autocorr_threshold = 1.96 / np.sqrt(n)
            results['autocorrelation'] = {
                'lag1_autocorr': lag1_autocorr,
                'threshold': autocorr_threshold,
                'significant': abs(lag1_autocorr) > autocorr_threshold
            }
    return results

# Compare resonances between different constants
def comparative_constant_analysis(zeros, tolerance=DEFAULT_TOLERANCE):
    multi_results = find_multi_tolerance_resonances(zeros, CONTROL_CONSTANTS)
    comparative_stats = {}
    for const_name, const_results in multi_results.items():
        if tolerance in const_results:
            resonances = const_results[tolerance]
            const_value = CONTROL_CONSTANTS[const_name]
            stats_result = enhanced_statistical_analysis(zeros, resonances, const_value, tolerance)
            comparative_stats[const_name] = {
                'constant_value': const_value,
                'resonance_count': len(resonances),
                'resonance_rate': len(resonances) / len(zeros) * 100,
                'stats': stats_result
            }
    return comparative_stats

# Analyze a batch of zeros
def analyze_batch_enhanced(zeros, batch_num):
    print(f"\n🔬 LOTE #{batch_num}: {len(zeros):,} zeros")
    if len(zeros) < MINIMUM_FOR_STATS:
        print(f"   📊 Necessário {MINIMUM_FOR_STATS - len(zeros):,} mais zeros para estatísticas")
        fine_structure_results = find_multi_tolerance_resonances(zeros, {'fine_structure': FINE_STRUCTURE})['fine_structure']
        tolerance_summary = {}
        for tolerance in TOLERANCE_LEVELS[:3]:
            if tolerance in fine_structure_results:
                resonances = fine_structure_results[tolerance]
                count = len(resonances)
                rate = count / len(zeros) * 100 if len(zeros) > 0 else 0
                tolerance_summary[tolerance] = {
                    'count': count,
                    'rate': rate,
                    'best_quality': min(r[2] for r in resonances) if count > 0 else None,
                    'resonances': resonances,
                    'stats': None
                }
        print(f"\n📊 RESSONÂNCIAS BÁSICAS:")
        print("| Tolerância | Contagem | Taxa (%) |")
        print("|------------|----------|----------|")
        for tolerance in TOLERANCE_LEVELS[:3]:
            if tolerance in tolerance_summary:
                data = tolerance_summary[tolerance]
                print(f"| {tolerance:8.0e} | {data['count']:8d} | {data['rate']:8.3f} |")
        return tolerance_summary, {}, None
    fine_structure_results = find_multi_tolerance_resonances(zeros, {'fine_structure': FINE_STRUCTURE})['fine_structure']
    print(f"\n📊 RESSONÂNCIAS POR TOLERÂNCIA:")
    print("| Tolerância | Contagem | Taxa (%) | Melhor Qualidade | Significância |")
    print("|------------|----------|----------|------------------|---------------|")
    tolerance_summary = {}
    for tolerance in TOLERANCE_LEVELS:
        if tolerance in fine_structure_results:
            resonances = fine_structure_results[tolerance]
            count = len(resonances)
            rate = count / len(zeros) * 100
            if count > 0:
                best_quality = min(r[2] for r in resonances)
                stats_result = enhanced_statistical_analysis(zeros, resonances, FINE_STRUCTURE, tolerance)
                sig_factor = stats_result['basic_stats']['significance_factor'] if stats_result else 0
                tolerance_summary[tolerance] = {
                    'count': count,
                    'rate': rate,
                    'best_quality': best_quality,
                    'significance': sig_factor,
                    'resonances': resonances,
                    'stats': stats_result
                }
                print(f"| {tolerance:8.0e} | {count:8d} | {rate:8.3f} | {best_quality:.3e} | {sig_factor:8.2f}x |")
            else:
                tolerance_summary[tolerance] = {
                    'count': 0, 'rate': 0, 'best_quality': None,
                    'significance': 0, 'resonances': [], 'stats': None
                }
                print(f"| {tolerance:8.0e} | {count:8d} | {rate:8.3f} |      N/A     |     N/A    |")
    print(f"\n🎛️ COMPARAÇÃO DE CONSTANTES (Tolerância: {DEFAULT_TOLERANCE}):")
    comparative_results = comparative_constant_analysis(zeros, DEFAULT_TOLERANCE)
    print("| Constante     | Valor      | Contagem | Taxa (%) | Significância |")
    print("|---------------|------------|----------|----------|---------------|")
    for const_name, results in comparative_results.items():
        count = results['resonance_count']
        rate = results['resonance_rate']
        sig_factor = results['stats']['basic_stats']['significance_factor'] if results['stats'] else 0
        value = results['constant_value']
        print(f"| {const_name:13s} | {value:10.6f} | {count:8d} | {rate:8.3f} | {sig_factor:8.2f}x |")
    best_overall = None
    if DEFAULT_TOLERANCE in tolerance_summary and tolerance_summary[DEFAULT_TOLERANCE]['resonances']:
        best_overall = min(tolerance_summary[DEFAULT_TOLERANCE]['resonances'], key=lambda x: x[2])
        print(f"\n💎 MELHOR RESSONÂNCIA (Tolerância: {DEFAULT_TOLERANCE}):")
        print(f"   Zero #{best_overall[0]:,} → γ = {best_overall[1]:.15f}")
        print(f"   γ mod α = {best_overall[2]:.15e}")
        print(f"   Energia prevista: {best_overall[1]/10:.2f} GeV")
    if DEFAULT_TOLERANCE in tolerance_summary and tolerance_summary[DEFAULT_TOLERANCE]['stats']:
        stats = tolerance_summary[DEFAULT_TOLERANCE]['stats']
        print(f"\n📈 TESTES ESTATÍSTICOS (Tolerância: {DEFAULT_TOLERANCE}):")
        if 'chi2_test' in stats:
            chi2 = stats['chi2_test']
            print(f"   Qui-quadrado: χ² = {chi2['statistic']:.3f}, p = {chi2['p_value']:.6f}, Significativo: {'Sim' if chi2['significant'] else 'Não'}")
        if 'binomial_test' in stats:
            binom = stats['binomial_test']
            print(f"   Binomial: p = {binom['p_value']:.6f}, Significativo: {'Sim' if binom['significant'] else 'Não'}")
        if 'ks_uniformity' in stats:
            ks = stats['ks_uniformity']
            print(f"   K-S: D = {ks['statistic']:.4f}, p = {ks['p_value']:.6f}, Uniforme: {'Não' if ks['significant'] else 'Sim'}")
    return tolerance_summary, comparative_results, best_overall

# Generate a comprehensive report
def generate_comprehensive_report(zeros, session_results, final_batch):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(RESULTS_DIR, f"Relatorio_Compreensivo_{timestamp}.txt")
    final_tolerance_analysis, final_comparative, final_best = analyze_batch_enhanced(zeros, final_batch)
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ZVT ESTRUTURA FINA HUNTER - RELATÓRIO COMPREENSIVO\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data: {datetime.now().isoformat()}\n")
        f.write(f"Versão: File Loader v2.1\n")
        f.write(f"Lote Final: #{final_batch}\n")
        f.write(f"Zeros Analisados: {len(zeros):,}\n")
        f.write(f"Arquivo de Origem: {ZEROS_FILE}\n\n")
        f.write("CONFIGURAÇÃO:\n")
        f.write(f"Constante de Estrutura Fina: {FINE_STRUCTURE:.15f}\n")
        f.write(f"Tolerâncias: {TOLERANCE_LEVELS}\n")
        f.write(f"Tolerância Padrão: {DEFAULT_TOLERANCE}\n")
        f.write(f"Precisão: {mp.dps} casas decimais\n\n")
        f.write("ANÁLISE MULTI-TOLERÂNCIA:\n")
        f.write("| Tolerância | Ressonâncias | Taxa (%) | Melhor Qualidade | Significância |\n")
        f.write("|------------|--------------|----------|------------------|---------------|\n")
        for tolerance in TOLERANCE_LEVELS:
            if tolerance in final_tolerance_analysis:
                data = final_tolerance_analysis[tolerance]
                best_quality = "N/A" if data['best_quality'] is None else f"{data['best_quality']:.3e}"
                significance = "N/A" if data['significance'] == 0 else f"{data['significance']:8.2f}"
                f.write(f"| {tolerance:8.0e} | {data['count']:10d} | {data['rate']:8.3f} | {best_quality:>11s} | {significance:>8s}x |\n")
        f.write(f"\nCOMPARAÇÃO DE CONSTANTES (Tolerância: {DEFAULT_TOLERANCE}):\n")
        f.write("| Constante     | Valor      | Ressonâncias | Taxa (%) | Significância |\n")
        f.write("|---------------|------------|--------------|----------|---------------|\n")
        for const_name, results in final_comparative.items():
            f.write(f"| {const_name:13s} | {results['constant_value']:10.6f} | {results['resonance_count']:10d} | {results['resonance_rate']:8.3f} | {results['stats']['basic_stats']['significance_factor']:8.2f}x |\n")
        if final_best:
            f.write(f"\nMELHOR RESSONÂNCIA:\n")
            f.write(f"Índice do Zero: #{final_best[0]:,}\n")
            f.write(f"Valor Gamma: {final_best[1]:.20f}\n")
            f.write(f"Qualidade: {final_best[2]:.20e}\n")
            f.write(f"Erro Relativo: {final_best[2]/FINE_STRUCTURE*100:.12f}%\n")
            f.write(f"Energia Prevista: {final_best[1]/10:.3f} GeV\n\n")
        if DEFAULT_TOLERANCE in final_tolerance_analysis and final_tolerance_analysis[DEFAULT_TOLERANCE]['stats']:
            stats = final_tolerance_analysis[DEFAULT_TOLERANCE]['stats']
            f.write("RESUMO ESTATÍSTICO:\n")
            basic = stats['basic_stats']
            f.write(f"  Zeros Totais: {basic['total_zeros']:,}\n")
            f.write(f"  Zeros Ressonantes: {basic['resonant_count']:,}\n")
            f.write(f"  Esperado por Acaso: {basic['expected_random']:.1f}\n")
            f.write(f"  Taxa de Ressonância: {basic['resonance_rate']*100:.6f}%\n")
            f.write(f"  Significância: {basic['significance_factor']:.3f}x\n")
            if 'chi2_test' in stats:
                f.write(f"  Qui-quadrado: χ² = {stats['chi2_test']['statistic']:.3f}, p = {stats['chi2_test']['p_value']:.6f}\n")
            if 'binomial_test' in stats:
                f.write(f"  Binomial: p = {stats['binomial_test']['p_value']:.6f}\n")
        f.write("\nRECOMENDAÇÕES:\n")
        f.write("1. Verificar resultados com diferentes tolerâncias\n")
        f.write("2. Investigar conexões com a constante de estrutura fina\n")
        f.write("3. Considerar cálculos com maior precisão\n")
        f.write("4. Analisar padrões de distribuição dos zeros\n")
        f.write("="*80 + "\n")
    print(f"📊 Relatório salvo: {report_file}")
    return report_file

# Main function
def infinite_hunter_file_loader():
    global shutdown_requested
    print(f"🚀 ZVT ESTRUTURA FINA HUNTER")
    print(f"α = {FINE_STRUCTURE:.12f}")
    print(f"Arquivo: {ZEROS_FILE}")
    print(f"Tolerâncias: {TOLERANCE_LEVELS}")
    print("🛑 Ctrl+C para parar")
    print("=" * 80)
    all_zeros = load_enhanced_cache()
    current_count = len(all_zeros)
    if current_count == 0:
        print("❌ Nenhum zero carregado. Verifique o arquivo.")
        return [], [], None
    print(f"📊 Zeros disponíveis: {current_count:,}")
    session_results = []
    best_overall = None
    batch_num = 1
    for i in range(0, current_count, INCREMENT):
        if shutdown_requested:
            break
        batch_start = i
        batch_end = min(i + INCREMENT, current_count)
        batch = all_zeros[batch_start:batch_end]
        print(f"\n🔬 LOTE #{batch_num}: Zeros {batch_start:,} a {batch_end:,}")
        start_time = time.time()
        tolerance_analysis, comparative_analysis, batch_best = analyze_batch_enhanced(batch, batch_num)
        if batch_best and (not best_overall or batch_best[2] < best_overall[2]):
            best_overall = batch_best
            print(f"    🎯 NOVO MELHOR GLOBAL!")
            print(f"    Zero #{best_overall[0]:,} → γ mod α = {best_overall[2]:.15e}")
        session_results.append({
            'batch': batch_num,
            'timestamp': datetime.now().isoformat(),
            'zeros_analyzed': batch_end,
            'batch_time': time.time() - start_time,
            'tolerance_analysis': tolerance_analysis,
            'comparative_analysis': comparative_analysis,
            'best_resonance': batch_best
        })
        batch_num += 1
    print(f"\n📊 Gerando relatório final...")
    generate_comprehensive_report(all_zeros, session_results, batch_num-1)
    return all_zeros, session_results, best_overall

# Main execution
if __name__ == "__main__":
    shutdown_requested = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    try:
        print("🌟 Iniciando ZVT Estrutura Fina Hunter")
        zeros, results, best = infinite_hunter_file_loader()
        if zeros and len(zeros) > 0:
            print(f"\n🎯 Análise Concluída!")
            print(f"📁 Resultados em: {RESULTS_DIR}/")
            print(f"💾 Cache: {CACHE_FILE}")
            print(f"📊 Estatísticas: {STATS_FILE}")
    except KeyboardInterrupt:
        print(f"\n⏹️ Análise interrompida. Progresso salvo.")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"\n🔬 Sessão concluída!")

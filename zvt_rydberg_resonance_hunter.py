#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZVT_RYDBERG_RESONANCE_HUNTER.py - Zeta/Rydberg resonance hunter with file loading
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
import math

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configuration
mp.dps = 50  # High precision

# Constante de Rydberg e quantidades relacionadas
R_INFINITY = 10973731.568160  # m⁻¹ (CODATA 2018)
C_LIGHT = 299792458  # m/s
H_PLANCK = 6.62607015e-34  # J⋅s
H_BAR = H_PLANCK / (2 * math.pi)
E_ELECTRON = 1.602176634e-19  # C
M_ELECTRON = 9.1093837015e-31  # kg
ALPHA = 7.2973525693e-3  # Estrutura fina

# Quantidades derivadas da constante de Rydberg
# Energia de Rydberg: E_Ry = hc R∞
E_RYDBERG = H_PLANCK * C_LIGHT * R_INFINITY  # ≈ 2.18e-18 J ≈ 13.6 eV

# Comprimento de onda característico: λ = 1/R∞
LAMBDA_RYDBERG = 1 / R_INFINITY  # ≈ 9.11e-8 m ≈ 91.1 nm

# Frequência de Rydberg: ν = c R∞
FREQ_RYDBERG = C_LIGHT * R_INFINITY  # ≈ 3.29e15 Hz

# Tempo característico: τ = 1/(c R∞)
TIME_RYDBERG = 1 / (C_LIGHT * R_INFINITY)  # ≈ 3.04e-16 s

# Escalas normalizadas para compatibilidade com zeros de Riemann
R_SCALED = R_INFINITY / 1e6      # ≈ 10.97 (escala dos zeros)
E_SCALED = E_RYDBERG * 1e18      # ≈ 2.18 (energia em escala)
FREQ_SCALED = FREQ_RYDBERG / 1e15  # ≈ 3.29 (frequência em escala)
LAMBDA_SCALED = LAMBDA_RYDBERG * 1e8  # ≈ 9.11 (comprimento de onda)

# Fatores relacionados às transições atômicas
LYMAN_FACTOR = R_INFINITY * (1 - 1/4) / 1e6    # Série de Lyman (≈ 8.23)
BALMER_FACTOR = R_INFINITY * (1/4 - 1/9) / 1e6  # Série de Balmer (≈ 1.52)
PASCHEN_FACTOR = R_INFINITY * (1/9 - 1/16) / 1e6 # Série de Paschen (≈ 0.53)

# Constante principal para análise
RYDBERG_CONSTANT = R_SCALED  # ≈ 10.97

TOLERANCE_LEVELS = [1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11]  # Tolerâncias relativas
DEFAULT_TOLERANCE = 1e-8
INCREMENT = 1000  # Batch size for processing

# Constantes de controle para validação estatística
CONTROL_CONSTANTS = {
    'rydberg_scaled': R_SCALED,          # ≈ 10.97
    'energy_scaled': E_SCALED,           # ≈ 2.18
    'freq_scaled': FREQ_SCALED,          # ≈ 3.29
    'lambda_scaled': LAMBDA_SCALED,      # ≈ 9.11
    'lyman_series': LYMAN_FACTOR,        # ≈ 8.23
    'balmer_series': BALMER_FACTOR,      # ≈ 1.52
    'paschen_series': PASCHEN_FACTOR,    # ≈ 0.53
    'alpha_scaled': ALPHA * 1000,        # ≈ 7.30
    'random_1': 10.5,
    'random_2': 11.3,
    'random_3': 8.7,
    'random_4': 12.4,
}

MAX_WORKERS = os.cpu_count()
CACHE_FILE = "zeta_zeros_cache.pkl"
STATS_FILE = "zvt_rydberg_stats.txt"
RESULTS_DIR = "zvt_rydberg_results"
ZEROS_FILE = os.path.expanduser("~/Downloads/zero.txt")  # Path to the zeros file

# Parameters for analysis
FRESH_START_ZEROS = 5000
MINIMUM_FOR_STATS = 1000

# Create results directory if it doesn't exist
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Global variable for shutdown control
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    print(f"\n⏸️ Shutdown solicitado. Completando lote atual e salvando...")
    shutdown_requested = True

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

def find_multi_tolerance_resonances(zeros, constants_dict=None):
    if constants_dict is None:
        constants_dict = {'rydberg_scaled': RYDBERG_CONSTANT}
    all_results = {}
    for const_name, const_value in constants_dict.items():
        all_results[const_name] = {}
        for tolerance in TOLERANCE_LEVELS:
            resonances = []
            for n, gamma in zeros:
                mod_val = gamma % const_value
                min_distance = min(mod_val, const_value - mod_val)
                relative_error = min_distance / const_value  # Erro relativo
                if relative_error < tolerance:
                    resonances.append((n, gamma, min_distance, tolerance, relative_error))
            all_results[const_name][tolerance] = resonances
    return all_results

def enhanced_statistical_analysis(zeros, resonances, constant_value, tolerance):
    if len(zeros) == 0 or len(resonances) == 0:
        return None
    total_zeros = len(zeros)
    resonant_count = len(resonances)
    expected_random = total_zeros * (2 * tolerance)  # Tolerância relativa
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
    p_expected = 2 * tolerance  # Tolerância já é relativa
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
    return results

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

def analyze_batch_enhanced(zeros, batch_num):
    print(f"\n🔬 LOTE #{batch_num}: {len(zeros):,} zeros")
    if len(zeros) < MINIMUM_FOR_STATS:
        print(f"   📊 Necessário {MINIMUM_FOR_STATS - len(zeros):,} mais zeros para estatísticas")
        rydberg_results = find_multi_tolerance_resonances(zeros, {'rydberg_scaled': RYDBERG_CONSTANT})['rydberg_scaled']
        tolerance_summary = {}
        for tolerance in TOLERANCE_LEVELS[:3]:
            if tolerance in rydberg_results:
                resonances = rydberg_results[tolerance]
                count = len(resonances)
                rate = count / len(zeros) * 100 if len(zeros) > 0 else 0
                tolerance_summary[tolerance] = {
                    'count': count,
                    'rate': rate,
                    'best_quality': min(r[4] for r in resonances) if count > 0 else None,  # Erro relativo
                    'resonances': resonances,
                    'stats': None
                }
        print(f"\n📊 RESSONÂNCIAS BÁSICAS (RYDBERG):")
        print("| Tolerância | Contagem | Taxa (%) |")
        print("|------------|----------|----------|")
        for tolerance in TOLERANCE_LEVELS[:3]:
            if tolerance in tolerance_summary:
                data = tolerance_summary[tolerance]
                print(f"| {tolerance:8.0e} | {data['count']:8d} | {data['rate']:8.3f} |")
        return tolerance_summary, {}, None
    
    rydberg_results = find_multi_tolerance_resonances(zeros, {'rydberg_scaled': RYDBERG_CONSTANT})['rydberg_scaled']
    print(f"\n📊 RESSONÂNCIAS RYDBERG POR TOLERÂNCIA (R∞/10⁶ = {RYDBERG_CONSTANT:.3f}):")
    print("| Tolerância | Contagem | Taxa (%) | Melhor Qualidade | Significância |")
    print("|------------|----------|----------|------------------|---------------|")
    tolerance_summary = {}
    for tolerance in TOLERANCE_LEVELS:
        if tolerance in rydberg_results:
            resonances = rydberg_results[tolerance]
            count = len(resonances)
            rate = count / len(zeros) * 100
            if count > 0:
                best_quality = min(r[4] for r in resonances)  # Erro relativo
                stats_result = enhanced_statistical_analysis(zeros, resonances, RYDBERG_CONSTANT, tolerance)
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
    
    print(f"\n🎛️ COMPARAÇÃO DE PARÂMETROS RYDBERG (Tolerância: {DEFAULT_TOLERANCE}):")
    comparative_results = comparative_constant_analysis(zeros, DEFAULT_TOLERANCE)
    print("| Parâmetro     | Valor          | Contagem | Taxa (%) | Significância |")
    print("|---------------|----------------|----------|----------|---------------|")
    for const_name, results in comparative_results.items():
        count = results['resonance_count']
        rate = results['resonance_rate']
        sig_factor = results['stats']['basic_stats']['significance_factor'] if results['stats'] else 0
        value = results['constant_value']
        value_str = f"{value:.3f}"
        print(f"| {const_name:13s} | {value_str:14s} | {count:8d} | {rate:8.3f} | {sig_factor:8.2f}x |")
    
    best_overall = None
    if DEFAULT_TOLERANCE in tolerance_summary and tolerance_summary[DEFAULT_TOLERANCE]['resonances']:
        best_overall = min(tolerance_summary[DEFAULT_TOLERANCE]['resonances'], key=lambda x: x[4])  # Por erro relativo
        print(f"\n💎 MELHOR RESSONÂNCIA RYDBERG (Tolerância: {DEFAULT_TOLERANCE}):")
        print(f"   Zero #{best_overall[0]:,} → γ = {best_overall[1]:.15f}")
        print(f"   γ mod (R∞/10⁶) = {best_overall[2]:.15e}")
        print(f"   Erro relativo: {best_overall[4]:.15e} ({best_overall[4]*100:.12f}%)")
        print(f"   Fator espectroscópico: γ = {best_overall[1]/RYDBERG_CONSTANT:.6f} × (R∞/10⁶)")
        print(f"   Número quântico efetivo: n_eff ≈ {math.sqrt(best_overall[1]/RYDBERG_CONSTANT):.3f}")
        print(f"   Energia correspondente: E ≈ -{13.6 * (best_overall[1]/RYDBERG_CONSTANT):.3f} eV")
        print(f"   Comprimento de onda: λ ≈ {91.1 / (best_overall[1]/RYDBERG_CONSTANT):.3f} nm")
    
    if DEFAULT_TOLERANCE in tolerance_summary and tolerance_summary[DEFAULT_TOLERANCE]['stats']:
        stats = tolerance_summary[DEFAULT_TOLERANCE]['stats']
        print(f"\n📈 TESTES ESTATÍSTICOS RYDBERG (Tolerância: {DEFAULT_TOLERANCE}):")
        if 'chi2_test' in stats:
            chi2 = stats['chi2_test']
            print(f"   Qui-quadrado: χ² = {chi2['statistic']:.3f}, p = {chi2['p_value']:.6f}, Significativo: {'Sim' if chi2['significant'] else 'Não'}")
        if 'binomial_test' in stats:
            binom = stats['binomial_test']
            print(f"   Binomial: p = {binom['p_value']:.6f}, Significativo: {'Sim' if binom['significant'] else 'Não'}")
    
    return tolerance_summary, comparative_results, best_overall

def generate_comprehensive_report(zeros, session_results, final_batch):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(RESULTS_DIR, f"Relatorio_Rydberg_{timestamp}.txt")
    final_tolerance_analysis, final_comparative, final_best = analyze_batch_enhanced(zeros, final_batch)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ZVT RYDBERG RESONANCE HUNTER - RELATÓRIO COMPREENSIVO\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data: {datetime.now().isoformat()}\n")
        f.write(f"Versão: Rydberg Hunter v1.0\n")
        f.write(f"Lote Final: #{final_batch}\n")
        f.write(f"Zeros Analisados: {len(zeros):,}\n")
        f.write(f"Arquivo de Origem: {ZEROS_FILE}\n\n")
        f.write("CONFIGURAÇÃO RYDBERG:\n")
        f.write(f"Constante de Rydberg (R∞): {R_INFINITY:.15f} m⁻¹\n")
        f.write(f"Energia de Rydberg: {E_RYDBERG:.15e} J ({E_RYDBERG/E_ELECTRON:.6f} eV)\n")
        f.write(f"Comprimento de onda de Rydberg: {LAMBDA_RYDBERG:.15e} m ({LAMBDA_RYDBERG*1e9:.1f} nm)\n")
        f.write(f"Frequência de Rydberg: {FREQ_RYDBERG:.15e} Hz\n")
        f.write(f"Tempo de Rydberg: {TIME_RYDBERG:.15e} s\n")
        f.write(f"Série de Lyman (fator): {LYMAN_FACTOR:.15f}\n")
        f.write(f"Série de Balmer (fator): {BALMER_FACTOR:.15f}\n")
        f.write(f"Série de Paschen (fator): {PASCHEN_FACTOR:.15f}\n")
        f.write(f"Tolerâncias: {TOLERANCE_LEVELS}\n")
        f.write(f"Tolerância Padrão: {DEFAULT_TOLERANCE}\n")
        f.write(f"Precisão: {mp.dps} casas decimais\n\n")
        f.write("ANÁLISE MULTI-TOLERÂNCIA (RYDBERG):\n")
        f.write("| Tolerância | Ressonâncias | Taxa (%) | Melhor Qualidade | Significância |\n")
        f.write("|------------|--------------|----------|------------------|---------------|\n")
        for tolerance in TOLERANCE_LEVELS:
            if tolerance in final_tolerance_analysis:
                data = final_tolerance_analysis[tolerance]
                best_quality = "N/A" if data['best_quality'] is None else f"{data['best_quality']:.3e}"
                significance = "N/A" if data['significance'] == 0 else f"{data['significance']:8.2f}"
                f.write(f"| {tolerance:8.0e} | {data['count']:10d} | {data['rate']:8.3f} | {best_quality:>11s} | {significance:>8s}x |\n")
        
        f.write(f"\nCOMPARAÇÃO DE PARÂMETROS RYDBERG (Tolerância: {DEFAULT_TOLERANCE}):\n")
        f.write("| Parâmetro     | Valor              | Ressonâncias | Taxa (%) | Significância |\n")
        f.write("|---------------|--------------------|--------------|----------|---------------|\n")
        for const_name, results in final_comparative.items():
            value_str = f"{results['constant_value']:.6f}"
            sig_factor = results['stats']['basic_stats']['significance_factor'] if results['stats'] else 0
            f.write(f"| {const_name:13s} | {value_str:18s} | {results['resonance_count']:10d} | {results['resonance_rate']:8.3f} | {sig_factor:8.2f}x |\n")
        
        if final_best:
            f.write(f"\nMELHOR RESSONÂNCIA RYDBERG:\n")
            f.write(f"Índice do Zero: #{final_best[0]:,}\n")
            f.write(f"Valor Gamma: {final_best[1]:.20f}\n")
            f.write(f"Qualidade Absoluta: {final_best[2]:.20e}\n")
            f.write(f"Qualidade Relativa: {final_best[4]:.20e}\n")
            f.write(f"Erro Relativo: {final_best[4]*100:.12f}%\n")
            f.write(f"Fator Espectroscópico: {final_best[1]/RYDBERG_CONSTANT:.6e}\n")
            f.write(f"Número Quântico Efetivo: {math.sqrt(final_best[1]/RYDBERG_CONSTANT):.6f}\n")
            f.write(f"Energia Atômica: {-13.6 * (final_best[1]/RYDBERG_CONSTANT):.6f} eV\n")
            f.write(f"Comprimento de Onda: {91.1 / (final_best[1]/RYDBERG_CONSTANT):.3f} nm\n")
            f.write(f"Relação Espectral: γ = {final_best[1]/RYDBERG_CONSTANT:.3e} × (R∞/10⁶)\n\n")
        else:
            f.write(f"\nNENHUMA RESSONÂNCIA RYDBERG ENCONTRADA nas tolerâncias testadas.\n\n")
        
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
        
        f.write("\nIMPLICAÇÕES PARA FÍSICA ATÔMICA:\n")
        f.write("1. Investigar conexões entre zeros de Riemann e espectroscopia atômica\n")
        f.write("2. Analisar relações com estrutura fina dos níveis de energia\n")
        f.write("3. Considerar implicações para teoria quântica do átomo de hidrogênio\n")
        f.write("4. Explorar conexões com séries espectrais e números quânticos\n")
        f.write("5. Avaliar potencial para predições espectrais baseadas em zeros\n")
        f.write("="*80 + "\n")
    
    print(f"📊 Relatório Rydberg salvo: {report_file}")
    return report_file

def infinite_hunter_rydberg():
    global shutdown_requested
    print(f"🚀 ZVT RYDBERG RESONANCE HUNTER")
    print(f"⚛️ Constante de Rydberg R∞ = {R_INFINITY:.6f} m⁻¹")
    print(f"🔬 Parâmetro Principal = {RYDBERG_CONSTANT:.3f}")
    print(f"📁 Arquivo: {ZEROS_FILE}")
    print(f"📊 Tolerâncias: {TOLERANCE_LEVELS}")
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
        if batch_best and (not best_overall or batch_best[4] < best_overall[4]):  # Comparar por erro relativo
            best_overall = batch_best
            print(f"    🎯 NOVO MELHOR GLOBAL RYDBERG!")
            print(f"    Zero #{best_overall[0]:,} → γ mod R∞ = {best_overall[2]:.15e}")
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
    
    print(f"\n📊 Gerando relatório final Rydberg...")
    generate_comprehensive_report(all_zeros, session_results, batch_num-1)
    return all_zeros, session_results, best_overall

def main():
    """Função principal"""
    global shutdown_requested
    print("🌟 Iniciando ZVT Rydberg Resonance Hunter")
    
    try:
        zeros, results, best = infinite_hunter_rydberg()
        
        if zeros and len(zeros) > 0:
            print(f"\n🎯 Análise Rydberg Concluída!")
            print(f"📁 Resultados em: {RESULTS_DIR}/")
            print(f"💾 Cache: {CACHE_FILE}")
            print(f"📊 Estatísticas: {STATS_FILE}")
            
            if best:
                print(f"🏆 Melhor ressonância: Zero #{best[0]:,} com qualidade relativa {best[4]:.3e}")
                print(f"⚛️ Fator espectroscópico: γ = {best[1]/RYDBERG_CONSTANT:.3e} × (R∞/10⁶)")
                print(f"🔬 Número quântico efetivo: n_eff ≈ {math.sqrt(best[1]/RYDBERG_CONSTANT):.3f}")
                print(f"💡 Energia: E ≈ {-13.6 * (best[1]/RYDBERG_CONSTANT):.3f} eV")
            else:
                print(f"📊 Nenhuma ressonância Rydberg excepcional encontrada")
                print(f"🤔 Zeros de Riemann podem ser independentes da espectroscopia atômica")
                
    except KeyboardInterrupt:
        print(f"\n⏹️ Análise interrompida. Progresso salvo.")
        
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        print(f"\n🔬 Sessão Rydberg concluída!")

if __name__ == "__main__":
    shutdown_requested = False
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
